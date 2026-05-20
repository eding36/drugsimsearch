from __future__ import annotations

import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from molsimsearch.dataset import MoleculeDataset, PropertyScaler, SimilarityAwareSampler, scaffold_split
from molsimsearch.loss import CombinedSimilarityLoss
from molsimsearch.model import MolGAT


def make_scheduler(
    optimizer: Adam,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_factor: float = 0.01,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_factor, cosine)

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: MolGAT,
    loader: DataLoader,
    optimizer: Adam,
    loss_fn: CombinedSimilarityLoss,
    scaler: PropertyScaler,
    grad_clip: float,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    total_tan = 0.0
    total_prop = 0.0
    total_pred = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        props_raw = batch.props.clone()
        batch.props = scaler.transform(batch.props)

        optimizer.zero_grad()
        embeddings = model(batch)
        loss, metrics = loss_fn(embeddings, batch.fp, props_raw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_tan += metrics["tanimoto_sim_mean"]
        total_prop += metrics["prop_sim_mean"]
        total_pred += metrics["pred_sim_mean"]
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "tanimoto_sim_mean": total_tan / n,
        "prop_sim_mean": total_prop / n,
        "pred_sim_mean": total_pred / n,
    }


def evaluate(
    model: MolGAT,
    loader: DataLoader,
    loss_fn: CombinedSimilarityLoss,
    scaler: PropertyScaler,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_pred, all_target = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            props_raw = batch.props.clone()
            batch.props = scaler.transform(batch.props)

            embeddings = model(batch)
            loss, _ = loss_fn(embeddings, batch.fp, props_raw)
            total_loss += loss.item()
            n_batches += 1

            n = embeddings.size(0)
            if n > 1:
                cos_sim = embeddings @ embeddings.T
                tan_mat = loss_fn._tanimoto_matrix(batch.fp)
                prop_mat = loss_fn._property_cosine_matrix(props_raw)
                target = loss_fn.tanimoto_weight * tan_mat + loss_fn.property_weight * prop_mat
                mask = loss_fn._upper_tri_mask(n, device)
                all_pred.append(cos_sim[mask].cpu().numpy())
                all_target.append(target[mask].cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    spearman_rho, pearson_r = 0.0, 0.0
    if all_pred:
        pred_arr = np.concatenate(all_pred)
        target_arr = np.concatenate(all_target)
        if len(pred_arr) > 1:
            spearman_rho = spearmanr(pred_arr, target_arr).statistic
            pearson_r = pearsonr(pred_arr, target_arr)[0]

    return {"loss": avg_loss, "spearman_rho": float(spearman_rho), "pearson_r": float(pearson_r)}


def compute_retrieval_map(
    model: MolGAT,
    dataset: MoleculeDataset,
    scaler: PropertyScaler,
    device: torch.device,
    tanimoto_threshold: float = 0.4,
    batch_size: int = 256,
    chunk_size: int = 512,
) -> float:
    """Vectorised MAP computation — processes `chunk_size` queries at a time to
    avoid building the full (n×n) matrix in memory."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embs, all_fps = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.props = scaler.transform(batch.props)
            all_embs.append(model(batch).cpu())
            all_fps.append(batch.fp.cpu())

    embs = torch.cat(all_embs, dim=0).float()   # (n, 128)
    fps  = torch.cat(all_fps,  dim=0).float()   # (n, 2048)
    fp_norms = fps.sum(dim=1)                   # (n,)
    n = embs.size(0)

    all_ap: list[float] = []

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            q_embs = embs[start:end].to(device)   # (c, 128)
            q_fps  = fps[start:end].to(device)    # (c, 2048)
            q_norms = fp_norms[start:end].to(device)

            db_embs  = embs.to(device)            # (n, 128)
            db_fps   = fps.to(device)             # (n, 2048)
            db_norms = fp_norms.to(device)

            # Cosine similarity — embeddings are already L2-normalised
            cos = q_embs @ db_embs.T              # (c, n)

            # Tanimoto
            inter = q_fps @ db_fps.T              # (c, n)
            union = q_norms.unsqueeze(1) + db_norms.unsqueeze(0) - inter
            tan = inter / (union + 1e-8)          # (c, n)

            # Mask self-scores
            for local_i in range(end - start):
                global_i = start + local_i
                cos[local_i, global_i] = -2.0
                tan[local_i, global_i] = 0.0

            pos_mask = (tan > tanimoto_threshold).float()   # (c, n)
            n_pos = pos_mask.sum(dim=1)                     # (c,)

            # Precision@rank for each hit, then sum and normalise
            # AP_i = sum_j( pos_mask[i,j] * cumhits[i,j] / rank[i,j] ) / n_pos_i
            # cumhits[i,j] = number of positives ranked above or at rank[i,j]
            # = (pos_mask[i] * (ranks[i] <= ranks[i,j])).sum() — expensive
            # Standard shortcut: sort by rank, cumsum, divide
            sort_idx = torch.argsort(cos, dim=1, descending=True)  # (c, n)
            sorted_pos = pos_mask.gather(1, sort_idx)               # (c, n)
            cumhits = torch.cumsum(sorted_pos, dim=1)               # (c, n)
            rank_range = torch.arange(1, n + 1, device=device).float().unsqueeze(0)
            precision_at_hit = (cumhits / rank_range) * sorted_pos  # (c, n)
            ap_scores = precision_at_hit.sum(dim=1) / n_pos.clamp(min=1)  # (c,)

            for local_i in range(end - start):
                if n_pos[local_i].item() > 0:
                    all_ap.append(ap_scores[local_i].item())

    return float(np.mean(all_ap)) if all_ap else 0.0


def plot_loss_curves(history: dict, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(len(history["train_loss"]))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["spearman"], label="Spearman ρ", color="green")
    ax2.plot(epochs, history["pearson"], label="Pearson r", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Correlation")
    ax2.set_title("Embedding Quality")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_checkpoint(
    path: str,
    epoch: int,
    model: MolGAT,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: PropertyScaler,
    val_loss: float,
    cfg: dict,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_loss": val_loss,
            "cfg": cfg,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: MolGAT,
    optimizer: Adam | None = None,
    scheduler: LambdaLR | None = None,
    scaler: PropertyScaler | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"], ckpt["val_loss"]


def train(cfg: dict) -> None:
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    l_cfg = cfg["loss"]
    d_cfg = cfg["data"]
    log_cfg = cfg["logging"]

    torch.manual_seed(t_cfg["seed"])
    np.random.seed(t_cfg["seed"])
    random.seed(t_cfg["seed"])

    if cfg.get("device", "auto") == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    dataset = MoleculeDataset(
        smiles_file=d_cfg["smiles_file"],
        fp_radius=l_cfg["fp_radius"],
        fp_nbits=l_cfg["fp_nbits"],
        smiles_col=d_cfg.get("smiles_col"),
        cache_dir=d_cfg.get("cache_dir", ".cache"),
    )
    print(f"Loaded {dataset.num_molecules} molecules")

    train_set, val_set = scaffold_split(dataset, t_cfg["val_split"], t_cfg["seed"])
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    use_sim_sampling = t_cfg.get("similarity_sampling", True) and len(train_set) >= 1000
    if use_sim_sampling:
        sampler = SimilarityAwareSampler(
            train_set,
            batch_size=t_cfg["batch_size"],
            n_clusters=t_cfg.get("sampler_clusters", 2000),
            mols_per_cluster=t_cfg.get("sampler_mols_per_cluster", 4),
            similar_fraction=t_cfg.get("sampler_similar_fraction", 0.25),
            seed=t_cfg["seed"],
        )
        train_loader = DataLoader(
            train_set, batch_sampler=sampler, num_workers=0, pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=t_cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
        )
    val_loader = DataLoader(
        val_set, batch_size=t_cfg["batch_size"], shuffle=False, num_workers=0
    )

    train_indices = train_set.indices
    all_props = np.stack([dataset[i].props.squeeze(0).numpy() for i in train_indices])
    scaler = PropertyScaler().fit(all_props)

    model = MolGAT.from_config(m_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, t_cfg["warmup_epochs"], t_cfg["epochs"])
    loss_fn = CombinedSimilarityLoss(
        tanimoto_weight=l_cfg["tanimoto_weight"],
        property_weight=l_cfg["property_weight"],
        use_upper_tri=l_cfg.get("use_upper_tri", True),
    )

    start_epoch = 0
    if cfg.get("resume"):
        start_epoch, _ = load_checkpoint(cfg["resume"], model, optimizer, scheduler, scaler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    save_dir = log_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Random-batch train eval loader (same distribution as val — for comparable loss)
    train_eval_loader = DataLoader(
        train_set, batch_size=t_cfg["batch_size"], shuffle=True, num_workers=0
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "spearman": [], "pearson": []}

    for epoch in range(start_epoch, t_cfg["epochs"]):
        train_epoch(model, train_loader, optimizer, loss_fn, scaler, t_cfg["grad_clip"], device)
        val_metrics = evaluate(model, val_loader, loss_fn, scaler, device)
        train_eval_metrics = evaluate(model, train_eval_loader, loss_fn, scaler, device)
        scheduler.step()

        history["train_loss"].append(train_eval_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["spearman"].append(val_metrics["spearman_rho"])
        history["pearson"].append(val_metrics["pearson_r"])

        if epoch % log_cfg.get("log_every", 1) == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train={train_eval_metrics['loss']:.4f} | "
                f"val={val_metrics['loss']:.4f} | "
                f"ρ={val_metrics['spearman_rho']:.3f} | "
                f"r={val_metrics['pearson_r']:.3f} | "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                os.path.join(save_dir, "best.ckpt"),
                epoch, model, optimizer, scheduler, scaler, best_val_loss, cfg,
            )

        if (epoch + 1) % log_cfg.get("checkpoint_every", 10) == 0:
            save_checkpoint(
                os.path.join(save_dir, f"epoch_{epoch:04d}.ckpt"),
                epoch, model, optimizer, scheduler, scaler, val_metrics["loss"], cfg,
            )

        if (epoch + 1) % log_cfg.get("plot_every", 25) == 0:
            plot_loss_curves(history, os.path.join(save_dir, "loss_curves.png"))

        map_every = log_cfg.get("map_every", 0)
        if map_every and (epoch + 1) % map_every == 0:
            tan_thresh = l_cfg.get("map_tanimoto_threshold", 0.6)
            map_score = compute_retrieval_map(model, dataset, scaler, device, tanimoto_threshold=tan_thresh)
            print(f"  MAP@(Tanimoto>{tan_thresh}): {map_score:.4f}")

    plot_loss_curves(history, os.path.join(save_dir, "loss_curves.png"))
    print(f"\nBest val loss: {best_val_loss:.4f}")

    tan_thresh = l_cfg.get("map_tanimoto_threshold", 0.6)
    print("Computing retrieval MAP...")
    map_score = compute_retrieval_map(model, dataset, scaler, device, tanimoto_threshold=tan_thresh)
    print(f"MAP@(Tanimoto>{tan_thresh}): {map_score:.4f}")
