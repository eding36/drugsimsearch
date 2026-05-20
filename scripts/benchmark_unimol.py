#!/usr/bin/env python3
"""
DUD benchmark: Uni-Mol vs MolGAT vs Tanimoto.

Usage:
    python scripts/benchmark_unimol.py \
        --checkpoint runs/zinc250k_remote/best.ckpt \
        --benchmark-dir benchmarking_platform
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from rdkit.ML.Scoring import Scoring as RDKitScoring
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from torch_geometric.loader import DataLoader

from molsimsearch.dataset import MoleculeDataset, PropertyScaler
from molsimsearch.model import MolGAT


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dat_gz(path: str) -> list[str]:
    smiles = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                smiles.append(parts[2])
    return smiles


def get_dud_targets(benchmark_dir: str) -> list[dict]:
    import glob
    base = os.path.join(benchmark_dir, "compounds", "DUD")
    targets = []
    for act_path in sorted(glob.glob(os.path.join(base, "*_actives.dat.gz"))):
        name = os.path.basename(act_path).replace("cmp_list_", "").replace("_actives.dat.gz", "")
        dec_path = act_path.replace("_actives.dat.gz", "_decoys.dat.gz")
        if os.path.exists(dec_path):
            targets.append({"name": name, "actives": act_path, "decoys": dec_path})
    return targets


# ── MolGAT embedding ──────────────────────────────────────────────────────────

def embed_molgat(smiles_list, model, scaler, fp_radius, fp_nbits, device, batch_size=256):
    dataset = MoleculeDataset.from_smiles_list(smiles_list, fp_radius, fp_nbits)
    if len(dataset) == 0:
        return torch.zeros(0, 128), torch.zeros(0, fp_nbits), []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, fps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.props = scaler.transform(batch.props)
            embs.append(model(batch).cpu())
            fps.append(batch.fp.cpu())
    valid = [d.smiles for d in dataset._data_list]
    return torch.cat(embs).float(), torch.cat(fps).float(), valid


# ── Uni-Mol embedding ─────────────────────────────────────────────────────────

def embed_unimol(smiles_list: list[str], unimol_repr, batch_size=64) -> tuple[np.ndarray, list[str]]:
    """Embed molecules with Uni-Mol, return (n, emb_dim) array and valid SMILES."""
    valid_smiles, all_embs = [], []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        try:
            out = unimol_repr.get_repr(batch, return_atomic_reprs=False)
            embs = np.array(out)  # list of (512,) arrays → (batch, 512)
            all_embs.append(embs)
            valid_smiles.extend(batch)
        except Exception:
            # Fall back one-by-one to skip bad molecules
            for smi in batch:
                try:
                    out = unimol_repr.get_repr([smi], return_atomic_reprs=False)
                    all_embs.append(np.array(out))
                    valid_smiles.append(smi)
                except Exception:
                    pass

    if not all_embs:
        return np.zeros((0, 512)), []

    embs = np.concatenate(all_embs, axis=0)
    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms, valid_smiles


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_metrics(scores: np.ndarray, labels: list[int]) -> dict | None:
    n, n_act = len(labels), sum(labels)
    if n_act == 0 or n_act == n:
        return None

    order = np.argsort(scores)[::-1]
    sorted_labels = np.array(labels)[order]

    auc = roc_auc_score(labels, scores)

    def ef_at(pct):
        n_top = max(1, int(n * pct / 100))
        return (sorted_labels[:n_top].sum() / n_act) / (n_top / n)

    scored = sorted(zip(scores.tolist(), labels), key=lambda x: -x[0])
    bedroc = RDKitScoring.CalcBEDROC([(s, l) for s, l in scored], col=1, alpha=20.0)

    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    tpr_at_1fpr = float(np.interp(0.01, fpr_arr, tpr_arr))

    return {
        "auc":         auc,
        "ef1":         ef_at(1),
        "ef5":         ef_at(5),
        "bedroc":      bedroc,
        "tpr_at_1fpr": tpr_at_1fpr,
        "avg_prec":    average_precision_score(labels, scores),
    }


def evaluate_target_three_way(act_molgat, act_fps, dec_molgat, dec_fps,
                               act_unimol, dec_unimol):
    """Leave-one-out evaluation comparing MolGAT, Uni-Mol, and Tanimoto."""
    n_act = len(act_molgat)
    if n_act < 2:
        return None

    keys = ["auc", "ef1", "ef5", "bedroc", "tpr_at_1fpr", "avg_prec"]
    accum = {k: {"molgat": [], "unimol": [], "tanimoto": []} for k in keys}

    for i in range(n_act):
        # MolGAT pool
        pool_mg  = torch.cat([act_molgat[:i], act_molgat[i+1:], dec_molgat])
        pool_fps = torch.cat([act_fps[:i],    act_fps[i+1:],    dec_fps])
        # Uni-Mol pool
        pool_um  = np.concatenate([act_unimol[:i], act_unimol[i+1:], dec_unimol], axis=0)

        labels = [1] * (n_act - 1) + [0] * len(dec_molgat)
        if len(set(labels)) < 2:
            continue

        q_mg  = act_molgat[i:i+1]
        q_fp  = act_fps[i:i+1]
        q_um  = act_unimol[i:i+1]

        # MolGAT cosine (L2-normed tensors)
        mg_scores  = (q_mg @ pool_mg.T).squeeze(0).numpy()
        # Tanimoto
        inter = (q_fp @ pool_fps.T).squeeze(0)
        union = q_fp.sum() + pool_fps.sum(dim=1) - inter
        tan_scores = (inter / (union + 1e-8)).numpy()
        # Uni-Mol cosine (already L2-normed)
        um_scores  = (q_um @ pool_um.T).squeeze(0)

        for method, scores in [("molgat", mg_scores),
                                ("unimol", um_scores),
                                ("tanimoto", tan_scores)]:
            m = calc_metrics(scores, labels)
            if m:
                for k in keys:
                    accum[k][method].append(m[k])

    if not accum["auc"]["molgat"]:
        return None

    result = {"n_actives": n_act, "n_decoys": len(dec_molgat)}
    for k in keys:
        for method in ("molgat", "unimol", "tanimoto"):
            vals = accum[k][method]
            result[f"{method}_{k}"] = np.mean(vals) if vals else 0.0
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--benchmark-dir", default="benchmarking_platform")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = torch.device(args.device)

    # Load MolGAT
    print("Loading MolGAT...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MolGAT.from_config(ckpt["cfg"]["model"]).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    scaler = PropertyScaler()
    scaler.load_state_dict(ckpt["scaler_state"])
    fp_cfg = ckpt["cfg"]["loss"]
    fp_radius, fp_nbits = fp_cfg["fp_radius"], fp_cfg["fp_nbits"]
    print(f"  Epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

    # Load Uni-Mol
    print("Loading Uni-Mol...")
    from unimol_tools import UniMolRepr
    unimol = UniMolRepr(data_type="molecule", remove_hs=False)
    print("  Uni-Mol ready")

    targets = get_dud_targets(args.benchmark_dir)
    print(f"\nRunning on {len(targets)} DUD targets...\n")

    hdr = (f"{'Target':<18}"
           f"  {'MolGAT':>7} {'UniMol':>7} {'Tanimoto':>9}"
           f"  {'MG_EF1':>7} {'UM_EF1':>7} {'TAN_EF1':>8}"
           f"  {'MG_BEDROC':>10} {'UM_BEDROC':>10} {'TAN_BEDROC':>11}"
           f"  {'ΔMG-Tan':>8} {'ΔUM-Tan':>8}")
    print(hdr)
    print("-" * len(hdr))

    all_results = []

    for t in targets:
        act_smiles = load_dat_gz(t["actives"])
        dec_smiles = load_dat_gz(t["decoys"])

        # MolGAT embeddings
        act_mg, act_fps, act_valid = embed_molgat(
            act_smiles, model, scaler, fp_radius, fp_nbits, device, args.batch_size)
        dec_mg, dec_fps, dec_valid = embed_molgat(
            dec_smiles, model, scaler, fp_radius, fp_nbits, device, args.batch_size)

        if len(act_mg) < 2 or len(dec_mg) == 0:
            print(f"{t['name']:<18}  skipped")
            continue

        # Uni-Mol embeddings (use same valid SMILES as MolGAT for fair comparison)
        print(f"  Embedding {t['name']} with Uni-Mol ({len(act_valid)} act, {len(dec_valid)} dec)...",
              end="\r", flush=True)
        act_um, _ = embed_unimol(act_valid, unimol, batch_size=64)
        dec_um, _ = embed_unimol(dec_valid, unimol, batch_size=64)

        if len(act_um) != len(act_mg) or len(dec_um) != len(dec_mg):
            print(f"{t['name']:<18}  skipped (Uni-Mol embedding mismatch)")
            continue

        res = evaluate_target_three_way(act_mg, act_fps, dec_mg, dec_fps, act_um, dec_um)
        if res is None:
            continue

        res["name"] = t["name"]
        all_results.append(res)

        d_mg  = res["molgat_auc"]   - res["tanimoto_auc"]
        d_um  = res["unimol_auc"]   - res["tanimoto_auc"]
        print(
            f"{t['name']:<18}"
            f"  {res['molgat_auc']:>7.3f} {res['unimol_auc']:>7.3f} {res['tanimoto_auc']:>9.3f}"
            f"  {res['molgat_ef1']:>7.2f} {res['unimol_ef1']:>7.2f} {res['tanimoto_ef1']:>8.2f}"
            f"  {res['molgat_bedroc']:>10.3f} {res['unimol_bedroc']:>10.3f} {res['tanimoto_bedroc']:>11.3f}"
            f"  {d_mg:>+8.3f} {d_um:>+8.3f}"
        )

    if not all_results:
        print("No results.")
        return

    def avg(k): return np.mean([r[k] for r in all_results])

    print("\n" + "=" * len(hdr))
    print(f"\n{'OVERALL':<18}"
          f"  {avg('molgat_auc'):>7.3f} {avg('unimol_auc'):>7.3f} {avg('tanimoto_auc'):>9.3f}"
          f"  {avg('molgat_ef1'):>7.2f} {avg('unimol_ef1'):>7.2f} {avg('tanimoto_ef1'):>8.2f}"
          f"  {avg('molgat_bedroc'):>10.3f} {avg('unimol_bedroc'):>10.3f} {avg('tanimoto_bedroc'):>11.3f}"
          f"  {avg('molgat_auc')-avg('tanimoto_auc'):>+8.3f}"
          f" {avg('unimol_auc')-avg('tanimoto_auc'):>+8.3f}")
    print(f"\n({len(all_results)}/{len(targets)} targets evaluated)")


if __name__ == "__main__":
    main()
