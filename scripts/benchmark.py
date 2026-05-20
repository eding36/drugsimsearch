#!/usr/bin/env python3
"""
Riniker & Landrum (2013) benchmark — evaluate model vs Tanimoto baseline.

Usage:
    python scripts/benchmark.py \
        --checkpoint runs/zinc250k_remote/best.ckpt \
        --benchmark-dir benchmarking_platform
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
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


def get_targets(benchmark_dir: str) -> list[dict]:
    """Return list of {name, actives_path, decoys_path} for DUD, MUV, ChEMBL."""
    import glob
    targets = []

    # DUD and MUV: per-target decoys
    for dataset in ("DUD", "MUV"):
        base = os.path.join(benchmark_dir, "compounds", dataset)
        for act_path in sorted(glob.glob(os.path.join(base, "*_actives.dat.gz"))):
            name = os.path.basename(act_path).replace("cmp_list_", "").replace("_actives.dat.gz", "")
            dec_path = act_path.replace("_actives.dat.gz", "_decoys.dat.gz")
            if os.path.exists(dec_path):
                targets.append({"name": name, "actives": act_path, "decoys": dec_path, "dataset": dataset})

    # ChEMBL: shared decoy pool
    chembl_base = os.path.join(benchmark_dir, "compounds", "ChEMBL")
    chembl_decoys = os.path.join(chembl_base, "cmp_list_ChEMBL_zinc_decoys.dat.gz")
    if os.path.exists(chembl_decoys):
        for act_path in sorted(glob.glob(os.path.join(chembl_base, "*_actives.dat.gz"))):
            name = os.path.basename(act_path).replace("cmp_list_", "").replace("_actives.dat.gz", "")
            targets.append({"name": name, "actives": act_path, "decoys": chembl_decoys, "dataset": "ChEMBL"})

    return targets


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_smiles(smiles_list: list[str], model, scaler, fp_radius, fp_nbits, device, batch_size=256):
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
    valid_smiles = [d.smiles for d in dataset._data_list]
    return torch.cat(embs).float(), torch.cat(fps).float(), valid_smiles


# ── Similarity metrics ────────────────────────────────────────────────────────

def tanimoto_scores(q_fp: torch.Tensor, ref_fps: torch.Tensor) -> np.ndarray:
    inter = (q_fp @ ref_fps.T).squeeze(0)
    union = q_fp.sum() + ref_fps.sum(dim=1) - inter
    return (inter / (union + 1e-8)).numpy()


def cosine_scores(q_emb: torch.Tensor, ref_embs: torch.Tensor) -> np.ndarray:
    return (q_emb @ ref_embs.T).squeeze(0).numpy()


def combined_scores(q_emb, ref_embs, q_fp, ref_fps, w=0.8) -> np.ndarray:
    tan = tanimoto_scores(q_fp, ref_fps)
    cos = cosine_scores(q_emb, ref_embs)
    return w * tan + (1 - w) * cos


# ── Per-target evaluation ─────────────────────────────────────────────────────

def calc_metrics(scores: np.ndarray, labels: list[int]) -> dict:
    """Compute AUC, EF@1%, EF@5%, BEDROC(α=20), TPR@1%FPR, AvgPrecision."""
    n = len(labels)
    n_act = sum(labels)
    if n_act == 0 or n_act == n:
        return None

    # Sort by score descending — required for EF and BEDROC
    order = np.argsort(scores)[::-1]
    sorted_labels = np.array(labels)[order]

    # AUC-ROC
    auc = roc_auc_score(labels, scores)

    # EF@k%: (actives in top k%) / (expected actives in top k% at random)
    def ef_at(pct):
        n_top = max(1, int(n * pct / 100))
        hits = sorted_labels[:n_top].sum()
        return (hits / n_act) / (n_top / n)

    ef1  = ef_at(1)
    ef5  = ef_at(5)

    # BEDROC α=20 (weights early hits heavily)
    # RDKit expects list of (score, label) sorted descending
    scored = [(float(scores[i]), int(labels[i])) for i in range(n)]
    scored.sort(key=lambda x: -x[0])
    bedroc = RDKitScoring.CalcBEDROC(scored, col=1, alpha=20.0)

    # TPR at 1% FPR
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    tpr_at_1fpr = float(np.interp(0.01, fpr_arr, tpr_arr))

    # Average precision (area under precision-recall curve)
    avg_prec = average_precision_score(labels, scores)

    return {
        "auc":         auc,
        "ef1":         ef1,
        "ef5":         ef5,
        "bedroc":      bedroc,
        "tpr_at_1fpr": tpr_at_1fpr,
        "avg_prec":    avg_prec,
    }


def evaluate_target(act_embs, act_fps, dec_embs, dec_fps):
    """Leave-one-out evaluation: AUC, EF@1%, EF@5%, BEDROC, TPR@1%FPR, AvgPrec."""
    n_act = len(act_embs)
    if n_act < 2:
        return None

    keys = ["auc", "ef1", "ef5", "bedroc", "tpr_at_1fpr", "avg_prec"]
    accum = {k: {"model": [], "tanimoto": [], "combined": []} for k in keys}

    for i in range(n_act):
        q_emb = act_embs[i:i+1]
        q_fp  = act_fps[i:i+1]

        pool_embs = torch.cat([act_embs[:i], act_embs[i+1:], dec_embs])
        pool_fps  = torch.cat([act_fps[:i],  act_fps[i+1:],  dec_fps])
        labels    = [1] * (n_act - 1) + [0] * len(dec_embs)

        for method, scores in [
            ("model",    cosine_scores(q_emb, pool_embs)),
            ("tanimoto", tanimoto_scores(q_fp, pool_fps)),
            ("combined", combined_scores(q_emb, pool_embs, q_fp, pool_fps)),
        ]:
            m = calc_metrics(scores, labels)
            if m:
                for k in keys:
                    accum[k][method].append(m[k])

    if not accum["auc"]["model"]:
        return None

    result = {"n_actives": n_act, "n_decoys": len(dec_embs)}
    for k in keys:
        for method in ("model", "tanimoto", "combined"):
            vals = accum[k][method]
            result[f"{method}_{k}"] = np.mean(vals) if vals else 0.0
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--benchmark-dir", default="benchmarking_platform")
    p.add_argument("--datasets", nargs="+", default=["DUD", "MUV", "ChEMBL"],
                   help="Which datasets to run (default: all three)")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MolGAT.from_config(ckpt["cfg"]["model"]).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    scaler = PropertyScaler()
    scaler.load_state_dict(ckpt["scaler_state"])
    fp_cfg = ckpt["cfg"]["loss"]
    fp_radius, fp_nbits = fp_cfg["fp_radius"], fp_cfg["fp_nbits"]
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}\n")

    targets = [t for t in get_targets(args.benchmark_dir) if t["dataset"] in args.datasets]
    print(f"Targets: {len(targets)} ({', '.join(args.datasets)})\n")

    results_by_dataset: dict[str, list[dict]] = {"DUD": [], "MUV": [], "ChEMBL": []}
    all_results = []

    # Per-target header
    hdr = (f"{'Target':<35} {'Act':>5} {'Dec':>6} "
           f"  {'AUC':>6} {'EF@1%':>6} {'EF@5%':>6} {'BEDROC':>7} {'TPR@1F':>7} {'AvgP':>6}"
           f"  {'TAN_AUC':>8} {'TAN_EF1':>8} {'TAN_BEDROC':>11}"
           f"  {'Δ AUC':>7}")
    print(hdr)
    print("-" * len(hdr))

    for t in targets:
        act_smiles = load_dat_gz(t["actives"])
        dec_smiles = load_dat_gz(t["decoys"])

        act_embs, act_fps, _ = embed_smiles(act_smiles, model, scaler, fp_radius, fp_nbits, device, args.batch_size)
        dec_embs, dec_fps, _ = embed_smiles(dec_smiles, model, scaler, fp_radius, fp_nbits, device, args.batch_size)

        if len(act_embs) < 2 or len(dec_embs) == 0:
            print(f"{t['name']:<35}  skipped (too few valid molecules)")
            continue

        res = evaluate_target(act_embs, act_fps, dec_embs, dec_fps)
        if res is None:
            continue

        res["name"] = t["name"]
        res["dataset"] = t["dataset"]
        all_results.append(res)
        results_by_dataset[t["dataset"]].append(res)

        d_auc = res["model_auc"] - res["tanimoto_auc"]
        print(
            f"{t['name']:<35} {res['n_actives']:>5} {res['n_decoys']:>6}"
            f"  {res['model_auc']:>6.3f} {res['model_ef1']:>6.2f} {res['model_ef5']:>6.2f}"
            f" {res['model_bedroc']:>7.3f} {res['model_tpr_at_1fpr']:>7.3f} {res['model_avg_prec']:>6.3f}"
            f"  {res['tanimoto_auc']:>8.3f} {res['tanimoto_ef1']:>8.2f} {res['tanimoto_bedroc']:>11.3f}"
            f"  {d_auc:>+7.3f}"
        )

    # Summary per dataset
    metrics = ["auc", "ef1", "ef5", "bedroc", "tpr_at_1fpr", "avg_prec"]
    methods = ["model", "tanimoto", "combined"]

    print(f"\n{'='*100}")
    print(f"\n{'Dataset':<10}  {'N':>4}  "
          f"{'Mod_AUC':>8} {'Mod_EF1':>8} {'Mod_EF5':>8} {'Mod_BEDROC':>11} {'Mod_TPR1F':>10} {'Mod_AvgP':>9}  "
          f"{'Tan_AUC':>8} {'Tan_EF1':>8} {'Tan_BEDROC':>11}  "
          f"{'ΔAUC':>6}")
    print("-" * 100)

    for ds in ("DUD", "MUV", "ChEMBL"):
        rs = results_by_dataset[ds]
        if not rs:
            continue
        def avg(key): return np.mean([r[key] for r in rs])
        print(
            f"{ds:<10}  {len(rs):>4}  "
            f"{avg('model_auc'):>8.3f} {avg('model_ef1'):>8.2f} {avg('model_ef5'):>8.2f}"
            f" {avg('model_bedroc'):>11.3f} {avg('model_tpr_at_1fpr'):>10.3f} {avg('model_avg_prec'):>9.3f}  "
            f"{avg('tanimoto_auc'):>8.3f} {avg('tanimoto_ef1'):>8.2f} {avg('tanimoto_bedroc'):>11.3f}  "
            f"{avg('model_auc')-avg('tanimoto_auc'):>+6.3f}"
        )

    if all_results:
        rs = all_results
        def avg(key): return np.mean([r[key] for r in rs])
        print("-" * 100)
        print(
            f"{'OVERALL':<10}  {len(rs):>4}  "
            f"{avg('model_auc'):>8.3f} {avg('model_ef1'):>8.2f} {avg('model_ef5'):>8.2f}"
            f" {avg('model_bedroc'):>11.3f} {avg('model_tpr_at_1fpr'):>10.3f} {avg('model_avg_prec'):>9.3f}  "
            f"{avg('tanimoto_auc'):>8.3f} {avg('tanimoto_ef1'):>8.2f} {avg('tanimoto_bedroc'):>11.3f}  "
            f"{avg('model_auc')-avg('tanimoto_auc'):>+6.3f}"
        )

    print(f"\nDone. {len(all_results)}/{len(targets)} targets evaluated.")


if __name__ == "__main__":
    main()
