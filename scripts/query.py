#!/usr/bin/env python3
"""
Query a molecule against a reference set and rank by similarity.

Usage:
    python scripts/query.py \
        --checkpoint runs/zinc250k_remote/best.ckpt \
        --query "CC(=O)Oc1ccccc1C(=O)O" \
        --reference data/sample.smi \
        --top-k 10
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch_geometric.loader import DataLoader

from molsimsearch.dataset import MoleculeDataset, PropertyScaler
from molsimsearch.featurizer import get_rdkit_properties
from molsimsearch.model import MolGAT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query molecule similarity against a reference set")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--query", required=True, help="SMILES string of the query molecule")
    p.add_argument("--reference", required=True, help="Path to reference .smi or .csv file")
    p.add_argument("--smiles-col", default=None, help="Column name if CSV input")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MolGAT.from_config(ckpt["cfg"]["model"]).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    scaler = PropertyScaler()
    scaler.load_state_dict(ckpt["scaler_state"])
    fp_cfg = ckpt["cfg"]["loss"]
    return model, scaler, fp_cfg["fp_radius"], fp_cfg["fp_nbits"]


def embed(smiles_list: list[str], model, scaler, fp_radius, fp_nbits, device, batch_size=256):
    dataset = MoleculeDataset.from_smiles_list(smiles_list, fp_radius, fp_nbits)
    valid_smiles = [d.smiles for d in dataset._data_list]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, fps, props = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.props = scaler.transform(batch.props)
            embs.append(model(batch).cpu())
            fps.append(batch.fp.cpu())
            props.append(batch.props.cpu())
    return (
        torch.cat(embs).float(),
        torch.cat(fps).float(),
        torch.cat(props).float(),
        valid_smiles,
    )


def tanimoto(q_fp: torch.Tensor, ref_fps: torch.Tensor) -> torch.Tensor:
    inter = q_fp @ ref_fps.T
    union = q_fp.sum() + ref_fps.sum(dim=1) - inter
    return (inter / (union + 1e-8)).squeeze(0)


def property_cosine(q_props: torch.Tensor, ref_props: torch.Tensor) -> torch.Tensor:
    q_n = F.normalize(q_props, p=2, dim=1)
    r_n = F.normalize(ref_props, p=2, dim=1)
    return (q_n @ r_n.T).squeeze(0)


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Validate query
    mol = Chem.MolFromSmiles(args.query)
    if mol is None:
        print(f"Invalid query SMILES: {args.query}")
        sys.exit(1)

    model, scaler, fp_radius, fp_nbits = load_model(args.checkpoint, device)

    # Load reference SMILES
    if args.reference.endswith(".csv") or args.reference.endswith(".tsv"):
        sep = "\t" if args.reference.endswith(".tsv") else ","
        df = pd.read_csv(args.reference, sep=sep)
        col = args.smiles_col or df.columns[0]
        ref_smiles = df[col].dropna().tolist()
    else:
        with open(args.reference) as f:
            ref_smiles = [l.strip().split()[0] for l in f if l.strip()]

    print(f"Embedding query + {len(ref_smiles)} reference molecules...")
    q_emb, q_fp, q_props, _ = embed([args.query], model, scaler, fp_radius, fp_nbits, device)
    r_emb, r_fp, r_props, valid_ref = embed(ref_smiles, model, scaler, fp_radius, fp_nbits, device)

    # Similarities
    model_sim  = (q_emb @ r_emb.T).squeeze(0)                    # cosine (L2-normed)
    tan_sim    = tanimoto(q_fp, r_fp)                             # structural only
    prop_sim   = property_cosine(q_props, r_props)                # property only
    combined   = 0.8 * tan_sim + 0.2 * prop_sim                  # ground-truth target

    top_k = min(args.top_k, len(valid_ref))
    top_idx = torch.argsort(model_sim, descending=True)[:top_k]

    print(f"\nQuery: {args.query}")
    print(f"{'Rank':<5} {'Model':>7} {'Tanimoto':>10} {'PropCos':>9} {'Combined':>10}  SMILES")
    print("-" * 90)
    for rank, idx in enumerate(top_idx, 1):
        i = idx.item()
        print(
            f"{rank:<5} {model_sim[i]:>7.3f} {tan_sim[i]:>10.3f} "
            f"{prop_sim[i]:>9.3f} {combined[i]:>10.3f}  {valid_ref[i]}"
        )

    # Summary statistics vs full reference distribution
    print(f"\nSimilarity to reference distribution (n={len(valid_ref)}):")
    for name, scores in [("Model cosine", model_sim), ("Tanimoto", tan_sim), ("Prop cosine", prop_sim)]:
        print(f"  {name:<16}  mean={scores.mean():.3f}  max={scores.max():.3f}  "
              f"p90={scores.quantile(0.9):.3f}  p99={scores.quantile(0.99):.3f}")


if __name__ == "__main__":
    main()
