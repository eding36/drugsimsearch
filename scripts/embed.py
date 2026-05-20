#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molsimsearch.dataset import MoleculeDataset, PropertyScaler
from molsimsearch.model import MolGAT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed molecules using a trained MolGAT checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--smiles", required=True, help="Path to .smi or .csv file")
    p.add_argument("--output", default="embeddings.csv", help="Output .csv or .npy file")
    p.add_argument("--smiles-col", default=None, help="Column name if CSV input")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def embed_molecules(
    smiles_list: list[str],
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "auto",
) -> tuple[np.ndarray, list[str]]:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MolGAT.from_config(ckpt["cfg"]["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    scaler = PropertyScaler()
    scaler.load_state_dict(ckpt["scaler_state"])

    dataset = MoleculeDataset.from_smiles_list(smiles_list)
    valid_smiles = [d.smiles for d in dataset._data_list]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            batch.props = scaler.transform(batch.props)
            emb = model(batch)
            all_embs.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embs) if all_embs else np.empty((0, 128))
    return embeddings, valid_smiles


def main() -> None:
    args = parse_args()

    if args.smiles.endswith(".csv") or args.smiles.endswith(".tsv"):
        sep = "\t" if args.smiles.endswith(".tsv") else ","
        df = pd.read_csv(args.smiles, sep=sep)
        col = args.smiles_col or df.columns[0]
        smiles_list = df[col].dropna().tolist()
    else:
        with open(args.smiles) as f:
            smiles_list = [line.strip().split()[0] for line in f if line.strip()]

    embeddings, valid_smiles = embed_molecules(
        smiles_list, args.checkpoint, args.batch_size, args.device
    )

    print(f"Embedded {len(valid_smiles)} / {len(smiles_list)} molecules")

    if args.output.endswith(".npy"):
        np.save(args.output, embeddings)
        print(f"Saved embeddings to {args.output}")
    else:
        cols = ["smiles"] + [f"emb_{i}" for i in range(embeddings.shape[1])]
        out_df = pd.DataFrame(
            np.column_stack([valid_smiles, embeddings]),
            columns=cols,
        )
        out_df.to_csv(args.output, index=False)
        print(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
    main()
