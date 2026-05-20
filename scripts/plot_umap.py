#!/usr/bin/env python3
"""
UMAP of patent SMILES using Morgan FP (Tanimoto/Jaccard), coloured by patent_id.

Usage:
    cd /Users/eding36/VSCodeProjects/drugSimSearch
    python3 scripts/plot_umap.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
from rdkit import Chem
from rdkit.Chem import AllChem

SMILES_CSV = "/Users/eding36/Downloads/combined_decimer_smiles_2026-05-17__simple.csv"
OUTPUT     = "runs/patent_embeddings/umap_tanimoto_by_patent.png"

_RGROUP_PAT = re.compile(r'\[R[^\]]*\]|\[X[^\]]*\]|\[R\]')


def _best_mol(smi: str):
    candidates = []
    for part in smi.split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            mol = Chem.MolFromSmiles(_RGROUP_PAT.sub("[H]", part))
        if mol is not None:
            candidates.append(mol)
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.GetNumHeavyAtoms())


def main():
    df = pd.read_csv(SMILES_CSV)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    df = df.dropna(subset=["smiles"])

    fps, patents = [], []
    for _, row in df.iterrows():
        mol = _best_mol(str(row["smiles"]).strip())
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(np.frombuffer(bv.ToBitString().encode(), dtype="u1") - ord("0"))
        patents.append(str(row["patent_id"]))

    # Deduplicate by FP
    seen, dedup_fps, dedup_patents = set(), [], []
    for fp, pid in zip(fps, patents):
        key = fp.tobytes()
        if key not in seen:
            seen.add(key)
            dedup_fps.append(fp)
            dedup_patents.append(pid)

    fp_arr = np.stack(dedup_fps).astype(np.float32)
    print(f"Running UMAP on {len(fp_arr)} molecules (jaccard)...")

    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=min(15, len(fp_arr) - 1),
                        min_dist=0.1, metric="jaccard")
    coords = reducer.fit_transform(fp_arr)

    unique_patents = sorted(set(dedup_patents))
    cmap = cm.tab10 if len(unique_patents) <= 10 else cm.tab20
    color_map = {p: cmap(i / max(len(unique_patents) - 1, 1))
                 for i, p in enumerate(unique_patents)}

    fig, ax = plt.subplots(figsize=(12, 9))
    for pid in unique_patents:
        mask = [i for i, p in enumerate(dedup_patents) if p == pid]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[pid]], label=pid,
                   alpha=0.75, s=40, edgecolors="none")

    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, framealpha=0.85, title="Patent ID")
    ax.set_title(f"Patent SMILES — UMAP (Tanimoto FP)  n={len(fp_arr)}", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
