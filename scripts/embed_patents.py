#!/usr/bin/env python3
"""
Embed patent SMILES with best.ckpt, compute pairwise cosine similarities,
and generate a UMAP cluster plot colored by patent_id.

Usage:
    cd /Users/eding36/VSCodeProjects/drugSimSearch
    python3 scripts/embed_patents.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molsimsearch.dataset import PropertyScaler
from molsimsearch.featurizer import mol_to_pyg_data
from molsimsearch.model import MolGAT

CHECKPOINT = "runs/zinc250k_remote/best.ckpt"
SMILES_CSV = "/Users/eding36/Downloads/combined_decimer_smiles_2026-05-17__simple.csv"
OUTPUT_DIR = "runs/patent_embeddings"
BATCH_SIZE = 64
FP_RADIUS = 2
FP_NBITS = 2048


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = MolGAT.from_config(cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    scaler = PropertyScaler()
    scaler.load_state_dict(ckpt["scaler_state"])
    return model, scaler


_RGROUP_PAT = re.compile(r'\[R[^\]]*\]|\[X[^\]]*\]|\[R\]')


def _strip_rgroups(smi: str) -> str:
    """Replace R-group tokens with [H] so dangling bonds get a hydrogen cap."""
    return _RGROUP_PAT.sub("[H]", smi)


def _best_mol(smi: str):
    """
    Parse a (possibly dot-separated) SMILES.
    For each fragment, try as-is first; if that fails, try after stripping R-groups.
    Returns the largest valid RDKit Mol, or None.
    """
    candidates = []
    for part in smi.split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            mol = Chem.MolFromSmiles(_strip_rgroups(part))
        if mol is not None:
            candidates.append(mol)
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.GetNumHeavyAtoms())


def preprocess_smiles(smiles_csv: str) -> tuple[list[str], list[str], list[str]]:
    """Load CSV, strip R-groups where needed, canonicalize + dedup.
    Returns (canonical_smiles, original_smiles, patent_ids)."""
    df = pd.read_csv(smiles_csv)
    print(f"Total rows: {len(df)}")

    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    df = df.dropna(subset=["smiles"])
    df = df[df["smiles"].str.strip() != ""]
    print(f"After error/null filter: {len(df)} rows")

    valid_canon: list[str] = []
    valid_original: list[str] = []
    valid_patents: list[str] = []
    rgroup_rescued = 0
    invalid = 0

    for _, row in df.iterrows():
        raw_smi = str(row["smiles"]).strip()
        pid = str(row["patent_id"])

        mol = _best_mol(raw_smi)
        if mol is None:
            invalid += 1
            continue

        has_rgroup = bool(_RGROUP_PAT.search(raw_smi))
        if has_rgroup:
            rgroup_rescued += 1

        canon = Chem.MolToSmiles(mol)
        valid_canon.append(canon)
        valid_original.append(raw_smi)
        valid_patents.append(pid)

    print(f"Valid: {len(valid_canon)}  (incl. {rgroup_rescued} R-group-stripped)  |  still invalid: {invalid}")

    seen: set[str] = set()
    dedup_canon: list[str] = []
    dedup_original: list[str] = []
    dedup_patents: list[str] = []
    for canon, orig, pid in zip(valid_canon, valid_original, valid_patents):
        if canon not in seen:
            seen.add(canon)
            dedup_canon.append(canon)
            dedup_original.append(orig)
            dedup_patents.append(pid)

    print(f"Unique canonical SMILES: {len(dedup_canon)}")
    return dedup_canon, dedup_original, dedup_patents


def build_data_list(canon_list: list[str], original_list: list[str], patents_list: list[str]):
    """Featurize canonical SMILES; return (data_list, filtered_canon, filtered_original, filtered_patents)."""
    data_list, filtered_canon, filtered_original, filtered_patents = [], [], [], []
    for canon, orig, pid in zip(canon_list, original_list, patents_list):
        d = mol_to_pyg_data(canon, FP_RADIUS, FP_NBITS)
        if d is not None:
            data_list.append(d)
            filtered_canon.append(canon)
            filtered_original.append(orig)
            filtered_patents.append(pid)
    print(f"Featurized: {len(data_list)} molecules")
    return data_list, filtered_canon, filtered_original, filtered_patents


@torch.no_grad()
def embed(data_list, model: MolGAT, scaler: PropertyScaler, device: torch.device) -> np.ndarray:
    from torch_geometric.data import Batch

    all_embs = []
    for start in range(0, len(data_list), BATCH_SIZE):
        chunk = data_list[start : start + BATCH_SIZE]
        batch = Batch.from_data_list(chunk).to(device)
        batch.props = scaler.transform(batch.props)
        emb = model(batch)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0).numpy()


def plot_umap(embeddings: np.ndarray, patents: list[str], out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) - 1), min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    unique_patents = sorted(set(patents))
    cmap = cm.get_cmap("tab10" if len(unique_patents) <= 10 else "tab20")
    color_map = {p: cmap(i / max(len(unique_patents) - 1, 1)) for i, p in enumerate(unique_patents)}

    fig, ax = plt.subplots(figsize=(13, 9))
    for p in unique_patents:
        idxs = [i for i, pid in enumerate(patents) if pid == p]
        ax.scatter(
            coords[idxs, 0], coords[idxs, 1],
            c=[color_map[p]], label=p,
            alpha=0.75, s=35, edgecolors="none",
        )

    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, framealpha=0.85, title="Patent ID")
    ax.set_title("Patent SMILES — UMAP (MolGAT embeddings)", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cluster plot: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    canon_list, original_list, patents_list = preprocess_smiles(SMILES_CSV)
    data_list, emb_canon, emb_original, emb_patents = build_data_list(canon_list, original_list, patents_list)

    print("Loading model...")
    model, scaler = load_model(CHECKPOINT, device)

    print("Embedding...")
    embeddings = embed(data_list, model, scaler, device)
    print(f"Embedding shape: {embeddings.shape}")

    # Save embeddings CSV
    emb_df = pd.DataFrame({"patent_id": emb_patents, "original": emb_original, "canonical": emb_canon})
    emb_cols = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    emb_df = pd.concat([emb_df, emb_cols], axis=1)
    emb_csv = os.path.join(OUTPUT_DIR, "embeddings.csv")
    emb_df.to_csv(emb_csv, index=False)
    print(f"Embeddings: {emb_csv}")

    # Pairwise cosine similarity (L2-normed → dot product)
    print("Computing pairwise similarities...")
    n = len(embeddings)
    cos_mat = embeddings @ embeddings.T  # (n, n)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "patent_id_1": emb_patents[i],
                "original_1": emb_original[i],
                "canonical_1": emb_canon[i],
                "patent_id_2": emb_patents[j],
                "original_2": emb_original[j],
                "canonical_2": emb_canon[j],
                "cosine_similarity": float(cos_mat[i, j]),
            })

    sim_df = pd.DataFrame(rows)
    sim_csv = os.path.join(OUTPUT_DIR, "pairwise_similarity.csv")
    sim_df.to_csv(sim_csv, index=False)
    print(f"Pairwise similarities: {sim_csv}  ({len(sim_df):,} pairs)")

    # Top 10 cross-patent similar pairs
    cross = sim_df[sim_df["patent_id_1"] != sim_df["patent_id_2"]]
    top10 = cross.nlargest(10, "cosine_similarity")
    print("\nTop 10 cross-patent similar pairs:")
    print(top10[["patent_id_1", "patent_id_2", "cosine_similarity", "canonical_1", "canonical_2"]].to_string(index=False))

    # UMAP plot
    print("\nGenerating UMAP cluster plot...")
    plot_umap(embeddings, emb_patents, os.path.join(OUTPUT_DIR, "umap_by_patent.png"))
    print("Done.")


if __name__ == "__main__":
    main()
