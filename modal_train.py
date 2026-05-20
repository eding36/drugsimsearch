#!/usr/bin/env python3
"""
Modal training script – train MolGAT on ZINC250k on a remote A10G GPU.

Start training:
    MODAL_PROFILE=chemicalbinding modal run modal_train.py
    MODAL_PROFILE=chemicalbinding modal run modal_train.py --epochs 100
    MODAL_PROFILE=chemicalbinding modal run modal_train.py --epochs 50 --batch-size 512

Download checkpoints after training:
    MODAL_PROFILE=chemicalbinding modal run modal_train.py::download_results
"""
from __future__ import annotations

import modal

APP_NAME = "drugsim-molgat"
REMOTE_DIR = "/root/drugSimSearch"
RUNS_DIR = "/root/runs"
LOCAL_PROJECT = "/Users/eding36/VSCodeProjects/drugSimSearch"

app = modal.App(APP_NAME)

# ── Container image (deps + project source + data) ─────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .run_commands(
        "pip install torch-geometric",
        "pip install torch-scatter torch-sparse "
        "-f https://data.pyg.org/whl/torch-2.4.1+cu121.html",
    )
    .pip_install(
        "rdkit>=2023.3.1",
        "numpy>=1.24",
        "pandas>=2.0",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "scipy>=1.11",
    )
    # Source code + data added at container startup (no image rebuild on code changes)
    .add_local_dir(
        LOCAL_PROJECT,
        remote_path=REMOTE_DIR,
        ignore=["runs", ".cache", "**/__pycache__", "**/*.pyc", ".git", ".eggs"],
    )
)

# ── Persistent volume for checkpoints + molecule cache ─────────────────────────
runs_vol = modal.Volume.from_name("drugsim-runs", create_if_missing=True)


# ── Training function ──────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,  # 12-hour ceiling
    volumes={RUNS_DIR: runs_vol},
)
def train_zinc250k(epochs: int = 200, batch_size: int = 256, resume: str | None = None) -> None:
    import os
    import sys

    sys.path.insert(0, REMOTE_DIR)

    import yaml

    from molsimsearch.train import train

    with open(f"{REMOTE_DIR}/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["smiles_file"] = f"{REMOTE_DIR}/data/zinc250k.csv"
    cfg["data"]["smiles_col"] = "smiles"
    cfg["data"]["cache_dir"] = f"{RUNS_DIR}/.cache"
    cfg["training"]["epochs"] = epochs
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["lr"] = 3e-4
    cfg["training"]["warmup_epochs"] = 5
    cfg["training"]["grad_clip"] = 1.0
    cfg["logging"]["save_dir"] = f"{RUNS_DIR}/zinc250k"
    cfg["logging"]["checkpoint_every"] = 5
    cfg["logging"]["log_every"] = 1
    cfg["logging"]["plot_every"] = 10
    cfg["device"] = "auto"

    if resume:
        cfg["resume"] = resume

    os.makedirs(f"{RUNS_DIR}/zinc250k", exist_ok=True)
    train(cfg)
    runs_vol.commit()


# ── Main entrypoint (default) ──────────────────────────────────────────────────
@app.local_entrypoint()
def main(epochs: int = 200, batch_size: int = 256, resume: str = ""):
    print("Launching MolGAT training on Modal (A10G GPU)")
    print(f"  Dataset : ZINC250k (249,455 molecules)")
    print(f"  Epochs  : {epochs}   Batch size: {batch_size}")
    print(f"  Output  : drugsim-runs volume → zinc250k/")
    train_zinc250k.remote(epochs=epochs, batch_size=batch_size, resume=resume or None)


# ── Download entrypoint (explicit) ────────────────────────────────────────────
# Usage: MODAL_PROFILE=chemicalbinding modal run modal_train.py::download_results
def download_results():
    import os

    dest = f"{LOCAL_PROJECT}/runs/zinc250k_remote"
    os.makedirs(dest, exist_ok=True)

    for entry in runs_vol.listdir("zinc250k"):
        name = entry.path.split("/")[-1]
        if name.endswith(".ckpt") or name in ("loss_curves.png", "config.yaml"):
            with open(f"{dest}/{name}", "wb") as fout:
                for chunk in runs_vol.read_file(f"zinc250k/{name}"):
                    fout.write(chunk)
            print(f"Downloaded: {name}")

    print(f"\nCheckpoints saved to {dest}")
