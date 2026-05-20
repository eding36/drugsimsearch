#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

# Allow running as `python scripts/train.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MolGAT molecular embedding model")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data", default=None, help="Path to SMILES file")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    overrides: dict = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.data is not None:
        overrides.setdefault("data", {})["smiles_file"] = args.data
    if args.save_dir is not None:
        overrides.setdefault("logging", {})["save_dir"] = args.save_dir
    if args.device != "auto":
        overrides["device"] = args.device
    if args.resume is not None:
        overrides["resume"] = args.resume

    cfg = deep_update(cfg, overrides)
    cfg.setdefault("device", "auto")

    save_dir = cfg["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    from molsimsearch.train import train
    train(cfg)


if __name__ == "__main__":
    main()
