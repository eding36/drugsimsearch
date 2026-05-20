# drugSimSearch — MolGAT Molecular Similarity Search

A Graph Attention Network that embeds molecules into a 128-dimensional space where cosine similarity reflects chemical similarity. With this package, you can embed (embed.py), cluster (plot_umap.py), and perform similarity searches on a large chemical library of your choice (query.py) of your molecules of interest! Trained on ZINC250k with a combined Tanimoto + physicochemical property objective.

---

## Architecture

### Overview

```
Molecule (SMILES)
    │
    ▼
[Featurizer]
    ├── Atom features  (56-dim per atom)
    ├── Bond features  (12-dim per bond)
    └── RDKit props    (12-dim per molecule)
    │
    ▼
[GAT Encoder] — 3 × GATv2Conv layers
    │
    ▼
global_mean_pool → graph_emb (256-dim)
    │
    ├── [Property MLP] ← RDKit props (12-dim)
    │       └── prop_emb (128-dim)
    │
    ▼
concat([graph_emb, prop_emb]) → (384-dim)
    │
    ▼
[Fusion MLP] → 128-dim
    │
    ▼
L2-normalize → embedding (128-dim, unit sphere)
```

---

### Atom Features (56-dim)

| Feature | Encoding | Dim |
|---|---|---|
| Atomic number | One-hot: H B C N O F Si P S Cl Se Br I + other | 14 |
| Degree | One-hot: 0–10 | 11 |
| Hybridization | One-hot: SP SP2 SP3 SP3D SP3D2 OTHER + other | 7 |
| Total H count | One-hot: 0–8 | 9 |
| Formal charge | One-hot: −3 to +3 + other | 8 |
| Is aromatic | Binary | 1 |
| Is in ring | Binary | 1 |
| Radical electrons | One-hot: 0–4 | 5 |

### Bond Features (12-dim)

| Feature | Encoding | Dim |
|---|---|---|
| Bond type | One-hot: SINGLE DOUBLE TRIPLE AROMATIC | 4 |
| Is conjugated | Binary | 1 |
| Is in ring | Binary | 1 |
| Stereo | One-hot: NONE ANY Z E CIS TRANS | 6 |

### RDKit Properties (12-dim)

MW, LogP, TPSA, HBD, HBA, RotatableBonds, NumRings, NumAromaticRings, FractionCSP3, HeavyAtomCount, MolMR, NumHeteroatoms

---

### GAT Encoder

3 stacked GATv2Conv layers with residual-style pass-through:

```
Layer 0: GATv2Conv(56 → 64 × 4 heads = 256) + LayerNorm(256) + ELU + Dropout(0.1)
Layer 1: GATv2Conv(256 → 64 × 4 heads = 256) + LayerNorm(256) + ELU + Dropout(0.1)
Layer 2: GATv2Conv(256 → 64 × 4 heads = 256) + LayerNorm(256) + ELU + Dropout(0.1)
```

`add_self_loops=False` on all layers (required when `edge_dim` is set).

After the encoder: `global_mean_pool` over all atom embeddings → **graph_emb (256-dim)**.

### Property MLP

```
Linear(12 → 64) → LayerNorm(64) → ELU
Linear(64 → 128) → LayerNorm(128) → ELU
→ prop_emb (128-dim)
```

### Fusion MLP

```
concat([graph_emb(256), prop_emb(128)]) → 384-dim
Linear(384 → 256) → LayerNorm(256) → ELU → Dropout(0.1)
Linear(256 → 128)
L2-normalize → embedding (128-dim)
```

---

### Loss Function

For each batch of N molecules, pairwise MSE on the upper triangle:

```
cos_sim_matrix  = embeddings @ embeddings.T        # predicted (L2-normed → dot product)
tanimoto_matrix = vectorized Tanimoto(Morgan FPs)  # structural target
prop_cos_matrix = F.normalize(props) @ .T          # physicochemical target

target = 0.8 × tanimoto_matrix + 0.2 × prop_cos_matrix
loss   = MSE(cos_sim_matrix[upper_tri], target[upper_tri])
```

The 0.8/0.2 weighting reflects that structural similarity should dominate with physicochemical properties as a secondary signal.

---

## Setup

```bash
pip install torch>=2.1.0 torch-geometric>=2.4.0 torch-scatter torch-sparse
pip install rdkit>=2023.3.1 numpy pandas tqdm pyyaml scikit-learn matplotlib scipy umap-learn
```

Or install the package:
```bash
pip install -e .
```

---

## Training Dataset

The model is trained on **ZINC250k** — 249,455 drug-like molecules (MW 250–500, LogP 0–5).

**Download:**
```bash
# Option 1: wget from popular ML mirror
wget https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv \
     -O data/zinc250k.csv

# Option 2: Kaggle (requires kaggle CLI)
kaggle datasets download basu369victor/zinc250k -p data/ --unzip
```

The file must be a CSV with a `smiles` column (or a plain `.smi` file with one SMILES per line). Place it at `data/zinc250k.csv`.

---

## Training

### Local

```bash
python3 scripts/train.py \
    --config configs/default.yaml \
    --data data/zinc250k.csv \
    --save-dir runs/my_run
```

Common overrides:
```bash
python3 scripts/train.py \
    --config configs/default.yaml \
    --data data/zinc250k.csv \
    --epochs 200 \
    --batch-size 256 \
    --lr 3e-4 \
    --save-dir runs/my_run
```

Resume from checkpoint:
```bash
python3 scripts/train.py \
    --config configs/default.yaml \
    --resume runs/my_run/best.ckpt
```

### Modal (A10G GPU, cloud)

```bash
MODAL_PROFILE=<your-profile> python3 -m modal run --detach modal_train.py
MODAL_PROFILE=<your-profile> python3 -m modal run --detach modal_train.py --epochs 200 --batch-size 256
```

Monitor:
```bash
MODAL_PROFILE=<your-profile> modal app logs <app-id>
```

Download results:
```bash
MODAL_PROFILE=<your-profile> python3 -m modal run modal_train.py::download_results
```

### Training Outputs

```
runs/my_run/
├── best.ckpt          # lowest val loss checkpoint
├── epoch_NNNN.ckpt    # periodic checkpoints (every 10 epochs)
├── loss_curves.png    # train/val loss + Spearman ρ
└── config.yaml        # resolved config snapshot
```

---

## Inference

### Embed a SMILES file

```bash
# From a .smi file (one SMILES per line)
python3 scripts/embed.py \
    --checkpoint runs/zinc250k_remote/best.ckpt \
    --smiles my_molecules.smi \
    --output embeddings.csv

# From a CSV
python3 scripts/embed.py \
    --checkpoint runs/zinc250k_remote/best.ckpt \
    --smiles my_molecules.csv \
    --smiles-col smiles \
    --output embeddings.csv

# As numpy array
python3 scripts/embed.py \
    --checkpoint runs/zinc250k_remote/best.ckpt \
    --smiles my_molecules.smi \
    --output embeddings.npy
```

Output CSV has columns: `smiles, emb_0, emb_1, … emb_127`.
Invalid SMILES are silently dropped; the script prints `Embedded N/M molecules`.

### Python API

```python
from scripts.embed import embed_molecules

smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",        # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # ibuprofen
]

embeddings, valid_smiles = embed_molecules(
    smiles,
    checkpoint_path="runs/zinc250k_remote/best.ckpt",
    batch_size=64,
    device="auto",   # "cuda", "cpu", or "auto"
)
# embeddings: np.ndarray (N, 128), L2-normalized
# cosine similarity: embeddings[i] @ embeddings[j]
```

### Query — find nearest neighbors

```bash
python3 scripts/query.py \
    --checkpoint runs/zinc250k_remote/best.ckpt \
    --query "CC(=O)Oc1ccccc1C(=O)O" \
    --database data/zinc250k.csv \
    --top-k 10
```

---

## Benchmarking

Evaluates retrieval performance against Tanimoto baseline on DUD (21 targets), MUV (17), and ChEMBL (80) using leave-one-out AUC, EF@1%, EF@5%, BEDROC(α=20), and average precision.

```bash
python3 scripts/benchmark.py \
    --checkpoint runs/zinc250k_remote/best.ckpt \
    --benchmark-dir benchmarking_platform
```

Reported results with epoch 145 checkpoint (val_loss=0.011):

| Dataset | MolGAT AUC | Tanimoto AUC | Δ AUC |
|---|---|---|---|
| DUD | 0.805 | 0.716 | +0.089 |
| MUV | 0.592 | 0.545 | +0.047 |
| ChEMBL | 0.696 | 0.573 | +0.123 |
| **Overall** | **0.683** | **0.594** | **+0.089** |

---

## Project Structure

```
drugSimSearch/
├── configs/
│   └── default.yaml          # all hyperparameters
├── data/
│   ├── zinc250k.csv          # training data (download separately)
│   └── sample.smi            # small test set (~200 molecules)
├── molsimsearch/
│   ├── featurizer.py         # atom/bond/property featurization
│   ├── dataset.py            # MoleculeDataset, PropertyScaler, SimilarityAwareSampler
│   ├── model.py              # MolGAT architecture
│   ├── loss.py               # CombinedSimilarityLoss
│   └── train.py              # training loop, evaluation, checkpointing
├── scripts/
│   ├── train.py              # CLI training entry point
│   ├── embed.py              # batch embed SMILES → CSV/NPY
│   ├── query.py              # nearest-neighbor search
│   ├── benchmark.py          # DUD/MUV/ChEMBL benchmark
│   ├── embed_patents.py      # patent SMILES embedding + pairwise similarity
│   └── plot_umap.py          # UMAP visualization
├── benchmarking_platform/
│   └── compounds/            # DUD, MUV, ChEMBL reference data
├── modal_train.py            # Modal cloud training (A10G GPU)
├── requirements.txt
└── pyproject.toml
```
