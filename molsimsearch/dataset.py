from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import Subset, random_split
from torch_geometric.data import Dataset

from molsimsearch.featurizer import mol_to_pyg_data


class MoleculeDataset(Dataset):
    def __init__(
        self,
        smiles_file: str,
        fp_radius: int = 2,
        fp_nbits: int = 2048,
        smiles_col: Optional[str] = None,
        cache_dir: str = ".cache",
        transform=None,
    ):
        super().__init__(transform=transform)
        self.smiles_file = smiles_file
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.smiles_col = smiles_col
        self.cache_dir = cache_dir
        self._data_list = self._load_or_process()

    def _load_smiles(self) -> list[str]:
        path = self.smiles_file
        if path.endswith(".csv") or path.endswith(".tsv"):
            sep = "\t" if path.endswith(".tsv") else ","
            df = pd.read_csv(path, sep=sep)
            col = self.smiles_col or df.columns[0]
            return df[col].dropna().tolist()
        else:
            smiles = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        smiles.append(line.split()[0])
            return smiles

    def _cache_path(self) -> str:
        with open(self.smiles_file, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()[:8]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{digest}_r{self.fp_radius}_b{self.fp_nbits}.pt")

    def _load_or_process(self):
        cache = self._cache_path()
        if os.path.exists(cache):
            return torch.load(cache, weights_only=False)
        smiles_list = self._load_smiles()
        data_list = []
        for smi in smiles_list:
            d = mol_to_pyg_data(smi, self.fp_radius, self.fp_nbits)
            if d is not None:
                data_list.append(d)
        torch.save(data_list, cache)
        return data_list

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int):
        return self._data_list[idx]

    @property
    def num_molecules(self) -> int:
        return len(self._data_list)

    @classmethod
    def from_smiles_list(
        cls,
        smiles_list: list[str],
        fp_radius: int = 2,
        fp_nbits: int = 2048,
    ) -> "MoleculeDataset":
        obj = cls.__new__(cls)
        Dataset.__init__(obj)
        obj.fp_radius = fp_radius
        obj.fp_nbits = fp_nbits
        obj._data_list = []
        for smi in smiles_list:
            d = mol_to_pyg_data(smi, fp_radius, fp_nbits)
            if d is not None:
                obj._data_list.append(d)
        return obj


class SimilarityAwareSampler(torch.utils.data.Sampler):
    """
    Builds batches with a controlled mix of similar and diverse molecules.

    Clustering uses a weighted combination of structural (Morgan FP) and
    physicochemical (RDKit properties) features, mirroring the 0.8/0.2
    tanimoto/property split used in the loss function.

    For each batch, `similar_fraction` of slots are filled by groups of
    `mols_per_cluster` molecules from the same cluster; the rest are random.

    Example with batch_size=256, similar_fraction=0.25, mols_per_cluster=4:
      - 64 slots:  16 groups × 4 molecules from the same cluster
      - 192 slots: random molecules from anywhere in the dataset
    """

    def __init__(
        self,
        dataset: Subset,
        batch_size: int,
        n_clusters: int = 2000,
        mols_per_cluster: int = 4,
        similar_fraction: float = 0.25,
        fp_weight: float = 0.8,
        prop_weight: float = 0.2,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.mols_per_cluster = mols_per_cluster
        self._seed = seed
        self.n_mols = len(dataset)
        self.n_batches = self.n_mols // batch_size

        n_similar = (int(batch_size * similar_fraction) // mols_per_cluster) * mols_per_cluster
        self.n_groups = n_similar // mols_per_cluster
        self.n_similar = n_similar
        self.n_random = batch_size - n_similar

        print("Building similarity-aware batch sampler (one-time)...")

        base = dataset.dataset if hasattr(dataset, "dataset") else dataset
        idx_map = dataset.indices if hasattr(dataset, "indices") else range(len(dataset))

        # Extract FPs and properties from in-memory data list
        fps = np.stack(
            [base._data_list[i].fp.squeeze(0).numpy() for i in idx_map],
            dtype=np.float32,
        )  # (n_mols, 2048)
        props = np.stack(
            [base._data_list[i].props.squeeze(0).numpy() for i in idx_map],
            dtype=np.float32,
        )  # (n_mols, 12)

        # Structural features: random projection 2048 → 64
        proj = SparseRandomProjection(n_components=64, random_state=seed)
        fps_reduced = proj.fit_transform(fps)                    # (n_mols, 64)
        fps_scaled = StandardScaler().fit_transform(fps_reduced) # zero-mean, unit-var

        # Property features: standardise each of the 12 properties
        props_scaled = StandardScaler().fit_transform(props)     # (n_mols, 12)

        # Combine with loss-matching weights: structure 0.8, properties 0.2
        # Scale dimensions proportionally so neither stream dominates by size
        struct_contrib = fps_scaled  * fp_weight   * (1.0 / fps_scaled.shape[1] ** 0.5)
        prop_contrib   = props_scaled * prop_weight * (1.0 / props_scaled.shape[1] ** 0.5)
        features = np.hstack([struct_contrib, prop_contrib])     # (n_mols, 76)

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(10_000, self.n_mols),
            n_init=3,
            max_iter=100,
        )
        labels = kmeans.fit_predict(features)

        clusters: dict[int, list[int]] = defaultdict(list)
        for pos, label in enumerate(labels):
            clusters[label].append(pos)

        self._valid = {k: np.array(v) for k, v in clusters.items() if len(v) >= mols_per_cluster}
        self._keys = np.array(list(self._valid.keys()))
        self._iter_count = 0

        covered = sum(len(v) for v in self._valid.values())
        print(f"  {len(self._valid)}/{n_clusters} usable clusters · {covered}/{self.n_mols} molecules covered")

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._iter_count)
        self._iter_count += 1
        for _ in range(self.n_batches):
            batch: list[int] = []
            used: set[int] = set()

            # Similar groups: sample n_groups clusters, take mols_per_cluster from each
            chosen = rng.choice(self._keys, size=self.n_groups, replace=False)
            for ck in chosen:
                picks = rng.choice(self._valid[int(ck)], size=self.mols_per_cluster, replace=False)
                batch.extend(picks.tolist())
                used.update(picks.tolist())

            # Random fill: shuffle all positions, skip already-used ones
            pool = rng.permutation(self.n_mols)
            random_picks = [int(p) for p in pool if p not in used]
            batch.extend(random_picks[: self.n_random])

            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.n_batches


def split_dataset(
    dataset: MoleculeDataset,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def scaffold_split(
    dataset: MoleculeDataset,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split by Murcko scaffold so val contains scaffolds unseen during training."""
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, data in enumerate(dataset._data_list):
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            key = "__invalid__"
        else:
            try:
                # Strip stereo before Murcko to avoid RDKit C++ assertion on bad bond stereo
                mol_clean = Chem.RWMol(mol)
                Chem.RemoveStereochemistry(mol_clean)
                key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol_clean.GetMol(), includeChirality=False)
            except Exception:
                key = data.smiles
        scaffold_to_indices[key].append(i)

    # Shuffle scaffold order with seed, then greedily fill val until quota
    scaffold_keys = list(scaffold_to_indices.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_keys)

    n_val_target = max(1, int(len(dataset) * val_split))
    train_idx: list[int] = []
    val_idx: list[int] = []
    for key in scaffold_keys:
        indices = scaffold_to_indices[key]
        if len(val_idx) < n_val_target:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    n_val_scaffolds = sum(1 for k in scaffold_keys if scaffold_to_indices[k][0] in set(val_idx))
    print(
        f"Scaffold split: {len(train_idx)} train / {len(val_idx)} val "
        f"({n_val_scaffolds} val-only scaffolds)"
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class PropertyScaler:
    def __init__(self):
        self._scaler = StandardScaler()

    def fit(self, props: np.ndarray) -> "PropertyScaler":
        self._scaler.fit(props)
        return self

    def transform(self, props: torch.Tensor) -> torch.Tensor:
        device = props.device
        arr = props.cpu().numpy()
        scaled = self._scaler.transform(arr).astype(np.float32)
        return torch.from_numpy(scaled).to(device)

    def state_dict(self) -> dict:
        return {
            "mean": self._scaler.mean_.copy(),
            "scale": self._scaler.scale_.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._scaler.mean_ = state["mean"]
        self._scaler.scale_ = state["scale"]
        self._scaler.n_features_in_ = len(state["mean"])
        self._scaler.var_ = state["scale"] ** 2
