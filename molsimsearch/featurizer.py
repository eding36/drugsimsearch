from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import BondStereo, BondType, HybridizationType
from rdkit import DataStructs
from torch_geometric.data import Data

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]  # H B C N O F Si P S Cl Se Br I
DEGREES = list(range(11))
HYBRIDIZATION_TYPES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.OTHER,
]
H_COUNTS = list(range(9))
FORMAL_CHARGES = list(range(-3, 4))
RADICAL_ELECTRONS = list(range(5))
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
STEREO_TYPES = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
]

# atom type(14) + degree(11) + hybridization(7) + h_count(9) + formal_charge(8)
# + is_aromatic(1) + is_in_ring(1) + radical_electrons(5) = 56
ATOM_FEAT_DIM = 56
# bond_type(4) + conjugated(1) + in_ring(1) + stereo(6) = 12
BOND_FEAT_DIM = 12
PROP_DIM = 12


def one_hot_encode(value, allowable_set: list, allow_unknown: bool = True) -> list[int]:
    if value in allowable_set:
        encoding = [int(value == s) for s in allowable_set]
    else:
        encoding = [0] * len(allowable_set)
    if allow_unknown:
        encoding.append(int(value not in allowable_set))
    return encoding


def get_atom_features(atom: rdchem.Atom) -> list[float]:
    return (
        one_hot_encode(atom.GetAtomicNum(), ATOM_TYPES, allow_unknown=True)       # 14
        + one_hot_encode(atom.GetDegree(), DEGREES, allow_unknown=False)           # 11
        + one_hot_encode(atom.GetHybridization(), HYBRIDIZATION_TYPES, allow_unknown=True)  # 7
        + one_hot_encode(atom.GetTotalNumHs(), H_COUNTS, allow_unknown=False)      # 9
        + one_hot_encode(atom.GetFormalCharge(), FORMAL_CHARGES, allow_unknown=True)  # 8
        + [int(atom.GetIsAromatic())]                                              # 1
        + [int(atom.IsInRing())]                                                   # 1
        + one_hot_encode(atom.GetNumRadicalElectrons(), RADICAL_ELECTRONS, allow_unknown=False)  # 5
    )


def get_bond_features(bond: rdchem.Bond) -> list[float]:
    return (
        one_hot_encode(bond.GetBondType(), BOND_TYPES, allow_unknown=False)        # 4
        + [int(bond.GetIsConjugated())]                                            # 1
        + [int(bond.IsInRing())]                                                   # 1
        + one_hot_encode(bond.GetStereo(), STEREO_TYPES, allow_unknown=False)      # 6
    )


def get_rdkit_properties(mol: rdchem.Mol) -> list[float]:
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        Crippen.MolMR(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
    ]


def get_morgan_fingerprint(mol: rdchem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    return fp


def mol_to_pyg_data(smiles: str, fp_radius: int = 2, fp_nbits: int = 2048) -> Optional[Data]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    x = torch.tensor(
        [get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32
    )

    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [feat, feat]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)

    # Store as 2D (1, dim) so PyG batch-concatenation produces (N, dim) not (N*dim,)
    fp = torch.tensor(get_morgan_fingerprint(mol, fp_radius, fp_nbits), dtype=torch.float32).unsqueeze(0)
    props = torch.tensor(get_rdkit_properties(mol), dtype=torch.float32).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, fp=fp, props=props, smiles=smiles)
