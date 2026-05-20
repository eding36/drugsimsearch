from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class PropertyMLP(nn.Module):
    def __init__(self, prop_dim: int = 12, hidden_dims: list[int] = [64, 128]):
        super().__init__()
        layers = []
        in_dim = prop_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ELU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, props: torch.Tensor) -> torch.Tensor:
        return self.net(props)


class MolGAT(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int = 56,
        bond_feat_dim: int = 12,
        prop_dim: int = 12,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 3,
        gat_dropout: float = 0.1,
        prop_hidden_dims: list[int] = [64, 128],
        fusion_hidden_dim: int = 256,
        embedding_dim: int = 128,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()

        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        in_dim = atom_feat_dim
        gat_out_dim = gat_hidden_dim * gat_heads  # 256

        for i in range(gat_layers):
            self.gat_convs.append(
                GATv2Conv(
                    in_dim,
                    gat_hidden_dim,
                    heads=gat_heads,
                    concat=True,
                    edge_dim=bond_feat_dim,
                    dropout=gat_dropout,
                    add_self_loops=False,
                )
            )
            self.gat_norms.append(nn.BatchNorm1d(gat_out_dim))
            in_dim = gat_out_dim

        self.prop_mlp = PropertyMLP(prop_dim, prop_hidden_dims)
        prop_out_dim = self.prop_mlp.out_dim

        fusion_in_dim = gat_out_dim + prop_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, embedding_dim),
        )

        self.act = nn.ELU()
        self.dropout = nn.Dropout(gat_dropout)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        props = data.props

        for conv, norm in zip(self.gat_convs, self.gat_norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = self.act(x)
            x = self.dropout(x)

        graph_emb = global_mean_pool(x, batch)
        prop_emb = self.prop_mlp(props)

        combined = torch.cat([graph_emb, prop_emb], dim=1)
        out = self.fusion(combined)
        return F.normalize(out, p=2, dim=1)

    @classmethod
    def from_config(cls, cfg: dict) -> "MolGAT":
        return cls(
            atom_feat_dim=cfg.get("atom_feat_dim", 56),
            bond_feat_dim=cfg.get("bond_feat_dim", 12),
            prop_dim=cfg.get("prop_dim", 12),
            gat_hidden_dim=cfg.get("gat_hidden_dim", 64),
            gat_heads=cfg.get("gat_heads", 4),
            gat_layers=cfg.get("gat_layers", 3),
            gat_dropout=cfg.get("gat_dropout", 0.1),
            prop_hidden_dims=cfg.get("prop_hidden_dims", [64, 128]),
            fusion_hidden_dim=cfg.get("fusion_hidden_dim", 256),
            embedding_dim=cfg.get("embedding_dim", 128),
            fusion_dropout=cfg.get("fusion_dropout", 0.1),
        )
