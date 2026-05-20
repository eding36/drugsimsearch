from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedSimilarityLoss(nn.Module):
    def __init__(
        self,
        tanimoto_weight: float = 0.8,
        property_weight: float = 0.2,
        use_upper_tri: bool = True,
    ):
        super().__init__()
        self.tanimoto_weight = tanimoto_weight
        self.property_weight = property_weight
        self.use_upper_tri = use_upper_tri

    def _cosine_sim_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings already L2-normalized, so cosine = dot product
        return embeddings @ embeddings.T

    def _tanimoto_matrix(self, fp_arr: torch.Tensor) -> torch.Tensor:
        fp_arr = fp_arr.float()
        inter = fp_arr @ fp_arr.T
        norms = fp_arr.sum(dim=1)
        union = norms.unsqueeze(1) + norms.unsqueeze(0) - inter
        return inter / (union + 1e-8)

    def _property_cosine_matrix(self, props_raw: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(props_raw, p=2, dim=1)
        return normalized @ normalized.T

    def _upper_tri_mask(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()

    def forward(
        self,
        embeddings: torch.Tensor,
        fp_arr: torch.Tensor,
        props_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        n = embeddings.size(0)
        cos_sim = self._cosine_sim_matrix(embeddings)
        tan_mat = self._tanimoto_matrix(fp_arr)
        prop_mat = self._property_cosine_matrix(props_raw)
        target = self.tanimoto_weight * tan_mat + self.property_weight * prop_mat

        if self.use_upper_tri and n > 1:
            mask = self._upper_tri_mask(n, embeddings.device)
            loss = F.mse_loss(cos_sim[mask], target[mask])
            pred_mean = cos_sim[mask].mean().item()
            tan_mean = tan_mat[mask].mean().item()
            prop_mean = prop_mat[mask].mean().item()
        else:
            loss = F.mse_loss(cos_sim, target)
            pred_mean = cos_sim.mean().item()
            tan_mean = tan_mat.mean().item()
            prop_mean = prop_mat.mean().item()

        metrics = {
            "tanimoto_sim_mean": tan_mean,
            "prop_sim_mean": prop_mean,
            "pred_sim_mean": pred_mean,
        }
        return loss, metrics
