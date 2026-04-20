from __future__ import annotations

import torch
from torch import nn


class ResidualBankHead(nn.Module):
    def __init__(self, hidden_dim: int, bank_dim: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bank_dim),
        )

    def forward(
        self,
        scene_emb: torch.Tensor,
        action_emb: torch.Tensor,
        atom_emb: torch.Tensor,
        residual_bank_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(torch.cat([scene_emb, action_emb, atom_emb], dim=-1))
        return torch.matmul(query, residual_bank_embeddings.T)
