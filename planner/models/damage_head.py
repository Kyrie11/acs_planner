from __future__ import annotations

import torch
from torch import nn


class DamageHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, scene_emb: torch.Tensor, action_emb: torch.Tensor, atom_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([scene_emb, action_emb, atom_emb], dim=-1)
        return self.net(x).squeeze(-1)
