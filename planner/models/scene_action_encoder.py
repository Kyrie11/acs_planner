from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EncoderOutput:
    scene_embedding: torch.Tensor
    action_embedding: torch.Tensor
    atom_embedding: torch.Tensor
    anchor_embedding: torch.Tensor


class SceneActionEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", {})
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        num_heads = int(model_cfg.get("num_heads", 8))
        num_layers = int(model_cfg.get("num_transformer_layers", 3))
        dropout = float(model_cfg.get("dropout", 0.1))
        self.hidden_dim = hidden_dim

        self.ego_proj = MLP(int(model_cfg.get("ego_dim", 8)), hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.agent_proj = MLP(int(model_cfg.get("agent_dim", 12)), hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.map_proj = MLP(int(model_cfg.get("map_dim", 8)), hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.action_proj = MLP(int(model_cfg.get("action_dim", 12)), hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.anchor_proj = MLP(int(model_cfg.get("anchor_dim", 16)), hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.atom_pool = MLP(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

    def masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    def forward(self, batch: Dict[str, torch.Tensor]) -> EncoderOutput:
        ego = self.ego_proj(batch["ego_history"])  # [B, T, H]
        agents = self.agent_proj(batch["agents"])  # [B, N, H]
        maps = self.map_proj(batch["map_polylines"])  # [B, M, H]
        action = self.action_proj(batch["action_features"])  # [B, H]
        anchors = self.anchor_proj(batch["atom_anchor_features"])  # [B, K, H]

        tokens = torch.cat([ego, agents, maps], dim=1)
        token_mask = None
        if all(k in batch for k in ["ego_mask", "agent_mask", "map_mask"]):
            token_mask = torch.cat([batch["ego_mask"], batch["agent_mask"], batch["map_mask"]], dim=1)
            src_key_padding_mask = ~token_mask.bool()
        else:
            src_key_padding_mask = None
        scene_tokens = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        scene_embedding = self.scene_norm(self.masked_mean(scene_tokens, token_mask))
        anchor_embedding = self.masked_mean(anchors, batch.get("atom_anchor_mask"))
        atom_embedding = self.atom_pool(torch.cat([scene_embedding, anchor_embedding], dim=-1))
        return EncoderOutput(
            scene_embedding=scene_embedding,
            action_embedding=action,
            atom_embedding=atom_embedding,
            anchor_embedding=anchor_embedding,
        )
