from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from planner.models.damage_head import DamageHead
from planner.models.mu_head import MuHead
from planner.models.residual_bank_head import ResidualBankHead
from planner.models.rho_head import RhoHead
from planner.models.scene_action_encoder import SceneActionEncoder


class ACSModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = SceneActionEncoder(config)
        hidden_dim = self.encoder.hidden_dim
        bank_dim = int(config.get("model", {}).get("residual_bank_dim", hidden_dim))
        self.rho_head = RhoHead(hidden_dim)
        self.damage_head = DamageHead(hidden_dim)
        self.mu_head = MuHead(hidden_dim)
        self.residual_bank_head = ResidualBankHead(hidden_dim, bank_dim)

    def forward(self, batch: Dict[str, torch.Tensor], residual_bank_embeddings: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        enc = self.encoder(batch)
        rho_logits = self.rho_head(enc.scene_embedding, enc.action_embedding, enc.atom_embedding)
        damage = self.damage_head(enc.scene_embedding, enc.action_embedding, enc.atom_embedding)
        mu_mean, mu_logvar = self.mu_head(enc.scene_embedding, enc.action_embedding, enc.atom_embedding)
        out = {
            "rho_logits": rho_logits,
            "damage": damage,
            "mu_mean": mu_mean,
            "mu_logvar": mu_logvar,
            "scene_embedding": enc.scene_embedding,
            "action_embedding": enc.action_embedding,
            "atom_embedding": enc.atom_embedding,
            "anchor_embedding": enc.anchor_embedding,
        }
        if residual_bank_embeddings is not None:
            out["residual_logits"] = self.residual_bank_head(
                enc.scene_embedding, enc.action_embedding, enc.atom_embedding, residual_bank_embeddings
            )
        return out
