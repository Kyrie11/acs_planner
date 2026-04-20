from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class ACSLoss:
    def __init__(self, config: dict):
        training_cfg = config.get("training", {})
        self.mu_weight = float(training_cfg.get("mu_loss_weight", 1.0))
        self.rho_weight = float(training_cfg.get("rho_loss_weight", 1.0))
        self.damage_weight = float(training_cfg.get("damage_loss_weight", 1.0))
        self.var_weight = float(training_cfg.get("var_loss_weight", 0.1))

    def __call__(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weight = batch["sample_weight"]
        rho_loss = F.binary_cross_entropy_with_logits(outputs["rho_logits"], batch["rho_target"], weight=weight)
        damage_loss = F.smooth_l1_loss(outputs["damage"], batch["damage_target"], reduction="none")
        damage_loss = (damage_loss * weight).mean()
        mu_error = outputs["mu_mean"] - batch["mu_target"]
        mu_var = torch.exp(outputs["mu_logvar"]).clamp_min(1e-4)
        mu_nll = 0.5 * (mu_error**2 / mu_var + outputs["mu_logvar"])
        mu_loss = (mu_nll * weight).mean()
        total = self.rho_weight * rho_loss + self.damage_weight * damage_loss + self.mu_weight * mu_loss
        return {
            "loss": total,
            "rho_loss": rho_loss.detach(),
            "damage_loss": damage_loss.detach(),
            "mu_loss": mu_loss.detach(),
        }
