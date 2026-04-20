from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from planner.actions.action_types import RefinedAction
from planner.evaluation.certification import CertificationBudget, hoeffding_radius


@dataclass
class ActionScore:
    action: RefinedAction
    fast_score: float
    cert_score: float
    cert_radius: float
    omission_upper: float

    @property
    def upper_bound(self) -> float:
        return self.cert_score + self.cert_radius + self.omission_upper

    @property
    def lower_bound(self) -> float:
        return self.cert_score - self.cert_radius


class RetainedEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.L_max = float(config["cost"]["L_max"])
        self.delta = float(config["ranking"].get("cert_confidence_delta", config.get("certification", {}).get("cert_confidence_delta", 0.05)))

    def aggregate_fast(self, rho: np.ndarray, mu_fast: np.ndarray) -> float:
        if rho.size == 0:
            return float("inf")
        return float(np.sum(rho * mu_fast))

    def aggregate_cert(self, rho: np.ndarray, mu_cert: np.ndarray, n_mc: int) -> tuple[float, float]:
        if rho.size == 0:
            return float("inf"), float(self.L_max)
        score = float(np.sum(rho * mu_cert))
        radius = float(np.sum(rho * hoeffding_radius(self.L_max, n_mc=n_mc, delta=self.delta / max(len(rho), 1))))
        return score, radius

    def rank(self, actions: Sequence[RefinedAction], fast_scores: Dict[str, float]) -> List[RefinedAction]:
        ranked = list(actions)
        ranked.sort(key=lambda a: fast_scores[a.action_id])
        return ranked

    def best_conservative(self, actions: Sequence[RefinedAction], scores: Dict[str, float]) -> RefinedAction | None:
        conservative = [a for a in actions if a.is_conservative]
        if not conservative:
            return None
        conservative.sort(key=lambda a: scores[a.action_id])
        return conservative[0]
