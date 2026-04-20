from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class CertificationBudget:
    n_mc: int
    delta: float
    L_max: float


@dataclass
class CertificationResult:
    passed: bool
    action_id: str
    upper_bound: float
    lower_bound_rivals: Dict[str, float]
    detail: Dict[str, float]


@dataclass
class ConformalCalibrator:
    quantile: float = 0.0

    def fit(self, residuals: Iterable[float], alpha: float = 0.05) -> None:
        residuals = np.asarray(list(residuals), dtype=np.float64)
        if residuals.size == 0:
            self.quantile = 0.0
            return
        self.quantile = float(np.quantile(residuals, 1.0 - alpha))

    def upper_bound(self, predicted: float) -> float:
        return float(predicted + self.quantile)


def hoeffding_radius(L_max: float, n_mc: int, delta: float) -> float:
    n_mc = max(n_mc, 1)
    delta = min(max(delta, 1e-8), 1.0 - 1e-8)
    return float(L_max * np.sqrt(np.log(2.0 / delta) / (2.0 * n_mc)))


def certify_winner(
    winner_id: str,
    cert_scores: Dict[str, float],
    cert_radii: Dict[str, float],
    omission_upper: Dict[str, float],
) -> CertificationResult:
    winner_upper = cert_scores[winner_id] + cert_radii[winner_id] + omission_upper.get(winner_id, 0.0)
    rivals_lower = {aid: cert_scores[aid] - cert_radii[aid] for aid in cert_scores if aid != winner_id}
    passed = all(winner_upper < lb for lb in rivals_lower.values()) if rivals_lower else True
    return CertificationResult(
        passed=passed,
        action_id=winner_id,
        upper_bound=winner_upper,
        lower_bound_rivals=rivals_lower,
        detail={
            "winner_score": cert_scores[winner_id],
            "winner_radius": cert_radii[winner_id],
            "winner_omission": omission_upper.get(winner_id, 0.0),
        },
    )
