from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def expected_cost(rho: np.ndarray, mu: np.ndarray) -> float:
    if rho.size == 0:
        return float("inf")
    return float(np.sum(rho * mu))


def renormalize_after_removal(rho: np.ndarray, remove_index: int) -> np.ndarray:
    if rho.size <= 1:
        return np.zeros_like(rho)
    out = rho.copy()
    removed = out[remove_index]
    out[remove_index] = 0.0
    denom = max(1e-6, 1.0 - removed)
    out /= denom
    return out


def omission_damage_targets(
    winner_cost: float,
    rival_cost: float,
    rho: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    margin = rival_cost - winner_cost
    targets = np.zeros_like(rho)
    for i in range(len(rho)):
        rho_minus = renormalize_after_removal(rho, i)
        cost_minus = expected_cost(rho_minus, mu)
        margin_minus = rival_cost - cost_minus
        targets[i] = max(0.0, margin - margin_minus)
    return targets
