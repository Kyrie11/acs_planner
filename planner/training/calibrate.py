from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from planner.common.io import load_pickle, save_pickle
from planner.evaluation.certification import ConformalCalibrator


def fit_omission_calibrator(predictions: Iterable[float], truths: Iterable[float], alpha: float = 0.05) -> ConformalCalibrator:
    residuals = [max(0.0, truth - pred) for pred, truth in zip(predictions, truths)]
    calibrator = ConformalCalibrator()
    calibrator.fit(residuals, alpha=alpha)
    return calibrator


def save_calibrator(calibrator: ConformalCalibrator, path: str | Path) -> None:
    save_pickle({"quantile": calibrator.quantile}, path)


def load_calibrator(path: str | Path) -> ConformalCalibrator:
    raw = load_pickle(path)
    return ConformalCalibrator(quantile=float(raw["quantile"]))
