from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from planner.common.config import load_yaml
from planner.common.io import load_pickle, load_torch
from planner.teacher.residual_bank import ResidualBank


def dominant_anchor_bucket(anchor_features: np.ndarray) -> str:
    if anchor_features.size == 0:
        return "default"
    # First channel stores anchor type id.
    type_id = int(np.round(np.median(anchor_features[:, 0])))
    mapping = {
        1: "branch",
        2: "conflict",
        3: "merge",
        4: "stop",
        5: "PED_CROSS",
        6: "ONCOMING_TURN",
        7: "PARKED_BYPASS",
        8: "YIELD_ZONE",
    }
    return mapping.get(type_id, "default")


def farthest_point_subset(vectors: np.ndarray, k: int) -> np.ndarray:
    if len(vectors) <= k:
        return vectors
    chosen = [0]
    distances = np.full((len(vectors),), np.inf, dtype=np.float64)
    for _ in range(1, k):
        last = vectors[chosen[-1]]
        distances = np.minimum(distances, np.linalg.norm(vectors - last[None, :], axis=-1))
        chosen.append(int(np.argmax(distances)))
    return vectors[np.asarray(chosen, dtype=np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    bank = ResidualBank(config)
    index = load_pickle(Path(args.dataset_root) / args.split / "index.pkl")
    if args.max_samples is not None:
        index = index[: args.max_samples]
    grouped: Dict[str, List[np.ndarray]] = {}
    for record in tqdm(index, desc="residual_bank"):
        data = load_torch(record["path"])
        anchor_feat = data["tensors"]["atom_anchor_features"].numpy()
        mask = data["tensors"]["atom_anchor_mask"].numpy().astype(bool)
        active = anchor_feat[mask]
        if active.size == 0:
            continue
        bucket = dominant_anchor_bucket(active)
        # Use the average anchor-state descriptor as a lightweight residual prototype.
        grouped.setdefault(bucket, []).append(active.mean(axis=0))
    max_per_bucket = int(config["residual"]["residual_bank_size_per_bucket"])
    for bucket, vectors in grouped.items():
        arr = np.asarray(vectors, dtype=np.float32)
        arr = farthest_point_subset(arr, max_per_bucket)
        for vec in arr:
            bank.add(bucket, vec, weight=1.0)
    bank.finalize(max_per_bucket=max_per_bucket)
    bank.save(args.output)
    print(f"saved residual bank to {args.output}")


if __name__ == "__main__":
    main()
