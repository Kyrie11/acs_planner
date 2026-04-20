from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from planner.common.io import load_pickle, save_pickle


@dataclass
class ResidualPrototype:
    bucket: str
    vector: np.ndarray
    weight: float


class ResidualBank:
    def __init__(self, config: dict):
        self.config = config
        self.prototypes: Dict[str, List[ResidualPrototype]] = {}
        self.embedding_dim = int(config.get("model", {}).get("residual_bank_dim", config.get("model", {}).get("hidden_dim", 256)))

    def add(self, bucket: str, vector: np.ndarray, weight: float = 1.0) -> None:
        self.prototypes.setdefault(bucket, []).append(ResidualPrototype(bucket, np.asarray(vector, dtype=np.float32), float(weight)))

    def finalize(self, max_per_bucket: int | None = None) -> None:
        max_per_bucket = max_per_bucket or int(self.config["residual"]["residual_bank_size_per_bucket"])
        for bucket, items in list(self.prototypes.items()):
            items.sort(key=lambda p: p.weight, reverse=True)
            self.prototypes[bucket] = items[:max_per_bucket]

    def bucket_embeddings(self, bucket: str) -> torch.Tensor:
        vectors = [proto.vector for proto in self.prototypes.get(bucket, [])]
        if not vectors:
            return torch.zeros((1, self.embedding_dim), dtype=torch.float32)
        arr = np.stack(vectors, axis=0)
        if arr.shape[1] != self.embedding_dim:
            padded = np.zeros((arr.shape[0], self.embedding_dim), dtype=np.float32)
            cols = min(arr.shape[1], self.embedding_dim)
            padded[:, :cols] = arr[:, :cols]
            arr = padded
        return torch.from_numpy(arr)

    def save(self, path: str | Path) -> None:
        serializable = {
            bucket: [{"vector": p.vector.tolist(), "weight": p.weight} for p in items]
            for bucket, items in self.prototypes.items()
        }
        save_pickle(serializable, path)

    @classmethod
    def load(cls, path: str | Path, config: dict) -> "ResidualBank":
        bank = cls(config)
        raw = load_pickle(path)
        for bucket, items in raw.items():
            for item in items:
                bank.add(bucket, np.asarray(item["vector"], dtype=np.float32), item["weight"])
        bank.finalize()
        return bank
