from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from planner.common.io import ensure_dir, save_pickle, save_torch


class CacheWriter:
    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)

    def write_sample(self, split: str, sample_id: str, tensor_dict: Dict[str, Any], meta: Dict[str, Any]) -> Path:
        out_dir = ensure_dir(self.root / split)
        path = out_dir / f"{sample_id}.pt"
        save_torch({"tensors": tensor_dict, "meta": meta}, path)
        return path

    def write_index(self, split: str, records: list[dict]) -> Path:
        out_dir = ensure_dir(self.root / split)
        path = out_dir / "index.pkl"
        save_pickle(records, path)
        return path

    def write_batch(self, split: str, batch_id: str, samples: List[dict]) -> Path:
        split_dir = ensure_dir(self.root / split)
        path = split_dir / f"{batch_id}.pt"
        save_torch(samples, path)
        return path