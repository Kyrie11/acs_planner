from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from planner.common.io import load_pickle, load_torch
from planner.training.feature_utils import collate_tensor_dict


class ACSTensorDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        index_path = self.root / split / "index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {index_path}")
        self.records: List[dict] = load_pickle(index_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        data = load_torch(record["path"]) if "path" in record else load_torch(self.root / self.split / record["file"])
        tensors = data["tensors"]
        meta = data["meta"]
        output = dict(tensors)
        output["rho_target"] = torch.tensor(float(meta.get("rho_target", 0.0)), dtype=torch.float32)
        output["damage_target"] = torch.tensor(float(meta.get("damage_target", 0.0)), dtype=torch.float32)
        output["mu_target"] = torch.tensor(float(meta.get("mu_target", 0.0)), dtype=torch.float32)
        output["sample_weight"] = torch.tensor(float(meta.get("sample_weight", 1.0)), dtype=torch.float32)
        output["residual_bucket"] = torch.tensor(int(meta.get("residual_bucket", 0)), dtype=torch.long)
        return output


def collate_acs_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    tensor_keys = [
        "ego_history",
        "ego_mask",
        "agents",
        "agent_mask",
        "map_polylines",
        "map_mask",
        "action_features",
        "atom_anchor_features",
        "atom_anchor_mask",
    ]
    out = collate_tensor_dict([{k: item[k] for k in tensor_keys} for item in batch])
    for key in ["rho_target", "damage_target", "mu_target", "sample_weight", "residual_bucket"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out
