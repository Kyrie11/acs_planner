from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from planner.common.config import load_yaml
from planner.models.acs_model import ACSModel
from planner.training.calibrate import fit_omission_calibrator, save_calibrator
from planner.training.dataset import ACSTensorDataset, collate_acs_batch


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ACSModel(ckpt.get("config", cfg))
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    dataset = ACSTensorDataset(args.data_root, split="val")
    loader = DataLoader(dataset, batch_size=int(cfg["training"].get("batch_size", 32)), shuffle=False, num_workers=int(cfg["training"].get("num_workers", 4)), collate_fn=collate_acs_batch)

    preds = []
    truths = []
    for batch in tqdm(loader, desc="calibrate"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        preds.extend(outputs["damage"].detach().cpu().tolist())
        truths.extend(batch["damage_target"].detach().cpu().tolist())

    calibrator = fit_omission_calibrator(preds, truths, alpha=float(cfg["training"]["calibration_alpha"]))
    save_calibrator(calibrator, args.output)
    print(f"saved calibrator to {args.output}; quantile={calibrator.quantile:.6f}")


if __name__ == "__main__":
    main()
