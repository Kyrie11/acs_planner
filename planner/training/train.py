from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from planner.common.config import load_yaml
from planner.common.io import ensure_dir
from planner.models.acs_model import ACSModel
from planner.training.dataset import ACSTensorDataset, collate_acs_batch
from planner.training.losses import ACSLoss


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    meters = {"loss": 0.0, "rho_loss": 0.0, "damage_loss": 0.0, "mu_loss": 0.0}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = loss_fn(outputs, batch)
        losses["loss"].backward()
        optimizer.step()
        bs = batch["rho_target"].shape[0]
        count += bs
        for k in meters:
            meters[k] += float(losses[k]) * bs
    return {k: v / max(count, 1) for k, v in meters.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    meters = {"loss": 0.0, "rho_loss": 0.0, "damage_loss": 0.0, "mu_loss": 0.0}
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        losses = loss_fn(outputs, batch)
        bs = batch["rho_target"].shape[0]
        count += bs
        for k in meters:
            meters[k] += float(losses[k]) * bs
    return {k: v / max(count, 1) for k, v in meters.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_yaml(args.config)
    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device)
    train_set = ACSTensorDataset(args.data_root, split="train")
    val_set = ACSTensorDataset(args.data_root, split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=collate_acs_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=collate_acs_batch,
    )
    model = ACSModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"].get("epochs", 20)),
    )
    loss_fn = ACSLoss(config)

    best_val = float("inf")
    history = []
    for epoch in range(int(config["training"].get("epochs", 20))):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(record)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"model": model.state_dict(), "config": config}, out_dir / "best_model.pt")
        torch.save({"model": model.state_dict(), "config": config, "history": history}, out_dir / "last_model.pt")


if __name__ == "__main__":
    main()
