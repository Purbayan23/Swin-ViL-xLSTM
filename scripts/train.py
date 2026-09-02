"""Configuration-driven full Pure U-Net training entry point.

Run this command only after the bounded sanity test passes. Checkpoints,
configuration, history, and experiment metadata are written to the configured
output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kvasir_seg import KvasirSegDataset
from src.losses.segmentation import BCESoftDiceLoss
from src.models.pure_unet import PureUNet
from src.training.checkpoint import save_checkpoint
from src.training.config import choose_device, load_config, project_path
from src.training.engine import evaluate, train_one_epoch
from src.utils.reproducibility import make_dataloader_generator, seed_everything, seed_worker


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_loader(config, split: str, shuffle: bool) -> DataLoader:
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset = KvasirSegDataset(
        data_root=project_path(config, dataset_config["root"]),
        manifest_path=project_path(config, dataset_config["manifest"]),
        split=split,
        image_size=dataset_config["image_size"],
        mask_threshold=dataset_config["mask_threshold"],
    )
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training_config["num_workers"]),
        pin_memory=bool(training_config["pin_memory"]),
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(int(config["seed"]) + (0 if shuffle else 1)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_pure_unet.json", type=Path)
    args = parser.parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())
    seed_everything(int(config["seed"]))
    device = choose_device(config)
    dataset_config = config["dataset"]
    train_loader = build_loader(config, "train", shuffle=True)
    validation_loader = build_loader(config, "validation", shuffle=False)
    model = PureUNet(**config["model"]).to(device)
    criterion = BCESoftDiceLoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["scheduler"]["t_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    checkpoint_dir = project_path(config, config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    write_json(checkpoint_dir / "config_snapshot.json", config)
    metadata = {
        "status": "running",
        "experiment_name": config["name"],
        "config_path": config["_config_path"],
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_root": str(project_path(config, dataset_config["root"])),
        "manifest": str(project_path(config, dataset_config["manifest"])),
        "device": str(device),
        "seed": int(config["seed"]),
        "model_parameters": parameter_count,
        "planned_epochs": int(config["training"]["epochs"]),
        "batch_size": int(config["training"]["batch_size"]),
    }
    write_json(checkpoint_dir / "experiment_metadata.json", metadata)
    history = []
    best_dice = float("-inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            threshold=float(config["dataset"]["prediction_threshold"]),
        )
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)
        write_json(checkpoint_dir / "training_history.json", {"history": history})
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            validation,
            config,
            int(config["seed"]),
        )
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            metadata.update(
                {
                    "best_epoch": epoch,
                    "best_validation_dice": best_dice,
                }
            )
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                validation,
                config,
                int(config["seed"]),
            )
        write_json(checkpoint_dir / "experiment_metadata.json", metadata)
        print(json.dumps(epoch_record, sort_keys=True))
    metadata.update(
        {
            "status": "completed",
            "completed_epochs": int(config["training"]["epochs"]),
        }
    )
    write_json(checkpoint_dir / "experiment_metadata.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
