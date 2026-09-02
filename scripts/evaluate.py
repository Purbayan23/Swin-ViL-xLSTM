"""Configuration-driven evaluation of the selected Pure U-Net checkpoint."""

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
from src.training.checkpoint import load_checkpoint
from src.training.config import choose_device, load_config, project_path
from src.training.engine import evaluate
from src.utils.reproducibility import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_pure_unet.json", type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    args = parser.parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())
    seed_everything(int(config["seed"]))
    device = choose_device(config)
    dataset_config = config["dataset"]
    dataset = KvasirSegDataset(
        data_root=project_path(config, dataset_config["root"]),
        manifest_path=project_path(config, dataset_config["manifest"]),
        split="test",
        image_size=dataset_config["image_size"],
        mask_threshold=dataset_config["mask_threshold"],
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    model = PureUNet(**config["model"]).to(device)
    checkpoint = args.checkpoint or (
        project_path(config, config["training"]["checkpoint_dir"]) / "best.pt"
    )
    checkpoint = Path(checkpoint).resolve()
    payload = load_checkpoint(checkpoint, model, map_location=device)
    metrics = evaluate(
        model,
        loader,
        BCESoftDiceLoss(**config["loss"]),
        device,
        threshold=float(dataset_config["prediction_threshold"]),
    )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "config_path": config["_config_path"],
        "dataset_root": str(project_path(config, dataset_config["root"])),
        "split": "test",
        "prediction_threshold": float(dataset_config["prediction_threshold"]),
        **metrics,
    }
    results_path = checkpoint.parent / "evaluation_test.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
