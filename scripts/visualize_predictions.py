"""Create reproducible qualitative prediction grids for a Pure U-Net run.

This is a post-hoc visualization workflow. It loads an existing checkpoint,
evaluates the frozen test split without gradients, and writes only qualitative
figures plus machine-readable selection metadata. It does not train, select a
checkpoint, or modify any checkpoint.

The default command evaluates the complete test split. ``--max-test-images``
exists only for a bounded implementation smoke test; such output is explicitly
marked as incomplete in its metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kvasir_seg import KvasirSegDataset
from src.metrics.segmentation import batch_metric_values
from src.models.pure_unet import PureUNet
from src.training.checkpoint import load_checkpoint
from src.training.config import choose_device, load_config, project_path
from src.utils.reproducibility import seed_everything


FIXED_SAMPLE_COUNT = 8
DIFFICULT_CASE_COUNT = 4
GRID_COLUMNS = ("Original RGB", "Ground Truth", "Prediction", "Disagreement Overlay")


def _resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_path(config, path)


def _metric_float(value: torch.Tensor) -> float:
    result = float(value.item())
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite metric encountered: {result}")
    return result


def _infer_sample(
    dataset: KvasirSegDataset,
    index: int,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    sample = dataset[index]
    image = sample["image"].unsqueeze(0).to(device)
    target = sample["mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits)
        metric_row = batch_metric_values(logits, target, threshold=threshold)[0]

    expected_image_shape = (1, 3, *dataset.image_size)
    expected_mask_shape = (1, 1, *dataset.image_size)
    if tuple(image.shape) != expected_image_shape:
        raise RuntimeError(f"unexpected image tensor shape: {tuple(image.shape)}")
    if tuple(target.shape) != expected_mask_shape:
        raise RuntimeError(f"unexpected mask tensor shape: {tuple(target.shape)}")
    if tuple(logits.shape) != expected_mask_shape:
        raise RuntimeError(f"unexpected model output shape: {tuple(logits.shape)}")

    prediction = (probabilities >= threshold).to(torch.uint8)[0, 0].cpu().numpy()
    truth = (target >= 0.5).to(torch.uint8)[0, 0].cpu().numpy()
    if not np.isin(prediction, (0, 1)).all():
        raise RuntimeError("prediction mask is not binary")
    if not np.isin(truth, (0, 1)).all():
        raise RuntimeError("ground-truth mask is not binary")

    return {
        "id": str(sample["id"]),
        "filename": str(dataset.entries[index]["image"]),
        "image": sample["image"].permute(1, 2, 0).numpy(),
        "truth": truth,
        "prediction": prediction,
        "dice": _metric_float(metric_row[0]),
        "iou": _metric_float(metric_row[1]),
        "precision": _metric_float(metric_row[2]),
        "recall": _metric_float(metric_row[3]),
    }


def _uint8_image(image: np.ndarray) -> Image.Image:
    array = np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _mask_image(mask: np.ndarray) -> Image.Image:
    array = (np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def _overlay_image(record: dict[str, Any]) -> Image.Image:
    base = np.clip(record["image"] * 255.0, 0.0, 255.0).astype(np.float32)
    truth = record["truth"].astype(bool)
    prediction = record["prediction"].astype(bool)
    true_positive = truth & prediction
    false_positive = ~truth & prediction
    false_negative = truth & ~prediction

    colors = np.zeros_like(base)
    colors[true_positive] = (40.0, 180.0, 80.0)
    colors[false_positive] = (220.0, 60.0, 60.0)
    colors[false_negative] = (60.0, 100.0, 220.0)
    highlighted = true_positive | false_positive | false_negative
    blended = base.copy()
    blended[highlighted] = 0.45 * base[highlighted] + 0.55 * colors[highlighted]
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def _save_grid(records: list[dict[str, Any]], path: Path, title: str) -> None:
    if not records:
        raise ValueError("cannot create a grid with no records")

    cell_width = int(records[0]["truth"].shape[1])
    cell_height = int(records[0]["truth"].shape[0])
    margin = 12
    header_height = 34
    row_label_height = 30
    gap = 6
    width = margin * 2 + cell_width * 4 + gap * 3
    height = margin * 2 + header_height + len(records) * (row_label_height + cell_height + gap)
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    _draw_text(draw, (margin, 2), title, font)

    x_positions = [margin + column * (cell_width + gap) for column in range(4)]
    header_y = margin + 16
    for x, label in zip(x_positions, GRID_COLUMNS):
        _draw_text(draw, (x, header_y), label, font)

    for row, record in enumerate(records):
        row_top = margin + header_height + row * (row_label_height + cell_height + gap)
        summary = f"{record['filename']}  |  Dice={record['dice']:.4f}  IoU={record['iou']:.4f}"
        _draw_text(draw, (margin, row_top), summary, font)
        cell_top = row_top + row_label_height
        cells = (
            _uint8_image(record["image"]),
            _mask_image(record["truth"]),
            _mask_image(record["prediction"]),
            _overlay_image(record),
        )
        for x, cell in zip(x_positions, cells):
            canvas.paste(cell, (x, cell_top))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _write_metadata(
    path: Path,
    config: dict[str, Any],
    dataset: KvasirSegDataset,
    checkpoint: Path,
    checkpoint_payload: dict[str, Any],
    device: torch.device,
    threshold: float,
    seed: int,
    model_parameters: int,
    selected: list[dict[str, Any]],
    difficult: list[dict[str, Any]],
    evaluated_count: int,
    full_test_split: bool,
    output_dir: Path,
) -> None:
    dataset_config = config["dataset"]
    metadata = {
        "workflow": "post_hoc_qualitative_prediction_visualization",
        "scientific_use": "qualitative_error_analysis_only",
        "checkpoint": {
            "path": str(checkpoint),
            "name": checkpoint.name,
            "epoch": int(checkpoint_payload["epoch"]),
        },
        "dataset": {
            "name": "Kvasir-SEG",
            "root": str(dataset.data_root),
            "manifest": str(dataset.manifest_path),
            "split": "test",
            "test_split_size": len(dataset),
            "images_evaluated": evaluated_count,
            "evaluation_scope": "full test split" if full_test_split else "bounded smoke-test subset",
            "augmentation": False,
        },
        "preprocessing": {
            "image_size": list(dataset.image_size),
            "mask_grayscale": True,
            "mask_threshold": dataset.mask_threshold,
            "mask_threshold_status": "engineering_choice_not_official_kvasir_seg_threshold",
            "mask_threshold_order": "before_resize",
            "mask_resize": "nearest_neighbor",
        },
        "model": {
            "name": "PureUNet",
            "parameters": model_parameters,
            "device": str(device),
        },
        "prediction": {
            "threshold": threshold,
            "inference": "torch.no_grad",
        },
        "selection": {
            "random_seed": seed,
            "fixed_sample_count": len(selected),
            "fixed_sample_method": "random sample from deterministically ID-sorted test candidates",
            "selected_filenames": [record["filename"] for record in selected],
            "difficult_case_count": len(difficult),
            "difficult_case_method": "lowest per-image Dice among evaluated test images; post-hoc only",
            "difficult_filenames": [record["filename"] for record in difficult],
            "difficult_cases": [
                {
                    "filename": record["filename"],
                    "dice": record["dice"],
                    "iou": record["iou"],
                }
                for record in difficult
            ],
        },
        "outputs": {
            "directory": str(output_dir),
            "fixed_sample_grid": str(output_dir / "fixed_sample_grid.png"),
            "lowest_dice_grid": str(output_dir / "lowest_dice_test_cases_grid.png"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_pure_unet.json", type=Path)
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="optional checkpoint override; default is <configured checkpoint_dir>/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="optional output override; default is a visualizations folder beside the checkpoint",
    )
    parser.add_argument(
        "--max-test-images",
        default=None,
        type=int,
        help="bounded smoke-test limit; omit for the required complete test-split evaluation",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path.resolve())
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(config)
    dataset_config = config["dataset"]
    dataset = KvasirSegDataset(
        data_root=project_path(config, dataset_config["root"]),
        manifest_path=project_path(config, dataset_config["manifest"]),
        split="test",
        image_size=dataset_config["image_size"],
        mask_threshold=int(dataset_config["mask_threshold"]),
    )

    if args.max_test_images is not None and args.max_test_images < FIXED_SAMPLE_COUNT:
        raise ValueError(
            f"--max-test-images must be at least {FIXED_SAMPLE_COUNT} to support the fixed sample"
        )
    evaluated_count = len(dataset) if args.max_test_images is None else min(
        len(dataset), args.max_test_images
    )
    if evaluated_count < FIXED_SAMPLE_COUNT:
        raise ValueError(f"test split must contain at least {FIXED_SAMPLE_COUNT} images")

    model = PureUNet(**config["model"]).to(device)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    checkpoint = (
        _resolve_path(config, args.checkpoint)
        if args.checkpoint is not None
        else project_path(config, config["training"]["checkpoint_dir"]) / "best.pt"
    )
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Provide the completed-run best.pt or use --checkpoint for a local smoke test."
        )
    checkpoint_payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    ordered_indices = sorted(
        range(len(dataset)), key=lambda index: str(dataset.entries[index]["id"])
    )[:evaluated_count]
    selection_rng = random.Random(seed)
    selected_indices = sorted(
        selection_rng.sample(ordered_indices, FIXED_SAMPLE_COUNT),
        key=lambda index: str(dataset.entries[index]["id"]),
    )
    selected_ids = {str(dataset.entries[index]["id"]) for index in selected_indices}

    records: list[dict[str, Any]] = []
    for index in ordered_indices:
        records.append(
            _infer_sample(
                dataset,
                index,
                model,
                device,
                threshold=float(dataset_config["prediction_threshold"]),
            )
        )
    record_by_id = {record["id"]: record for record in records}
    selected = [record_by_id[str(dataset.entries[index]["id"])] for index in selected_indices]
    difficult = sorted(records, key=lambda record: (record["dice"], record["id"]))[:DIFFICULT_CASE_COUNT]
    if not selected_ids.issubset(record_by_id):
        raise RuntimeError("fixed sample selection was not evaluated")

    output_dir = (
        _resolve_path(config, args.output_dir)
        if args.output_dir is not None
        else checkpoint.parent / "visualizations"
    ).resolve()
    _save_grid(
        selected,
        output_dir / "fixed_sample_grid.png",
        "Deterministic fixed test sample (seed=42; overlay TP=green, FP=red, FN=blue)",
    )
    _save_grid(
        difficult,
        output_dir / "lowest_dice_test_cases_grid.png",
        "Lowest-Dice test cases (post-hoc; overlay TP=green, FP=red, FN=blue)",
    )
    _write_metadata(
        output_dir / "visualization_metadata.json",
        config,
        dataset,
        checkpoint,
        checkpoint_payload,
        device,
        float(dataset_config["prediction_threshold"]),
        seed,
        model_parameters,
        selected,
        difficult,
        evaluated_count,
        args.max_test_images is None and evaluated_count == len(dataset),
        output_dir,
    )

    result = {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "device": str(device),
        "model_parameters": model_parameters,
        "test_split_size": len(dataset),
        "images_evaluated": evaluated_count,
        "fixed_sample_count": len(selected),
        "difficult_case_count": len(difficult),
        "output_dir": str(output_dir),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
