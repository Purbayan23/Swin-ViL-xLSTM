"""Inference-only replay of the existing A1 forensic instrumentation.

This tool loads one saved A1 checkpoint, runs fixed samples from one frozen
Kvasir-SEG split, and writes only compact tensor statistics.  It does not
create checkpoints, construct an optimizer or scheduler, or perform updates.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kvasir_seg import KvasirSegDataset
from src.models.factory import build_model
from src.training.config import choose_device, load_config, project_path
from src.utils.reproducibility import seed_everything


OBSERVED_VIL_MEASUREMENTS = (
    "input",
    "Q",
    "K",
    "V",
    "max_log_decay",
    "exp_neg_max_log_decay",
    "raw_mlstm_output",
    "normalized_combination",
    "final_vil_residual_output",
)


def _load_forensic_module() -> ModuleType:
    module_name = "_a1_forensic_instrumentation"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    script_path = Path(__file__).with_name("train_a1_forensic.py")
    specification = importlib.util.spec_from_file_location(module_name, script_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load forensic instrumentation from {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


FORENSIC = _load_forensic_module()


def _resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_path(config, path)


def _load_checkpoint_payload(checkpoint: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(
            f"checkpoint does not contain the expected model_state mapping: {checkpoint}"
        )
    return payload


def load_replay_model(
    checkpoint: Path,
    config_path: Path | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module]:
    """Load the model and metadata without constructing training objects."""

    payload = _load_checkpoint_payload(checkpoint, device)
    if config_path is not None:
        config = load_config(config_path)
    else:
        checkpoint_config = payload.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError(
                "--config is required when the checkpoint has no embedded config"
            )
        config = copy.deepcopy(checkpoint_config)
        config.setdefault("_config_path", str(checkpoint))
        config.setdefault("_project_root", str(PROJECT_ROOT))
    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return config, payload, model


def select_sample_indices(dataset: KvasirSegDataset, indices: list[int]) -> list[int]:
    """Validate and preserve deterministic user-specified dataset indices."""

    if not indices:
        raise ValueError("at least one sample index is required")
    if len(set(indices)) != len(indices):
        raise ValueError("sample indices must be unique")
    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(
                f"sample index {index} is outside split of length {len(dataset)}"
            )
    return list(indices)


def _directional_statistics(
    details: dict[str, dict[str, Any]],
    direction: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    prefix = f"vil/{direction}_block_"
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for key, stats in details.items():
        if not key.startswith(prefix):
            continue
        _, label, measurement = key.split("/", 2)
        if measurement in OBSERVED_VIL_MEASUREMENTS:
            blocks.setdefault(label, {})[measurement] = copy.deepcopy(stats)
    return {label: blocks[label] for label in sorted(blocks)}


def _first_directional_block(
    statistics: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]] | None:
    if not statistics:
        return None
    return statistics[sorted(statistics)[0]]


def _safe_ratio(
    numerator: dict[str, Any] | None,
    denominator: dict[str, Any] | None,
) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator_value = numerator.get("abs_max")
    denominator_value = denominator.get("abs_max")
    if numerator_value is None or denominator_value in (None, 0.0):
        return None
    return float(numerator_value) / float(denominator_value)


def collect_sample_replay(
    *,
    dataset: KvasirSegDataset,
    index: int,
    model: torch.nn.Module,
    device: torch.device,
    observer: Any,
    epoch: int | None,
    global_step: int | None,
    split: str,
) -> dict[str, Any]:
    """Run one fixed sample through the existing instrumented A1 model."""

    sample = dataset[index]
    image = sample["image"].unsqueeze(0).to(device)
    observer.begin_forward(
        split=split,
        epoch=int(epoch) if epoch is not None else -1,
        batch_index=index,
        global_step=int(global_step) if global_step is not None else -1,
        sample_ids=[str(sample["id"])],
    )
    failure: dict[str, Any] | None = None
    with torch.no_grad():
        try:
            model(image)
        except FORENSIC.ForensicFailure as error:
            failure = {
                "phase": error.phase,
                "stage": error.stage,
                "tensor_name": error.tensor_name,
                "direction": error.direction,
                "stats": copy.deepcopy(error.stats),
                "classification": FORENSIC.classify_failure(error),
                "value_type": FORENSIC.classify_nonfinite_value(error.stats),
            }
    observer.finish_forward()
    details = copy.deepcopy(observer.current_details)
    cnn_bottleneck = details.get("global/cnn_bottleneck_input")
    forward_blocks = _directional_statistics(details, "forward")
    reverse_blocks = _directional_statistics(details, "reverse")
    forward = _first_directional_block(forward_blocks)
    reverse = _first_directional_block(reverse_blocks)
    result = {
        "index": int(index),
        "id": str(sample["id"]),
        "statistics": {
            "cnn_bottleneck": cnn_bottleneck,
            "forward": forward_blocks,
            "reverse": reverse_blocks,
        },
        "amplification_ratios": {
            "forward_final_abs_max_over_cnn_bottleneck_abs_max": _safe_ratio(
                forward.get("final_vil_residual_output") if forward else None,
                cnn_bottleneck,
            ),
            "reverse_input_abs_max_over_cnn_bottleneck_abs_max": _safe_ratio(
                reverse.get("input") if reverse else None,
                cnn_bottleneck,
            ),
            "reverse_final_abs_max_over_reverse_input_abs_max": _safe_ratio(
                reverse.get("final_vil_residual_output") if reverse else None,
                reverse.get("input") if reverse else None,
            ),
        },
    }
    if failure is not None:
        result["failure"] = failure
    observer.clear()
    return result


def replay_checkpoint(
    *,
    checkpoint: Path,
    config_path: Path | None,
    sample_indices: list[int],
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    config, payload, model = load_replay_model(checkpoint, config_path, device)
    dataset_config = config["dataset"]
    dataset = KvasirSegDataset(
        data_root=_resolve_path(config, dataset_config["root"]),
        manifest_path=_resolve_path(config, dataset_config["manifest"]),
        split=split,
        image_size=dataset_config["image_size"],
        mask_threshold=int(dataset_config["mask_threshold"]),
    )
    selected_indices = select_sample_indices(dataset, sample_indices)
    observer = FORENSIC.ForensicObserver()
    instrumentation = FORENSIC.A1Instrumentation(model, observer)
    instrumentation.install()
    instrumentation.activate()
    try:
        samples = [
            collect_sample_replay(
                dataset=dataset,
                index=index,
                model=model,
                device=device,
                observer=observer,
                epoch=payload.get("epoch"),
                global_step=payload.get("global_step"),
                split=split,
            )
            for index in selected_indices
        ]
    finally:
        instrumentation.deactivate()
        instrumentation.close()
    return {
        "mode": "inference_only_checkpoint_replay",
        "inference_only": True,
        "checkpoint": {
            "path": str(checkpoint),
            "name": checkpoint.name,
        },
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "seed": payload.get("seed", config.get("seed")),
        "config_path": str(config.get("_config_path", config_path or "embedded_checkpoint_config")),
        "dataset": {
            "name": "Kvasir-SEG",
            "root": str(_resolve_path(config, dataset_config["root"])),
            "manifest": str(_resolve_path(config, dataset_config["manifest"])),
            "split": split,
            "split_size": len(dataset),
            "image_size": list(dataset.image_size),
            "mask_threshold": int(dataset.mask_threshold),
            "prediction_threshold": float(dataset_config["prediction_threshold"]),
        },
        "sample_indices": selected_indices,
        "sample_ids": [sample["id"] for sample in samples],
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "checkpoint_creation": "none",
        "samples": samples,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--sample-indices", required=True, nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    config_path = None
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
        config_path = config_path.resolve()
        config_for_device = load_config(config_path)
    else:
        config_for_device = {"device": args.device}
    if args.device == "auto":
        device = choose_device(config_for_device)
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    checkpoint = checkpoint.resolve()
    seed = None
    if config_path is not None:
        seed = int(config_for_device["seed"])
    seed_everything(seed if seed is not None else 42)
    result = replay_checkpoint(
        checkpoint=checkpoint,
        config_path=config_path,
        sample_indices=args.sample_indices,
        split=args.split,
        device=device,
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    write_json(output.resolve(), result)
    print(json.dumps({"output": str(output.resolve()), "samples": len(result["samples"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
