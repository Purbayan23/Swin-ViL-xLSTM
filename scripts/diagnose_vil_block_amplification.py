"""Bounded, diagnostic-only A0/A1 ViL block amplification comparison.

This script instruments module outputs and selected intermediate tensors through
forward hooks. It does not alter production model code, mLSTM equations,
training configuration, or checkpoint files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kvasir_seg import KvasirSegDataset
from src.losses.segmentation import BCESoftDiceLoss
from src.metrics.segmentation import batch_metric_values
from src.models.factory import build_model
from src.training.checkpoint import load_checkpoint
from src.training.config import choose_device, load_config, project_path
from src.utils.reproducibility import (
    make_dataloader_generator,
    seed_everything,
    seed_worker,
)
import src.models.vil_bottleneck_unet as vil_module


STAGE_NAMES = (
    "vil_input",
    "layernorm_output",
    "x_mlstm_conv_output",
    "x_mlstm_conv_act_output",
    "queries",
    "keys",
    "values",
    "raw_parallel_mlstm_output",
    "mlstm_output",
    "h_state_plus_learnable_skip_x_mlstm_conv_act",
    "silu_z",
    "gated_product",
    "proj_down_input",
    "proj_down_output",
    "final_vil_residual_output",
)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite_mask = torch.isfinite(value)
    nan_mask = torch.isnan(value)
    inf_mask = torch.isinf(value)
    finite_values = value[finite_mask]
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "finite_count": int(finite_mask.sum().item()),
        "nan_count": int(nan_mask.sum().item()),
        "inf_count": int(inf_mask.sum().item()),
        "nonfinite_count": int((~finite_mask).sum().item()),
        "finite": bool(finite_mask.all().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
                "abs_max": float(finite_values.abs().max().item()),
            }
        )
    else:
        result.update({"min": None, "max": None, "mean": None, "std": None, "abs_max": None})
    return result


class BlockCapture:
    def __init__(self) -> None:
        self.blocks: dict[str, dict[str, Any]] = {}
        self.active_block: str | None = None
        self.active_call: dict[str, Any] | None = None
        self.aux: dict[str, torch.Tensor] = {}

    def begin(self, block_name: str, value: torch.Tensor) -> None:
        block = self.blocks.setdefault(block_name, {"calls": []})
        self.active_block = block_name
        self.active_call = {"stages": []}
        block["calls"].append(self.active_call)
        self.aux = {}
        self.add("vil_input", value)

    def add(self, label: str, value: torch.Tensor) -> None:
        if self.active_call is None:
            return
        self.active_call["stages"].append({"tensor": label, **tensor_stats(value)})

    def end(self) -> None:
        self.active_block = None
        self.active_call = None
        self.aux = {}


def _block_map(model: torch.nn.Module, model_label: str) -> dict[str, torch.nn.Module]:
    processor = model.bottleneck_processor
    if model_label == "a0":
        return {
            f"a0_block_{index}": block
            for index, block in enumerate(processor.blocks)
        }
    return {
        **{
            f"a1_forward_{index}": block
            for index, block in enumerate(processor.forward_blocks)
        },
        **{
            f"a1_reverse_{index}": block
            for index, block in enumerate(processor.reverse_blocks)
        },
    }


def install_block_hooks(
    model: torch.nn.Module,
    model_label: str,
    capture: BlockCapture,
) -> list[Any]:
    handles: list[Any] = []
    for block_name, block in _block_map(model, model_label).items():
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, inputs, name=block_name: capture.begin(name, inputs[0])
            )
        )
        handles.append(
            block.norm.register_forward_hook(
                lambda _module, _inputs, output: capture.add("layernorm_output", output)
            )
        )

        def on_proj_up(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            _, z = torch.chunk(output.detach(), chunks=2, dim=-1)
            silu_z = F.silu(z)
            capture.aux["silu_z"] = silu_z
            capture.add("silu_z", silu_z)

        handles.append(block.proj_up.register_forward_hook(on_proj_up))

        def on_conv(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            output_detached = output.detach()
            act = F.silu(output_detached)
            capture.aux["x_mlstm_conv_act"] = act
            capture.add("x_mlstm_conv_output", output_detached)
            capture.add("x_mlstm_conv_act_output", act)

        handles.append(block.conv1d.register_forward_hook(on_conv))
        for label, module in (
            ("queries", block.q_proj),
            ("keys", block.k_proj),
            ("values", block.v_proj),
        ):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, stage=label: capture.add(
                        stage, output
                    )
                )
            )

        def on_mlstm(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            output_detached = output.detach()
            capture.aux["mlstm_output"] = output_detached
            capture.add("mlstm_output", output_detached)
            x_act = capture.aux.get("x_mlstm_conv_act")
            if x_act is not None:
                skip = output_detached + block.learnable_skip.detach() * x_act
                capture.aux["h_state_plus_skip"] = skip
                capture.add("h_state_plus_learnable_skip_x_mlstm_conv_act", skip)

        handles.append(block.mlstm.register_forward_hook(on_mlstm))

        def on_proj_down_pre(
            _module: torch.nn.Module, inputs: tuple[Any, ...]
        ) -> None:
            value = inputs[0]
            if torch.is_tensor(value):
                capture.add("gated_product", value)
                capture.add("proj_down_input", value)

        handles.append(block.proj_down.register_forward_pre_hook(on_proj_down_pre))
        handles.append(
            block.proj_down.register_forward_hook(
                lambda _module, _inputs, output: capture.add("proj_down_output", output)
            )
        )

        def on_block_output(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
        ) -> None:
            capture.add("final_vil_residual_output", output)
            capture.end()

        handles.append(block.register_forward_hook(on_block_output))
    return handles


def install_raw_mlstm_capture(capture: BlockCapture) -> Any:
    original = vil_module._parallel_stabilized_mlstm

    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        output = original(*args, **kwargs)
        capture.add("raw_parallel_mlstm_output", output)
        return output

    vil_module._parallel_stabilized_mlstm = wrapped
    return original


def _loader(config: dict[str, Any], seed: int) -> DataLoader:
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset = KvasirSegDataset(
        data_root=project_path(config, dataset_config["root"]),
        manifest_path=project_path(config, dataset_config["manifest"]),
        split="train",
        image_size=dataset_config["image_size"],
        mask_threshold=dataset_config["mask_threshold"],
    )
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        pin_memory=bool(training_config["pin_memory"]),
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(seed),
    )


def _first_nonfinite(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> tuple[str | None, dict[str, Any] | None, float, int, int]:
    first_name = None
    first_stats = None
    maximum = 0.0
    nan_count = 0
    inf_count = 0
    for name, value in named_tensors:
        stats = tensor_stats(value)
        nan_count += stats["nan_count"]
        inf_count += stats["inf_count"]
        maximum = max(maximum, stats["abs_max"] or 0.0)
        if first_name is None and not stats["finite"]:
            first_name = name
            first_stats = stats
    return first_name, first_stats, maximum, nan_count, inf_count


def gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    values = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    values = [(name, value) for name, value in values if value is not None]
    first, stats, maximum, nan_count, inf_count = _first_nonfinite(values)
    return {
        "all_finite": first is None,
        "first_nonfinite_parameter": first,
        "first_nonfinite_stats": stats,
        "maximum_abs_gradient": maximum,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "gradient_parameter_count": len(values),
    }


def parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    values = list(model.named_parameters())
    first, stats, maximum, nan_count, inf_count = _first_nonfinite(values)
    return {
        "all_finite": first is None,
        "first_nonfinite_parameter": first,
        "first_nonfinite_stats": stats,
        "maximum_abs_parameter": maximum,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def optimizer_summary(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    values: list[tuple[str, torch.Tensor]] = []
    for parameter, state in optimizer.state.items():
        name = names.get(id(parameter), "<unnamed>")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value):
                values.append((f"{name}.{key}", value))
    first, stats, maximum, nan_count, inf_count = _first_nonfinite(values)
    return {
        "all_finite": first is None,
        "first_nonfinite_state": first,
        "first_nonfinite_stats": stats,
        "maximum_abs_state": maximum,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "state_tensor_count": len(values),
    }


def block_maxima(blocks: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for block_name, block in blocks.items():
        maxima: dict[str, float | None] = {}
        for call in block["calls"]:
            for stage in call["stages"]:
                value = stage["abs_max"]
                label = stage["tensor"]
                if value is not None:
                    maxima[label] = max(maxima.get(label, 0.0) or 0.0, value)
        result[block_name] = maxima
    return result


def run_model(
    model_label: str,
    config_path: Path,
    checkpoint_path: Path,
    max_batches: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    expected_name = "unet_vil_bottleneck" if model_label == "a0" else "unet_vil_bottleneck_a1"
    if config["model"]["name"] != expected_name:
        raise ValueError(f"{model_label} config must use model.name={expected_name!r}")
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(config)
    loader = _loader(config, seed)
    model = build_model(config).to(device)
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
    payload = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        map_location=device,
    )
    capture = BlockCapture()
    handles = install_block_hooks(model, model_label, capture)
    original_parallel = install_raw_mlstm_capture(capture)
    records: list[dict[str, Any]] = []
    error: str | None = None
    try:
        model.train()
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            record: dict[str, Any] = {
                "batch_index": batch_index,
                "sample_ids": [str(value) for value in batch["id"]],
                "input": tensor_stats(images),
                "target": tensor_stats(masks),
            }
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            record["output_shape"] = list(logits.shape)
            record["loss"] = tensor_stats(loss)
            record["metrics"] = batch_metric_values(
                logits.detach(),
                masks,
                threshold=float(config["dataset"]["prediction_threshold"]),
            ).mean(dim=0).tolist()
            if not torch.isfinite(loss).item():
                record["status"] = "nonfinite_loss"
                records.append(record)
                break
            loss.backward()
            record["gradients"] = gradient_summary(model)
            if not record["gradients"]["all_finite"]:
                record["status"] = "nonfinite_gradient"
                records.append(record)
                break
            optimizer.step()
            record["parameters_after_optimizer_step"] = parameter_summary(model)
            record["optimizer_state_after_step"] = optimizer_summary(model, optimizer)
            record["learning_rate"] = [float(group["lr"]) for group in optimizer.param_groups]
            record["status"] = "pass"
            records.append(record)
            if not record["parameters_after_optimizer_step"]["all_finite"]:
                break
            if not record["optimizer_state_after_step"]["all_finite"]:
                break
            if batch_index >= max_batches:
                break
    except Exception as exc:
        error = repr(exc)
    finally:
        vil_module._parallel_stabilized_mlstm = original_parallel
        for handle in handles:
            handle.remove()
    return {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "seed": seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": int(config["training"]["batch_size"]),
        "blocks": capture.blocks,
        "block_stage_maxima": block_maxima(capture.blocks),
        "records": records,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-a0",
        type=Path,
        default=PROJECT_ROOT / "configs/sanity_vil_bottleneck.json",
    )
    parser.add_argument(
        "--checkpoint-a0",
        type=Path,
        default=PROJECT_ROOT / "experiments/sanity/architecture_a_vil_bottleneck_seed42.pt",
    )
    parser.add_argument(
        "--config-a1",
        type=Path,
        default=PROJECT_ROOT / "configs/sanity_vil_bottleneck_a1.json",
    )
    parser.add_argument(
        "--checkpoint-a1",
        type=Path,
        default=PROJECT_ROOT / "experiments/sanity/architecture_a1_alternating_vil_bottleneck_seed42.pt",
    )
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic": "bounded A0/A1 ViL block amplification comparison",
        "not_full_training": True,
        "models": {},
    }
    try:
        report["models"]["a0"] = run_model(
            "a0",
            args.config_a0.resolve(),
            args.checkpoint_a0.resolve(),
            args.max_batches,
        )
        report["models"]["a1"] = run_model(
            "a1",
            args.config_a1.resolve(),
            args.checkpoint_a1.resolve(),
            args.max_batches,
        )
        report["status"] = "pass"
    except Exception as exc:
        report["status"] = "diagnostic_error"
        report["error"] = repr(exc)
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
