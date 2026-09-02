"""Bounded, read-only numerical-stability diagnostic for Architecture A1.

This script deliberately does not save model checkpoints or alter the project
implementation.  It restores a supplied checkpoint, continues the configured
training loop for a bounded number of epochs, records tensor-level numerical
statistics, and stops at the first unexpected non-finite value.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
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
from src.metrics.segmentation import METRIC_NAMES, batch_metric_values
from src.models.factory import build_model
from src.training.checkpoint import load_checkpoint
from src.training.config import choose_device, load_config, project_path
from src.utils.reproducibility import (
    make_dataloader_generator,
    seed_everything,
    seed_worker,
)
import src.models.vil_bottleneck_unet as vil_module


class NonFiniteDetected(RuntimeError):
    """Raised internally to stop immediately at the first unexpected value."""


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def tensor_stats(tensor: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, Any]:
    """Describe a tensor while separating expected causal-mask -inf values."""

    value = tensor.detach()
    if valid_mask is not None:
        mask = valid_mask.to(device=value.device, dtype=torch.bool)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(value)
        intentional_negative_inf = (~mask) & torch.isneginf(value)
    else:
        mask = torch.ones_like(value, dtype=torch.bool)
        intentional_negative_inf = torch.zeros_like(value, dtype=torch.bool)

    nan_mask = torch.isnan(value)
    inf_mask = torch.isinf(value)
    unexpected_nan = nan_mask
    unexpected_inf = inf_mask & ~intentional_negative_inf
    unexpected = unexpected_nan | unexpected_inf
    finite_mask = torch.isfinite(value)
    finite_values = value[finite_mask]

    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "finite": bool(finite_mask.all().item()),
        "finite_excluding_expected_mask": not bool(unexpected.any().item()),
        "nan_count": int(nan_mask.sum().item()),
        "inf_count": int(inf_mask.sum().item()),
        "intentional_mask_negative_inf_count": int(
            intentional_negative_inf.sum().item()
        ),
        "unexpected_nan_count": int(unexpected_nan.sum().item()),
        "unexpected_inf_count": int(unexpected_inf.sum().item()),
        "unexpected_nonfinite": bool(unexpected.any().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "abs_max": float(finite_values.abs().max().item()),
            }
        )
    else:
        result.update({"min": None, "max": None, "mean": None, "abs_max": None})
    return result


def _first_nonfinite_parameter(
    model: torch.nn.Module,
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any]]:
    first_name = None
    first_stats = None
    all_stats: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        stats = tensor_stats(parameter)
        all_stats[name] = stats
        if first_name is None and stats["unexpected_nonfinite"]:
            first_name = name
            first_stats = stats
    return first_name, first_stats, all_stats


def _gradient_summary(
    model: torch.nn.Module,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None, dict[str, Any]]:
    first_name = None
    first_stats = None
    all_stats: dict[str, Any] = {}
    none_gradients: list[str] = []
    maximum_abs_finite = 0.0
    total_nan = 0
    total_inf = 0
    finite_gradient_count = 0
    gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            none_gradients.append(name)
            continue
        gradient_count += 1
        stats = tensor_stats(parameter.grad)
        all_stats[name] = stats
        total_nan += stats["nan_count"]
        total_inf += stats["inf_count"]
        if stats["abs_max"] is not None:
            maximum_abs_finite = max(maximum_abs_finite, stats["abs_max"])
        if stats["unexpected_nonfinite"] and first_name is None:
            first_name = name
            first_stats = stats
        if not stats["unexpected_nonfinite"]:
            finite_gradient_count += 1
    summary = {
        "gradient_parameter_count": gradient_count,
        "finite_gradient_parameter_count": finite_gradient_count,
        "none_gradient_count": len(none_gradients),
        "none_gradients": none_gradients,
        "all_finite": first_name is None,
        "maximum_abs_finite_gradient": maximum_abs_finite,
        "nan_count": total_nan,
        "inf_count": total_inf,
    }
    return summary, first_name, first_stats, all_stats


def _optimizer_state_summary(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], tuple[str, str] | None, dict[str, Any] | None, dict[str, Any]]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    first_key: tuple[str, str] | None = None
    first_stats = None
    all_stats: dict[str, Any] = {}
    state_tensor_count = 0
    finite_state_tensor_count = 0
    maximum_abs_finite = 0.0
    total_nan = 0
    total_inf = 0
    for parameter, state in optimizer.state.items():
        parameter_name = parameter_names.get(id(parameter), "<unnamed>")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if not torch.is_tensor(value):
                continue
            state_tensor_count += 1
            stats = tensor_stats(value)
            label = f"{parameter_name}.{key}"
            all_stats[label] = stats
            total_nan += stats["nan_count"]
            total_inf += stats["inf_count"]
            if stats["abs_max"] is not None:
                maximum_abs_finite = max(maximum_abs_finite, stats["abs_max"])
            if stats["unexpected_nonfinite"] and first_key is None:
                first_key = (parameter_name, key)
                first_stats = stats
            if not stats["unexpected_nonfinite"]:
                finite_state_tensor_count += 1
    summary = {
        "state_tensor_count": state_tensor_count,
        "finite_state_tensor_count": finite_state_tensor_count,
        "all_finite": first_key is None,
        "maximum_abs_finite_state": maximum_abs_finite,
        "nan_count": total_nan,
        "inf_count": total_inf,
    }
    return summary, first_key, first_stats, all_stats


class DiagnosticState:
    def __init__(self) -> None:
        self.epoch = None
        self.batch_index = None
        self.phase = None
        self.current_block = None
        self.current_record: dict[str, Any] | None = None
        self.current_forward: dict[str, Any] | None = None
        self.first_failure: dict[str, Any] | None = None

    def begin_batch(self, phase: str, epoch: int, batch_index: int, sample_ids: Any) -> dict[str, Any]:
        self.phase = phase
        self.epoch = epoch
        self.batch_index = batch_index
        self.current_block = None
        self.current_forward = {"stages": [], "mlstm_calls": []}
        self.current_record = {
            "phase": phase,
            "epoch": epoch,
            "batch_index": batch_index,
            "sample_ids": [str(value) for value in sample_ids],
            "forward": self.current_forward,
        }
        return self.current_record

    def add_stage(
        self,
        label: str,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if self.current_forward is None:
            return
        stats = tensor_stats(value, valid_mask=valid_mask)
        self.current_forward["stages"].append({"tensor": label, **stats})
        if stats["unexpected_nonfinite"]:
            self.fail(
                kind="forward_tensor",
                operation=label,
                block=self.current_block,
                tensor_stats=stats,
            )

    def fail(self, **event: Any) -> None:
        if self.first_failure is None:
            if self.current_forward is not None:
                event.setdefault("mLSTM_calls", self.current_forward["mlstm_calls"])
            self.first_failure = {
                "epoch": self.epoch,
                "batch_index": self.batch_index,
                "phase": self.phase,
                **event,
            }
        raise NonFiniteDetected(str(event))


def _build_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
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
        generator=make_dataloader_generator(
            int(config["seed"]) + (0 if shuffle else 1)
        ),
    )


def _install_hooks(model: torch.nn.Module, state: DiagnosticState) -> list[Any]:
    processor = model.bottleneck_processor
    handles: list[Any] = []

    def pre_hook(label: str, block: str | None = None):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if block is not None:
                state.current_block = block
            value = inputs[0]
            if torch.is_tensor(value):
                state.add_stage(label, value)

        return hook

    def post_hook(label: str, block: str | None = None):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output
            if torch.is_tensor(value):
                state.add_stage(label, value)
            if block is not None:
                state.current_block = None

        return hook

    handles.append(model.bottleneck.register_forward_pre_hook(pre_hook("cnn_bottleneck_input")))

    for index, block in enumerate(processor.forward_blocks):
        block_name = f"forward_{index}"
        handles.append(
            block.register_forward_pre_hook(
                pre_hook(f"forward_vil_block_{index}_input", block_name)
            )
        )
        handles.append(
            block.register_forward_hook(
                post_hook(f"forward_vil_block_{index}_output", block_name)
            )
        )
        handles.append(
            block.mlstm.register_forward_hook(
                post_hook(f"forward_mlstm_{index}_output")
            )
        )

    for index, block in enumerate(processor.reverse_blocks):
        block_name = f"reverse_{index}"
        handles.append(
            block.register_forward_pre_hook(
                pre_hook(f"reversed_sequence_{index}", block_name)
            )
        )
        handles.append(
            block.register_forward_pre_hook(
                pre_hook(f"reverse_vil_block_{index}_input", block_name)
            )
        )
        handles.append(
            block.register_forward_hook(
                post_hook(f"reverse_vil_block_{index}_output", block_name)
            )
        )
        handles.append(
            block.mlstm.register_forward_hook(
                post_hook(f"reverse_mlstm_{index}_output")
            )
        )

    handles.append(
        processor.register_forward_hook(post_hook("restored_spatial_bottleneck"))
    )
    for index, decoder in enumerate(model.decoders):
        handles.append(decoder.register_forward_hook(post_hook(f"decoder_{index}_output")))
    handles.append(model.head.register_forward_hook(post_hook("final_logits")))
    return handles


def _install_mlstm_instrumentation(state: DiagnosticState):
    original = vil_module._parallel_stabilized_mlstm

    def instrumented_parallel(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        input_gate: torch.Tensor,
        forget_gate: torch.Tensor,
        causal_mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        call: dict[str, Any] = {
            "block": state.current_block,
            "stages": [],
        }
        if state.current_forward is not None:
            state.current_forward["mlstm_calls"].append(call)

        def add(label: str, value: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
            stats = tensor_stats(value, valid_mask=valid_mask)
            call["stages"].append({"tensor": label, **stats})
            if stats["unexpected_nonfinite"]:
                state.fail(
                    kind="mLSTM_forward_operation",
                    operation=label,
                    block=state.current_block,
                    tensor_stats=stats,
                    max_log_decay=_find_stage(call, "max_log_decay"),
                    exp_neg_max_log_decay=_find_stage(call, "exp_neg_max_log_decay"),
                )

        with torch.no_grad():
            q = queries.detach()
            k = keys.detach()
            v = values.detach()
            ig = input_gate.detach()
            fg = forget_gate.detach()
            add("queries", q)
            add("keys", k)
            add("values", v)
            add("input_gate", ig)
            add("forget_gate", fg)
            log_forget = F.logsigmoid(fg)
            add("log_forget", log_forget)
            log_forget_cumsum = torch.cat(
                [
                    torch.zeros_like(log_forget[:, :, :1]),
                    torch.cumsum(log_forget, dim=-2),
                ],
                dim=-2,
            )
            add("cumulative_log_decay", log_forget_cumsum)
            repeated = log_forget_cumsum.expand(
                -1, -1, -1, log_forget_cumsum.shape[-2]
            )
            log_forget_matrix = repeated - repeated.transpose(-2, -1)
            add("log_forget_matrix_unmasked", log_forget_matrix)
            valid_mask = causal_mask.view(1, 1, *causal_mask.shape).expand(
                log_forget_matrix.shape[0], log_forget_matrix.shape[1], -1, -1
            )
            log_forget_matrix = torch.where(
                valid_mask,
                log_forget_matrix[:, :, 1:, 1:],
                torch.full_like(log_forget_matrix[:, :, 1:, 1:], -float("inf")),
            )
            add("log_forget_matrix_masked", log_forget_matrix, valid_mask)
            log_decay = log_forget_matrix + ig.transpose(-2, -1)
            add("log_decay", log_decay, valid_mask)
            max_log_decay = log_decay.amax(dim=-1, keepdim=True)
            add("max_log_decay", max_log_decay)
            decay = torch.exp(log_decay - max_log_decay)
            add("stabilized_decay", decay)
            keys_scaled = k / math.sqrt(q.shape[-1])
            add("keys_scaled", keys_scaled)
            combination = (q @ keys_scaled.transpose(-2, -1)) * decay
            add("combination", combination)
            normalizer_exp = torch.exp(-max_log_decay)
            add("exp_neg_max_log_decay", normalizer_exp)
            normalizer = torch.maximum(
                combination.sum(dim=-1, keepdim=True).abs(),
                normalizer_exp,
            )
            add("normalization_denominator", normalizer)
            normalized_combination = combination / (normalizer + eps)
            add("normalized_combination", normalized_combination)
            result = normalized_combination @ v
            add("mlstm_output", result)
        return original(queries, keys, values, input_gate, forget_gate, causal_mask, eps)

    vil_module._parallel_stabilized_mlstm = instrumented_parallel
    return original


def _find_stage(call: dict[str, Any], label: str) -> dict[str, Any] | None:
    for stage in reversed(call.get("stages", [])):
        if stage.get("tensor") == label:
            return stage
    return None


def _record_loss(record: dict[str, Any], loss: torch.Tensor) -> None:
    record["loss"] = tensor_stats(loss)


def _epoch_summary(
    epoch: int,
    batch_records: list[dict[str, Any]],
    validation: dict[str, float] | None,
    learning_rate_before_scheduler_step: list[float],
    learning_rate_after_scheduler_step: list[float],
) -> dict[str, Any]:
    train_losses = [
        (record["loss"]["mean"], record.get("batch_size", 1))
        for record in batch_records
        if record["phase"] == "train"
        and record.get("loss", {}).get("mean") is not None
    ]
    train_sample_count = sum(batch_size for _, batch_size in train_losses)
    train_loss_total = sum(loss * batch_size for loss, batch_size in train_losses)
    return {
        "epoch": epoch,
        "train_batch_count": sum(record["phase"] == "train" for record in batch_records),
        "validation_batch_count": sum(
            record["phase"] == "validation" for record in batch_records
        ),
        "train_loss_mean": (
            train_loss_total / train_sample_count if train_sample_count else None
        ),
        "validation": validation,
        "learning_rate_before_scheduler_step": learning_rate_before_scheduler_step,
        "learning_rate_after_scheduler_step": learning_rate_after_scheduler_step,
    }


def _aggregate_validation(records: list[dict[str, Any]]) -> dict[str, float]:
    """Match ``training.engine.evaluate`` without an extra forward pass."""

    loss_total = 0.0
    sample_total = 0
    metric_total = torch.zeros(len(METRIC_NAMES), dtype=torch.float64)
    metric_square_total = torch.zeros(len(METRIC_NAMES), dtype=torch.float64)
    for record in records:
        batch_size = int(record["batch_size"])
        loss_total += float(record["loss"]["mean"]) * batch_size
        values = torch.tensor(record["metric_values"], dtype=torch.float64)
        metric_total += values.sum(dim=0)
        metric_square_total += (values * values).sum(dim=0)
        sample_total += batch_size
    denominator = max(sample_total, 1)
    means = metric_total / denominator
    variances = (metric_square_total / denominator - means * means).clamp_min(0.0)
    result = {"loss": loss_total / denominator}
    for index, name in enumerate(METRIC_NAMES):
        result[name] = float(means[index].item())
        result[f"{name}_std"] = float(variances[index].sqrt().item())
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/colab_vil_bottleneck_a1.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="number of continuation epochs after the restored checkpoint epoch",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="optional diagnostic-only batch cap per phase and epoch",
    )
    parser.add_argument(
        "--anomaly-detection",
        action="store_true",
        help="also enable torch autograd anomaly detection during backward",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when supplied")

    config = load_config((PROJECT_ROOT / args.config).resolve())
    if config["model"]["name"] != "unet_vil_bottleneck_a1":
        raise ValueError("diagnostic requires model.name=unet_vil_bottleneck_a1")
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if checkpoint_path == output_path:
        raise ValueError("--output must not overwrite --checkpoint")
    checkpoint_preview = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_seed = int(checkpoint_preview.get("seed", config["seed"]))
    seed_everything(checkpoint_seed)
    device = choose_device(config)

    state = DiagnosticState()
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic": "A1 numerical-stability continuation from supplied checkpoint",
        "exact_epoch26_replay": False,
        "configuration": {
            "config_path": str(config["_config_path"]),
            "checkpoint_path": str(checkpoint_path),
            "output_path": str(output_path),
            "requested_continuation_epochs": args.epochs,
            "max_batches": args.max_batches,
            "anomaly_detection": args.anomaly_detection,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "checkpoint": {
            "fields": sorted(checkpoint_preview.keys()),
            "checkpoint_epoch": int(checkpoint_preview.get("epoch", -1)),
            "checkpoint_seed": checkpoint_seed,
            "config_seed": int(config["seed"]),
            "seed_matches_config": checkpoint_seed == int(config["seed"]),
            "restored": {
                "model_state": "model_state" in checkpoint_preview,
                "optimizer_state": "optimizer_state" in checkpoint_preview,
                "scheduler_state": "scheduler_state" in checkpoint_preview,
                "epoch": "epoch" in checkpoint_preview,
                "validation_metrics": "validation_metrics" in checkpoint_preview,
                "config": "config" in checkpoint_preview,
                "seed": "seed" in checkpoint_preview,
                "rng_state": "rng_state" in checkpoint_preview,
            },
        },
        "epochs": [],
        "batches": [],
        "failure": None,
    }

    if int(checkpoint_preview.get("epoch", -1)) < 0:
        raise ValueError("checkpoint does not contain a valid epoch")
    start_epoch = int(checkpoint_preview["epoch"])
    end_epoch = start_epoch + args.epochs
    train_loader = _build_loader(config, "train", shuffle=True)
    validation_loader = _build_loader(config, "validation", shuffle=False)
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
    loaded = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        map_location=device,
    )
    report["checkpoint"]["loaded_epoch"] = int(loaded["epoch"])
    report["checkpoint"]["loaded_seed"] = int(loaded["seed"])
    report["checkpoint"]["loaded_validation_metrics"] = loaded.get("validation_metrics")
    report["checkpoint"]["scheduler_last_lr_after_restore"] = [
        float(value) for value in scheduler.get_last_lr()
    ]
    report["model"] = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "features": list(config["model"]["features"]),
        "input_size": list(config["dataset"]["image_size"]),
        "batch_size": int(config["training"]["batch_size"]),
    }

    handles = _install_hooks(model, state)
    original_parallel = _install_mlstm_instrumentation(state)
    stop_requested = False
    started = time.perf_counter()

    def run_train_batch(epoch: int, batch_index: int, batch: dict[str, Any]) -> dict[str, Any]:
        record = state.begin_batch("train", epoch, batch_index, batch["id"])
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        record["batch_size"] = int(images.shape[0])
        record["input"] = tensor_stats(images)
        record["target"] = tensor_stats(masks)
        optimizer.zero_grad(set_to_none=True)
        try:
            logits = model(images)
            record["output_shape"] = list(logits.shape)
            loss = criterion(logits, masks)
            _record_loss(record, loss)
            if not torch.isfinite(loss).item():
                state.fail(
                    kind="loss",
                    operation="BCE_plus_soft_Dice_loss",
                    tensor_stats=record["loss"],
                )
            anomaly_context = (
                torch.autograd.detect_anomaly(check_nan=True)
                if args.anomaly_detection
                else nullcontext()
            )
            with anomaly_context:
                loss.backward()
        except NonFiniteDetected:
            raise
        except Exception as exc:
            record["backward_exception"] = repr(exc)
            state.fail(
                kind="backward_exception",
                operation="loss.backward",
                exception=repr(exc),
            )
        gradient_summary, first_name, first_stats, all_gradient_stats = _gradient_summary(model)
        record["gradients"] = gradient_summary
        if first_name is not None:
            record["gradients"]["all_parameter_stats"] = all_gradient_stats
            state.fail(
                kind="gradient",
                operation="loss.backward",
                parameter=first_name,
                tensor_stats=first_stats,
                maximum_abs_gradient=gradient_summary["maximum_abs_finite_gradient"],
                nan_count=gradient_summary["nan_count"],
                inf_count=gradient_summary["inf_count"],
            )
        try:
            optimizer.step()
        except Exception as exc:
            record["optimizer_exception"] = repr(exc)
            state.fail(
                kind="optimizer_step_exception",
                operation="optimizer.step",
                exception=repr(exc),
            )
        first_parameter, first_parameter_stats, all_parameter_stats = _first_nonfinite_parameter(model)
        parameter_summary = {
            "all_finite": first_parameter is None,
            "first_nonfinite_parameter": first_parameter,
            "maximum_abs_finite_parameter": max(
                (stats["abs_max"] or 0.0) for stats in all_parameter_stats.values()
            ),
        }
        record["parameters_after_optimizer_step"] = parameter_summary
        if first_parameter is not None:
            record["parameters_after_optimizer_step"]["all_parameter_stats"] = all_parameter_stats
            state.fail(
                kind="parameter",
                operation="optimizer.step",
                parameter=first_parameter,
                tensor_stats=first_parameter_stats,
            )
        state_summary, first_state, first_state_stats, all_state_stats = _optimizer_state_summary(
            model, optimizer
        )
        record["optimizer_state_after_step"] = state_summary
        if first_state is not None:
            record["optimizer_state_after_step"]["all_state_stats"] = all_state_stats
            state.fail(
                kind="optimizer_state",
                operation="optimizer.step",
                parameter=first_state[0],
                state_tensor=first_state[1],
                tensor_stats=first_state_stats,
            )
        record["learning_rate"] = [float(group["lr"]) for group in optimizer.param_groups]
        return record

    def run_validation_batch(
        epoch: int, batch_index: int, batch: dict[str, Any]
    ) -> dict[str, Any]:
        record = state.begin_batch("validation", epoch, batch_index, batch["id"])
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        record["batch_size"] = int(images.shape[0])
        record["input"] = tensor_stats(images)
        record["target"] = tensor_stats(masks)
        with torch.no_grad():
            logits = model(images)
            record["output_shape"] = list(logits.shape)
            loss = criterion(logits, masks)
            metric_values = batch_metric_values(
                logits,
                masks,
                threshold=float(config["dataset"]["prediction_threshold"]),
            ).cpu()
        _record_loss(record, loss)
        record["metric_values"] = metric_values.tolist()
        if not torch.isfinite(loss).item():
            state.fail(
                kind="loss",
                operation="validation_BCE_plus_soft_Dice_loss",
                tensor_stats=record["loss"],
            )
        return record

    try:
        for epoch in range(start_epoch + 1, end_epoch + 1):
            model.train()
            epoch_records: list[dict[str, Any]] = []
            for batch_index, batch in enumerate(train_loader, start=1):
                record = run_train_batch(epoch, batch_index, batch)
                epoch_records.append(record)
                report["batches"].append(record)
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
            model.eval()
            validation_records: list[dict[str, Any]] = []
            for batch_index, batch in enumerate(validation_loader, start=1):
                record = run_validation_batch(epoch, batch_index, batch)
                validation_records.append(record)
                report["batches"].append(record)
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
            validation_result = _aggregate_validation(validation_records)
            learning_rate_before_scheduler_step = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            scheduler.step()
            learning_rate_after_scheduler_step = [
                float(value) for value in scheduler.get_last_lr()
            ]
            if not all(math.isfinite(value) for value in learning_rate_after_scheduler_step):
                state.fail(
                    kind="scheduler",
                    operation="scheduler.step",
                    learning_rates=learning_rate_after_scheduler_step,
                )
            report["epochs"].append(
                _epoch_summary(
                    epoch,
                    epoch_records + validation_records,
                    validation_result,
                    learning_rate_before_scheduler_step,
                    learning_rate_after_scheduler_step,
                )
            )
    except NonFiniteDetected as exc:
        report["status"] = "nonfinite_detected"
        report["failure"] = state.first_failure or {"exception": str(exc)}
        if state.current_record is not None and state.current_record not in report["batches"]:
            report["batches"].append(state.current_record)
    except Exception as exc:
        report["status"] = "diagnostic_error"
        report["failure"] = {
            "epoch": state.epoch,
            "batch_index": state.batch_index,
            "phase": state.phase,
            "kind": "unexpected_diagnostic_exception",
            "exception": repr(exc),
        }
        if state.current_record is not None and state.current_record not in report["batches"]:
            report["batches"].append(state.current_record)
    else:
        report["status"] = "completed_bounded_diagnostic_no_nonfinite"
    finally:
        vil_module._parallel_stabilized_mlstm = original_parallel
        for handle in handles:
            handle.remove()

    report["elapsed_seconds"] = time.perf_counter() - started
    report["continuation"] = {
        "restored_epoch": start_epoch,
        "first_epoch_checked": start_epoch + 1,
        "last_epoch_checked": (
            state.epoch
            if state.first_failure is not None
            else end_epoch
        ),
        "requested_last_epoch": end_epoch,
        "same_epoch26_replay": False,
        "rng_state_restored": False,
        "rng_limitation": (
            "The checkpoint contains a seed but no Python/NumPy/PyTorch/DataLoader RNG states; "
            "the continuation is controlled from seed 42 but is not an exact replay of the "
            "original post-epoch-15 batch order."
        ),
    }
    write_json(output_path, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(output_path),
        "restored_epoch": start_epoch,
        "last_epoch_checked": report["continuation"]["last_epoch_checked"],
        "failure": report["failure"],
    }, sort_keys=True))
    return 1 if report["status"] in {"nonfinite_detected", "diagnostic_error"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
