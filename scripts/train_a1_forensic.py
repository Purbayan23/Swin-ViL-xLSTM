"""Forensic A1 training run with reversible numerical instrumentation.

This script deliberately preserves the existing A1 implementation and training
protocol.  It adds only observational hooks, reproducibility-state capture,
bounded execution, and fail-fast forensic artifacts.

Examples
--------
Bounded local validation::

    python scripts/train_a1_forensic.py \
        --config configs/vil_bottleneck_a1.json \
        --output experiments/runs/architecture_a1_forensic_seed42_sanity \
        --max-epochs 1 --max-batches 1 --max-validation-batches 1 \
        --sanity-check

Full Colab forensic run (do not execute automatically)::

    python scripts/train_a1_forensic.py \
        --config configs/colab_vil_bottleneck_a1.json \
        --output /content/drive/MyDrive/Project_ViL/experiments/runs/architecture_a1_forensic_seed42
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kvasir_seg import KvasirSegDataset
from src.losses.segmentation import BCESoftDiceLoss
from src.metrics.segmentation import METRIC_NAMES, batch_metric_values
from src.models.factory import build_model
from src.models import vil_bottleneck_a1n as vil_a1n_module
from src.models import vil_bottleneck_unet as vil_module
from src.models.vil_bottleneck_unet import ViLMLSTMBlock
from src.training.config import choose_device, load_config, project_path
from src.utils.reproducibility import make_dataloader_generator, seed_everything, seed_worker


class ForensicFailure(RuntimeError):
    """Expected fail-fast exception carrying the first observed failure."""

    def __init__(
        self,
        *,
        phase: str,
        stage: str,
        tensor_name: str,
        stats: dict[str, Any] | None = None,
        direction: str | None = None,
        message: str | None = None,
    ) -> None:
        self.phase = phase
        self.stage = stage
        self.tensor_name = tensor_name
        self.stats = stats or {}
        self.direction = direction
        super().__init__(message or f"non-finite value at {phase}:{stage}:{tensor_name}")


WarningConditionKey = tuple[str, str, str, float]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def environment_summary(device: torch.device) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
    }


def capture_rng_state(
    train_generator: torch.Generator | None = None,
    validation_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "dataloader_train": train_generator.get_state() if train_generator is not None else None,
        "dataloader_validation": (
            validation_generator.get_state() if validation_generator is not None else None
        ),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def save_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_validation_dice: float,
    train_generator: torch.Generator | None,
    validation_generator: torch.Generator | None,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "forensic_state_format": 1,
        "label": label,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_dice": float(best_validation_dice),
        "rng_state": capture_rng_state(train_generator, validation_generator),
        "seed": int(config["seed"]),
    }
    torch.save(payload, path)


def _finite_values(tensor: torch.Tensor) -> torch.Tensor:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    if finite.any():
        return detached[finite].to(dtype=torch.float64)
    return torch.empty(0, dtype=torch.float64, device=detached.device)


def tensor_stats(
    tensor: torch.Tensor,
    *,
    intentional_neg_inf: int = 0,
) -> dict[str, Any]:
    detached = tensor.detach()
    total = int(detached.numel())
    nan_count = int(torch.isnan(detached).sum().item()) if detached.is_floating_point() else 0
    pos_inf_count = int(torch.isposinf(detached).sum().item()) if detached.is_floating_point() else 0
    neg_inf_count = int(torch.isneginf(detached).sum().item()) if detached.is_floating_point() else 0
    finite_count = int(torch.isfinite(detached).sum().item()) if detached.is_floating_point() else total
    unexpected_neg_inf = max(neg_inf_count - int(intentional_neg_inf), 0)
    unexpected_nonfinite = nan_count + pos_inf_count + unexpected_neg_inf
    finite_values = _finite_values(detached) if detached.is_floating_point() else detached.reshape(-1).to(torch.float64)
    if finite_values.numel():
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        absolute_maximum = float(finite_values.abs().max().item())
        mean = float(finite_values.mean().item())
        standard_deviation = float(finite_values.std(unbiased=False).item())
    else:
        minimum = maximum = absolute_maximum = mean = standard_deviation = None
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": total,
        "finite_count": finite_count,
        "finite_fraction": (finite_count / total) if total else 1.0,
        "num_nan": nan_count,
        "num_pos_inf": pos_inf_count,
        "num_neg_inf": neg_inf_count,
        "intentional_neg_inf": int(intentional_neg_inf),
        "unexpected_nonfinite": unexpected_nonfinite,
        "min": minimum,
        "max": maximum,
        "abs_max": absolute_maximum,
        "mean": mean,
        "std": standard_deviation,
    }


class ForensicObserver:
    """Collect scalar tensor summaries and raise at the first unexpected value."""

    ABS_THRESHOLDS = (1e2, 1e3, 1e4, 1e5, 1e6)
    LOG_DECAY_THRESHOLDS = (50.0, 70.0, 80.0, 88.0)

    def __init__(self) -> None:
        self.current: dict[str, Any] | None = None
        self.current_details: dict[str, dict[str, Any]] = {}
        self.block_values: dict[str, dict[str, torch.Tensor]] = {}
        self.max_observed: dict[str, dict[str, float | int | None]] = {}
        self.current_epoch: int | None = None
        self.epoch_max_observed: dict[str, dict[str, Any]] = {}
        # Warning episodes persist across forward passes.  Each condition is
        # keyed by its scope, statistic, and threshold, so crossing a higher
        # threshold is a new episode even if a lower threshold remains active.
        self.warning_active: dict[WarningConditionKey, bool] = {}
        self.warning_episode_ids: dict[WarningConditionKey, int] = {}
        self.warning_lifecycle_events: list[dict[str, Any]] = []
        self.warning_event_sink: Any | None = None
        self.last_forward_detail: dict[str, Any] | None = None
        self.last_compact: dict[str, Any] = {}
        self.last_warning: bool = False
        self.last_warning_reasons: list[str] = []
        self.last_warning_crossings: list[str] = []
        self.last_warning_crossing_details: list[dict[str, Any]] = []
        self.current_warning_lifecycle_events: list[dict[str, Any]] = []

    def begin_forward(
        self,
        *,
        split: str,
        epoch: int,
        batch_index: int,
        global_step: int,
        sample_ids: list[str],
    ) -> None:
        self.current = {
            "split": split,
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "global_step": int(global_step),
            "sample_ids": list(sample_ids),
        }
        self.current_details = {}
        self.block_values = {}
        self.last_warning = False
        self.last_warning_reasons = []
        self.last_warning_crossings = []
        self.last_warning_crossing_details = []
        self.current_warning_lifecycle_events = []

    def begin_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.epoch_max_observed = {}

    def _record_epoch_observation(self, key: str, stats: dict[str, Any]) -> None:
        if self.current_epoch is None:
            return
        summary = self.epoch_max_observed.setdefault(key, _summary_template())
        _merge_scalar_summary(summary, stats)

    def _scope_key(self, scope: str, name: str) -> str:
        return f"{scope}/{name}"

    def set_warning_event_sink(self, sink: Any | None) -> None:
        """Send lifecycle events directly to a bounded JSONL artifact."""

        self.warning_event_sink = sink

    def _warning_context(self) -> dict[str, Any]:
        current = self.current or {}
        return {
            "epoch": int(current.get("epoch", -1)),
            "batch": int(current.get("batch_index", -1)),
            "batch_index": int(current.get("batch_index", -1)),
            "global_step": int(current.get("global_step", -1)),
            "split": str(current.get("split", "unknown")),
        }

    def _emit_warning_lifecycle_event(
        self,
        event_type: str,
        condition_key: WarningConditionKey,
        *,
        previous_state: bool | None,
        current_state: bool,
        episode_id: int | None,
        **extra: Any,
    ) -> dict[str, Any]:
        split, scope, metric, threshold = condition_key
        event = {
            "event_type": event_type,
            **self._warning_context(),
            "scope": scope,
            "metric": metric,
            "threshold": float(threshold),
            "previous_state": previous_state,
            "current_state": bool(current_state),
            "episode_id": episode_id,
            **extra,
        }
        # The split is part of the stable key and is repeated explicitly in
        # the event for simple downstream auditing.
        event["split"] = split
        self.warning_lifecycle_events.append(event)
        self.current_warning_lifecycle_events.append(event)
        if self.warning_event_sink is not None:
            self.warning_event_sink(event)
        return event

    def _observe_warning_condition(
        self,
        *,
        scope: str,
        metric: str,
        threshold: float,
        active: bool,
        reason: str,
    ) -> None:
        condition_key: WarningConditionKey = (
            str((self.current or {}).get("split", "unknown")),
            str(scope),
            str(metric),
            float(threshold),
        )
        key_exists = condition_key in self.warning_active
        was_active = self.warning_active.get(condition_key, False)
        episode_id = self.warning_episode_ids.get(condition_key)

        if not key_exists:
            self._emit_warning_lifecycle_event(
                "warning_state_created",
                condition_key,
                previous_state=None,
                current_state=active,
                episode_id=None,
            )

        if active and not was_active:
            episode_id = self.warning_episode_ids.get(condition_key, 0) + 1
            self.warning_episode_ids[condition_key] = episode_id
            self._emit_warning_lifecycle_event(
                "warning_episode_started",
                condition_key,
                previous_state=False,
                current_state=True,
                episode_id=episode_id,
                reason=reason,
            )
            self.last_warning_crossings.append(reason)
            self.last_warning_crossing_details.append(
                {
                    "split": condition_key[0],
                    "scope": condition_key[1],
                    "metric": condition_key[2],
                    "threshold": condition_key[3],
                    "episode_id": episode_id,
                    "reason": reason,
                }
            )

        self.warning_active[condition_key] = bool(active)
        if active:
            self._emit_warning_lifecycle_event(
                "warning_condition_true",
                condition_key,
                previous_state=was_active if key_exists else False,
                current_state=True,
                episode_id=episode_id,
                reason=reason,
            )
        elif was_active:
            self._emit_warning_lifecycle_event(
                "warning_state_reset",
                condition_key,
                previous_state=True,
                current_state=False,
                episode_id=episode_id,
            )

    def _warning_check(self, scope: str, name: str, stats: dict[str, Any]) -> None:
        absolute_maximum = stats.get("abs_max")
        if absolute_maximum is not None:
            for threshold in self.ABS_THRESHOLDS:
                metric = f"{name}:abs_max"
                active = absolute_maximum > threshold
                reason = f"{scope}/{name}:abs_max>{threshold:g}"
                self._observe_warning_condition(
                    scope=scope,
                    metric=metric,
                    threshold=float(threshold),
                    active=active,
                    reason=reason,
                )
                if active:
                    if reason not in self.last_warning_reasons:
                        self.last_warning_reasons.append(reason)
        if name == "max_log_decay" and stats.get("min") is not None:
            negative_minimum = -float(stats["min"])
            for threshold in self.LOG_DECAY_THRESHOLDS:
                metric = f"{name}:-min"
                active = negative_minimum > threshold
                reason = f"{scope}/{name}:-min>{threshold:g}"
                self._observe_warning_condition(
                    scope=scope,
                    metric=metric,
                    threshold=float(threshold),
                    active=active,
                    reason=reason,
                )
                if active:
                    if reason not in self.last_warning_reasons:
                        self.last_warning_reasons.append(reason)
        self.last_warning = bool(self.last_warning_crossings)

    def record_warning_checkpoint_saved(self, checkpoint_path: Path | str) -> None:
        """Record one lifecycle event for each crossing captured in a file."""

        for crossing in self.last_warning_crossing_details:
            condition_key: WarningConditionKey = (
                str(crossing["split"]),
                str(crossing["scope"]),
                str(crossing["metric"]),
                float(crossing["threshold"]),
            )
            self._emit_warning_lifecycle_event(
                "warning_checkpoint_saved",
                condition_key,
                previous_state=True,
                current_state=True,
                episode_id=int(crossing["episode_id"]),
                reason=str(crossing["reason"]),
                checkpoint_path=str(checkpoint_path),
            )

    def record_optimization_telemetry(
        self,
        *,
        gradients: dict[str, Any],
        parameters_before: dict[str, Any],
        parameters_after: dict[str, Any],
        optimizer_before: dict[str, Any],
        optimizer_after: dict[str, Any],
    ) -> None:
        """Add scalar optimization summaries to the current epoch only."""

        summaries = {
            "optimization/gradient": gradients,
            "optimization/parameters_before_step": parameters_before,
            "optimization/parameters_after_step": parameters_after,
            "optimization/adamw_exp_avg_before_step": optimizer_before["exp_avg"],
            "optimization/adamw_exp_avg_sq_before_step": optimizer_before["exp_avg_sq"],
            "optimization/adamw_exp_avg_after_step": optimizer_after["exp_avg"],
            "optimization/adamw_exp_avg_sq_after_step": optimizer_after["exp_avg_sq"],
        }
        for key, summary in summaries.items():
            self._record_epoch_observation(key, summary)

    def epoch_summary(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.epoch_max_observed)

    def record(
        self,
        scope: str,
        name: str,
        tensor: torch.Tensor,
        *,
        intentional_neg_inf: int = 0,
    ) -> dict[str, Any]:
        if self.current is None:
            return tensor_stats(tensor, intentional_neg_inf=intentional_neg_inf)
        stats = tensor_stats(tensor, intentional_neg_inf=intentional_neg_inf)
        key = self._scope_key(scope, name)
        self.current_details[key] = stats
        maximums = self.max_observed.setdefault(
            key,
            {
                "abs_max": None,
                "max": None,
                "min": None,
                "num_nan_max": 0,
                "num_pos_inf_max": 0,
                "num_neg_inf_max": 0,
            },
        )
        for field in ("abs_max", "max"):
            value = stats.get(field)
            if value is not None:
                previous = maximums[field]
                maximums[field] = float(value) if previous is None else max(float(previous), float(value))
        value = stats.get("min")
        if value is not None:
            previous = maximums["min"]
            maximums["min"] = float(value) if previous is None else min(float(previous), float(value))
        for field in ("num_nan", "num_pos_inf", "num_neg_inf"):
            maximums[f"{field}_max"] = max(
                int(maximums[f"{field}_max"] or 0), int(stats[field])
            )
        self._record_epoch_observation(key, stats)
        self._warning_check(scope, name, stats)
        if stats["unexpected_nonfinite"]:
            raise ForensicFailure(
                phase="forward",
                stage=name,
                tensor_name=key,
                stats=stats,
                direction=scope if scope.startswith("vil/") else None,
            )
        return stats

    def record_global(self, name: str, tensor: torch.Tensor) -> dict[str, Any]:
        return self.record("global", name, tensor)

    def enter_block(self, label: str, input_tensor: torch.Tensor) -> None:
        self.block_values[label] = {}
        self.record(f"vil/{label}", "input", input_tensor)

    def store_block_value(self, label: str, name: str, tensor: torch.Tensor) -> None:
        self.block_values.setdefault(label, {})[name] = tensor.detach()

    def observe_mlstm(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        input_gate: torch.Tensor,
        forget_gate: torch.Tensor,
        causal_mask: torch.Tensor,
        eps: float,
    ) -> None:
        """Mirror the existing function only for detached observation.

        The production function is still called separately and its return value
        is returned unchanged by the wrapper.  This copy is under no_grad and
        exists solely to expose intermediate statistics.
        """

        scope = f"vil/{self._active_block_label()}"
        with torch.no_grad():
            _, _, _, head_dim = queries.shape
            log_forget = F.logsigmoid(forget_gate)
            self.record(scope, "log_forget", log_forget)
            log_forget_cumsum = torch.cat(
                [
                    torch.zeros_like(log_forget[:, :, :1]),
                    torch.cumsum(log_forget, dim=-2),
                ],
                dim=-2,
            )
            self.record(scope, "cumulative_log_decay", log_forget_cumsum)
            repeated = log_forget_cumsum.expand(
                -1, -1, -1, log_forget_cumsum.shape[-2]
            )
            log_forget_matrix = repeated - repeated.transpose(-2, -1)
            masked_log_forget_matrix = torch.where(
                causal_mask,
                log_forget_matrix[:, :, 1:, 1:],
                torch.full_like(log_forget_matrix[:, :, 1:, 1:], -float("inf")),
            )
            expected_masked = (
                int((~causal_mask).sum().item())
                * int(masked_log_forget_matrix.shape[0])
                * int(masked_log_forget_matrix.shape[1])
            )
            self.record(
                scope,
                "log_forget_matrix",
                masked_log_forget_matrix,
                intentional_neg_inf=expected_masked,
            )
            log_decay = masked_log_forget_matrix + input_gate.transpose(-2, -1)
            self.record(
                scope,
                "log_decay",
                log_decay,
                intentional_neg_inf=expected_masked,
            )
            max_log_decay = log_decay.amax(dim=-1, keepdim=True)
            self.record(scope, "max_log_decay", max_log_decay)
            stabilized_decay = torch.exp(log_decay - max_log_decay)
            self.record(scope, "stabilized_decay", stabilized_decay)
            keys_scaled = keys / math.sqrt(head_dim)
            qk_matrix = queries @ keys_scaled.transpose(-2, -1)
            combination = qk_matrix * stabilized_decay
            self.record(scope, "combination", combination)
            row_sum = combination.sum(dim=-1, keepdim=True).abs()
            exp_neg_max_log_decay = torch.exp(-max_log_decay)
            self.record(scope, "exp_neg_max_log_decay", exp_neg_max_log_decay)
            normalizer = torch.maximum(row_sum, exp_neg_max_log_decay)
            self.record(scope, "normalizer", normalizer)
            normalized_combination = combination / (normalizer + eps)
            self.record(scope, "normalized_combination", normalized_combination)
            raw_output = normalized_combination @ values
            self.record(scope, "raw_mlstm_output", raw_output)

    def _active_block_label(self) -> str:
        if self.block_values:
            return next(reversed(self.block_values))
        return "unknown"

    def finish_forward(self) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
        if self.current is None:
            return {}, None, False
        compact_blocks: dict[str, Any] = {}
        for key, stats in self.current_details.items():
            if not key.startswith("vil/"):
                continue
            parts = key.split("/", 2)
            if len(parts) != 3:
                continue
            label, name = parts[1], parts[2]
            if label not in compact_blocks:
                compact_blocks[label] = {
                    "max_abs_max": None,
                    "min_max_log_decay": None,
                    "max_exp_neg_max_log_decay": None,
                    "max_final_output": None,
                }
            if stats.get("abs_max") is not None:
                compact_blocks[label]["max_abs_max"] = max(
                    compact_blocks[label]["max_abs_max"] or 0.0,
                    float(stats["abs_max"]),
                )
            if name == "max_log_decay" and stats.get("min") is not None:
                compact_blocks[label]["min_max_log_decay"] = float(stats["min"])
            if name == "exp_neg_max_log_decay" and stats.get("max") is not None:
                compact_blocks[label]["max_exp_neg_max_log_decay"] = float(stats["max"])
            if name == "final_vil_residual_output" and stats.get("abs_max") is not None:
                compact_blocks[label]["max_final_output"] = float(stats["abs_max"])
        compact = {
            **self.current,
            "warnings": list(self.last_warning_reasons),
            "warning_crossings": list(self.last_warning_crossings),
            "warning_crossed": self.last_warning,
            "warning_lifecycle_event_count": len(self.current_warning_lifecycle_events),
            "vil": compact_blocks,
        }
        detail = None
        if self.last_warning:
            detail = {
                **self.current,
                "event_type": "warning",
                "warnings": list(self.last_warning_reasons),
                "warning_crossings": list(self.last_warning_crossings),
                "warning_lifecycle_events": copy.deepcopy(self.current_warning_lifecycle_events),
                "stages": copy.deepcopy(self.current_details),
            }
        self.last_compact = compact
        self.last_forward_detail = detail
        return compact, detail, self.last_warning

    def failure_detail(self, failure: ForensicFailure) -> dict[str, Any]:
        context = dict(self.current or {})
        return {
            **context,
            "event_type": "nonfinite",
            "phase": failure.phase,
            "stage": failure.stage,
            "tensor_name": failure.tensor_name,
            "direction": failure.direction,
            "failure_stats": failure.stats,
            "warnings": list(self.last_warning_reasons),
            "warning_crossings": list(self.last_warning_crossings),
            "warning_lifecycle_events": copy.deepcopy(self.current_warning_lifecycle_events),
            "stages": copy.deepcopy(self.current_details),
        }

    def clear(self) -> None:
        self.current = None
        self.current_details = {}
        self.block_values = {}
        self.current_warning_lifecycle_events = []


class CheckpointBudget:
    """Bound ordinary forensic checkpoint files without suppressing failure saves."""

    def __init__(
        self,
        *,
        max_warning_checkpoints: int,
        max_total_checkpoints: int,
        event_sink: Any | None = None,
    ) -> None:
        if max_warning_checkpoints < 0:
            raise ValueError("max_warning_checkpoints must be non-negative")
        if max_total_checkpoints < 0:
            raise ValueError("max_total_checkpoints must be non-negative")
        self.max_warning_checkpoints = int(max_warning_checkpoints)
        self.max_total_checkpoints = int(max_total_checkpoints)
        self.warning_checkpoints_saved = 0
        self._ordinary_paths: set[str] = set()
        self._exhausted_reasons: set[str] = set()
        self.event_sink = event_sink

    @property
    def ordinary_checkpoints_saved(self) -> int:
        return len(self._ordinary_paths)

    @property
    def warning_budget_exhausted(self) -> bool:
        return self.warning_checkpoints_saved >= self.max_warning_checkpoints

    @property
    def total_budget_exhausted(self) -> bool:
        return self.ordinary_checkpoints_saved >= self.max_total_checkpoints

    def _path_key(self, path: Path | str) -> str:
        return str(Path(path).resolve())

    def _emit_exhausted(
        self,
        *,
        reason: str,
        checkpoint_kind: str,
        context: dict[str, Any] | None,
    ) -> None:
        if reason in self._exhausted_reasons:
            return
        self._exhausted_reasons.add(reason)
        event = {
            "event_type": "checkpoint_budget_exhausted",
            "budget": reason,
            "checkpoint_kind": checkpoint_kind,
            "max_warning_checkpoints": self.max_warning_checkpoints,
            "max_total_checkpoints": self.max_total_checkpoints,
            "warning_checkpoints_saved": self.warning_checkpoints_saved,
            "ordinary_checkpoints_saved": self.ordinary_checkpoints_saved,
            **(context or {}),
        }
        if self.event_sink is not None:
            self.event_sink(event)

    def can_save(
        self,
        *,
        checkpoint_kind: str,
        path: Path | str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        path_key = self._path_key(path)
        if checkpoint_kind == "warning" and self.warning_budget_exhausted:
            self._emit_exhausted(
                reason="warning_checkpoints",
                checkpoint_kind=checkpoint_kind,
                context=context,
            )
            return False
        # Rewriting best.pt does not consume another ordinary file slot.
        if path_key not in self._ordinary_paths and self.total_budget_exhausted:
            self._emit_exhausted(
                reason="total_checkpoints",
                checkpoint_kind=checkpoint_kind,
                context=context,
            )
            return False
        return True

    def record_saved(self, *, checkpoint_kind: str, path: Path | str) -> None:
        path_key = self._path_key(path)
        if path_key not in self._ordinary_paths:
            self._ordinary_paths.add(path_key)
        if checkpoint_kind == "warning":
            self.warning_checkpoints_saved += 1

    def summary(self) -> dict[str, Any]:
        return {
            "max_warning_checkpoints": self.max_warning_checkpoints,
            "max_total_checkpoints": self.max_total_checkpoints,
            "warning_checkpoints_saved": self.warning_checkpoints_saved,
            "ordinary_checkpoints_saved": self.ordinary_checkpoints_saved,
            "warning_budget_exhausted": self.warning_budget_exhausted,
            "total_budget_exhausted": self.total_budget_exhausted,
        }


_ACTIVE_OBSERVER: ForensicObserver | None = None


class A1Instrumentation:
    """Install observational hooks and a reversible mLSTM stage wrapper."""

    def __init__(self, model: nn.Module, observer: ForensicObserver) -> None:
        self.model = model
        self.observer = observer
        self.handles: list[Any] = []
        self.original_parallel = vil_module._parallel_stabilized_mlstm
        self.original_a1n_callback = vil_a1n_module.get_diagnostic_callback()
        self.a1n_callback_active = False
        self.installed = False

    def _output_hook(self, scope: str, stage: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            self.observer.record(scope, stage, output)

        return hook

    def _observe_a1n_mlstm(
        self,
        _cell: nn.Module,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        if self.observer.current is None:
            return
        scope = f"vil/{self.observer._active_block_label()}"
        for name in (
            "cumulative_log_decay",
            "combination",
            "normalized_combination",
            "max_log_decay",
            "log_normalizer",
            "inverse_normalizer",
        ):
            tensor = diagnostics.get(name)
            if tensor is not None:
                self.observer.record(scope, name, tensor)

    def install(self) -> None:
        global _ACTIVE_OBSERVER
        if self.installed:
            return

        def wrapped_parallel(*args: Any, **kwargs: Any) -> torch.Tensor:
            observer = _ACTIVE_OBSERVER
            if observer is not None and observer.current is not None:
                observer.observe_mlstm(*args, **kwargs)
            return self.original_parallel(*args, **kwargs)

        vil_module._parallel_stabilized_mlstm = wrapped_parallel

        processor = getattr(self.model, "bottleneck_processor", None)
        if processor is not None:
            def processor_pre(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                self.observer.record_global("cnn_bottleneck_input", inputs[0])

            def processor_post(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: torch.Tensor,
            ) -> None:
                self.observer.record_global("restored_spatial_bottleneck", output)

            self.handles.append(
                processor.register_forward_pre_hook(processor_pre)
            )
            self.handles.append(processor.register_forward_hook(processor_post))

        for index, decoder in enumerate(getattr(self.model, "decoders", [])):
            self.handles.append(
                decoder.register_forward_hook(
                    self._output_hook("global", f"decoder_stage_{index}_output")
                )
            )
        head = getattr(self.model, "head", None)
        if head is not None:
            self.handles.append(
                head.register_forward_hook(
                    self._output_hook("global", "segmentation_logits")
                )
            )

        for module_name, block in self.model.named_modules():
            if not isinstance(block, ViLMLSTMBlock):
                continue
            if ".forward_blocks." in module_name:
                direction = "forward"
            elif ".reverse_blocks." in module_name:
                direction = "reverse"
            else:
                direction = "vil"
            block_index = module_name.rsplit(".", 1)[-1]
            label = f"{direction}_block_{block_index}"
            scope = f"vil/{label}"

            def block_pre(_module: nn.Module, inputs: tuple[Any, ...], label=label) -> None:
                self.observer.enter_block(label, inputs[0])

            def block_post(
                module: ViLMLSTMBlock,
                _inputs: tuple[Any, ...],
                output: torch.Tensor,
                label=label,
                scope=scope,
            ) -> None:
                values = self.observer.block_values.get(label, {})
                projection_up = values.get("projection_up")
                convolution = values.get("convolution")
                mlstm_output = values.get("mlstm_output")
                if projection_up is not None and convolution is not None and mlstm_output is not None:
                    with torch.no_grad():
                        x_mlstm, z = torch.chunk(projection_up, chunks=2, dim=-1)
                        del x_mlstm
                        convolution_act = F.silu(convolution)
                        skip_combination = mlstm_output + module.learnable_skip.detach() * convolution_act
                        silu_gate = F.silu(z)
                        gated_product = skip_combination * silu_gate
                    self.observer.record(scope, "skip_combination", skip_combination)
                    self.observer.record(scope, "silu_gate", silu_gate)
                    self.observer.record(scope, "gated_product", gated_product)
                self.observer.record(scope, "final_vil_residual_output", output)

            self.handles.append(block.register_forward_pre_hook(block_pre))
            self.handles.append(block.register_forward_hook(block_post))
            self.handles.append(
                block.norm.register_forward_hook(
                    self._output_hook(scope, "layer_norm_output")
                )
            )

            def projection_up_hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: torch.Tensor,
                label=label,
                scope=scope,
            ) -> None:
                self.observer.store_block_value(label, "projection_up", output)
                self.observer.record(scope, "expanded_projection_output", output)
                x_mlstm, z = torch.chunk(output.detach(), chunks=2, dim=-1)
                self.observer.record(scope, "mlstm_branch", x_mlstm)
                self.observer.record(scope, "z_branch", z)

            self.handles.append(block.proj_up.register_forward_hook(projection_up_hook))

            def convolution_hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: torch.Tensor,
                label=label,
                scope=scope,
            ) -> None:
                self.observer.store_block_value(label, "convolution", output)
                self.observer.record(scope, "causal_convolution_output", output)
                self.observer.record(scope, "mlstm_branch_silu", F.silu(output.detach()))

            self.handles.append(block.conv1d.register_forward_hook(convolution_hook))
            self.handles.append(
                block.q_proj.register_forward_hook(
                    self._output_hook(scope, "Q")
                )
            )
            self.handles.append(
                block.k_proj.register_forward_hook(
                    self._output_hook(scope, "K")
                )
            )
            self.handles.append(
                block.v_proj.register_forward_hook(
                    self._output_hook(scope, "V")
                )
            )

            def mlstm_hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: torch.Tensor,
                label=label,
                scope=scope,
            ) -> None:
                self.observer.store_block_value(label, "mlstm_output", output)
                self.observer.record(scope, "post_mlstm_normalization_output", output)

            self.handles.append(block.mlstm.register_forward_hook(mlstm_hook))
            self.handles.append(
                block.proj_down.register_forward_hook(
                    self._output_hook(scope, "projection_output")
                )
            )
        self.installed = True

    def activate(self) -> None:
        global _ACTIVE_OBSERVER
        _ACTIVE_OBSERVER = self.observer
        if not self.a1n_callback_active:
            self.original_a1n_callback = vil_a1n_module.get_diagnostic_callback()
            vil_a1n_module.set_diagnostic_callback(self._observe_a1n_mlstm)
            self.a1n_callback_active = True

    def deactivate(self) -> None:
        global _ACTIVE_OBSERVER
        _ACTIVE_OBSERVER = None
        if self.a1n_callback_active:
            vil_a1n_module.set_diagnostic_callback(self.original_a1n_callback)
            self.a1n_callback_active = False

    def close(self) -> None:
        global _ACTIVE_OBSERVER
        _ACTIVE_OBSERVER = None
        if self.a1n_callback_active:
            vil_a1n_module.set_diagnostic_callback(self.original_a1n_callback)
            self.a1n_callback_active = False
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.installed:
            vil_module._parallel_stabilized_mlstm = self.original_parallel
        self.installed = False


def build_loader(
    config: dict[str, Any],
    split: str,
    shuffle: bool,
) -> tuple[DataLoader, torch.Generator]:
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset = KvasirSegDataset(
        data_root=project_path(config, dataset_config["root"]),
        manifest_path=project_path(config, dataset_config["manifest"]),
        split=split,
        image_size=dataset_config["image_size"],
        mask_threshold=dataset_config["mask_threshold"],
    )
    generator = make_dataloader_generator(int(config["seed"]) + (0 if shuffle else 1))
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training_config["num_workers"]),
        pin_memory=bool(training_config["pin_memory"]),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, generator


def validate_forensic_config(config: dict[str, Any]) -> None:
    model_name = str(config["model"]["name"]).lower()
    if model_name not in {"unet_vil_bottleneck_a1", "unet_vil_bottleneck_a1n"}:
        raise ValueError(
            "train_a1_forensic.py requires an A1 or A1-N model name, "
            f"got {model_name!r}"
        )
    if int(config["seed"]) != 42:
        raise ValueError(f"forensic A1 requires seed 42, got {config['seed']}")
    if int(config["training"]["batch_size"]) != 4:
        raise ValueError("forensic A1 requires batch_size=4")
    if float(config["optimizer"]["learning_rate"]) != 1e-3:
        raise ValueError("forensic A1 requires learning_rate=1e-3")
    if float(config["optimizer"]["weight_decay"]) != 1e-4:
        raise ValueError("forensic A1 requires weight_decay=1e-4")
    if str(config["scheduler"]["name"]) != "CosineAnnealingLR":
        raise ValueError("forensic A1 requires CosineAnnealingLR")
    if int(config["scheduler"]["t_max"]) != 100:
        raise ValueError("forensic A1 requires scheduler t_max=100")


def scalar_loss_components(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: BCESoftDiceLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        target_cast = target.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits.detach(), target_cast)
        probabilities = torch.sigmoid(logits.detach())
        reduce_dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * target_cast).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + target_cast.sum(dim=reduce_dims)
        dice = (2.0 * intersection + criterion.dice_epsilon) / (
            denominator + criterion.dice_epsilon
        )
        dice_loss = 1.0 - dice.mean()
        return (
            criterion.bce_weight * bce,
            criterion.dice_weight * dice_loss,
            probabilities,
        )


def _merge_scalar_summary(
    destination: dict[str, Any],
    source: dict[str, Any],
) -> None:
    """Merge scalar tensor/optimization statistics without retaining tensors."""

    for field in ("abs_max", "max"):
        value = source.get(field)
        if value is not None:
            previous = destination.get(field)
            destination[field] = float(value) if previous is None else max(float(previous), float(value))
    value = source.get("min")
    if value is not None:
        previous = destination.get("min")
        destination["min"] = float(value) if previous is None else min(float(previous), float(value))
    for field in (
        "num_nan",
        "num_pos_inf",
        "num_neg_inf",
        "unexpected_nonfinite",
        "nonfinite_elements",
        "parameter_count",
        "state_tensor_count",
        "state_tensors_with_nonfinite",
    ):
        value = source.get(field)
        if value is not None:
            destination[field] = int(destination.get(field, 0)) + int(value)
    for field in (
        "parameter_name_with_max_abs",
        "state_name_with_max_abs",
        "first_nonfinite",
    ):
        if destination.get(field) is None and source.get(field) is not None:
            destination[field] = source[field]


def _summary_template() -> dict[str, Any]:
    return {
        "abs_max": None,
        "max": None,
        "min": None,
        "num_nan": 0,
        "num_pos_inf": 0,
        "num_neg_inf": 0,
        "unexpected_nonfinite": 0,
    }


def _named_tensor_summary(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    include_l2: bool = False,
) -> dict[str, Any]:
    summary = _summary_template()
    summary.update(
        {
            "l2_norm": 0.0 if include_l2 else None,
            "parameter_count": 0,
            "nonfinite_elements": 0,
            "parameters_with_nonfinite": 0,
            "parameter_name_with_max_abs": None,
            "first_nonfinite": None,
        }
    )
    total_squared = 0.0
    for name, tensor in named_tensors:
        stats = tensor_stats(tensor)
        previous_abs_max = summary["abs_max"]
        summary["parameter_count"] += 1
        _merge_scalar_summary(summary, stats)
        summary["nonfinite_elements"] += int(stats["unexpected_nonfinite"])
        if stats["unexpected_nonfinite"]:
            summary["parameters_with_nonfinite"] += 1
            if summary["first_nonfinite"] is None:
                summary["first_nonfinite"] = {"parameter_name": name, "stats": stats}
        if stats["abs_max"] is not None and (
            previous_abs_max is None or float(stats["abs_max"]) > float(previous_abs_max)
        ):
            summary["parameter_name_with_max_abs"] = name
        if include_l2:
            finite_tensor = tensor.detach()
            if finite_tensor.is_floating_point():
                finite_tensor = finite_tensor[torch.isfinite(finite_tensor)]
            if finite_tensor.numel():
                total_squared += float((finite_tensor.to(torch.float64) ** 2).sum().item())
    if include_l2:
        summary["l2_norm"] = math.sqrt(total_squared)
    summary["has_nan"] = bool(summary["num_nan"])
    summary["has_pos_inf"] = bool(summary["num_pos_inf"])
    summary["has_neg_inf"] = bool(summary["num_neg_inf"])
    summary["has_inf"] = bool(summary["num_pos_inf"] or summary["num_neg_inf"])
    return summary


def gradient_summary(model: nn.Module) -> dict[str, Any]:
    named_gradients = [
        (name, parameter.grad.detach())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    summary = _named_tensor_summary(named_gradients, include_l2=True)
    summary["max_abs"] = summary["abs_max"] or 0.0
    summary["parameters_with_nan"] = sum(
        1
        for name, tensor in named_gradients
        if tensor_stats(tensor)["num_nan"]
    )
    summary["parameters_with_inf"] = sum(
        1
        for name, tensor in named_gradients
        if tensor_stats(tensor)["num_pos_inf"] or tensor_stats(tensor)["num_neg_inf"]
    )
    return summary


def parameter_summary(model: nn.Module) -> dict[str, Any]:
    named_parameters = [(name, parameter.detach()) for name, parameter in model.named_parameters()]
    summary = _named_tensor_summary(named_parameters)
    summary["max_abs"] = summary["abs_max"] or 0.0
    return summary


def parameter_finiteness(model: nn.Module) -> dict[str, Any] | None:
    summary = parameter_summary(model)
    return summary["first_nonfinite"]


def optimizer_state_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    state_summaries = {
        "exp_avg": _summary_template(),
        "exp_avg_sq": _summary_template(),
    }
    for state_name in state_summaries:
        state_summaries[state_name].update(
            {
                "state_tensor_count": 0,
                "state_tensors_with_nonfinite": 0,
                "parameter_name_with_max_abs": None,
                "state_name_with_max_abs": state_name,
                "first_nonfinite": None,
                "nonfinite_elements": 0,
            }
        )
    for parameter, state in optimizer.state.items():
        parameter_name = parameter_names.get(id(parameter), "unknown_parameter")
        for state_name, summary in state_summaries.items():
            value = state.get(state_name)
            if value is None or not torch.is_tensor(value):
                continue
            stats = tensor_stats(value)
            previous_abs_max = summary["abs_max"]
            summary["state_tensor_count"] += 1
            _merge_scalar_summary(summary, stats)
            summary["nonfinite_elements"] += int(stats["unexpected_nonfinite"])
            if stats["unexpected_nonfinite"]:
                summary["state_tensors_with_nonfinite"] += 1
                if summary["first_nonfinite"] is None:
                    summary["first_nonfinite"] = {
                        "parameter_name": parameter_name,
                        "state_name": state_name,
                        "stats": stats,
                    }
            if stats["abs_max"] is not None and (
                previous_abs_max is None or float(stats["abs_max"]) > float(previous_abs_max)
            ):
                summary["parameter_name_with_max_abs"] = parameter_name
    for summary in state_summaries.values():
        summary["has_nan"] = bool(summary["num_nan"])
        summary["has_pos_inf"] = bool(summary["num_pos_inf"])
        summary["has_neg_inf"] = bool(summary["num_neg_inf"])
        summary["has_inf"] = bool(summary["num_pos_inf"] or summary["num_neg_inf"])
    return {
        "has_state": any(
            summary["state_tensor_count"] for summary in state_summaries.values()
        ),
        "exp_avg": state_summaries["exp_avg"],
        "exp_avg_sq": state_summaries["exp_avg_sq"],
        "max_exp_avg_abs": state_summaries["exp_avg"]["abs_max"],
        "max_exp_avg_sq_abs": state_summaries["exp_avg_sq"]["abs_max"],
        "exp_avg_parameter_name_with_max_abs": state_summaries["exp_avg"][
            "parameter_name_with_max_abs"
        ],
        "exp_avg_sq_parameter_name_with_max_abs": state_summaries["exp_avg_sq"][
            "parameter_name_with_max_abs"
        ],
        "first_nonfinite": next(
            (
                summary["first_nonfinite"]
                for summary in state_summaries.values()
                if summary["first_nonfinite"] is not None
            ),
            None,
        ),
        "state_tensors_with_nonfinite": sum(
            summary["state_tensors_with_nonfinite"] for summary in state_summaries.values()
        ),
        "nonfinite_elements": sum(
            summary["nonfinite_elements"] for summary in state_summaries.values()
        ),
    }


def optimizer_state_finiteness(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any] | None:
    return optimizer_state_summary(model, optimizer)["first_nonfinite"]


def scheduler_state_finiteness(scheduler: Any) -> dict[str, Any] | None:
    for name, value in scheduler.state_dict().items():
        if isinstance(value, float) and not math.isfinite(value):
            return {"state_name": name, "value": value}
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, float) and not math.isfinite(item):
                    return {"state_name": f"{name}[{index}]", "value": item}
    return None


def classify_failure(failure: ForensicFailure) -> str:
    classifications = {
        "forward": "forward_activation",
        "loss": "loss",
        "gradient": "gradient",
        "optimizer_state_before_step": "optimizer_state_before_step",
        "optimizer_state_after_step": "optimizer_state_after_step",
        "parameter": (
            "parameter_before_step"
            if failure.stage == "before_optimizer_step"
            else "parameter_after_step"
        ),
        "scheduler_state": "scheduler_state",
    }
    return classifications.get(failure.phase, "other")


def classify_nonfinite_value(stats: dict[str, Any]) -> str:
    if int(stats.get("num_nan", 0)):
        return "NaN"
    if int(stats.get("num_pos_inf", 0)):
        return "+Inf"
    if int(stats.get("num_neg_inf", 0)):
        return "-Inf"
    return "nonfinite"


def compare_nested_tensors(left: Any, right: Any) -> float:
    maximum = 0.0
    if torch.is_tensor(left) and torch.is_tensor(right):
        maximum = float((left.detach().to(torch.float64) - right.detach().to(torch.float64)).abs().max().item())
    elif isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() & right.keys():
            maximum = max(maximum, compare_nested_tensors(left[key], right[key]))
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        for left_item, right_item in zip(left, right):
            maximum = max(maximum, compare_nested_tensors(left_item, right_item))
    return maximum


def run_forensic_self_tests() -> dict[str, Any]:
    """Exercise warning episodes and failure retention without model training."""

    observer = ForensicObserver()

    def observe_value(
        value: float,
        name: str = "test_metric",
        *,
        split: str = "self_test",
        epoch: int = 0,
        batch_index: int = 0,
    ) -> tuple[bool, bool, dict[str, Any]]:
        observer.begin_forward(
            split=split,
            epoch=epoch,
            batch_index=batch_index,
            global_step=epoch * 100 + batch_index,
            sample_ids=["self_test"],
        )
        observer.record("vil/forward_block_0", name, torch.tensor([value]))
        compact, detail, crossed = observer.finish_forward()
        observer.clear()
        return crossed, detail is not None, compact

    below_crossed, below_detail, compact = observe_value(50.0)
    first_crossed, first_detail, _ = observe_value(101.0)
    sustained_crossed, sustained_detail, _ = observe_value(110.0)
    reset_crossed, reset_detail, _ = observe_value(50.0)
    recrossed, recross_detail, _ = observe_value(101.0)

    log_observer = ForensicObserver()

    def observe_log_decay(value: float) -> bool:
        log_observer.begin_forward(
            split="self_test",
            epoch=0,
            batch_index=0,
            global_step=0,
            sample_ids=["self_test"],
        )
        log_observer.record(
            "vil/forward_block_0",
            "max_log_decay",
            torch.tensor([value]),
        )
        _, _, crossed = log_observer.finish_forward()
        log_observer.clear()
        return crossed

    log_first = observe_log_decay(-51.0)
    log_sustained = observe_log_decay(-55.0)
    log_reset = observe_log_decay(-49.0)
    log_recross = observe_log_decay(-51.0)

    def simulate_stream(
        values: list[tuple[str, int, int, float]],
        *,
        name: str = "test_metric",
    ) -> tuple[int, ForensicObserver]:
        stream_observer = ForensicObserver()
        checkpoint_count = 0
        for split, epoch, batch_index, value in values:
            stream_observer.begin_forward(
                split=split,
                epoch=epoch,
                batch_index=batch_index,
                global_step=epoch * 100 + batch_index,
                sample_ids=["self_test"],
            )
            stream_observer.record(
                "vil/forward_block_0",
                name,
                torch.tensor([value]),
            )
            _, _, crossed = stream_observer.finish_forward()
            if crossed:
                checkpoint_count += 1
                stream_observer.record_warning_checkpoint_saved(
                    f"synthetic_warning_{checkpoint_count}.pt"
                )
            stream_observer.clear()
        return checkpoint_count, stream_observer

    continuous_values = [
        ("train", epoch, batch_index, 101.0)
        for epoch in range(1, 11)
        for batch_index in range(10)
    ]
    continuous_checkpoint_count, _ = simulate_stream(continuous_values)
    epoch_boundary_values = [
        ("train", 1, batch_index, 101.0) for batch_index in range(10)
    ] + [
        ("train", 2, batch_index, 101.0) for batch_index in range(10)
    ]
    epoch_boundary_checkpoint_count, _ = simulate_stream(epoch_boundary_values)
    split_values = [
        ("train", 1, batch_index, 101.0) for batch_index in range(10)
    ] + [
        ("validation", 1, batch_index, 101.0) for batch_index in range(10)
    ] + [
        ("train", 2, batch_index, 101.0) for batch_index in range(10)
    ] + [
        ("validation", 2, batch_index, 101.0) for batch_index in range(10)
    ]
    split_checkpoint_count, split_observer = simulate_stream(split_values)
    recross_values = [
        ("train", 1, 0, 50.0),
        ("train", 1, 1, 101.0),
        ("train", 1, 2, 110.0),
        ("train", 1, 3, 50.0),
        ("train", 1, 4, 101.0),
    ]
    recross_checkpoint_count, recross_observer = simulate_stream(recross_values)
    directional_observer = ForensicObserver()
    directional_checkpoint_count = 0
    for batch_index in range(10):
        directional_observer.begin_forward(
            split="train",
            epoch=1,
            batch_index=batch_index,
            global_step=batch_index,
            sample_ids=["self_test"],
        )
        directional_observer.record(
            "vil/forward_block_0",
            "test_metric",
            torch.tensor([101.0]),
        )
        directional_observer.record(
            "vil/reverse_block_0",
            "test_metric",
            torch.tensor([101.0]),
        )
        _, _, crossed = directional_observer.finish_forward()
        if crossed:
            directional_checkpoint_count += 1
        directional_observer.clear()
    lifecycle_types = {
        event["event_type"] for event in recross_observer.warning_lifecycle_events
    }
    lifecycle_fields_present = all(
        {
            "epoch",
            "batch",
            "split",
            "scope",
            "metric",
            "threshold",
            "previous_state",
            "current_state",
            "episode_id",
        }.issubset(event)
        for event in recross_observer.warning_lifecycle_events
    )
    train_condition_key = (
        "train",
        "vil/forward_block_0",
        "test_metric:abs_max",
        100.0,
    )
    validation_condition_key = (
        "validation",
        "vil/forward_block_0",
        "test_metric:abs_max",
        100.0,
    )

    nonfinite_observer = ForensicObserver()
    nonfinite_observer.begin_forward(
        split="self_test",
        epoch=0,
        batch_index=0,
        global_step=0,
        sample_ids=["self_test"],
    )
    try:
        nonfinite_observer.record(
            "vil/forward_block_0",
            "nonfinite_test",
            torch.tensor([float("nan")]),
        )
    except ForensicFailure as error:
        nonfinite_detected = (
            error.stats["num_nan"] == 1
            and error.stats["num_pos_inf"] == 0
            and error.stats["num_neg_inf"] == 0
        )
    else:
        nonfinite_detected = False

    with tempfile.TemporaryDirectory(prefix="a1_forensic_self_test_") as directory:
        directory_path = Path(directory)
        test_model = nn.Linear(2, 1)
        test_optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3)
        test_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            test_optimizer,
            T_max=100,
            eta_min=1e-6,
        )
        train_generator = make_dataloader_generator(42)
        validation_generator = make_dataloader_generator(43)
        failure_observer = ForensicObserver()
        failure_observer.begin_forward(
            split="self_test",
            epoch=1,
            batch_index=2,
            global_step=2,
            sample_ids=["self_test"],
        )
        failure = ForensicFailure(
            phase="forward",
            stage="self_test",
            tensor_name="self_test_tensor",
            stats=tensor_stats(torch.tensor([float("inf")])),
        )
        failure_report_and_state(
            output_dir=directory_path,
            failure=failure,
            observer=failure_observer,
            model=test_model,
            optimizer=test_optimizer,
            scheduler=test_scheduler,
            config={"seed": 42},
            epoch=1,
            batch_index=2,
            global_step=2,
            sample_ids=["self_test"],
            learning_rate=1e-3,
            gradient_info=None,
            best_validation_dice=float("-inf"),
            train_generator=train_generator,
            validation_generator=validation_generator,
            recent_history=[],
        )
        failure_checkpoint_behavior = all(
            (directory_path / "checkpoints" / filename).is_file()
            for filename in ("pre_failure_step.pt", "failure_state.pt")
        ) and (directory_path / "failure_report.json").is_file()

    warning_budget_events: list[dict[str, Any]] = []
    warning_budget = CheckpointBudget(
        max_warning_checkpoints=20,
        max_total_checkpoints=40,
        event_sink=warning_budget_events.append,
    )
    warning_budget_saved = 0
    for index in range(100):
        warning_path = f"warning_{index}.pt"
        if warning_budget.can_save(
            checkpoint_kind="warning",
            path=warning_path,
            context={"epoch": 1, "batch": index, "split": "train"},
        ):
            warning_budget.record_saved(checkpoint_kind="warning", path=warning_path)
            warning_budget_saved += 1

    total_budget_events: list[dict[str, Any]] = []
    total_budget = CheckpointBudget(
        max_warning_checkpoints=100,
        max_total_checkpoints=40,
        event_sink=total_budget_events.append,
    )
    total_budget_saved = 0
    for index in range(100):
        regular_path = f"regular_{index}.pt"
        if total_budget.can_save(
            checkpoint_kind="regular",
            path=regular_path,
            context={"epoch": index + 1, "batch": -1, "split": "epoch"},
        ):
            total_budget.record_saved(checkpoint_kind="regular", path=regular_path)
            total_budget_saved += 1

    monitor_model = nn.Linear(2, 1)
    monitor_optimizer = torch.optim.AdamW(monitor_model.parameters(), lr=1e-3)
    monitor_loss = monitor_model(torch.ones(1, 2)).sum()
    monitor_loss.backward()
    monitor_gradient = gradient_summary(monitor_model)
    monitor_parameters_before = parameter_summary(monitor_model)
    monitor_optimizer_before = optimizer_state_summary(monitor_model, monitor_optimizer)
    monitor_optimizer.step()
    monitor_parameters_after = parameter_summary(monitor_model)
    monitor_optimizer_after = optimizer_state_summary(monitor_model, monitor_optimizer)
    optimization_monitoring_finite = (
        not monitor_gradient["has_nan"]
        and not monitor_gradient["has_inf"]
        and not monitor_parameters_before["has_nan"]
        and not monitor_parameters_before["has_inf"]
        and not monitor_parameters_after["has_nan"]
        and not monitor_parameters_after["has_inf"]
        and not monitor_optimizer_after["exp_avg"]["has_nan"]
        and not monitor_optimizer_after["exp_avg"]["has_inf"]
        and not monitor_optimizer_after["exp_avg_sq"]["has_nan"]
        and not monitor_optimizer_after["exp_avg_sq"]["has_inf"]
    )

    results = {
        "below_threshold_no_checkpoint": not below_crossed and not below_detail,
        "first_crossing_one_event": first_crossed and first_detail,
        "sustained_above_no_event": not sustained_crossed and not sustained_detail,
        "below_threshold_resets_episode": not reset_crossed and not reset_detail,
        "later_recross_one_event": recrossed and recross_detail,
        "log_decay_crossing": log_first and not log_sustained and not log_reset and log_recross,
        "100_consecutive_above_one_checkpoint": continuous_checkpoint_count == 1,
        "epoch_boundary_continuity_one_checkpoint": epoch_boundary_checkpoint_count == 1,
        "train_validation_independent_episodes": (
            split_checkpoint_count == 2
            and split_observer.warning_episode_ids.get(train_condition_key) == 1
            and split_observer.warning_episode_ids.get(validation_condition_key) == 1
        ),
        "false_true_true_false_true_two_checkpoints": recross_checkpoint_count == 2,
        "forward_reverse_independent_episodes": (
            directional_checkpoint_count == 1
            and directional_observer.warning_episode_ids.get(
                ("train", "vil/forward_block_0", "test_metric:abs_max", 100.0)
            ) == 1
            and directional_observer.warning_episode_ids.get(
                ("train", "vil/reverse_block_0", "test_metric:abs_max", 100.0)
            ) == 1
        ),
        "lifecycle_events_present": {
            "warning_state_created",
            "warning_episode_started",
            "warning_condition_true",
            "warning_state_reset",
            "warning_checkpoint_saved",
        }.issubset(lifecycle_types),
        "lifecycle_event_fields_present": lifecycle_fields_present,
        "warning_budget_twenty_of_one_hundred": (
            warning_budget_saved == 20
            and len(warning_budget_events) == 1
            and warning_budget_events[0]["event_type"] == "checkpoint_budget_exhausted"
        ),
        "global_budget_caps_ordinary_checkpoints": (
            total_budget_saved == 40
            and total_budget.ordinary_checkpoints_saved == 40
            and len(total_budget_events) == 1
        ),
        "optimization_monitoring_finite": optimization_monitoring_finite,
        "optimization_monitoring_has_scalar_maxima": (
            monitor_gradient["max_abs"] > 0.0
            and monitor_gradient["parameter_name_with_max_abs"] is not None
            and monitor_optimizer_after["max_exp_avg_abs"] is not None
            and monitor_optimizer_after["max_exp_avg_sq_abs"] is not None
        ),
        "nonfinite_detection": nonfinite_detected,
        "failure_checkpoint_behavior": failure_checkpoint_behavior,
        "compact_logging_present": "vil" in compact and "warning_crossed" in compact,
    }
    return {"passed": all(results.values()), "checks": results}


def run_instrumentation_equivalence(
    model: nn.Module,
    config: dict[str, Any],
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Verify observational instrumentation does not alter outputs or updates."""

    model_without = copy.deepcopy(model).to(device)
    model_with = copy.deepcopy(model).to(device)
    images = batch["image"].to(device)[:1]
    masks = batch["mask"].to(device)[:1]
    criterion_without = BCESoftDiceLoss(**config["loss"])
    criterion_with = BCESoftDiceLoss(**config["loss"])

    model_without.eval()
    with torch.no_grad():
        output_without = model_without(images)

    observer = ForensicObserver()
    instrumentation = A1Instrumentation(model_with, observer)
    instrumentation.install()
    instrumentation.activate()
    observer.begin_forward(
        split="equivalence",
        epoch=0,
        batch_index=0,
        global_step=0,
        sample_ids=[str(batch["id"][0])],
    )
    model_with.eval()
    with torch.no_grad():
        output_with = model_with(images)
    observer.finish_forward()
    output_difference = float((output_without - output_with).abs().max().item())
    observer.clear()
    instrumentation.deactivate()
    instrumentation.close()

    model_without.train()
    model_with.train()
    optimizer_without = torch.optim.AdamW(
        model_without.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    optimizer_with = torch.optim.AdamW(
        model_with.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler_without = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_without,
        T_max=int(config["scheduler"]["t_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    scheduler_with = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_with,
        T_max=int(config["scheduler"]["t_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    optimizer_without.zero_grad(set_to_none=True)
    loss_without = criterion_without(model_without(images), masks)
    loss_without.backward()
    optimizer_without.step()
    scheduler_without.step()

    observer = ForensicObserver()
    instrumentation = A1Instrumentation(model_with, observer)
    instrumentation.install()
    instrumentation.activate()
    observer.begin_forward(
        split="equivalence",
        epoch=0,
        batch_index=0,
        global_step=0,
        sample_ids=[str(batch["id"][0])],
    )
    optimizer_with.zero_grad(set_to_none=True)
    loss_with = criterion_with(model_with(images), masks)
    observer.record_global("equivalence_loss", loss_with.detach())
    loss_with.backward()
    observer.finish_forward()
    optimizer_with.step()
    scheduler_with.step()
    parameter_difference = compare_nested_tensors(model_without.state_dict(), model_with.state_dict())
    gradient_difference = 0.0
    for (_, left), (_, right) in zip(
        model_without.named_parameters(), model_with.named_parameters()
    ):
        if left.grad is not None and right.grad is not None:
            gradient_difference = max(
                gradient_difference,
                float((left.grad.detach().to(torch.float64) - right.grad.detach().to(torch.float64)).abs().max().item()),
            )
    optimizer_difference = compare_nested_tensors(
        optimizer_without.state_dict(), optimizer_with.state_dict()
    )
    scheduler_difference = compare_nested_tensors(
        scheduler_without.state_dict(), scheduler_with.state_dict()
    )
    instrumentation.deactivate()
    instrumentation.close()
    return {
        "output_max_abs_difference": output_difference,
        "loss_abs_difference": float(abs(loss_without.item() - loss_with.item())),
        "gradient_max_abs_difference": gradient_difference,
        "parameter_max_abs_difference_after_step": parameter_difference,
        "optimizer_state_max_abs_difference": optimizer_difference,
        "scheduler_state_max_abs_difference": scheduler_difference,
        "passed": all(
            value == 0.0
            for value in (
                output_difference,
                float(abs(loss_without.item() - loss_with.item())),
                gradient_difference,
                parameter_difference,
                optimizer_difference,
                scheduler_difference,
            )
        ),
    }


def record_loss(
    observer: ForensicObserver,
    logits: torch.Tensor,
    masks: torch.Tensor,
    criterion: BCESoftDiceLoss,
) -> torch.Tensor:
    bce_component, dice_component, probabilities = scalar_loss_components(logits, masks, criterion)
    observer.record_global("loss_logits", logits)
    observer.record_global("loss_probabilities", probabilities)
    for name, component in (
        ("bce_component", bce_component),
        ("soft_dice_component", dice_component),
    ):
        if not torch.isfinite(component.detach()).all():
            raise ForensicFailure(
                phase="loss",
                stage="loss_component",
                tensor_name=f"global/{name}",
                stats=tensor_stats(component.detach()),
            )
    observer.record_global("bce_component", bce_component)
    observer.record_global("soft_dice_component", dice_component)
    loss = criterion(logits, masks)
    if not torch.isfinite(loss.detach()).all():
        raise ForensicFailure(
            phase="loss",
            stage="total_loss",
            tensor_name="global/total_loss",
            stats=tensor_stats(loss.detach()),
        )
    observer.record_global("total_loss", loss.detach())
    return loss


def evaluate_forensic(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: BCESoftDiceLoss,
    device: torch.device,
    threshold: float,
    observer: ForensicObserver,
    epoch: int,
    global_step: int,
    max_batches: int | None,
) -> tuple[dict[str, float], int, list[dict[str, Any]]]:
    model.eval()
    loss_total = 0.0
    sample_total = 0
    metric_total = torch.zeros(len(METRIC_NAMES), dtype=torch.float64)
    metric_square_total = torch.zeros(len(METRIC_NAMES), dtype=torch.float64)
    compact_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            sample_ids = [str(value) for value in batch["id"]]
            observer.begin_forward(
                split="validation",
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
                sample_ids=sample_ids,
            )
            logits = model(images)
            compact, detail, _warning = observer.finish_forward()
            if detail is not None:
                compact_records.append({"warning": detail, "compact": compact})
            loss = record_loss(observer, logits, masks, criterion)
            values = batch_metric_values(logits, masks, threshold=threshold).cpu()
            if not torch.isfinite(values).all():
                stats = tensor_stats(values)
                raise ForensicFailure(
                    phase="loss/metrics",
                    stage="metrics",
                    tensor_name="validation_metrics",
                    stats=stats,
                )
            batch_size = images.shape[0]
            loss_total += float(loss.item()) * batch_size
            metric_total += values.sum(dim=0)
            metric_square_total += (values * values).sum(dim=0)
            sample_total += batch_size
            observer.clear()
    if sample_total == 0:
        raise RuntimeError("validation loader produced no samples")
    result = {"loss": loss_total / sample_total}
    means = metric_total / sample_total
    variances = (metric_square_total / sample_total - means * means).clamp_min(0.0)
    for index, name in enumerate(METRIC_NAMES):
        result[name] = float(means[index].item())
        result[f"{name}_std"] = float(variances[index].sqrt().item())
    return result, global_step, compact_records


def failure_report_and_state(
    *,
    output_dir: Path,
    failure: ForensicFailure,
    observer: ForensicObserver,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    epoch: int,
    batch_index: int,
    global_step: int,
    sample_ids: list[str],
    learning_rate: float,
    gradient_info: dict[str, Any] | None,
    best_validation_dice: float,
    train_generator: torch.Generator,
    validation_generator: torch.Generator,
    recent_history: list[dict[str, Any]],
    checkpoint_budget_summary: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    pre_failure = checkpoint_dir / "pre_failure_step.pt"
    failure_state = checkpoint_dir / "failure_state.pt"
    save_training_state(
        pre_failure,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        epoch=epoch,
        global_step=global_step,
        best_validation_dice=best_validation_dice,
        train_generator=train_generator,
        validation_generator=validation_generator,
        label="best_effort_pre_failure_state",
    )
    save_training_state(
        failure_state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        epoch=epoch,
        global_step=global_step,
        best_validation_dice=best_validation_dice,
        train_generator=train_generator,
        validation_generator=validation_generator,
        label="failure_state",
    )
    detailed = observer.failure_detail(failure)
    report = {
        "status": "nonfinite_detected",
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "global_step": int(global_step),
        "sample_ids": sample_ids,
        "phase": failure.phase,
        "classification": classify_failure(failure),
        "stage": failure.stage,
        "tensor_name": failure.tensor_name,
        "direction": failure.direction,
        "value_type": classify_nonfinite_value(failure.stats),
        "num_nan": int(failure.stats.get("num_nan", 0)),
        "num_pos_inf": int(failure.stats.get("num_pos_inf", 0)),
        "num_neg_inf": int(failure.stats.get("num_neg_inf", 0)),
        "loss": detailed.get("loss", None),
        "learning_rate": learning_rate,
        "gradient": gradient_info,
        "gradient_norm": gradient_info.get("l2_norm") if gradient_info else None,
        "parameter_name": (
            failure.tensor_name
            if classify_failure(failure)
            in {
                "gradient",
                "optimizer_state_before_step",
                "optimizer_state_after_step",
                "parameter_before_step",
                "parameter_after_step",
            }
            else (
                gradient_info.get("first_nonfinite", {}).get("parameter_name")
                if gradient_info and gradient_info.get("first_nonfinite")
                else None
            )
        ),
        "pre_failure_checkpoint": str(pre_failure),
        "failure_state_checkpoint": str(failure_state),
        "python_rng_saved": True,
        "numpy_rng_saved": True,
        "torch_rng_saved": True,
        "cuda_rng_saved": torch.cuda.is_available(),
        "dataloader_rng_saved": True,
        "failure_stats": failure.stats,
        "detailed_statistics": detailed,
        "optimization_telemetry": {
            key: detailed.get(key)
            for key in (
                "gradient",
                "parameters_before_step",
                "parameters_after_step",
                "optimizer_state_before_step",
                "optimizer_state_after_step",
                "learning_rate",
            )
            if key in detailed
        },
        "maximum_activation_values": observer.max_observed,
        "epoch_numerical_summary": observer.epoch_summary(),
        "immediately_preceding_step": recent_history[-1] if recent_history else None,
        "recent_history": recent_history,
        "checkpoint_budget": checkpoint_budget_summary,
        "historical_epoch27_is_not_exactly_replayed": True,
    }
    write_json(output_dir / "failure_report.json", report)


def train_forensic(
    *,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    max_epochs: int | None,
    max_batches: int | None,
    max_validation_batches: int | None,
    sanity_check: bool,
    anomaly_detection: bool,
    checkpoint_every: int,
    max_warning_checkpoints: int,
    max_total_checkpoints: int,
) -> int:
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    experiment_type = (
        "forensic_a1n_log_domain_normalization"
        if str(config["model"]["name"]).lower() == "unet_vil_bottleneck_a1n"
        else "forensic_a1_numerical_investigation"
    )
    for directory in (output_dir / "checkpoints", output_dir / "diagnostics"):
        directory.mkdir(parents=True, exist_ok=False)
    events_path = output_dir / "forensic_events.jsonl"
    warning_state_events_path = output_dir / "warning_state_events.jsonl"
    events_path.touch()
    warning_state_events_path.touch()
    checkpoint_budget = CheckpointBudget(
        max_warning_checkpoints=max_warning_checkpoints,
        max_total_checkpoints=max_total_checkpoints,
        event_sink=lambda event: append_jsonl(warning_state_events_path, event),
    )

    write_json(output_dir / "config_snapshot.json", config)
    model = build_model(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
    train_loader, train_generator = build_loader(config, "train", shuffle=True)
    validation_loader, validation_generator = build_loader(config, "validation", shuffle=False)
    write_json(
        output_dir / "experiment_metadata.json",
        {
            "status": "running",
            "experiment_type": experiment_type,
            "historical_epoch27_not_exactly_replayed": True,
            "fresh_initialization": True,
            "config_path": config["_config_path"],
            "output_dir": str(output_dir),
            "device": str(device),
            "environment": environment_summary(device),
            "seed": int(config["seed"]),
            "model_name": config["model"]["name"],
            "model_parameters": parameter_count,
            "expected_a1_parameters": 5645937,
            "batch_size": int(config["training"]["batch_size"]),
            "planned_epochs": int(config["training"]["epochs"]),
            "max_epochs": max_epochs,
            "max_batches": max_batches,
            "max_validation_batches": max_validation_batches,
            "anomaly_detection": anomaly_detection,
            "regular_checkpoint_interval_epochs": checkpoint_every,
            "warning_state_events_path": str(warning_state_events_path),
            "max_warning_checkpoints": max_warning_checkpoints,
            "max_total_checkpoints": max_total_checkpoints,
            "checkpoint_budget": checkpoint_budget.summary(),
            "checkpoint_policy": (
                "one full regular checkpoint every checkpoint interval; one overwritten "
                "best.pt; warning checkpoints subject to max-warning-checkpoints and "
                "max-total-checkpoints; failure checkpoints only on failure"
            ),
            "instrumentation_is_observational": True,
            "existing_a1_artifacts_modified": False,
            "reference_files_modified": False,
        },
    )
    torch.save(
        {
            "label": "startup_after_seed_model_and_loader_initialization",
            "rng_state": capture_rng_state(train_generator, validation_generator),
            "seed": int(config["seed"]),
        },
        output_dir / "diagnostics" / "reproducibility_state_startup.pt",
    )

    sanity_results: dict[str, Any] | None = None
    if sanity_check:
        warning_tests = run_forensic_self_tests()
        if not warning_tests["passed"]:
            raise RuntimeError(f"warning/self-tests failed: {warning_tests}")
        train_generator_before_sanity = train_generator.get_state()
        first_batch = next(iter(train_loader))
        equivalence_results = run_instrumentation_equivalence(
            model=model,
            config=config,
            batch=first_batch,
            device=device,
        )
        sanity_results = {
            "warning_threshold_tests": warning_tests,
            "instrumentation_equivalence": equivalence_results,
        }
        write_json(
            output_dir / "diagnostics" / "sanity_tests.json",
            sanity_results,
        )
        if not equivalence_results["passed"]:
            raise RuntimeError(f"instrumentation equivalence failed: {equivalence_results}")
        # The equivalence check is a validation-only operation.  Restore the
        # DataLoader generator so the bounded training run retains the same
        # shuffle behavior it would have had without this check.
        train_generator.set_state(train_generator_before_sanity)

    observer = ForensicObserver()
    observer.set_warning_event_sink(
        lambda event: append_jsonl(warning_state_events_path, event)
    )
    instrumentation = A1Instrumentation(model, observer)
    instrumentation.install()
    instrumentation.activate()
    torch.autograd.set_detect_anomaly(anomaly_detection)
    history: list[dict[str, Any]] = []
    rolling_history: list[dict[str, Any]] = []
    best_validation_dice = float("-inf")
    global_step = 0
    completed_epochs = 0
    failure: ForensicFailure | None = None

    try:
        epoch_budget = int(config["training"]["epochs"])
        if max_epochs is not None:
            epoch_budget = min(epoch_budget, int(max_epochs))
        for epoch in range(1, epoch_budget + 1):
            observer.begin_epoch(epoch)
            model.train()
            train_loss_total = 0.0
            train_samples = 0
            train_batch_count = 0
            for batch_index, batch in enumerate(train_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                train_batch_count += 1
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                sample_ids = [str(value) for value in batch["id"]]
                optimizer.zero_grad(set_to_none=True)
                observer.begin_forward(
                    split="train",
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=global_step,
                    sample_ids=sample_ids,
                )
                observer.current["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                try:
                    logits = model(images)
                    compact_forward, detail_forward, warning = observer.finish_forward()
                    loss = record_loss(observer, logits, masks, criterion)
                    if not torch.isfinite(loss.detach()).all():
                        stats = tensor_stats(loss.detach())
                        raise ForensicFailure(
                            phase="loss",
                            stage="total_loss",
                            tensor_name="global/total_loss",
                            stats=stats,
                        )
                    observer.current["loss"] = float(loss.detach().item())
                    try:
                        loss.backward()
                    except RuntimeError as error:
                        raise ForensicFailure(
                            phase="backward",
                            stage="autograd",
                            tensor_name="backward",
                            message=str(error),
                        ) from error
                    gradients = gradient_summary(model)
                    observer.current["gradient"] = gradients
                    parameters_before = parameter_summary(model)
                    optimizer_state_before = optimizer_state_summary(model, optimizer)
                    observer.current["parameters_before_step"] = parameters_before
                    observer.current["optimizer_state_before_step"] = optimizer_state_before
                    if gradients["first_nonfinite"] is not None:
                        failure_info = gradients["first_nonfinite"]
                        raise ForensicFailure(
                            phase="gradient",
                            stage="gradient",
                            tensor_name=failure_info["parameter_name"],
                            stats=failure_info["stats"],
                        )
                    if parameters_before["first_nonfinite"] is not None:
                        failure_info = parameters_before["first_nonfinite"]
                        raise ForensicFailure(
                            phase="parameter",
                            stage="before_optimizer_step",
                            tensor_name=failure_info["parameter_name"],
                            stats=failure_info["stats"],
                        )
                    if optimizer_state_before["first_nonfinite"] is not None:
                        failure_info = optimizer_state_before["first_nonfinite"]
                        raise ForensicFailure(
                            phase="optimizer_state_before_step",
                            stage="before_optimizer_step",
                            tensor_name=failure_info["parameter_name"],
                            stats=failure_info["stats"],
                        )
                    warning_checkpoint_path = (
                        output_dir
                        / "checkpoints"
                        / f"warning_e{epoch:03d}_b{batch_index:04d}_s{global_step:06d}.pt"
                    )
                    if warning and checkpoint_budget.can_save(
                        checkpoint_kind="warning",
                        path=warning_checkpoint_path,
                        context={
                            "epoch": epoch,
                            "batch": batch_index,
                            "batch_index": batch_index,
                            "global_step": global_step,
                            "split": "train",
                        },
                    ):
                        save_training_state(
                            warning_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config,
                            epoch=epoch,
                            global_step=global_step,
                            best_validation_dice=best_validation_dice,
                            train_generator=train_generator,
                            validation_generator=validation_generator,
                            label="warning_before_optimizer_step",
                        )
                        checkpoint_budget.record_saved(
                            checkpoint_kind="warning",
                            path=warning_checkpoint_path,
                        )
                        observer.record_warning_checkpoint_saved(warning_checkpoint_path)
                    optimizer.step()
                    parameters_after = parameter_summary(model)
                    observer.current["parameters_after_step"] = parameters_after
                    if parameters_after["first_nonfinite"] is not None:
                        failure_info = parameters_after["first_nonfinite"]
                        raise ForensicFailure(
                            phase="parameter",
                            stage="after_optimizer_step",
                            tensor_name=failure_info["parameter_name"],
                            stats=failure_info["stats"],
                        )
                    optimizer_state_after = optimizer_state_summary(model, optimizer)
                    observer.current["optimizer_state_after_step"] = optimizer_state_after
                    if optimizer_state_after["first_nonfinite"] is not None:
                        failure_info = optimizer_state_after["first_nonfinite"]
                        raise ForensicFailure(
                            phase="optimizer_state_after_step",
                            stage="after_optimizer_step",
                            tensor_name=failure_info["parameter_name"],
                            stats=failure_info["stats"],
                        )
                    observer.record_optimization_telemetry(
                        gradients=gradients,
                        parameters_before=parameters_before,
                        parameters_after=parameters_after,
                        optimizer_before=optimizer_state_before,
                        optimizer_after=optimizer_state_after,
                    )
                    step_record = {
                        **compact_forward,
                        "loss": float(loss.detach().item()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_norm": gradients["l2_norm"],
                        "gradient_max_abs": gradients["max_abs"],
                        "parameter_abs_max": max(
                            float(tensor_stats(parameter.detach())["abs_max"] or 0.0)
                            for parameter in model.parameters()
                        ),
                        "warning": warning,
                        "optimization": {
                            "gradient": gradients,
                            "parameters_before_step": parameters_before,
                            "parameters_after_step": parameters_after,
                            "optimizer_state_before_step": optimizer_state_before,
                            "optimizer_state_after_step": optimizer_state_after,
                        },
                    }
                    rolling_history.append(step_record)
                    rolling_history = rolling_history[-20:]
                    if detail_forward is not None:
                        detail_forward["step_summary"] = step_record
                        append_jsonl(events_path, detail_forward)
                    train_loss_total += float(loss.detach().item()) * images.shape[0]
                    train_samples += images.shape[0]
                    observer.clear()
                    global_step += 1
                except ForensicFailure as caught:
                    failure = caught
                    raise

            if train_batch_count == 0:
                raise RuntimeError("training loader produced no batches")
            validation, _, validation_warning_records = evaluate_forensic(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                threshold=float(config["dataset"]["prediction_threshold"]),
                observer=observer,
                epoch=epoch,
                global_step=global_step,
                max_batches=max_validation_batches,
            )
            for warning_record in validation_warning_records:
                append_jsonl(events_path, warning_record["warning"])
            scheduler.step()
            scheduler_failure = scheduler_state_finiteness(scheduler)
            if scheduler_failure is not None:
                raise ForensicFailure(
                    phase="scheduler_state",
                    stage="after_scheduler_step",
                    tensor_name=scheduler_failure["state_name"],
                    stats=tensor_stats(torch.tensor([float(scheduler_failure["value"])])),
                )
            if not math.isfinite(validation["dice"]):
                raise RuntimeError("validation Dice became non-finite")
            completed_epochs = epoch
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss_total / max(train_samples, 1),
                **validation,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "global_step": global_step,
                "train_batches": train_batch_count,
                "numerical_summary": observer.epoch_summary(),
            }
            history.append(epoch_record)
            write_json(output_dir / "training_history.json", {"history": history})
            regular_checkpoint_path = output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
            if epoch % checkpoint_every == 0 and checkpoint_budget.can_save(
                checkpoint_kind="regular",
                path=regular_checkpoint_path,
                context={
                    "epoch": epoch,
                    "batch": -1,
                    "batch_index": -1,
                    "global_step": global_step,
                    "split": "epoch",
                },
            ):
                save_training_state(
                    regular_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    best_validation_dice=best_validation_dice,
                    train_generator=train_generator,
                    validation_generator=validation_generator,
                    label="regular_epoch_checkpoint",
                )
                checkpoint_budget.record_saved(
                    checkpoint_kind="regular",
                    path=regular_checkpoint_path,
                )
            if validation["dice"] > best_validation_dice:
                best_validation_dice = validation["dice"]
                best_checkpoint_path = output_dir / "checkpoints" / "best.pt"
                if checkpoint_budget.can_save(
                    checkpoint_kind="best",
                    path=best_checkpoint_path,
                    context={
                        "epoch": epoch,
                        "batch": -1,
                        "batch_index": -1,
                        "global_step": global_step,
                        "split": "validation",
                    },
                ):
                    save_training_state(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        epoch=epoch,
                        global_step=global_step,
                        best_validation_dice=best_validation_dice,
                        train_generator=train_generator,
                        validation_generator=validation_generator,
                        label="best_validation_dice_checkpoint",
                    )
                    checkpoint_budget.record_saved(
                        checkpoint_kind="best",
                        path=best_checkpoint_path,
                    )
            write_json(
                output_dir / "experiment_metadata.json",
                {
                    "status": "running",
                    "experiment_type": experiment_type,
                    "historical_epoch27_not_exactly_replayed": True,
                    "fresh_initialization": True,
                    "config_path": config["_config_path"],
                    "output_dir": str(output_dir),
                    "device": str(device),
                    "environment": environment_summary(device),
                    "seed": int(config["seed"]),
                    "model_name": config["model"]["name"],
                    "model_parameters": parameter_count,
                    "expected_a1_parameters": 5645937,
                    "best_epoch": max(
                        (record["epoch"] for record in history if record["dice"] == best_validation_dice),
                        default=None,
                    ),
                    "best_validation_dice": best_validation_dice,
                    "completed_epochs": completed_epochs,
                    "global_step": global_step,
                    "sanity_check": sanity_results,
                    "regular_checkpoint_interval_epochs": checkpoint_every,
                    "warning_state_events_path": str(warning_state_events_path),
                    "max_warning_checkpoints": max_warning_checkpoints,
                    "max_total_checkpoints": max_total_checkpoints,
                    "checkpoint_budget": checkpoint_budget.summary(),
                    "checkpoint_policy": (
                        "one full regular checkpoint every checkpoint interval; one overwritten "
                        "best.pt; warning checkpoints subject to max-warning-checkpoints and "
                        "max-total-checkpoints; failure checkpoints only on failure"
                    ),
                    "instrumentation_is_observational": True,
                    "existing_a1_artifacts_modified": False,
                    "reference_files_modified": False,
                },
            )
            print(json.dumps(epoch_record, sort_keys=True))

        status = "completed_without_nonfinite" if completed_epochs == int(config["training"]["epochs"]) else "bounded_completed_without_nonfinite"
        write_json(
            output_dir / "experiment_metadata.json",
            {
                "status": status,
                "experiment_type": experiment_type,
                "historical_epoch27_not_exactly_replayed": True,
                "fresh_initialization": True,
                "config_path": config["_config_path"],
                "output_dir": str(output_dir),
                "device": str(device),
                "environment": environment_summary(device),
                "seed": int(config["seed"]),
                "model_name": config["model"]["name"],
                "model_parameters": parameter_count,
                "expected_a1_parameters": 5645937,
                "best_epoch": max(
                    (record["epoch"] for record in history if record["dice"] == best_validation_dice),
                    default=None,
                ),
                "best_validation_dice": best_validation_dice,
                "completed_epochs": completed_epochs,
                "global_step": global_step,
                "sanity_check": sanity_results,
                "regular_checkpoint_interval_epochs": checkpoint_every,
                "warning_state_events_path": str(warning_state_events_path),
                "max_warning_checkpoints": max_warning_checkpoints,
                "max_total_checkpoints": max_total_checkpoints,
                "checkpoint_budget": checkpoint_budget.summary(),
                "checkpoint_policy": (
                    "one full regular checkpoint every checkpoint interval; one overwritten "
                    "best.pt; warning checkpoints subject to max-warning-checkpoints and "
                    "max-total-checkpoints; failure checkpoints only on failure"
                ),
                "instrumentation_is_observational": True,
                "existing_a1_artifacts_modified": False,
                "reference_files_modified": False,
            },
        )
        write_json(
            output_dir / "run_summary.json",
            {
                "status": status,
                "best_epoch": max(
                    (record["epoch"] for record in history if record["dice"] == best_validation_dice),
                    default=None,
                ),
                "best_validation_dice": best_validation_dice,
                "final_metrics": history[-1] if history else None,
                "completed_epochs": completed_epochs,
                "global_step": global_step,
                "warning_count": sum(1 for line in events_path.read_text(encoding="utf-8").splitlines() if line),
                "warning_lifecycle_event_count": sum(
                    1
                    for line in warning_state_events_path.read_text(encoding="utf-8").splitlines()
                    if line
                ),
                "regular_checkpoint_interval_epochs": checkpoint_every,
                "warning_state_events_path": str(warning_state_events_path),
                "max_warning_checkpoints": max_warning_checkpoints,
                "max_total_checkpoints": max_total_checkpoints,
                "checkpoint_budget": checkpoint_budget.summary(),
                "checkpoint_policy": (
                    "one full regular checkpoint every checkpoint interval; one overwritten "
                    "best.pt; warning checkpoints subject to max-warning-checkpoints and "
                    "max-total-checkpoints; failure checkpoints only on failure"
                ),
                "max_observed": observer.max_observed,
                "warning_events_are_detailed_scalar_only": True,
                "historical_epoch27_not_exactly_replayed": True,
            },
        )
        return 0
    except ForensicFailure as caught:
        failure = caught
        current = observer.current or {}
        gradient_info = current.get("gradient") if current else None
        failure_detail = observer.failure_detail(caught)
        append_jsonl(events_path, failure_detail)
        failure_report_and_state(
            output_dir=output_dir,
            failure=caught,
            observer=observer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=int(current.get("epoch", completed_epochs + 1)),
            batch_index=int(current.get("batch_index", -1)),
            global_step=int(current.get("global_step", global_step)),
            sample_ids=list(current.get("sample_ids", [])),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            gradient_info=gradient_info,
            best_validation_dice=best_validation_dice,
            train_generator=train_generator,
            validation_generator=validation_generator,
            recent_history=rolling_history,
            checkpoint_budget_summary=checkpoint_budget.summary(),
        )
        write_json(
            output_dir / "experiment_metadata.json",
            {
                "status": "nonfinite_detected",
                "completed_epochs": completed_epochs,
                "global_step": global_step,
                "failure_stage": caught.stage,
                "failure_phase": caught.phase,
                "failure_classification": classify_failure(caught),
                "warning_state_events_path": str(warning_state_events_path),
                "max_warning_checkpoints": max_warning_checkpoints,
                "max_total_checkpoints": max_total_checkpoints,
                "checkpoint_budget": checkpoint_budget.summary(),
                "historical_epoch27_not_exactly_replayed": True,
                "existing_a1_artifacts_modified": False,
                "reference_files_modified": False,
            },
        )
        print(
            f"Non-finite failure observed in forensic A1 run at "
            f"epoch {current.get('epoch')}, batch {current.get('batch_index')}: "
            f"{caught.phase}/{caught.stage}/{caught.tensor_name}"
        )
        return 0
    finally:
        instrumentation.deactivate()
        instrumentation.close()
        torch.autograd.set_detect_anomaly(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vil_bottleneck_a1.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/runs/architecture_a1_forensic_seed42"))
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--anomaly-detection", action="store_true")
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=10,
        help="regular full-checkpoint interval; warning and best checkpoints are budgeted, failure checkpoints are exempt",
    )
    parser.add_argument(
        "--max-warning-checkpoints",
        type=int,
        default=20,
        help="maximum number of full warning checkpoints; warning telemetry continues after exhaustion",
    )
    parser.add_argument(
        "--max-total-checkpoints",
        type=int,
        default=40,
        help="maximum number of ordinary checkpoint files; failure checkpoints are exempt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    validate_forensic_config(config)
    output_dir = args.output.resolve() if args.output.is_absolute() else (PROJECT_ROOT / args.output).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing forensic output directory: {output_dir}"
        )
    seed_everything(int(config["seed"]))
    device = choose_device(config)
    return train_forensic(
        config=config,
        output_dir=output_dir,
        device=device,
        max_epochs=args.max_epochs,
        max_batches=args.max_batches,
        max_validation_batches=args.max_validation_batches,
        sanity_check=args.sanity_check,
        anomaly_detection=args.anomaly_detection,
        checkpoint_every=args.checkpoint_every_epochs,
        max_warning_checkpoints=args.max_warning_checkpoints,
        max_total_checkpoints=args.max_total_checkpoints,
    )


if __name__ == "__main__":
    raise SystemExit(main())
