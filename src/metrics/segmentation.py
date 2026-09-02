"""Binary Dice, IoU, precision, and recall with explicit empty-case rules."""

from __future__ import annotations

import torch


METRIC_NAMES = ("dice", "iou", "precision", "recall")


def batch_metric_values(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return one row per image: Dice, IoU, precision, recall.

    If both prediction and target are empty, the applicable metric is 1. If
    only one side is empty, the applicable metric is 0. This prevents NaNs.
    """

    if logits.shape != target.shape:
        raise ValueError(f"logits/target shape mismatch: {logits.shape} vs {target.shape}")
    probabilities = torch.sigmoid(logits)
    prediction = probabilities >= threshold
    truth = target >= 0.5
    reduce_dims = tuple(range(1, prediction.ndim))
    tp = (prediction & truth).sum(dim=reduce_dims).to(torch.float64)
    fp = (prediction & ~truth).sum(dim=reduce_dims).to(torch.float64)
    fn = (~prediction & truth).sum(dim=reduce_dims).to(torch.float64)

    dice_denominator = 2.0 * tp + fp + fn
    iou_denominator = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    both_empty = (tp + fp + fn) == 0

    dice = torch.where(
        both_empty,
        torch.ones_like(tp),
        2.0 * tp / dice_denominator.clamp_min(1.0),
    )
    iou = torch.where(
        both_empty,
        torch.ones_like(tp),
        tp / iou_denominator.clamp_min(1.0),
    )
    precision = torch.where(
        precision_denominator == 0,
        torch.where(both_empty, torch.ones_like(tp), torch.zeros_like(tp)),
        tp / precision_denominator.clamp_min(1.0),
    )
    recall = torch.where(
        recall_denominator == 0,
        torch.where(both_empty, torch.ones_like(tp), torch.zeros_like(tp)),
        tp / recall_denominator.clamp_min(1.0),
    )
    return torch.stack((dice, iou, precision, recall), dim=1)


def summarize_metric_values(values: torch.Tensor) -> dict[str, float]:
    if values.ndim != 2 or values.shape[1] != len(METRIC_NAMES):
        raise ValueError(f"expected [N,4] metric values, got {values.shape}")
    means = values.mean(dim=0)
    standard_deviations = values.std(dim=0, unbiased=False)
    result: dict[str, float] = {}
    for index, name in enumerate(METRIC_NAMES):
        result[name] = float(means[index].item())
        result[f"{name}_std"] = float(standard_deviations[index].item())
    return result
