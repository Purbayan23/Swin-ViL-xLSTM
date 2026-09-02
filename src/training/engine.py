"""Minimal model-agnostic training/evaluation loops."""

from __future__ import annotations

from typing import Any

import torch

from src.metrics.segmentation import METRIC_NAMES, batch_metric_values


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    loss_total = 0.0
    sample_total = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        loss_total += float(loss.detach().item()) * batch_size
        sample_total += batch_size
    return loss_total / max(sample_total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    sample_total = 0
    metric_total = torch.zeros(len(METRIC_NAMES), dtype=torch.float64)
    metric_square_total = torch.zeros(len(METRIC_NAMES), dtype=torch.float64)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        values = batch_metric_values(logits, masks, threshold=threshold).cpu()
        batch_size = images.shape[0]
        loss_total += float(loss.item()) * batch_size
        metric_total += values.sum(dim=0)
        metric_square_total += (values * values).sum(dim=0)
        sample_total += batch_size
    result = {"loss": loss_total / max(sample_total, 1)}
    denominator = max(sample_total, 1)
    means = metric_total / denominator
    variances = (metric_square_total / denominator - means * means).clamp_min(0.0)
    for index, name in enumerate(METRIC_NAMES):
        result[name] = float(means[index].item())
        result[f"{name}_std"] = float(variances[index].sqrt().item())
    return result
