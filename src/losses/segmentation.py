"""Numerically stable BCE plus soft-Dice loss for logits."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BCESoftDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative and not both zero")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_epsilon = float(dice_epsilon)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(f"logits/target shape mismatch: {logits.shape} vs {target.shape}")
        target = target.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probabilities = torch.sigmoid(logits)
        reduce_dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * target).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.dice_epsilon) / (
            denominator + self.dice_epsilon
        )
        soft_dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * soft_dice_loss
