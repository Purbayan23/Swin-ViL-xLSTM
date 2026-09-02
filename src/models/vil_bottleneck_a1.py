"""Architecture A1: alternating ViL/mLSTM bottleneck ablation.

The directional pair follows the source Vision-LSTM ``ViLBlockPair`` pattern:
an independent top-left-to-bottom-right block is followed by an independent
bottom-right-to-top-left block. The reverse block is run on a flipped token
sequence and its result is flipped back before the next operation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .pure_unet import PureUNet
from .vil_bottleneck_unet import ViLMLSTMBlock


class AlternatingViLMLSTMBottleneck(nn.Module):
    """Apply independent forward/reverse ViL blocks while preserving positions."""

    def __init__(
        self,
        channels: int,
        depth: int = 1,
        expansion: int = 2,
        qkv_block_size: int = 4,
        conv_kernel_size: int = 4,
        proj_bias: bool = False,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.channels = int(channels)
        self.depth = int(depth)
        block_kwargs = {
            "dim": self.channels,
            "expansion": expansion,
            "qkv_block_size": qkv_block_size,
            "conv_kernel_size": conv_kernel_size,
            "proj_bias": proj_bias,
            "conv_bias": conv_bias,
            "norm_eps": norm_eps,
        }
        self.forward_blocks = nn.ModuleList(
            [ViLMLSTMBlock(**block_kwargs) for _ in range(self.depth)]
        )
        self.reverse_blocks = nn.ModuleList(
            [ViLMLSTMBlock(**block_kwargs) for _ in range(self.depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W] input, got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        tokens = x.flatten(start_dim=2).transpose(1, 2)
        for forward_block, reverse_block in zip(self.forward_blocks, self.reverse_blocks):
            tokens = forward_block(tokens)
            reverse_tokens = torch.flip(tokens, dims=(1,))
            reverse_tokens = reverse_block(reverse_tokens)
            tokens = torch.flip(reverse_tokens, dims=(1,))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class A1AlternatingBottleneckUNet(PureUNet):
    """Pure U-Net with the independent alternating A1 bottleneck pair."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256, 256),
        negative_slope: float = 0.01,
        instance_norm_eps: float = 1e-5,
        vil_bottleneck: dict | None = None,
    ) -> None:
        vil_config = dict(vil_bottleneck or {})
        # Match Pure U-Net initialization exactly under a shared seed. The
        # extra A1 blocks consume RNG only after the common U-Net is built.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            negative_slope=negative_slope,
            instance_norm_eps=instance_norm_eps,
            bottleneck_processor=None,
        )
        self.bottleneck_processor = AlternatingViLMLSTMBottleneck(
            channels=int(features[-1]), **vil_config
        )


__all__ = ["A1AlternatingBottleneckUNet", "AlternatingViLMLSTMBottleneck"]
