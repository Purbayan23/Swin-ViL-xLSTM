"""Pure convolutional U-Net specified by BASELINE_SPECIFICATION_V1.md."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    """Two Conv2d -> InstanceNorm2d -> LeakyReLU operations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.01,
        instance_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(
                out_channels, affine=True, eps=instance_norm_eps
            ),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(
                out_channels, affine=True, eps=instance_norm_eps
            ),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PureUNet(nn.Module):
    """NCHW Pure U-Net with one foreground-logit output channel."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256, 256),
        negative_slope: float = 0.01,
        instance_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        features = tuple(int(value) for value in features)
        if len(features) != 5:
            raise ValueError("Pure U-Net V1 requires five feature widths")
        if any(value <= 0 for value in features):
            raise ValueError(f"feature widths must be positive, got {features}")

        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for next_channels in features[:-1]:
            self.encoders.append(
                ConvNormAct(
                    current_channels,
                    next_channels,
                    negative_slope=negative_slope,
                    instance_norm_eps=instance_norm_eps,
                )
            )
            current_channels = next_channels

        self.bottleneck = ConvNormAct(
            features[-2],
            features[-1],
            negative_slope=negative_slope,
            instance_norm_eps=instance_norm_eps,
        )

        self.up_projections = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = features[-1]
        for skip_channels in reversed(features[:-1]):
            self.up_projections.append(
                nn.Conv2d(current_channels, skip_channels, kernel_size=1, bias=True)
            )
            self.decoders.append(
                ConvNormAct(
                    skip_channels * 2,
                    skip_channels,
                    negative_slope=negative_slope,
                    instance_norm_eps=instance_norm_eps,
                )
            )
            current_channels = skip_channels

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for projection, decoder, skip in zip(
            self.up_projections, self.decoders, reversed(skips)
        ):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if x.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    f"decoder/skip spatial mismatch: {x.shape[-2:]} vs {skip.shape[-2:]}"
                )
            x = projection(x)
            x = torch.cat((x, skip), dim=1)
            x = decoder(x)
        return self.head(x)
