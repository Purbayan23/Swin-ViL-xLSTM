"""Configuration-driven model construction."""

from __future__ import annotations

from typing import Any

from .pure_unet import PureUNet
from .vil_bottleneck_a1 import A1AlternatingBottleneckUNet
from .vil_bottleneck_unet import ViLBottleneckUNet


def build_model(config: dict[str, Any]):
    """Build a model while keeping the existing Pure U-Net default unchanged."""

    model_config = dict(config["model"])
    model_name = str(model_config.pop("name", "pure_unet")).lower()
    if model_name in {"pure_unet", "pure-u-net", "pure cnn u-net"}:
        if "vil_bottleneck" in model_config:
            raise ValueError("vil_bottleneck settings require model.name='unet_vil_bottleneck'")
        return PureUNet(**model_config)
    if model_name == "unet_vil_bottleneck":
        vil_config = model_config.pop("vil_bottleneck", {})
        return ViLBottleneckUNet(**model_config, vil_bottleneck=vil_config)
    if model_name == "unet_vil_bottleneck_a1":
        vil_config = model_config.pop("vil_bottleneck_a1", {})
        return A1AlternatingBottleneckUNet(**model_config, vil_bottleneck=vil_config)
    raise ValueError(f"unknown model.name: {model_name!r}")


__all__ = ["build_model"]
