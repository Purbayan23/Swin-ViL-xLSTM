"""Deterministic Kvasir-SEG image and mask preprocessing."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


def _size_tuple(size: Sequence[int]) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"image_size must contain (height, width), got {size}")
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {size}")
    return height, width


def preprocess_image(image: Image.Image, image_size: Sequence[int]) -> np.ndarray:
    """Return an RGB float32 CHW image in [0, 1]."""

    height, width = _size_tuple(image_size)
    resized = image.convert("RGB").resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32)
    array = np.ascontiguousarray(array.transpose(2, 0, 1) / 255.0)
    if array.shape != (3, height, width):
        raise RuntimeError(f"unexpected image shape after preprocessing: {array.shape}")
    return array


def preprocess_mask(
    mask: Image.Image,
    image_size: Sequence[int],
    threshold: int = 128,
) -> np.ndarray:
    """Return a binary float32 CHW mask after threshold-then-nearest resize.

    The threshold is an engineering choice for the locally decoded JPEG masks,
    not an official Kvasir-SEG threshold.
    """

    height, width = _size_tuple(image_size)
    if not 0 <= threshold <= 255:
        raise ValueError(f"mask threshold must be in [0, 255], got {threshold}")

    gray = np.asarray(mask.convert("L"), dtype=np.uint8)
    binary_255 = (gray >= threshold).astype(np.uint8) * 255
    binary_image = Image.fromarray(binary_255, mode="L")
    resized = binary_image.resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    array = (np.asarray(resized, dtype=np.uint8) > 0).astype(np.float32)
    array = np.ascontiguousarray(array[None, ...])
    if array.shape != (1, height, width):
        raise RuntimeError(f"unexpected mask shape after preprocessing: {array.shape}")
    return array
