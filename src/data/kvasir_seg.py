"""Kvasir-SEG dataset backed by the frozen filename-stem manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import preprocess_image, preprocess_mask


class KvasirSegDataset(Dataset):
    """Load one frozen Kvasir-SEG split.

    The dataset deliberately uses only image/mask pairs from the manifest.
    ``kavsir_bboxes.json`` is not used as a segmentation target.
    """

    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        split: str,
        image_size: Sequence[int] = (224, 224),
        mask_threshold: int = 128,
        verify_pairs: bool = True,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.image_size = tuple(int(v) for v in image_size)
        self.mask_threshold = int(mask_threshold)

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if split not in manifest.get("splits", {}):
            raise KeyError(f"split {split!r} is absent from {self.manifest_path}")
        self.entries: list[dict[str, Any]] = list(manifest["splits"][split])
        if not self.entries:
            raise ValueError(f"split {split!r} is empty")
        if verify_pairs:
            self._verify_manifest_pairs()

    def _verify_manifest_pairs(self) -> None:
        for entry in self.entries:
            sample_id = str(entry["id"])
            image_path = self.data_root / entry["image"]
            mask_path = self.data_root / entry["mask"]
            if image_path.stem != sample_id or mask_path.stem != sample_id:
                raise ValueError(f"manifest stem mismatch for {sample_id}")
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"missing image/mask for {sample_id}: {image_path}, {mask_path}"
                )
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(
                        f"image/mask size mismatch for {sample_id}: "
                        f"{image.size} vs {mask.size}"
                    )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample_id = str(entry["id"])
        image_path = self.data_root / entry["image"]
        mask_path = self.data_root / entry["mask"]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"image/mask size mismatch for {sample_id}: "
                    f"{image.size} vs {mask.size}"
                )
            image_array = preprocess_image(image, self.image_size)
            mask_array = preprocess_mask(mask, self.image_size, self.mask_threshold)

        return {
            "image": torch.from_numpy(image_array),
            "mask": torch.from_numpy(mask_array),
            "id": sample_id,
        }
