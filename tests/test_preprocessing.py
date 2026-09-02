"""Lightweight preprocessing tests; does not require PyTorch."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_image, preprocess_mask


class PreprocessingTest(unittest.TestCase):
    def test_threshold_then_nearest_resize_is_binary(self) -> None:
        values = np.array(
            [[0, 8, 246, 255], [1, 7, 247, 254]], dtype=np.uint8
        )
        rgb = np.stack((values, values, values), axis=-1)
        result = preprocess_mask(Image.fromarray(rgb, mode="RGB"), (8, 8), 128)
        self.assertEqual(result.shape, (1, 8, 8))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(set(result.ravel().tolist()), {0.0, 1.0})

    def test_real_pair_contract(self) -> None:
        image_path = PROJECT_ROOT / "data" / "Kvasir-SEG" / "images"
        image_path = sorted(image_path.glob("*.jpg"))[0]
        mask_path = PROJECT_ROOT / "data" / "Kvasir-SEG" / "masks" / image_path.name
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image_result = preprocess_image(image, (224, 224))
            mask_result = preprocess_mask(mask, (224, 224), 128)
        self.assertEqual(image_result.shape, (3, 224, 224))
        self.assertEqual(image_result.dtype, np.float32)
        self.assertEqual(mask_result.shape, (1, 224, 224))
        self.assertEqual(mask_result.dtype, np.float32)
        self.assertTrue(set(mask_result.ravel().tolist()).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
