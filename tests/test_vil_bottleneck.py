"""Focused CPU tests for the Architecture A ViL/mLSTM bottleneck."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vil_bottleneck_unet import (
    ViLBottleneckUNet,
    ViLMLSTMBlock,
    ViLMLSTMBottleneck,
)
from src.data.kvasir_seg import KvasirSegDataset
from src.training.config import load_config, project_path


class VilBottleneckTest(unittest.TestCase):
    def test_frozen_test_split_and_binary_mask_contract(self) -> None:
        config = load_config(PROJECT_ROOT / "configs/vil_bottleneck.json")
        dataset_config = config["dataset"]
        dataset = KvasirSegDataset(
            data_root=project_path(config, dataset_config["root"]),
            manifest_path=project_path(config, dataset_config["manifest"]),
            split="test",
            image_size=dataset_config["image_size"],
            mask_threshold=dataset_config["mask_threshold"],
        )
        self.assertEqual(len(dataset), 150)
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 224, 224))
        self.assertEqual(tuple(sample["mask"].shape), (1, 224, 224))
        self.assertEqual(sample["mask"].dtype, torch.float32)
        self.assertTrue(set(sample["mask"].unique().tolist()).issubset({0.0, 1.0}))

    def test_round_trip_preserves_row_major_feature_layout(self) -> None:
        bottleneck = ViLMLSTMBottleneck(channels=4, depth=1)
        bottleneck.blocks = nn.ModuleList([nn.Identity()])
        features = torch.arange(4 * 2 * 4, dtype=torch.float32).reshape(1, 4, 2, 4)
        restored = bottleneck(features)
        self.assertEqual(tuple(restored.shape), (1, 4, 2, 4))
        self.assertTrue(torch.equal(restored, features))

    def test_mlstm_block_backward_has_finite_gradients(self) -> None:
        torch.manual_seed(42)
        block = ViLMLSTMBlock(dim=8, qkv_block_size=4, conv_kernel_size=4)
        features = torch.randn(2, 16, 8, requires_grad=True)
        output = block(features)
        self.assertEqual(tuple(output.shape), (2, 16, 8))
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        for parameter in block.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_seeded_block_execution_is_repeatable(self) -> None:
        torch.manual_seed(42)
        first_block = ViLMLSTMBlock(dim=8, qkv_block_size=4, conv_kernel_size=4)
        first_input = torch.randn(1, 16, 8)
        first_output = first_block(first_input)
        torch.manual_seed(42)
        second_block = ViLMLSTMBlock(dim=8, qkv_block_size=4, conv_kernel_size=4)
        second_input = torch.randn(1, 16, 8)
        second_output = second_block(second_input)
        self.assertTrue(torch.equal(first_input, second_input))
        self.assertTrue(torch.equal(first_output, second_output))

    def test_full_model_forward_and_parameter_count(self) -> None:
        torch.manual_seed(42)
        model = ViLBottleneckUNet()
        output = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 1, 224, 224))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 5_611_617)


if __name__ == "__main__":
    unittest.main()
