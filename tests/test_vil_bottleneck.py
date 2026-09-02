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
from src.models.pure_unet import PureUNet
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

    def test_source_aligned_qkv_and_matrix_lstm_head_semantics(self) -> None:
        block = ViLMLSTMBlock(dim=256, expansion=2, qkv_block_size=4)
        self.assertEqual(block.inner_dim, 512)
        self.assertEqual(block.qkv_num_heads, 128)
        self.assertEqual(block.qkv_head_dim, 4)
        self.assertEqual(block.mlstm_num_heads, 4)
        self.assertEqual(block.mlstm_head_dim, 128)
        self.assertEqual(block.q_proj.num_heads, 128)
        self.assertEqual(block.q_proj.head_dim, 4)
        self.assertEqual(block.mlstm.num_heads, 4)
        self.assertEqual(block.mlstm.head_dim, 128)
        self.assertEqual(block.mlstm.input_gate.out_features, 4)
        self.assertEqual(block.mlstm.output_norm.num_heads, 4)

    def test_architecture_a_preserves_pure_unet_initialization(self) -> None:
        torch.manual_seed(42)
        pure = PureUNet()
        torch.manual_seed(42)
        architecture_a = ViLBottleneckUNet()

        pure_parameters = dict(pure.named_parameters())
        architecture_a_parameters = dict(
            (name, parameter)
            for name, parameter in architecture_a.named_parameters()
            if not name.startswith("bottleneck_processor.")
        )
        self.assertEqual(set(pure_parameters), set(architecture_a_parameters))
        maximum_difference = 0.0
        mismatched = 0
        for name, parameter in pure_parameters.items():
            difference = (parameter - architecture_a_parameters[name]).abs().max().item()
            maximum_difference = max(maximum_difference, difference)
            mismatched += int(difference != 0.0)
        self.assertEqual(mismatched, 0)
        self.assertEqual(maximum_difference, 0.0)

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
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 5_230_441)

    def test_architecture_a_config_matches_frozen_protocol(self) -> None:
        config = load_config(PROJECT_ROOT / "configs/vil_bottleneck.json")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["dataset"]["image_size"], [224, 224])
        self.assertEqual(config["dataset"]["mask_threshold"], 128)
        self.assertEqual(config["dataset"]["prediction_threshold"], 0.5)
        self.assertEqual(config["dataset"]["manifest"], "data/splits/kvasir_seg_seed42_70_15_15.json")
        self.assertEqual(config["model"]["in_channels"], 3)
        self.assertEqual(config["model"]["out_channels"], 1)
        self.assertEqual(config["model"]["features"], [32, 64, 128, 256, 256])
        vil_config = config["model"]["vil_bottleneck"]
        self.assertEqual(vil_config["depth"], 1)
        self.assertEqual(vil_config["expansion"], 2)
        self.assertEqual(vil_config["qkv_block_size"], 4)
        self.assertEqual(config["loss"], {"bce_weight": 0.5, "dice_weight": 0.5, "dice_epsilon": 1e-6})
        self.assertEqual(config["training"]["epochs"], 100)
        self.assertEqual(config["training"]["batch_size"], 4)
        self.assertEqual(config["optimizer"]["name"], "AdamW")
        self.assertEqual(config["optimizer"]["learning_rate"], 1e-3)
        self.assertEqual(config["optimizer"]["weight_decay"], 1e-4)
        self.assertEqual(config["scheduler"]["name"], "CosineAnnealingLR")
        self.assertEqual(config["scheduler"]["t_max"], 100)
        self.assertEqual(config["scheduler"]["eta_min"], 1e-6)


if __name__ == "__main__":
    unittest.main()
