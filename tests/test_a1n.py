import unittest
from pathlib import Path

import torch

from src.models.factory import build_model
from src.models.vil_bottleneck_a1n import _parallel_log_domain_mlstm
from src.models.vil_bottleneck_unet import _parallel_stabilized_mlstm
from src.training.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def mlstm_inputs(
    *,
    sequence_length: int = 4,
    input_gate_value: float = 0.0,
    query_value: float = 0.25,
):
    queries = torch.full((1, 2, sequence_length, 3), query_value)
    keys = torch.full_like(queries, query_value)
    values = torch.arange(1, 1 + queries.numel(), dtype=torch.float32).reshape_as(queries)
    input_gate = torch.full((1, 2, sequence_length, 1), input_gate_value)
    forget_gate = torch.zeros_like(input_gate)
    causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
    return queries, keys, values, input_gate, forget_gate, causal_mask


class A1NTest(unittest.TestCase):
    def test_log_domain_matches_original_for_safe_finite_range(self):
        inputs = mlstm_inputs(input_gate_value=-0.5)
        original = _parallel_stabilized_mlstm(*inputs, eps=1e-6)
        log_domain, diagnostics = _parallel_log_domain_mlstm(*inputs, eps=1e-6)
        self.assertTrue(torch.isfinite(log_domain).all())
        self.assertTrue(torch.allclose(original, log_domain, rtol=1e-5, atol=1e-7))
        self.assertTrue(torch.isfinite(diagnostics["log_normalizer"]).all())
        self.assertTrue(torch.isfinite(diagnostics["inverse_normalizer"]).all())

    def test_negative_max_log_decay_overflow_is_avoided(self):
        inputs = mlstm_inputs(input_gate_value=-1000.0)
        _, diagnostics = _parallel_log_domain_mlstm(*inputs, eps=1e-6)
        original_exp_term = torch.exp(-diagnostics["max_log_decay"])
        log_domain, _ = _parallel_log_domain_mlstm(*inputs, eps=1e-6)
        self.assertTrue(torch.isinf(original_exp_term).any())
        self.assertTrue(torch.isfinite(diagnostics["inverse_normalizer"]).all())
        self.assertTrue(torch.isfinite(log_domain).all())

    def test_zero_and_small_combination_sum_are_finite(self):
        zero_inputs = mlstm_inputs(query_value=0.0)
        zero_output, zero_diagnostics = _parallel_log_domain_mlstm(*zero_inputs)
        self.assertTrue(torch.isfinite(zero_output).all())
        self.assertTrue(torch.equal(zero_output, torch.zeros_like(zero_output)))
        self.assertTrue(torch.isfinite(zero_diagnostics["log_normalizer"]).all())

        small_inputs = mlstm_inputs(query_value=1e-20)
        small_output, _ = _parallel_log_domain_mlstm(*small_inputs)
        self.assertTrue(torch.isfinite(small_output).all())

    def test_a1n_has_a1_initialization_and_safe_inference_equivalence(self):
        a1_config = load_config(PROJECT_ROOT / "configs/vil_bottleneck_a1.json")
        a1n_config = load_config(PROJECT_ROOT / "configs/vil_bottleneck_a1n.json")
        torch.manual_seed(42)
        a1 = build_model(a1_config).eval()
        torch.manual_seed(42)
        a1n = build_model(a1n_config).eval()
        for (_, a1_parameter), (_, a1n_parameter) in zip(
            a1.named_parameters(), a1n.named_parameters()
        ):
            self.assertTrue(torch.equal(a1_parameter, a1n_parameter))
        self.assertEqual(
            sum(parameter.numel() for parameter in a1.parameters()),
            sum(parameter.numel() for parameter in a1n.parameters()),
        )
        torch.manual_seed(123)
        inputs = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            a1_output = a1(inputs)
            a1n_output = a1n(inputs)
        self.assertTrue(torch.isfinite(a1n_output).all())
        self.assertTrue(torch.allclose(a1_output, a1n_output, rtol=1e-5, atol=1e-6))

    def test_a1n_does_not_materialize_exp_log_exp_term(self):
        source = Path(PROJECT_ROOT / "src/models/vil_bottleneck_a1n.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("torch.exp(log_exp_term)", source)
        self.assertNotIn("torch.exp(-max_log_decay)", source)


if __name__ == "__main__":
    unittest.main()
