"""A1-N: diagnostic log-domain evaluation of the existing A1 mLSTM normalizer.

The model structure and parameters are inherited from A1.  Each existing
MatrixLSTMCell instance is given an alternate forward method that evaluates
the same normalization in the log domain.  No architectural, gating, or
initialization change is introduced.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Any, Callable

import torch
from torch.nn import functional as F

from .vil_bottleneck_a1 import A1AlternatingBottleneckUNet
from .vil_bottleneck_unet import MatrixLSTMCell


DiagnosticCallback = Callable[[MatrixLSTMCell, dict[str, torch.Tensor]], None]
_DIAGNOSTIC_CALLBACK: DiagnosticCallback | None = None


def set_diagnostic_callback(callback: DiagnosticCallback | None) -> None:
    global _DIAGNOSTIC_CALLBACK
    _DIAGNOSTIC_CALLBACK = callback


def get_diagnostic_callback() -> DiagnosticCallback | None:
    return _DIAGNOSTIC_CALLBACK


def _parallel_log_domain_mlstm(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    causal_mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the existing mLSTM normalization without ``exp(-max_log_decay)``."""

    _, _, _, head_dim = queries.shape
    log_forget = F.logsigmoid(forget_gate)
    log_forget_cumsum = torch.cat(
        [torch.zeros_like(log_forget[:, :, :1]), torch.cumsum(log_forget, dim=-2)],
        dim=-2,
    )
    repeated = log_forget_cumsum.expand(-1, -1, -1, log_forget_cumsum.shape[-2])
    log_forget_matrix = repeated - repeated.transpose(-2, -1)
    log_forget_matrix = torch.where(
        causal_mask,
        log_forget_matrix[:, :, 1:, 1:],
        torch.full_like(log_forget_matrix[:, :, 1:, 1:], -float("inf")),
    )

    log_decay = log_forget_matrix + input_gate.transpose(-2, -1)
    max_log_decay = log_decay.amax(dim=-1, keepdim=True)
    decay = torch.exp(log_decay - max_log_decay)
    keys_scaled = keys / math.sqrt(head_dim)
    combination = (queries @ keys_scaled.transpose(-2, -1)) * decay
    combination_sum_abs = combination.sum(dim=-1, keepdim=True).abs()
    log_abs_combination = torch.log(combination_sum_abs)
    log_exp_term = -max_log_decay
    log_normalizer = torch.maximum(log_abs_combination, log_exp_term)
    inverse_normalizer = torch.exp(-log_normalizer)

    # This is algebraically equivalent to combination/(normalizer + eps):
    # (combination * 1/normalizer) / (1 + eps/normalizer).
    normalized_combination = (combination * inverse_normalizer) / (
        1.0 + eps * inverse_normalizer
    )
    output = normalized_combination @ values
    diagnostics = {
        # These tensors are exposed for the existing forensic observer.  The
        # returned values are the values used by this path; no separate
        # numerical approximation is made for instrumentation.
        "cumulative_log_decay": log_forget_cumsum,
        "combination": combination,
        "normalized_combination": normalized_combination,
        "log_abs_combination": log_abs_combination,
        "log_normalizer": log_normalizer,
        "inverse_normalizer": inverse_normalizer,
        "max_log_decay": max_log_decay,
    }
    return output, diagnostics


def _log_domain_matrix_lstm_forward(
    self: MatrixLSTMCell,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical [B,S,D] shapes")
    batch, sequence_length, _ = q.shape
    gate_input = torch.cat((q, k, v), dim=-1)
    q = q.reshape(batch, sequence_length, self.num_heads, -1).transpose(1, 2)
    k = k.reshape(batch, sequence_length, self.num_heads, -1).transpose(1, 2)
    v = v.reshape(batch, sequence_length, self.num_heads, -1).transpose(1, 2)
    input_gate = self.input_gate(gate_input).transpose(-1, -2).unsqueeze(-1)
    forget_gate = self.forget_gate(gate_input).transpose(-1, -2).unsqueeze(-1)
    h_state, diagnostics = _parallel_log_domain_mlstm(
        q,
        k,
        v,
        input_gate,
        forget_gate,
        self._causal_mask(sequence_length, q.device),
        eps=self.eps,
    )
    callback = get_diagnostic_callback()
    if callback is not None:
        callback(self, diagnostics)
    normalized = self.output_norm(h_state)
    return normalized.transpose(1, 2).reshape(batch, sequence_length, self.dim)


class A1NAlternatingBottleneckUNet(A1AlternatingBottleneckUNet):
    """A1 with only the mLSTM normalization evaluated in the log domain."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for module in self.modules():
            if isinstance(module, MatrixLSTMCell):
                # Bind without constructing or initializing any replacement
                # module; all A1 parameters and their RNG-derived values stay
                # identical.
                module.forward = MethodType(_log_domain_matrix_lstm_forward, module)


__all__ = [
    "A1NAlternatingBottleneckUNet",
    "_parallel_log_domain_mlstm",
    "get_diagnostic_callback",
    "set_diagnostic_callback",
]
