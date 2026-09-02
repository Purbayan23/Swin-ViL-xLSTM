"""Minimal project-local ViL/mLSTM bottleneck for Architecture A.

The sequence block follows the feature-processing pattern used by the
xLSTM-UNet bottleneck implementation: a row-major spatial feature map is
flattened to tokens, processed by a residual mLSTM block, and reshaped back.
This is deliberately not the complete VisionLSTM/VisionLSTM2 image model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .pure_unet import PureUNet


def _small_init_(parameter: torch.Tensor, dim: int) -> None:
    with torch.no_grad():
        parameter.normal_(mean=0.0, std=math.sqrt(2.0 / (5.0 * dim)))


def _wang_init_(parameter: torch.Tensor, dim: int, num_blocks: int = 1) -> None:
    with torch.no_grad():
        parameter.normal_(mean=0.0, std=2.0 / num_blocks / math.sqrt(dim))


class ResidualLayerNorm(nn.Module):
    """LayerNorm with the residual-weight initialization used by ViL."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=1.0 + self.weight,
            bias=None,
            eps=self.eps,
        )


class HeadwiseLinear(nn.Module):
    """Independent square linear projections for each Q/K/V headwise group."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by positive num_heads, got {dim}, {num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(self.dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _small_init_(self.weight, self.dim)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        leading_shape = x.shape[:-1]
        x_heads = x.reshape(-1, self.num_heads, self.head_dim)
        output = torch.einsum("nhd,hod->nho", x_heads, self.weight)
        output = output.reshape(*leading_shape, self.dim)
        if self.bias is not None:
            output = output + self.bias
        return output


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise 1-D convolution used by the reference mLSTM layer."""

    def __init__(self, dim: int, kernel_size: int = 4, bias: bool = True) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.padding = int(kernel_size) - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=dim,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,S,D] input, got {tuple(x.shape)}")
        result = self.conv(x.transpose(1, 2))
        if self.padding:
            result = result[:, :, :-self.padding]
        return result.transpose(1, 2)


class HeadwiseLayerNorm(nn.Module):
    """Per-head group normalization used on the mLSTM output."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got {dim}, {num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.num_heads or x.shape[-1] * self.num_heads != self.dim:
            raise ValueError(f"expected [B,{self.num_heads},S,{self.dim // self.num_heads}], got {tuple(x.shape)}")
        batch, _, sequence_length, _ = x.shape
        flattened = x.transpose(1, 2).reshape(batch * sequence_length, self.dim)
        normalized = F.group_norm(
            flattened,
            num_groups=self.num_heads,
            weight=1.0 + self.weight,
            bias=None,
            eps=self.eps,
        )
        return normalized.reshape(batch, sequence_length, self.num_heads, -1).transpose(1, 2)


def _parallel_stabilized_mlstm(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    causal_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the stabilized parallel matrix-memory mLSTM recurrence."""

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
    normalizer = torch.maximum(
        combination.sum(dim=-1, keepdim=True).abs(),
        torch.exp(-max_log_decay),
    )
    return (combination / (normalizer + eps)) @ values


class MatrixLSTMCell(nn.Module):
    """Parallel matrix-memory LSTM cell adapted from the reference ViL block."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got {dim}, {num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.eps = float(eps)
        self.input_gate = nn.Linear(3 * dim, num_heads)
        self.forget_gate = nn.Linear(3 * dim, num_heads)
        self.output_norm = HeadwiseLayerNorm(dim, num_heads)
        self._causal_masks: dict[tuple[int, str], torch.Tensor] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.forget_gate.weight)
        nn.init.zeros_(self.input_gate.weight)
        with torch.no_grad():
            self.forget_gate.bias.copy_(torch.linspace(3.0, 6.0, self.num_heads))
            self.input_gate.bias.normal_(mean=0.0, std=0.1)

    def _causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        key = (sequence_length, str(device))
        if key not in self._causal_masks:
            self._causal_masks[key] = torch.tril(
                torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
            )
        return self._causal_masks[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have identical [B,S,D] shapes")
        batch, sequence_length, _ = q.shape
        gate_input = torch.cat((q, k, v), dim=-1)
        q = q.reshape(batch, sequence_length, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch, sequence_length, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch, sequence_length, self.num_heads, -1).transpose(1, 2)
        input_gate = self.input_gate(gate_input).transpose(-1, -2).unsqueeze(-1)
        forget_gate = self.forget_gate(gate_input).transpose(-1, -2).unsqueeze(-1)
        h_state = _parallel_stabilized_mlstm(
            q,
            k,
            v,
            input_gate,
            forget_gate,
            self._causal_mask(sequence_length, q.device),
            eps=self.eps,
        )
        normalized = self.output_norm(h_state)
        return normalized.transpose(1, 2).reshape(batch, sequence_length, self.dim)


class ViLMLSTMBlock(nn.Module):
    """One residual top-left-to-bottom-right ViL/mLSTM sequence block."""

    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        qkv_block_size: int = 4,
        conv_kernel_size: int = 4,
        proj_bias: bool = False,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if expansion <= 0 or qkv_block_size <= 0:
            raise ValueError("expansion and qkv_block_size must be positive")
        inner_dim = int(expansion) * int(dim)
        if inner_dim % qkv_block_size != 0:
            raise ValueError(
                f"expanded dimension {inner_dim} must be divisible by qkv_block_size {qkv_block_size}"
            )
        self.dim = int(dim)
        self.inner_dim = inner_dim
        self.qkv_block_size = int(qkv_block_size)
        # The reference uses two distinct groupings of the same expanded
        # embedding: Q/K/V projections use 128 groups of width 4 here, while
        # the matrix-LSTM cell uses 4 heads of width 128.
        self.qkv_num_heads = inner_dim // self.qkv_block_size
        self.qkv_head_dim = self.qkv_block_size
        self.mlstm_num_heads = self.qkv_block_size
        self.mlstm_head_dim = inner_dim // self.mlstm_num_heads
        self.norm = ResidualLayerNorm(dim, eps=norm_eps)
        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
        self.q_proj = HeadwiseLinear(inner_dim, self.qkv_num_heads, bias=proj_bias)
        self.k_proj = HeadwiseLinear(inner_dim, self.qkv_num_heads, bias=proj_bias)
        self.v_proj = HeadwiseLinear(inner_dim, self.qkv_num_heads, bias=proj_bias)
        self.conv1d = CausalDepthwiseConv1d(inner_dim, conv_kernel_size, bias=conv_bias)
        self.mlstm = MatrixLSTMCell(inner_dim, self.mlstm_num_heads)
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _small_init_(self.proj_up.weight, self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        _wang_init_(self.proj_down.weight, self.dim)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
        _small_init_(self.q_proj.weight, self.dim)
        _small_init_(self.k_proj.weight, self.dim)
        _small_init_(self.v_proj.weight, self.dim)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        nn.init.ones_(self.learnable_skip)
        self.mlstm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"expected [B,S,{self.dim}] input, got {tuple(x.shape)}")
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        residual = x
        x_inner = self.proj_up(self.norm(x))
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)
        x_mlstm_conv_act = F.silu(self.conv1d(x_mlstm))
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_state = self.mlstm(q=q, k=k, v=v)
        h_state = h_state + self.learnable_skip * x_mlstm_conv_act
        return residual + self.proj_down(h_state * F.silu(z))


class ViLMLSTMBottleneck(nn.Module):
    """Map ``[B,C,H,W]`` to a row-major sequence and back."""

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
        self.blocks = nn.ModuleList(
            [
                ViLMLSTMBlock(
                    dim=self.channels,
                    expansion=expansion,
                    qkv_block_size=qkv_block_size,
                    conv_kernel_size=conv_kernel_size,
                    proj_bias=proj_bias,
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W] input, got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        tokens = x.flatten(start_dim=2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class ViLBottleneckUNet(PureUNet):
    """Pure U-Net with exactly one changed component: a ViL/mLSTM bottleneck."""

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
        # Initialize the shared U-Net first.  This preserves the exact Pure
        # U-Net RNG draw order under a shared seed; the additional processor
        # is initialized afterward from the subsequent RNG state.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            negative_slope=negative_slope,
            instance_norm_eps=instance_norm_eps,
            bottleneck_processor=None,
        )
        self.bottleneck_processor = ViLMLSTMBottleneck(
            channels=int(features[-1]), **vil_config
        )


__all__ = ["ViLBottleneckUNet", "ViLMLSTMBottleneck", "ViLMLSTMBlock"]
