"""Shared components used across all supported architectures."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Applies the transform: ``x * rsqrt(mean(x^2) + eps) * weight``

    Used for layer norms (dim=hidden_size) and QK-norms (dim=head_dim).

    Args:
        dim: The dimension to normalize over (last axis).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Upcast to float32 for numerical stability with bfloat16/float16 inputs.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Cast normed back before weight multiply to match HF rounding convention.
        return self.weight.to(input_dtype) * normed.to(input_dtype)


# ---------------------------------------------------------------------------
# RoPE — Rotary Position Embeddings
# ---------------------------------------------------------------------------


def _vanilla_inv_freq(head_dim: int, theta: float) -> Tensor:
    """Standard theta^(-2i/d) inverse frequencies."""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))


def _llama3_scaled_inv_freq(
    head_dim: int,
    theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> Tensor:
    """Llama 3 frequency-dependent scaling.

    Divides the frequency spectrum into three bands:
    - High frequencies (short wavelength): unchanged.
    - Low frequencies (long wavelength): scaled down by ``factor``.
    - Medium frequencies: smoothly interpolated between the two.
    """
    inv_freq = _vanilla_inv_freq(head_dim, theta)
    wavelen = 2 * math.pi / inv_freq

    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor

    # Low-freq band gets full scaling; high-freq band is untouched.
    inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    # Medium band: smooth interpolation.
    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq_scaled / factor + smooth * inv_freq_scaled
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_scaled)


def _linear_scaled_inv_freq(head_dim: int, theta: float, factor: float) -> Tensor:
    """Uniform frequency scaling by ``factor``."""
    return _vanilla_inv_freq(head_dim, theta) / factor


def build_rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    rope_scaling: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for rotary position embeddings.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base frequency (e.g. 10000, 500000, 1000000).
        rope_scaling: Optional scaling config dict.  Recognised ``rope_type``
            values: ``"llama3"`` (frequency-dependent), ``"linear"`` (uniform).
            ``None`` or ``"default"`` uses vanilla RoPE.

    Returns:
        ``(cos, sin)`` each of shape ``[max_seq_len, head_dim]``, in float32.

    Note:
        Tables are precomputed in float32 for precision during the
        trigonometric computation.  ``load_model`` casts them to the
        model dtype (e.g. bf16) once at load time, matching HF's
        behavior.
    """
    if rope_scaling is None or rope_scaling.get("rope_type") in (None, "default"):
        inv_freq = _vanilla_inv_freq(head_dim, theta)
    elif rope_scaling["rope_type"] == "llama3":
        inv_freq = _llama3_scaled_inv_freq(
            head_dim,
            theta,
            factor=rope_scaling["factor"],
            low_freq_factor=rope_scaling["low_freq_factor"],
            high_freq_factor=rope_scaling["high_freq_factor"],
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        )
    elif rope_scaling["rope_type"] == "linear":
        inv_freq = _linear_scaled_inv_freq(head_dim, theta, factor=rope_scaling["factor"])
    else:
        raise ValueError(f"Unknown rope_type: {rope_scaling['rope_type']!r}")

    positions = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    """Swap and negate halves: [a, b] -> [-b, a]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor ``[batch, num_heads, seq_len, head_dim]``.
        k: Key tensor ``[batch, num_kv_heads, seq_len, head_dim]``.
        cos: Cosine table ``[seq_len, head_dim]`` (pre-sliced to actual seq_len).
        sin: Sine table ``[seq_len, head_dim]``.

    Returns:
        ``(q_rotated, k_rotated)`` with the same shapes as the inputs.
    """
    # Broadcast cos/sin over batch and head dimensions.
    cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim]
    sin = sin[None, None, :, :]
    q_rotated = q * cos + _rotate_half(q) * sin
    k_rotated = k * cos + _rotate_half(k) * sin
    return q_rotated.to(q.dtype), k_rotated.to(k.dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head attention with GQA, optional QK-norm, and RoPE.

    Projection dimensions use ``num_heads * head_dim`` (not ``hidden_size``),
    since ``head_dim`` can be decoupled from ``hidden_size // num_heads``
    (e.g. Gemma 3 1B: hidden_size=1152, num_heads=4, head_dim=256).

    Args:
        hidden_size: Model hidden dimension (input/output size).
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Dimension of each attention head.
        bias: Whether to use bias in Q/K/V/O projections.
        qk_norm: Whether to apply RMSNorm to Q and K per-head.
        rms_norm_eps: Epsilon for QK-norm RMSNorm layers.
        scale: Attention scaling factor.  Defaults to ``head_dim ** -0.5``.
            Pass ``query_pre_attn_scalar ** -0.5`` for Gemma 3.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[batch, seq_len, hidden_size]``.
            cos: RoPE cosine table ``[seq_len, head_dim]``.
            sin: RoPE sine table ``[seq_len, head_dim]``.
            mask: Attention mask ``[1, 1, seq_len, seq_len]`` (additive, float).

        Returns:
            Output tensor ``[batch, seq_len, hidden_size]``.
        """
        batch, seq_len, _ = x.shape

        # Project and reshape to [batch, num_heads, seq_len, head_dim].
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Optional QK-norm (per-head, before RoPE).
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Rotary position embeddings.
        q, k = apply_rope(q, k, cos, sin)

        # GQA: expand K/V heads to match Q heads.
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

        # Reshape back to [batch, seq_len, num_heads * head_dim] and project.
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Gated MLP
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, nn.Module] = {
    "silu": nn.SiLU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
}


class GatedMLP(nn.Module):
    """Gated MLP: ``down_proj(act_fn(gate_proj(x)) * up_proj(x))``.

    Uses SiLU activation for SwiGLU (Llama 3, Qwen 3) or GELU-tanh for
    GeGLU (Gemma 3).

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: Inner MLP dimension.
        bias: Whether to use bias in projections.
        act_fn: Activation function name (``"silu"`` or ``"gelu_pytorch_tanh"``).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        act_fn: str = "silu",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        if act_fn not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {act_fn!r}. Choose from {sorted(_ACTIVATIONS)}")
        self.act_fn = _ACTIVATIONS[act_fn]

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def causal_mask(
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Standard lower-triangular causal mask.

    Uses the float additive convention: ``0.0`` for attend, ``-inf`` for mask.

    Args:
        seq_len: Sequence length.
        dtype: Output tensor dtype.
        device: Output tensor device.

    Returns:
        Mask of shape ``[1, 1, seq_len, seq_len]``.
    """
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device), diagonal=1
    )[None, None, :, :]


def sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Causal mask that also masks positions beyond the sliding window.

    Each query position can attend to at most ``window_size`` previous
    positions (including itself).  Uses the float additive convention.

    Args:
        seq_len: Sequence length.
        window_size: Maximum attention window.
        dtype: Output tensor dtype.
        device: Output tensor device.

    Returns:
        Mask of shape ``[1, 1, seq_len, seq_len]``.
    """
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device), diagonal=1
    )
    # Mask positions further than window_size below the diagonal.
    window = torch.tril(
        torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device),
        diagonal=-(window_size),
    )
    return (causal + window)[None, None, :, :]
