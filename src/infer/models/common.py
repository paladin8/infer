"""Shared components used across all supported architectures."""

from __future__ import annotations

import math
from typing import Any

import torch
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
        return (self.weight * normed).to(input_dtype)


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
        ``(cos, sin)`` each of shape ``[max_seq_len, head_dim]``.
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
