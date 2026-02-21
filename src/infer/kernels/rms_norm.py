"""Fused RMSNorm Triton kernel.

Replaces 7+ CUDA kernel launches (cast→pow→mean→rsqrt→mul→cast→mul)
with a single kernel.  Supports both standard RMSNorm and Gemma 3's
``(1 + weight)`` variant.

Precision note: the standard variant matches PyTorch's rounding convention
(``weight.to(input_dtype) * normed.to(input_dtype)``) by casting normed
back to the input dtype before the weight multiply.  The Gemma variant
computes entirely in f32 then casts, matching HF's Gemma3RMSNorm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rms_norm_kernel(
    X,
    W,
    Y,
    stride_x_row,
    stride_y_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    GEMMA_STYLE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm kernel.

    Each program instance processes one row (the last dimension).
    """
    row_idx = tl.program_id(0)
    x_ptr = X + row_idx * stride_x_row
    y_ptr = Y + row_idx * stride_y_row

    # Load the row in float32 for numerical stability.
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS: rsqrt(mean(x^2) + eps).
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rrms = tl.rsqrt(mean_sq + eps)

    # Normalize.
    normed = x * rrms

    # Apply weight — match PyTorch's precision convention per variant.
    if GEMMA_STYLE:
        # Gemma3: (1 + w_f32) * normed_f32, then cast to input dtype.
        w = tl.load(W + offsets, mask=mask, other=0.0).to(tl.float32)
        out = (1.0 + w) * normed
    else:
        # Standard: weight.to(input_dtype) * normed.to(input_dtype)
        # This matches HF's Llama/Qwen3 RMSNorm rounding convention.
        w = tl.load(W + offsets, mask=mask, other=0.0).to(INPUT_DTYPE)
        out = w * normed.to(INPUT_DTYPE)

    tl.store(y_ptr + offsets, out.to(INPUT_DTYPE), mask=mask)


# Map torch dtypes to triton dtype constants.
_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def triton_rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    gemma_style: bool = False,
) -> Tensor:
    """Fused RMSNorm using Triton.

    Args:
        x: Input tensor of any shape, normalized over the last dimension.
        weight: Weight tensor of shape ``(N,)`` where N is the last dim of x.
        eps: Epsilon for numerical stability.
        gemma_style: If True, use ``(1 + weight) * normed`` (Gemma 3 convention).

    Returns:
        Normalized tensor with the same shape and dtype as ``x``.
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.reshape(-1, N)
    num_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)

    # Round up to next power of 2 for the block size.
    BLOCK_SIZE = triton.next_power_of_2(N)

    input_dtype = _DTYPE_MAP.get(x.dtype, tl.float32)

    _rms_norm_kernel[(num_rows,)](
        x_2d,
        weight,
        y,
        x_2d.stride(0),
        y.stride(0),
        N=N,
        eps=eps,
        GEMMA_STYLE=gemma_style,
        INPUT_DTYPE=input_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(orig_shape)
