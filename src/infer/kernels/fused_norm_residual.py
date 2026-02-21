"""Fused residual add + RMSNorm Triton kernel.

Fuses the common pattern ``x = residual + x; normed = rmsnorm(x)``
into a single kernel, eliminating one HBM round-trip.  Outputs both
the combined residual (needed as input to the next sub-layer) and the
normalized result.

Precision note: matches PyTorch's rounding conventions exactly.
See ``rms_norm.py`` for details on per-variant precision handling.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fused_residual_rms_norm_kernel(
    RESIDUAL,
    X,
    W,
    OUT_COMBINED,
    OUT_NORMED,
    stride_res_row,
    stride_x_row,
    stride_comb_row,
    stride_norm_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    GEMMA_STYLE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused residual + RMSNorm.

    Each program processes one row.  Computes:
        combined = residual + x
        normed = rms_norm(combined, weight)
    """
    row_idx = tl.program_id(0)
    res_ptr = RESIDUAL + row_idx * stride_res_row
    x_ptr = X + row_idx * stride_x_row
    comb_ptr = OUT_COMBINED + row_idx * stride_comb_row
    norm_ptr = OUT_NORMED + row_idx * stride_norm_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load residual and x in original dtype.
    residual = tl.load(res_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Combined: compute in input dtype (matches PyTorch bf16+bf16→bf16).
    combined_native = residual + x

    # Store combined (in input dtype for the next sub-layer).
    tl.store(comb_ptr + offsets, combined_native, mask=mask)

    # RMSNorm on combined (upcast to f32 for precision).
    combined_f32 = combined_native.to(tl.float32)
    sq = combined_f32 * combined_f32
    mean_sq = tl.sum(sq, axis=0) / N
    rrms = tl.rsqrt(mean_sq + eps)
    normed = combined_f32 * rrms

    # Apply weight — match PyTorch's precision convention per variant.
    if GEMMA_STYLE:
        # Gemma3: (1 + w_f32) * normed_f32, then cast to input dtype.
        w = tl.load(W + offsets, mask=mask, other=0.0).to(tl.float32)
        out = (1.0 + w) * normed
    else:
        # Standard: weight.to(input_dtype) * normed.to(input_dtype)
        w = tl.load(W + offsets, mask=mask, other=0.0).to(INPUT_DTYPE)
        out = w * normed.to(INPUT_DTYPE)

    tl.store(norm_ptr + offsets, out.to(INPUT_DTYPE), mask=mask)


# Map torch dtypes to triton dtype constants.
_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def triton_fused_residual_rms_norm(
    residual: Tensor,
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    gemma_style: bool = False,
) -> tuple[Tensor, Tensor]:
    """Fused residual add + RMSNorm.

    Computes:
        combined = residual + x
        normed = rms_norm(combined, weight)

    Args:
        residual: Residual tensor, any shape.
        x: Input tensor, same shape as residual.
        weight: RMSNorm weight of shape ``(N,)``.
        eps: Epsilon for numerical stability.
        gemma_style: If True, use ``(1 + weight) * normed``.

    Returns:
        ``(combined, normed)`` — both same shape/dtype as inputs.
    """
    orig_shape = residual.shape
    N = orig_shape[-1]
    residual_2d = residual.reshape(-1, N)
    x_2d = x.reshape(-1, N)
    num_rows = residual_2d.shape[0]

    combined = torch.empty_like(residual_2d)
    normed = torch.empty_like(residual_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    input_dtype = _DTYPE_MAP.get(residual.dtype, tl.float32)

    _fused_residual_rms_norm_kernel[(num_rows,)](
        residual_2d,
        x_2d,
        weight,
        combined,
        normed,
        residual_2d.stride(0),
        x_2d.stride(0),
        combined.stride(0),
        normed.stride(0),
        N=N,
        eps=eps,
        GEMMA_STYLE=gemma_style,
        INPUT_DTYPE=input_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return combined.reshape(orig_shape), normed.reshape(orig_shape)
