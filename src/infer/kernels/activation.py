"""Fused gated activation Triton kernel.

Fuses ``act_fn(gate) * up`` into a single kernel, eliminating the
intermediate activation tensor from HBM.  Supports SiLU (SwiGLU) and
GELU-tanh (GeGLU) activations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fused_gated_activation_kernel(
    GATE,
    UP,
    OUT,
    stride_gate_row,
    stride_up_row,
    stride_out_row,
    N: tl.constexpr,
    USE_GELU: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute act_fn(gate) * up in a single kernel.

    Each program processes one row of size N (the intermediate_size).
    """
    row_idx = tl.program_id(0)
    gate_ptr = GATE + row_idx * stride_gate_row
    up_ptr = UP + row_idx * stride_up_row
    out_ptr = OUT + row_idx * stride_out_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    if USE_GELU:
        # GELU with tanh approximation:
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        activated = 0.5 * gate * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    else:
        # SiLU: silu(x) = x * sigmoid(x)
        activated = gate * tl.sigmoid(gate)

    result = activated * up.to(tl.float32)
    tl.store(out_ptr + offsets, result.to(up.dtype), mask=mask)


def triton_fused_gated_activation(
    gate: Tensor,
    up: Tensor,
    use_gelu: bool = False,
) -> Tensor:
    """Fused gated activation: ``act_fn(gate) * up``.

    Args:
        gate: Gate projection output, shape ``[*, intermediate_size]``.
        up: Up projection output, same shape as gate.
        use_gelu: If True, use GELU-tanh (GeGLU). Otherwise SiLU (SwiGLU).

    Returns:
        Result tensor with same shape and dtype as inputs.
    """
    orig_shape = gate.shape
    N = orig_shape[-1]
    gate_2d = gate.reshape(-1, N)
    up_2d = up.reshape(-1, N)
    num_rows = gate_2d.shape[0]

    out = torch.empty_like(gate_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _fused_gated_activation_kernel[(num_rows,)](
        gate_2d,
        up_2d,
        out,
        gate_2d.stride(0),
        up_2d.stride(0),
        out.stride(0),
        N=N,
        USE_GELU=use_gelu,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)
