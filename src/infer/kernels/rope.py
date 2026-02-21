"""Fused RoPE (Rotary Position Embedding) Triton kernel.

Applies the rotation to Q and K tensors in a single kernel launch,
replacing the PyTorch slice→cat→negate→mul→add sequence and avoiding
a second launch overhead.

Precision note: the rotation ``x * cos + rotate_half(x) * sin`` compiles
to FMA (fused multiply-add) instructions inside the Triton kernel.  FMA
uses a single rounding step for ``a*b + c``, whereas HuggingFace's
PyTorch implementation rounds each intermediate (``q * cos``, then
``rotate_half(q) * sin``) to bf16 before adding.  This produces a small
per-element difference (~0.016 max in bf16), but the Triton result is
actually **closer to the fp64 ground truth** (~9% of elements closer vs
~5% for PyTorch, with ~7% lower total absolute error).  The difference
is amplified by downstream RMSNorm layers — architectures with more
norms per block (e.g. Gemma 3's 4 sandwich norms) show larger block-level
divergence despite identical per-element RoPE error.
"""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_kernel(
    Q,  # [batch, num_heads, seq_len, head_dim]
    K,  # [batch, num_kv_heads, seq_len, head_dim]
    COS,  # [seq_len, head_dim]
    SIN,  # [seq_len, head_dim]
    Q_OUT,
    K_OUT,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_qo_batch,
    stride_qo_head,
    stride_qo_seq,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_ko_batch,
    stride_ko_head,
    stride_ko_seq,
    stride_cos_seq,
    seq_len,
    num_q_heads,
    total_heads,  # num_q_heads + num_kv_heads
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE rotation to one head at one position.

    The grid covers all Q heads followed by all K heads:
    ``(batch * (num_q_heads + num_kv_heads) * seq_len,)``.
    Programs with ``head_idx < num_q_heads`` process Q; the rest process K.
    """
    pid = tl.program_id(0)

    # Decompose pid into (batch, head, seq) indices.
    seq_idx = pid % seq_len
    head_idx = (pid // seq_len) % total_heads
    batch_idx = pid // (seq_len * total_heads)

    # Select Q or K pointers and strides based on head_idx.
    is_q = head_idx < num_q_heads
    if is_q:
        in_offset = batch_idx * stride_q_batch + head_idx * stride_q_head + seq_idx * stride_q_seq
        out_offset = (
            batch_idx * stride_qo_batch + head_idx * stride_qo_head + seq_idx * stride_qo_seq
        )
        IN = Q
        OUT = Q_OUT
    else:
        k_head = head_idx - num_q_heads
        in_offset = batch_idx * stride_k_batch + k_head * stride_k_head + seq_idx * stride_k_seq
        out_offset = batch_idx * stride_ko_batch + k_head * stride_ko_head + seq_idx * stride_ko_seq
        IN = K
        OUT = K_OUT

    cos_offset = seq_idx * stride_cos_seq

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HALF_DIM

    # Load first half and second half.
    x_first = tl.load(IN + in_offset + offsets, mask=mask, other=0.0)
    x_second = tl.load(IN + in_offset + HALF_DIM + offsets, mask=mask, other=0.0)

    # Load cos/sin for this position (first half only — the table is
    # constructed as cat([freqs, freqs], dim=-1) so both halves are identical).
    cos_val = tl.load(COS + cos_offset + offsets, mask=mask, other=0.0)
    sin_val = tl.load(SIN + cos_offset + offsets, mask=mask, other=0.0)

    # RoPE rotation: out = x * cos + rotate_half(x) * sin
    # rotate_half([a, b]) = [-b, a]
    out_first = x_first * cos_val + (-x_second) * sin_val
    out_second = x_second * cos_val + x_first * sin_val

    tl.store(OUT + out_offset + offsets, out_first, mask=mask)
    tl.store(OUT + out_offset + HALF_DIM + offsets, out_second, mask=mask)


def triton_apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings using a single fused Triton kernel.

    Processes both Q and K in one launch.  Handles non-contiguous inputs
    (e.g. after ``.transpose(1, 2)``) by passing separate input/output strides.

    Args:
        q: Query tensor ``[batch, num_heads, seq_len, head_dim]``.
        k: Key tensor ``[batch, num_kv_heads, seq_len, head_dim]``.
        cos: Cosine table ``[seq_len, head_dim]``.
        sin: Sine table ``[seq_len, head_dim]``.

    Returns:
        ``(q_rotated, k_rotated)`` with the same shapes and dtypes as inputs.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be 4-D [batch, heads, seq, head_dim], got shape {q.shape}")
    if k.ndim != 4:
        raise ValueError(f"k must be 4-D [batch, heads, seq, head_dim], got shape {k.shape}")
    if cos.ndim != 2:
        raise ValueError(
            f"cos must be 2-D [seq_len, head_dim], got shape {cos.shape}. "
            "Did you forget to squeeze the batch dimension?"
        )
    if sin.ndim != 2:
        raise ValueError(
            f"sin must be 2-D [seq_len, head_dim], got shape {sin.shape}. "
            "Did you forget to squeeze the batch dimension?"
        )
    if q.stride(-1) != 1:
        raise ValueError(
            f"q must be contiguous in the last dimension (stride[-1]=1), got stride {q.stride()}"
        )
    if k.stride(-1) != 1:
        raise ValueError(
            f"k must be contiguous in the last dimension (stride[-1]=1), got stride {k.stride()}"
        )
    if cos.stride(-1) != 1:
        raise ValueError(f"cos must be contiguous in the last dimension, got stride {cos.stride()}")

    batch, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape
    half_dim = head_dim // 2
    total_heads = num_heads + num_kv_heads

    q_out = q.new_empty(q.shape)
    k_out = k.new_empty(k.shape)

    BLOCK_SIZE = triton.next_power_of_2(half_dim)

    grid = (batch * total_heads * seq_len,)
    _rope_kernel[grid](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        cos.stride(0),
        seq_len,
        num_heads,
        total_heads,
        HALF_DIM=half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q_out, k_out
