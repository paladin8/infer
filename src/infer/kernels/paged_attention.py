"""Triton paged attention kernel for decode.

Fuses the gather and attention computation for decode operations (single
query token per sequence).  Reads K/V directly from scattered block pools
instead of gathering into contiguous tensors before SDPA, eliminating
``O(batch * seq_len * head_dim)`` gather copy per layer.

Decode-only: prefill always uses standard SDPA.  Supports sliding-window
attention via the ``window_size`` parameter.
"""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _paged_attention_kernel(
    OUT,  # [batch, num_q_heads, head_dim]
    Q,  # [batch, num_q_heads, 1, head_dim]
    K_POOL,  # [total_blocks, num_kv_heads, block_size, head_dim]
    V_POOL,  # [total_blocks, num_kv_heads, block_size, head_dim]
    PAGE_TABLE,  # [batch, max_num_blocks]  int32
    SEQ_LENS,  # [batch]  int32
    stride_out_batch,
    stride_out_head,
    stride_q_batch,
    stride_q_head,
    stride_k_block,
    stride_k_head,
    stride_k_pos,
    stride_v_block,
    stride_v_head,
    stride_v_pos,
    stride_pt_batch,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
    BLOCK_HD: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_BS: tl.constexpr,  # next_power_of_2(BLOCK_SIZE)
    WINDOW_SIZE: tl.constexpr,  # 0 = full attention, >0 = sliding window
):
    """Paged attention for one (batch, q_head) pair.

    Grid: ``(batch, num_q_heads)``.

    Uses online softmax (streaming softmax) to iterate over KV blocks
    without materialising the full attention matrix.  Float32 accumulation
    for numerical stability.

    When ``WINDOW_SIZE > 0``, only the last ``WINDOW_SIZE`` positions
    contribute to attention.  Blocks entirely before the window are skipped,
    and the block straddling the window boundary has per-position masking.
    """
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    # GQA: map query head to KV head.
    kv_head_idx = q_head_idx // GQA_GROUP_SIZE

    # Sequence length for this batch element.
    seq_len = tl.load(SEQ_LENS + batch_idx)

    # Sliding window: compute the start of the attention window.
    window_start = tl.maximum(0, seq_len - WINDOW_SIZE) if WINDOW_SIZE > 0 else 0

    # Load query vector [HEAD_DIM] → float32.
    q_offset = batch_idx * stride_q_batch + q_head_idx * stride_q_head
    dim_offsets = tl.arange(0, BLOCK_HD)
    dim_mask = dim_offsets < HEAD_DIM
    q = tl.load(Q + q_offset + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    # Online softmax running state (explicit Triton scalars).
    m_prev = tl.full([], float("-inf"), dtype=tl.float32)  # running max
    l_prev = tl.full([], 0.0, dtype=tl.float32)  # running sum of exp(scores)
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)  # weighted V accumulator

    pos_offsets = tl.arange(0, BLOCK_BS)

    for block_idx in range(MAX_NUM_BLOCKS):
        start_pos = block_idx * BLOCK_SIZE
        # Triton has no break; use valid_len=0 to mask everything for
        # blocks past the sequence boundary (all scores become -inf,
        # contributing zero to the output via softmax).
        remaining = seq_len - start_pos
        valid_len = tl.maximum(0, tl.minimum(BLOCK_SIZE, remaining))

        # Skip blocks entirely before the sliding window.
        block_end_pos = start_pos + BLOCK_SIZE
        skip = block_end_pos <= window_start if WINDOW_SIZE > 0 else False

        if not skip:
            # Physical block ID from page table.
            block_id = tl.load(PAGE_TABLE + batch_idx * stride_pt_batch + block_idx).to(tl.int64)

            # --- Load K block [BLOCK_BS, BLOCK_HD] → float32 ---
            k_base = block_id * stride_k_block + kv_head_idx * stride_k_head
            k_ptrs = K_POOL + k_base + pos_offsets[:, None] * stride_k_pos + dim_offsets[None, :]
            k_mask = (pos_offsets[:, None] < valid_len) & dim_mask[None, :]
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Attention scores: q @ k^T * scale → [BLOCK_BS].
            scores = tl.sum(q[None, :] * k, axis=1) * SCALE
            scores = tl.where(pos_offsets < valid_len, scores, float("-inf"))

            # Sliding window: mask out positions before the window start.
            if WINDOW_SIZE > 0:
                abs_positions = start_pos + pos_offsets
                scores = tl.where(abs_positions >= window_start, scores, float("-inf"))

            # Online softmax update.
            m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
            exp_scores = tl.exp(scores - m_new)
            correction = tl.exp(m_prev - m_new)
            l_new = correction * l_prev + tl.sum(exp_scores, axis=0)

            # --- Load V block [BLOCK_BS, BLOCK_HD] → float32 ---
            v_base = block_id * stride_v_block + kv_head_idx * stride_v_head
            v_ptrs = V_POOL + v_base + pos_offsets[:, None] * stride_v_pos + dim_offsets[None, :]
            v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Accumulate weighted V.
            acc = correction * acc + tl.sum(exp_scores[:, None] * v, axis=0)

            m_prev = m_new
            l_prev = l_new

    # Final normalisation (guard against l_prev==0 from empty sequences).
    safe_l = tl.maximum(l_prev, 1e-6)
    out = tl.where(dim_mask, acc / safe_l, 0.0)

    # Store output.
    out_offset = batch_idx * stride_out_batch + q_head_idx * stride_out_head
    tl.store(OUT + out_offset + dim_offsets, out, mask=dim_mask)


def triton_paged_attention(
    q: Tensor,
    k_pool: Tensor,
    v_pool: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    *,
    scale: float,
    max_num_blocks: int,
    window_size: int = 0,
) -> Tensor:
    """Paged attention for decode step.

    Reads K/V directly from scattered block pools using the page table,
    avoiding the O(batch * seq_len * head_dim) gather.  Handles GQA
    internally.

    Args:
        q: Query tensor ``[batch, num_q_heads, 1, head_dim]``.
        k_pool: Key block pool for one layer,
            ``[total_blocks, num_kv_heads, block_size, head_dim]``.
        v_pool: Value block pool, same shape as ``k_pool``.
        page_table: Per-sequence block IDs ``[batch, max_num_blocks]``,
            dtype ``int32``.
        seq_lens: Per-sequence token counts ``[batch]``, dtype ``int32``.
        scale: Attention scaling factor (typically ``head_dim ** -0.5``).
        max_num_blocks: Compile-time upper bound on blocks per sequence
            (``page_table.shape[1]``).
        window_size: Sliding window size. ``0`` means full attention.
            When positive, only the last ``window_size`` positions
            contribute to the output.

    Returns:
        Attention output ``[batch, num_q_heads, 1, head_dim]``.
    """
    batch, num_q_heads, _, head_dim = q.shape
    _, num_kv_heads, block_size, _ = k_pool.shape
    gqa_group_size = num_q_heads // num_kv_heads

    out = q.new_empty(batch, num_q_heads, head_dim)

    BLOCK_HD = triton.next_power_of_2(head_dim)
    BLOCK_BS = triton.next_power_of_2(block_size)

    # More warps for wider head dims to keep the SMs busy.
    num_warps = 8 if BLOCK_HD >= 128 else 4

    grid = (batch, num_q_heads)
    _paged_attention_kernel[grid](
        out,
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out.stride(0),
        out.stride(1),
        q.stride(0),
        q.stride(1),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        v_pool.stride(0),
        v_pool.stride(1),
        v_pool.stride(2),
        page_table.stride(0),
        GQA_GROUP_SIZE=gqa_group_size,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        SCALE=scale,
        MAX_NUM_BLOCKS=max_num_blocks,
        BLOCK_HD=BLOCK_HD,
        BLOCK_BS=BLOCK_BS,
        WINDOW_SIZE=window_size,
        num_warps=num_warps,
    )

    return out.unsqueeze(2)
