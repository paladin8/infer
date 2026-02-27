"""Unit tests for Triton paged attention sliding window extension.

Tests correctness of the ``window_size`` parameter against a reference
SDPA implementation with an explicit sliding-window mask.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_windowed_attention(
    q: Tensor,
    k_pool: Tensor,
    v_pool: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    scale: float,
    window_size: int,
) -> Tensor:
    """Gather K/V from blocks, apply sliding-window mask, run SDPA in float32."""
    batch, num_q_heads, _, head_dim = q.shape
    _, num_kv_heads, block_size, _ = k_pool.shape
    gqa_group = num_q_heads // num_kv_heads
    max_seq_len = int(seq_lens.max().item())

    gathered_k = q.new_zeros(batch, num_kv_heads, max_seq_len, head_dim)
    gathered_v = q.new_zeros(batch, num_kv_heads, max_seq_len, head_dim)

    for b in range(batch):
        sl = int(seq_lens[b].item())
        positions = torch.arange(sl, device=q.device)
        block_indices = positions // block_size
        offsets = positions % block_size
        block_ids = page_table[b, block_indices].long()
        gathered_k[b, :, :sl, :] = k_pool[block_ids, :, offsets, :].permute(1, 0, 2)
        gathered_v[b, :, :sl, :] = v_pool[block_ids, :, offsets, :].permute(1, 0, 2)

    # GQA expansion.
    if gqa_group > 1:
        gathered_k = gathered_k.repeat_interleave(gqa_group, dim=1)
        gathered_v = gathered_v.repeat_interleave(gqa_group, dim=1)

    # Build mask: padding + sliding window.
    mask = torch.zeros(batch, 1, 1, max_seq_len, device=q.device, dtype=torch.float32)
    for b in range(batch):
        sl = int(seq_lens[b].item())
        # Padding.
        if sl < max_seq_len:
            mask[b, :, :, sl:] = float("-inf")
        # Sliding window.
        if window_size > 0:
            window_start = max(0, sl - window_size)
            if window_start > 0:
                mask[b, :, :, :window_start] = float("-inf")

    out = F.scaled_dot_product_attention(
        q.float(), gathered_k.float(), gathered_v.float(), attn_mask=mask, scale=scale
    )
    return out.to(q.dtype)


def _make_pool_and_page_table(
    *,
    batch: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    seq_lens: list[int],
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create random K/V pool, page table, and seq_lens tensor."""
    device = "cuda"
    max_blocks_per_seq = max(math.ceil(sl / block_size) for sl in seq_lens)
    total_blocks = batch * max_blocks_per_seq + 10

    k_pool = torch.randn(
        total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    v_pool = torch.randn(
        total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )

    page_table = torch.zeros(batch, max_blocks_per_seq, dtype=torch.int32, device=device)
    next_block = 0
    for b in range(batch):
        n_blocks = math.ceil(seq_lens[b] / block_size)
        for j in range(n_blocks):
            page_table[b, j] = next_block
            next_block += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return k_pool, v_pool, page_table, seq_lens_t


def _run_windowed_test(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    seq_lens: list[int],
    window_size: int,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Run Triton kernel with window_size and compare against reference."""
    from infer.kernels.paged_attention import triton_paged_attention

    k_pool, v_pool, page_table, seq_lens_t = _make_pool_and_page_table(
        batch=batch,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        seq_lens=seq_lens,
        dtype=dtype,
    )
    q = torch.randn(batch, num_q_heads, 1, head_dim, dtype=dtype, device="cuda")
    scale = head_dim**-0.5
    max_num_blocks = page_table.shape[1]

    result = triton_paged_attention(
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens_t,
        scale=scale,
        max_num_blocks=max_num_blocks,
        window_size=window_size,
    )
    reference = _reference_windowed_attention(
        q, k_pool, v_pool, page_table, seq_lens_t, scale, window_size
    )

    assert result.shape == reference.shape == (batch, num_q_heads, 1, head_dim)
    torch.testing.assert_close(result.float(), reference.float(), atol=atol, rtol=rtol)


class TestPagedAttentionWindow:
    """Tests for sliding window extension of triton_paged_attention."""

    def test_full_attention_window_zero(self) -> None:
        """window_size=0 gives identical results to full attention."""
        _run_windowed_test(
            batch=2,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            seq_lens=[48, 32],
            window_size=0,
        )

    def test_sliding_window_32(self) -> None:
        """Sliding window of 32 on a 100-token sequence."""
        _run_windowed_test(
            batch=1,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            seq_lens=[100],
            window_size=32,
        )

    def test_window_covers_entire_sequence(self) -> None:
        """window_size >= seq_len: identical to full attention."""
        _run_windowed_test(
            batch=2,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            seq_lens=[20, 30],
            window_size=100,
        )

    def test_window_size_1(self) -> None:
        """window_size=1: attend only to current (last) position."""
        _run_windowed_test(
            batch=2,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            seq_lens=[50, 30],
            window_size=1,
        )

    def test_window_boundary_at_block_start(self) -> None:
        """Window boundary falls exactly at a block boundary."""
        # block_size=16, seq_len=64, window=32 → window starts at 32 = 2*16.
        _run_windowed_test(
            batch=1,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            seq_lens=[64],
            window_size=32,
        )

    def test_window_boundary_mid_block(self) -> None:
        """Window boundary falls in the middle of a block."""
        # block_size=16, seq_len=60, window=25 → window starts at 35.
        _run_windowed_test(
            batch=1,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            seq_lens=[60],
            window_size=25,
        )

    def test_variable_seq_lens_with_window(self) -> None:
        """Batch with different seq_lens, all using the same window."""
        _run_windowed_test(
            batch=4,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            block_size=16,
            seq_lens=[10, 80, 50, 120],
            window_size=32,
        )

    def test_gqa_with_window(self) -> None:
        """GQA combined with sliding window."""
        _run_windowed_test(
            batch=2,
            num_q_heads=16,
            num_kv_heads=2,
            head_dim=128,
            block_size=16,
            seq_lens=[64, 48],
            window_size=16,
        )

    def test_head_dim_256_with_window(self) -> None:
        """Gemma-style head_dim=256 with sliding window."""
        _run_windowed_test(
            batch=2,
            num_q_heads=4,
            num_kv_heads=1,
            head_dim=256,
            block_size=16,
            seq_lens=[50, 30],
            window_size=20,
        )

    def test_small_block_size_with_window(self) -> None:
        """Block size=8 with sliding window."""
        _run_windowed_test(
            batch=2,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=8,
            seq_lens=[40, 60],
            window_size=16,
        )
