"""Unit tests for the Triton paged attention kernel.

Tests correctness against a reference implementation that gathers K/V from
the block pool and runs standard scaled dot-product attention.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference_paged_attention(
    q: Tensor,
    k_pool: Tensor,
    v_pool: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    scale: float,
) -> Tensor:
    """Gather K/V from blocks, expand GQA heads, run SDPA in float32."""
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
        # k_pool[block_ids, :, offsets, :] → [sl, kv_heads, head_dim]
        gathered_k[b, :, :sl, :] = k_pool[block_ids, :, offsets, :].permute(1, 0, 2)
        gathered_v[b, :, :sl, :] = v_pool[block_ids, :, offsets, :].permute(1, 0, 2)

    # GQA expansion.
    if gqa_group > 1:
        gathered_k = gathered_k.repeat_interleave(gqa_group, dim=1)
        gathered_v = gathered_v.repeat_interleave(gqa_group, dim=1)

    # Mask for variable lengths.
    mask = torch.zeros(batch, 1, 1, max_seq_len, device=q.device, dtype=torch.float32)
    for b in range(batch):
        sl = int(seq_lens[b].item())
        if sl < max_seq_len:
            mask[b, :, :, sl:] = float("-inf")

    out = F.scaled_dot_product_attention(
        q.float(), gathered_k.float(), gathered_v.float(), attn_mask=mask, scale=scale
    )
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_pool_and_page_table(
    *,
    batch: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    seq_lens: list[int],
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create random K/V pool, page table, and seq_lens tensor.

    Returns (k_pool, v_pool, page_table, seq_lens_tensor).
    """
    device = "cuda"
    max_blocks_per_seq = max(math.ceil(sl / block_size) for sl in seq_lens)
    total_blocks = batch * max_blocks_per_seq + 10  # extra headroom

    k_pool = torch.randn(
        total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    v_pool = torch.randn(
        total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )

    # Allocate non-overlapping blocks for each sequence.
    page_table = torch.zeros(batch, max_blocks_per_seq, dtype=torch.int32, device=device)
    next_block = 0
    for b in range(batch):
        n_blocks = math.ceil(seq_lens[b] / block_size)
        for j in range(n_blocks):
            page_table[b, j] = next_block
            next_block += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return k_pool, v_pool, page_table, seq_lens_t


def _run_and_compare(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    seq_lens: list[int],
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Run both Triton and reference, assert close."""
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
        q, k_pool, v_pool, page_table, seq_lens_t, scale=scale, max_num_blocks=max_num_blocks
    )
    reference = _reference_paged_attention(q, k_pool, v_pool, page_table, seq_lens_t, scale)

    assert result.shape == reference.shape == (batch, num_q_heads, 1, head_dim)
    torch.testing.assert_close(result.float(), reference.float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPagedAttentionKernel:
    """Correctness tests for triton_paged_attention."""

    def test_mha_single_batch(self) -> None:
        """Multi-head attention (no GQA), single sequence."""
        _run_and_compare(
            batch=1, num_q_heads=4, num_kv_heads=4, head_dim=64, block_size=16, seq_lens=[48]
        )

    def test_gqa_4_to_1(self) -> None:
        """GQA with 4:1 ratio (4 query heads per KV head)."""
        _run_and_compare(
            batch=2, num_q_heads=8, num_kv_heads=2, head_dim=64, block_size=16, seq_lens=[32, 64]
        )

    def test_gqa_8_to_1(self) -> None:
        """GQA with 8:1 ratio (Qwen3-style)."""
        _run_and_compare(
            batch=2, num_q_heads=16, num_kv_heads=2, head_dim=128, block_size=16, seq_lens=[50, 30]
        )

    def test_mqa(self) -> None:
        """Multi-query attention: 1 KV head."""
        _run_and_compare(
            batch=2, num_q_heads=8, num_kv_heads=1, head_dim=64, block_size=16, seq_lens=[20, 40]
        )

    def test_single_token(self) -> None:
        """seq_len=1: single token in cache, one partial block."""
        _run_and_compare(
            batch=1, num_q_heads=4, num_kv_heads=4, head_dim=64, block_size=16, seq_lens=[1]
        )

    def test_partial_last_block(self) -> None:
        """Last block not full (seq_len not divisible by block_size)."""
        _run_and_compare(
            batch=2, num_q_heads=4, num_kv_heads=2, head_dim=64, block_size=16, seq_lens=[25, 7]
        )

    def test_exact_block_boundary(self) -> None:
        """seq_len exactly divisible by block_size."""
        _run_and_compare(
            batch=1, num_q_heads=4, num_kv_heads=4, head_dim=64, block_size=16, seq_lens=[64]
        )

    def test_variable_seq_lens(self) -> None:
        """Batch with highly variable sequence lengths."""
        _run_and_compare(
            batch=4,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            block_size=16,
            seq_lens=[5, 100, 33, 64],
        )

    def test_head_dim_128(self) -> None:
        """Common head_dim=128 (Llama, Qwen)."""
        _run_and_compare(
            batch=3,
            num_q_heads=24,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            seq_lens=[60, 120, 45],
        )

    def test_head_dim_256(self) -> None:
        """Larger head_dim=256 (Gemma 3)."""
        _run_and_compare(
            batch=2,
            num_q_heads=4,
            num_kv_heads=1,
            head_dim=256,
            block_size=16,
            seq_lens=[30, 50],
        )

    def test_block_size_8(self) -> None:
        """Smaller block size."""
        _run_and_compare(
            batch=2, num_q_heads=4, num_kv_heads=2, head_dim=64, block_size=8, seq_lens=[20, 35]
        )

    def test_block_size_32(self) -> None:
        """Larger block size."""
        _run_and_compare(
            batch=2, num_q_heads=4, num_kv_heads=2, head_dim=64, block_size=32, seq_lens=[50, 100]
        )

    def test_long_sequence(self) -> None:
        """Longer sequence spanning many blocks."""
        _run_and_compare(
            batch=1,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            block_size=16,
            seq_lens=[512],
        )

    def test_float16(self) -> None:
        """float16 dtype."""
        _run_and_compare(
            batch=2,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=16,
            seq_lens=[32, 48],
            dtype=torch.float16,
        )

    def test_output_shape(self) -> None:
        """Verify output shape matches [batch, num_q_heads, 1, head_dim]."""
        from infer.kernels.paged_attention import triton_paged_attention

        k_pool, v_pool, page_table, seq_lens_t = _make_pool_and_page_table(
            batch=3, num_kv_heads=2, block_size=16, head_dim=64, seq_lens=[10, 20, 15]
        )
        q = torch.randn(3, 8, 1, 64, dtype=torch.bfloat16, device="cuda")
        out = triton_paged_attention(
            q, k_pool, v_pool, page_table, seq_lens_t, scale=64.0**-0.5, max_num_blocks=2
        )
        assert out.shape == (3, 8, 1, 64)
        assert out.dtype == torch.bfloat16


class TestPagedDecodeCacheViewWriteOnly:
    """Tests for write_only() and kernel tensor properties on PagedDecodeCacheView."""

    def test_write_only_writes_to_pool(self) -> None:
        """write_only() stores K/V to the correct pool position."""
        from infer.cache.paged import PagedKVCachePool

        pool = PagedKVCachePool(
            k=torch.zeros(2, 10, 1, 4, 8, device="cuda"),
            v=torch.zeros(2, 10, 1, 4, 8, device="cuda"),
            total_blocks=10,
            block_size=4,
        )
        seq_id = pool.allocate_slot(initial_tokens=4)
        pool.seq_lens[seq_id] = 3  # already prefilled 3 tokens
        view = pool.decode_view([seq_id])

        k = torch.ones(1, 1, 1, 8, device="cuda")
        v = torch.ones(1, 1, 1, 8, device="cuda") * 2
        view.write_only(0, k, v)

        # Token at position 3 → block 0, offset 3.
        block_id = pool.page_tables[seq_id][0]
        assert pool.k[0, block_id, 0, 3, 0].item() == 1.0
        assert pool.v[0, block_id, 0, 3, 0].item() == 2.0

    def test_kernel_tensors_shape(self) -> None:
        """page_table_tensor and seq_lens_tensor have correct shapes."""
        from infer.cache.paged import PagedKVCachePool

        pool = PagedKVCachePool(
            k=torch.zeros(1, 20, 1, 4, 8, device="cuda"),
            v=torch.zeros(1, 20, 1, 4, 8, device="cuda"),
            total_blocks=20,
            block_size=4,
        )
        s0 = pool.allocate_slot(initial_tokens=8)
        pool.seq_lens[s0] = 6
        s1 = pool.allocate_slot(initial_tokens=4)
        pool.seq_lens[s1] = 3

        view = pool.decode_view([s0, s1])
        k = torch.randn(2, 1, 1, 8, device="cuda")
        v = torch.randn(2, 1, 1, 8, device="cuda")
        view.write_only(0, k, v)

        pt = view.page_table_tensor
        sl = view.seq_lens_tensor
        assert pt.shape == (2, 2)  # seq 0 has 2 blocks, seq 1 has 1 block → max=2
        assert sl.shape == (2,)
        assert sl[0].item() == 7  # 6 + 1
        assert sl[1].item() == 4  # 3 + 1
        assert pt.dtype == torch.int32
        assert sl.dtype == torch.int32

    def test_mixed_update_and_write_only(self) -> None:
        """update() and write_only() can be called on different layers."""
        from infer.cache.paged import PagedKVCachePool

        pool = PagedKVCachePool(
            k=torch.zeros(3, 10, 1, 4, 8, device="cuda"),
            v=torch.zeros(3, 10, 1, 4, 8, device="cuda"),
            total_blocks=10,
            block_size=4,
        )
        seq_id = pool.allocate_slot(initial_tokens=4)
        pool.seq_lens[seq_id] = 2
        view = pool.decode_view([seq_id])

        k = torch.randn(1, 1, 1, 8, device="cuda")
        v = torch.randn(1, 1, 1, 8, device="cuda")

        # Layer 0: Triton path (write_only).
        view.write_only(0, k, v)
        assert view.page_table_tensor is not None

        # Layer 1: SDPA path (update).
        cached_k, _cached_v = view.update(1, k, v)
        assert cached_k.shape == (1, 1, 3, 8)  # seq_len=2 + 1 new token

        # Layer 2: Triton path again.
        view.write_only(2, k, v)
