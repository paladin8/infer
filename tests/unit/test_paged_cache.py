"""Unit tests for PagedKVCachePool and paged cache views."""

from __future__ import annotations

import pytest
import torch

from infer.cache.paged import (
    PagedBatchedPrefillCacheView,
    PagedKVCachePool,
    PagedPrefillCacheView,
)
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
BLOCK_SIZE = 16
TOTAL_BLOCKS = 32
DTYPE = torch.float32
DEVICE = "cpu"


@pytest.fixture()
def pool() -> PagedKVCachePool:
    """A small PagedKVCachePool for testing."""
    shape = (NUM_LAYERS, TOTAL_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    return PagedKVCachePool(k, v, TOTAL_BLOCKS, BLOCK_SIZE)


@pytest.fixture()
def config() -> ModelConfig:
    """A minimal ModelConfig for testing from_model_config."""
    return ModelConfig(
        model_type="llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=NUM_KV_HEADS,
        vocab_size=100,
        max_position_embeddings=128,
        head_dim=HEAD_DIM,
    )


# ---------------------------------------------------------------------------
# PagedKVCachePool — allocation and management
# ---------------------------------------------------------------------------


class TestPagedPoolAllocation:
    def test_allocate_with_initial_tokens(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=50)
        # ceil(50 / 16) = 4 blocks
        assert len(pool.page_tables[seq_id]) == 4
        assert pool.seq_lens[seq_id] == 0
        assert pool.allocator.num_allocated() == 4

    def test_allocate_zero_initial_tokens(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=0)
        assert len(pool.page_tables[seq_id]) == 0
        assert pool.seq_lens[seq_id] == 0

    def test_allocate_default(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot()
        assert len(pool.page_tables[seq_id]) == 0
        assert pool.seq_lens[seq_id] == 0

    def test_monotonic_seq_ids(self, pool: PagedKVCachePool) -> None:
        ids = [pool.allocate_slot() for _ in range(5)]
        assert ids == [0, 1, 2, 3, 4]

    def test_allocate_exact_block_boundary(self, pool: PagedKVCachePool) -> None:
        """initial_tokens exactly divisible by block_size."""
        seq_id = pool.allocate_slot(initial_tokens=32)
        assert len(pool.page_tables[seq_id]) == 2  # 32 / 16 = 2

    def test_block_exhaustion_on_allocate(self, pool: PagedKVCachePool) -> None:
        """Exhaust all blocks, then try to allocate more."""
        pool.allocate_slot(initial_tokens=TOTAL_BLOCKS * BLOCK_SIZE)
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            pool.allocate_slot(initial_tokens=BLOCK_SIZE)

    def test_exhaustion_cleans_up_partial_state(self, pool: PagedKVCachePool) -> None:
        """Failed allocation does not leave orphaned state."""
        pool.allocate_slot(initial_tokens=(TOTAL_BLOCKS - 1) * BLOCK_SIZE)
        initial_tables = len(pool.page_tables)
        with pytest.raises(RuntimeError):
            pool.allocate_slot(initial_tokens=2 * BLOCK_SIZE)
        # No new page table or seq_lens entry was left behind.
        assert len(pool.page_tables) == initial_tables

    def test_seq_id_gap_after_failed_allocation(self, pool: PagedKVCachePool) -> None:
        """_next_seq_id increments even on failure, next success gets a new ID."""
        pool.allocate_slot(initial_tokens=(TOTAL_BLOCKS - 1) * BLOCK_SIZE)  # seq 0
        with pytest.raises(RuntimeError):
            pool.allocate_slot(initial_tokens=2 * BLOCK_SIZE)  # seq 1 fails
        # Free seq 0 to make room.
        pool.free_slot(0)
        seq_id = pool.allocate_slot(initial_tokens=BLOCK_SIZE)
        assert seq_id == 2  # skipped 1


class TestPagedPoolDeallocation:
    def test_free_returns_blocks(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=50)
        num_blocks = len(pool.page_tables[seq_id])
        free_before = pool.allocator.num_free()
        pool.free_slot(seq_id)
        assert pool.allocator.num_free() == free_before + num_blocks
        assert seq_id not in pool.page_tables
        assert seq_id not in pool.seq_lens

    def test_free_nonexistent_raises(self, pool: PagedKVCachePool) -> None:
        with pytest.raises(KeyError):
            pool.free_slot(999)

    def test_free_multiple_sequences(self, pool: PagedKVCachePool) -> None:
        s0 = pool.allocate_slot(initial_tokens=16)
        s1 = pool.allocate_slot(initial_tokens=32)
        s2 = pool.allocate_slot(initial_tokens=48)
        pool.free_slot(s1)
        assert s0 in pool.page_tables
        assert s1 not in pool.page_tables
        assert s2 in pool.page_tables
        # Only s1's blocks were freed (2 blocks).
        assert pool.allocator.num_allocated() == 1 + 3  # s0 + s2

    def test_free_slot_with_zero_blocks(self, pool: PagedKVCachePool) -> None:
        """Freeing a slot with no blocks (initial_tokens=0) works."""
        seq_id = pool.allocate_slot(initial_tokens=0)
        pool.free_slot(seq_id)
        assert seq_id not in pool.page_tables


class TestPagedPoolQueries:
    def test_get_seq_len(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot()
        assert pool.get_seq_len(seq_id) == 0
        pool.seq_lens[seq_id] = 42
        assert pool.get_seq_len(seq_id) == 42

    def test_free_token_capacity(self, pool: PagedKVCachePool) -> None:
        assert pool.free_token_capacity() == TOTAL_BLOCKS * BLOCK_SIZE
        pool.allocate_slot(initial_tokens=50)  # 4 blocks
        assert pool.free_token_capacity() == (TOTAL_BLOCKS - 4) * BLOCK_SIZE

    def test_free_slot_count(self, pool: PagedKVCachePool) -> None:
        assert pool.free_slot_count() == TOTAL_BLOCKS
        pool.allocate_slot(initial_tokens=50)
        assert pool.free_slot_count() == TOTAL_BLOCKS - 4

    def test_active_seq_count(self, pool: PagedKVCachePool) -> None:
        assert pool.active_seq_count() == 0
        pool.allocate_slot()
        pool.allocate_slot()
        assert pool.active_seq_count() == 2

    def test_is_paged(self, pool: PagedKVCachePool) -> None:
        assert pool.is_paged() is True


class TestFromModelConfig:
    def test_shape_and_capacity(self, config: ModelConfig) -> None:
        pool = PagedKVCachePool.from_model_config(
            config, total_blocks=20, block_size=8, dtype=DTYPE, device=DEVICE
        )
        expected = (NUM_LAYERS, 20, NUM_KV_HEADS, 8, HEAD_DIM)
        assert pool.k.shape == expected
        assert pool.v.shape == expected
        assert pool.allocator.num_free() == 20
        assert pool.block_size == 8


class TestAuditAndReclaim:
    def test_no_leaks(self, pool: PagedKVCachePool) -> None:
        pool.allocate_slot(initial_tokens=32)
        pool.allocate_slot(initial_tokens=48)
        assert pool.audit_blocks() == []

    def test_leaked_blocks_detected(self, pool: PagedKVCachePool) -> None:
        """Blocks allocated directly via allocator (not in page tables) are leaked."""
        pool.allocate_slot(initial_tokens=16)
        leaked = pool.allocator.allocate(5, owner=999)
        audit = pool.audit_blocks()
        assert set(audit) == set(leaked)

    def test_reclaim_leaked_blocks(self, pool: PagedKVCachePool) -> None:
        pool.allocator.allocate(5, owner=999)
        free_before = pool.allocator.num_free()
        reclaimed = pool.reclaim_leaked_blocks()
        assert reclaimed == 5
        assert pool.allocator.num_free() == free_before + 5
        assert pool.audit_blocks() == []

    def test_reclaim_after_lost_tracking(self, pool: PagedKVCachePool) -> None:
        """Simulate a tracking loss: allocate, then delete tracking state."""
        seq_id = pool.allocate_slot(initial_tokens=32)
        num_blocks = len(pool.page_tables[seq_id])
        # Simulate tracking loss.
        del pool.page_tables[seq_id]
        del pool.seq_lens[seq_id]
        # Those blocks are now leaked.
        assert len(pool.audit_blocks()) == num_blocks
        reclaimed = pool.reclaim_leaked_blocks()
        assert reclaimed == num_blocks

    def test_reclaim_when_no_leaks(self, pool: PagedKVCachePool) -> None:
        pool.allocate_slot(initial_tokens=32)
        assert pool.reclaim_leaked_blocks() == 0


# ---------------------------------------------------------------------------
# PagedPrefillCacheView
# ---------------------------------------------------------------------------


class TestPagedPrefillCacheView:
    def test_scatter_write_single_block(self, pool: PagedKVCachePool) -> None:
        """Prefill within a single block."""
        seq_id = pool.allocate_slot(initial_tokens=10)
        view = pool.prefill_view(seq_id)

        k = torch.randn(1, NUM_KV_HEADS, 10, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 10, HEAD_DIM)
        ret_k, ret_v = view.update(layer_idx=0, k=k, v=v)

        # Returns input directly.
        assert ret_k is k
        assert ret_v is v

        # Verify data in pool: all 10 tokens in block 0, positions 0-9.
        block_id = pool.page_tables[seq_id][0]
        for pos in range(10):
            torch.testing.assert_close(
                pool.k[0, block_id, :, pos, :],
                k[0, :, pos, :],
            )

    def test_scatter_write_cross_block(self, pool: PagedKVCachePool) -> None:
        """Prefill spanning multiple blocks."""
        seq_id = pool.allocate_slot(initial_tokens=40)
        view = pool.prefill_view(seq_id)
        blocks = pool.page_tables[seq_id]
        assert len(blocks) == 3  # ceil(40/16)

        k = torch.randn(1, NUM_KV_HEADS, 40, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 40, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)

        # Block 0: positions 0-15
        for pos in range(16):
            torch.testing.assert_close(
                pool.k[0, blocks[0], :, pos, :],
                k[0, :, pos, :],
            )
        # Block 1: positions 0-15 (logical 16-31)
        for pos in range(16):
            torch.testing.assert_close(
                pool.k[0, blocks[1], :, pos, :],
                k[0, :, 16 + pos, :],
            )
        # Block 2: positions 0-7 (logical 32-39), 8-15 untouched
        for pos in range(8):
            torch.testing.assert_close(
                pool.k[0, blocks[2], :, pos, :],
                k[0, :, 32 + pos, :],
            )
        # Remaining positions in block 2 should be zeros.
        torch.testing.assert_close(
            pool.k[0, blocks[2], :, 8:, :],
            torch.zeros(NUM_KV_HEADS, BLOCK_SIZE - 8, HEAD_DIM),
        )

    def test_advance_updates_pool(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=32)
        view = pool.prefill_view(seq_id)

        k = torch.randn(1, NUM_KV_HEADS, 32, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 32, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)
        view.advance(32)

        assert view.seq_len == 32
        assert pool.seq_lens[seq_id] == 32

    def test_multiple_layers(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=8)
        view = pool.prefill_view(seq_id)

        k0 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM)
        v0 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM)
        k1 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM) * 2
        v1 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM) * 2

        view.update(layer_idx=0, k=k0, v=v0)
        view.update(layer_idx=1, k=k1, v=v1)

        block_id = pool.page_tables[seq_id][0]
        # pool.k[layer, block] is [kv_heads, block_size, head_dim]
        # k0[0] is [kv_heads, 8, head_dim] — compare directly
        torch.testing.assert_close(pool.k[0, block_id, :, :8, :], k0[0])
        torch.testing.assert_close(pool.k[1, block_id, :, :8, :], k1[0])

    def test_is_paged(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=16)
        view = pool.prefill_view(seq_id)
        assert view.is_paged() is True

    def test_satisfies_protocol(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=16)
        view: PagedPrefillCacheView = pool.prefill_view(seq_id)
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.seq_len == 0


# ---------------------------------------------------------------------------
# PagedDecodeCacheView
# ---------------------------------------------------------------------------


def _prefill_sequence(pool: PagedKVCachePool, prompt_len: int) -> int:
    """Helper: allocate and prefill a sequence, return seq_id."""
    seq_id = pool.allocate_slot(initial_tokens=prompt_len)
    view = pool.prefill_view(seq_id)
    k = torch.randn(1, NUM_KV_HEADS, prompt_len, HEAD_DIM)
    v = torch.randn(1, NUM_KV_HEADS, prompt_len, HEAD_DIM)
    for layer in range(NUM_LAYERS):
        view.update(layer_idx=layer, k=k, v=v)
    view.advance(prompt_len)
    return seq_id


class TestPagedDecodeCacheView:
    def test_single_decode_step(self, pool: PagedKVCachePool) -> None:
        """Prefill, then one decode step."""
        seq_id = _prefill_sequence(pool, prompt_len=20)
        decode = pool.decode_view([seq_id])

        assert decode.seq_len == 20

        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, cached_v = decode.update(layer_idx=0, k=k, v=v)

        # Gathered: [1, kv_heads, 21, head_dim]
        assert cached_k.shape == (1, NUM_KV_HEADS, 21, HEAD_DIM)
        assert cached_v.shape == (1, NUM_KV_HEADS, 21, HEAD_DIM)

        # Verify the new token was written at position 20.
        block_idx = 20 // BLOCK_SIZE  # 1
        offset = 20 % BLOCK_SIZE  # 4
        block_id = pool.page_tables[seq_id][block_idx]
        torch.testing.assert_close(
            pool.k[0, block_id, :, offset, :],
            k[0, :, 0, :],
        )

    def test_multi_sequence_batched_decode(self, pool: PagedKVCachePool) -> None:
        s0 = _prefill_sequence(pool, prompt_len=10)
        s1 = _prefill_sequence(pool, prompt_len=30)
        decode = pool.decode_view([s0, s1])

        assert decode.seq_len == 30

        k = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, _cached_v = decode.update(layer_idx=0, k=k, v=v)

        # max_len = 30 + 1 = 31
        assert cached_k.shape == (2, NUM_KV_HEADS, 31, HEAD_DIM)

    def test_advance_updates_all(self, pool: PagedKVCachePool) -> None:
        s0 = _prefill_sequence(pool, prompt_len=10)
        s1 = _prefill_sequence(pool, prompt_len=20)
        decode = pool.decode_view([s0, s1])

        k = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        decode.update(layer_idx=0, k=k, v=v)
        decode.advance(1)

        assert pool.seq_lens[s0] == 11
        assert pool.seq_lens[s1] == 21
        assert decode.seq_len == 21

    def test_gather_index_caching(self, pool: PagedKVCachePool) -> None:
        """Gather indices computed on layer 0 are reused on layer 1."""
        s0 = _prefill_sequence(pool, prompt_len=10)
        decode = pool.decode_view([s0])

        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)

        decode.update(layer_idx=0, k=k, v=v)
        block_ids_after_0 = decode._gather_block_ids

        decode.update(layer_idx=1, k=k, v=v)
        block_ids_after_1 = decode._gather_block_ids

        # Same tensor object (not recomputed).
        assert block_ids_after_0 is block_ids_after_1

    def test_lazy_block_allocation(self, pool: PagedKVCachePool) -> None:
        """Decode at exact block boundary triggers lazy allocation."""
        seq_id = _prefill_sequence(pool, prompt_len=16)
        # Exactly 1 block. Decode needs block 1.
        blocks_before = len(pool.page_tables[seq_id])
        assert blocks_before == 1

        decode = pool.decode_view([seq_id])
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        decode.update(layer_idx=0, k=k, v=v)

        assert len(pool.page_tables[seq_id]) == 2

    def test_no_new_block_within_block(self, pool: PagedKVCachePool) -> None:
        """Decode within an existing block does not allocate."""
        seq_id = _prefill_sequence(pool, prompt_len=15)
        # 1 block, position 15 is still within block 0 (offsets 0-15).
        blocks_before = len(pool.page_tables[seq_id])
        assert blocks_before == 1

        decode = pool.decode_view([seq_id])
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        decode.update(layer_idx=0, k=k, v=v)

        assert len(pool.page_tables[seq_id]) == 1

    def test_block_allocation_failure_during_decode(self, pool: PagedKVCachePool) -> None:
        """RuntimeError when pool is exhausted at decode time."""
        # Allocate all but 1 block, then prefill to fill that block exactly.
        pool.allocate_slot(initial_tokens=(TOTAL_BLOCKS - 1) * BLOCK_SIZE)
        seq_id = pool.allocate_slot(initial_tokens=BLOCK_SIZE)
        pool.seq_lens[seq_id] = BLOCK_SIZE  # pretend prefill advanced

        decode = pool.decode_view([seq_id])
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            decode.update(layer_idx=0, k=k, v=v)

    def test_empty_view(self, pool: PagedKVCachePool) -> None:
        decode = pool.decode_view([])
        assert decode.seq_len == 0

    def test_is_paged(self, pool: PagedKVCachePool) -> None:
        decode = pool.decode_view([])
        assert decode.is_paged() is True

    def test_gather_correctness_vs_naive(self, pool: PagedKVCachePool) -> None:
        """Vectorized gather matches a naive per-sequence loop."""
        s0 = _prefill_sequence(pool, prompt_len=20)
        s1 = _prefill_sequence(pool, prompt_len=10)
        decode = pool.decode_view([s0, s1])

        k = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, _cached_v = decode.update(layer_idx=0, k=k, v=v)

        # Naive gather for verification.
        for i, seq_id in enumerate([s0, s1]):
            seq_len = decode.slot_seq_lens[i] + 1
            for pos in range(seq_len):
                blk_idx = pos // BLOCK_SIZE
                off = pos % BLOCK_SIZE
                blk_id = pool.page_tables[seq_id][blk_idx]
                expected_k = pool.k[0, blk_id, :, off, :]
                torch.testing.assert_close(
                    cached_k[i, :, pos, :],
                    expected_k,
                )

    def test_decode_after_batched_prefill(self, pool: PagedKVCachePool) -> None:
        """Decode works correctly after sequences were batch-prefilled."""
        s0 = pool.allocate_slot(initial_tokens=5)
        s1 = pool.allocate_slot(initial_tokens=10)
        prompt_lens = [5, 10]

        bpv = pool.batched_prefill_view([s0, s1], prompt_lens)
        padded_len = 10
        k = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        for layer in range(NUM_LAYERS):
            bpv.update(layer_idx=layer, k=k, v=v)
        bpv.advance(padded_len)

        assert pool.seq_lens[s0] == 5
        assert pool.seq_lens[s1] == 10

        decode = pool.decode_view([s0, s1])
        assert decode.seq_len == 10

        dk = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        dv = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, _ = decode.update(layer_idx=0, k=dk, v=dv)
        assert cached_k.shape == (2, NUM_KV_HEADS, 11, HEAD_DIM)


# ---------------------------------------------------------------------------
# PagedBatchedPrefillCacheView
# ---------------------------------------------------------------------------


class TestPagedBatchedPrefillCacheView:
    def test_padding_aware_writes(self, pool: PagedKVCachePool) -> None:
        """Only actual tokens are written, not padding."""
        s0 = pool.allocate_slot(initial_tokens=10)
        s1 = pool.allocate_slot(initial_tokens=20)
        prompt_lens = [10, 20]

        view = pool.batched_prefill_view([s0, s1], prompt_lens)
        padded_len = 20
        k = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)

        # Seq 0: only 10 tokens written (1 block of 16, positions 0-9 used).
        block_id_s0 = pool.page_tables[s0][0]
        for pos in range(10):
            torch.testing.assert_close(
                pool.k[0, block_id_s0, :, pos, :],
                k[0, :, pos, :],
            )
        # Positions 10-15 in block 0 should still be zeros.
        torch.testing.assert_close(
            pool.k[0, block_id_s0, :, 10:, :],
            torch.zeros(NUM_KV_HEADS, BLOCK_SIZE - 10, HEAD_DIM),
        )

        # Seq 1: all 20 tokens written across 2 blocks.
        blocks_s1 = pool.page_tables[s1]
        for pos in range(16):
            torch.testing.assert_close(
                pool.k[0, blocks_s1[0], :, pos, :],
                k[1, :, pos, :],
            )
        for pos in range(4):
            torch.testing.assert_close(
                pool.k[0, blocks_s1[1], :, pos, :],
                k[1, :, 16 + pos, :],
            )

    def test_returns_input_kv(self, pool: PagedKVCachePool) -> None:
        s0 = pool.allocate_slot(initial_tokens=8)
        view = pool.batched_prefill_view([s0], [8])
        k = torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM)
        ret_k, ret_v = view.update(layer_idx=0, k=k, v=v)
        assert ret_k is k
        assert ret_v is v

    def test_advance_sets_actual_prompt_lens(self, pool: PagedKVCachePool) -> None:
        s0 = pool.allocate_slot(initial_tokens=5)
        s1 = pool.allocate_slot(initial_tokens=10)
        prompt_lens = [5, 10]
        view = pool.batched_prefill_view([s0, s1], prompt_lens)

        padded_len = 10
        k = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)
        view.advance(padded_len)

        assert pool.seq_lens[s0] == 5
        assert pool.seq_lens[s1] == 10
        assert view.seq_len == padded_len

    def test_is_paged(self, pool: PagedKVCachePool) -> None:
        s0 = pool.allocate_slot(initial_tokens=8)
        view = pool.batched_prefill_view([s0], [8])
        assert view.is_paged() is True

    def test_satisfies_protocol(self, pool: PagedKVCachePool) -> None:
        s0 = pool.allocate_slot(initial_tokens=8)
        view: PagedBatchedPrefillCacheView = pool.batched_prefill_view([s0], [8])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)

    def test_multiple_layers(self, pool: PagedKVCachePool) -> None:
        s0 = pool.allocate_slot(initial_tokens=8)
        view = pool.batched_prefill_view([s0], [8])

        k0 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM)
        v0 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM)
        k1 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM) * 2
        v1 = torch.ones(1, NUM_KV_HEADS, 8, HEAD_DIM) * 2

        view.update(layer_idx=0, k=k0, v=v0)
        view.update(layer_idx=1, k=k1, v=v1)

        block_id = pool.page_tables[s0][0]
        # Layer 0: 1s
        for pos in range(8):
            torch.testing.assert_close(
                pool.k[0, block_id, :, pos, :],
                k0[0, :, pos, :],
            )
        # Layer 1: 2s
        for pos in range(8):
            torch.testing.assert_close(
                pool.k[1, block_id, :, pos, :],
                k1[0, :, pos, :],
            )


# ---------------------------------------------------------------------------
# Block size edge cases
# ---------------------------------------------------------------------------


class TestBlockSizeOne:
    """Tests with block_size=1 — one token per block."""

    @pytest.fixture()
    def pool_bs1(self) -> PagedKVCachePool:
        shape = (NUM_LAYERS, 64, NUM_KV_HEADS, 1, HEAD_DIM)
        k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
        v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
        return PagedKVCachePool(k, v, total_blocks=64, block_size=1)

    def test_prefill_allocates_one_block_per_token(self, pool_bs1: PagedKVCachePool) -> None:
        seq_id = pool_bs1.allocate_slot(initial_tokens=5)
        assert len(pool_bs1.page_tables[seq_id]) == 5

    def test_prefill_and_decode(self, pool_bs1: PagedKVCachePool) -> None:
        seq_id = pool_bs1.allocate_slot(initial_tokens=5)
        pv = pool_bs1.prefill_view(seq_id)
        k = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM)
        for layer in range(NUM_LAYERS):
            pv.update(layer_idx=layer, k=k, v=v)
        pv.advance(5)
        assert pool_bs1.seq_lens[seq_id] == 5

        decode = pool_bs1.decode_view([seq_id])
        dk = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        dv = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, _ = decode.update(layer_idx=0, k=dk, v=dv)
        # 5 + 1 = 6 positions gathered
        assert cached_k.shape == (1, NUM_KV_HEADS, 6, HEAD_DIM)
        # New block was allocated for the decode token.
        assert len(pool_bs1.page_tables[seq_id]) == 6


# ---------------------------------------------------------------------------
# Multi-step decode
# ---------------------------------------------------------------------------


class TestMultiStepDecode:
    """Verify correct behavior across multiple decode steps (the production pattern)."""

    def test_two_decode_steps(self, pool: PagedKVCachePool) -> None:
        """prefill → decode step 1 → advance → new decode view → decode step 2."""
        seq_id = _prefill_sequence(pool, prompt_len=10)

        # Step 1.
        decode1 = pool.decode_view([seq_id])
        k1 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v1 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k1, _ = decode1.update(layer_idx=0, k=k1, v=v1)
        assert cached_k1.shape == (1, NUM_KV_HEADS, 11, HEAD_DIM)
        decode1.advance(1)
        assert pool.seq_lens[seq_id] == 11

        # Step 2: new decode view picks up updated seq_len.
        decode2 = pool.decode_view([seq_id])
        assert decode2.seq_len == 11
        k2 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v2 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k2, _ = decode2.update(layer_idx=0, k=k2, v=v2)
        assert cached_k2.shape == (1, NUM_KV_HEADS, 12, HEAD_DIM)
        decode2.advance(1)
        assert pool.seq_lens[seq_id] == 12

        # Verify step 2 wrote at position 11.
        block_idx = 11 // BLOCK_SIZE
        offset = 11 % BLOCK_SIZE
        block_id = pool.page_tables[seq_id][block_idx]
        torch.testing.assert_close(
            pool.k[0, block_id, :, offset, :],
            k2[0, :, 0, :],
        )

    def test_multi_step_crosses_block_boundary(self, pool: PagedKVCachePool) -> None:
        """Decode across a block boundary over multiple steps."""
        seq_id = _prefill_sequence(pool, prompt_len=15)  # 1 block, 15 of 16 used

        # Step 1: position 15 → still in block 0.
        d1 = pool.decode_view([seq_id])
        k1 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v1 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        d1.update(layer_idx=0, k=k1, v=v1)
        d1.advance(1)
        assert len(pool.page_tables[seq_id]) == 1  # no new block yet

        # Step 2: position 16 → needs block 1.
        d2 = pool.decode_view([seq_id])
        k2 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v2 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        d2.update(layer_idx=0, k=k2, v=v2)
        d2.advance(1)
        assert len(pool.page_tables[seq_id]) == 2  # new block allocated
        assert pool.seq_lens[seq_id] == 17

        # Verify write at position 16 (block 1, offset 0).
        block_id = pool.page_tables[seq_id][1]
        torch.testing.assert_close(
            pool.k[0, block_id, :, 0, :],
            k2[0, :, 0, :],
        )

    def test_decode_at_two_times_block_size(self, pool: PagedKVCachePool) -> None:
        """Decode after prefilling exactly 2 * block_size tokens."""
        seq_id = _prefill_sequence(pool, prompt_len=2 * BLOCK_SIZE)
        assert len(pool.page_tables[seq_id]) == 2

        decode = pool.decode_view([seq_id])
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        decode.update(layer_idx=0, k=k, v=v)

        # Should have allocated block 2.
        assert len(pool.page_tables[seq_id]) == 3
        block_id = pool.page_tables[seq_id][2]
        torch.testing.assert_close(
            pool.k[0, block_id, :, 0, :],
            k[0, :, 0, :],
        )


# ---------------------------------------------------------------------------
# Multi-layer decode data correctness
# ---------------------------------------------------------------------------


class TestMultiLayerDecode:
    """Verify that layers > 0 write correctly and gather uses cached indices."""

    def test_layer1_write_and_gather(self, pool: PagedKVCachePool) -> None:
        """Layer 1 writes different data than layer 0, both gathered correctly."""
        seq_id = _prefill_sequence(pool, prompt_len=10)
        decode = pool.decode_view([seq_id])

        k0 = torch.ones(1, NUM_KV_HEADS, 1, HEAD_DIM) * 3.0
        v0 = torch.ones(1, NUM_KV_HEADS, 1, HEAD_DIM) * 4.0
        k1 = torch.ones(1, NUM_KV_HEADS, 1, HEAD_DIM) * 5.0
        v1 = torch.ones(1, NUM_KV_HEADS, 1, HEAD_DIM) * 6.0

        cached_k0, cached_v0 = decode.update(layer_idx=0, k=k0, v=v0)
        cached_k1, cached_v1 = decode.update(layer_idx=1, k=k1, v=v1)

        # Both should have shape [1, kv_heads, 11, head_dim].
        assert cached_k0.shape == (1, NUM_KV_HEADS, 11, HEAD_DIM)
        assert cached_k1.shape == (1, NUM_KV_HEADS, 11, HEAD_DIM)

        # Verify the new token at position 10.
        # Layer 0: should be 3.0
        torch.testing.assert_close(
            cached_k0[0, :, 10, :],
            torch.full((NUM_KV_HEADS, HEAD_DIM), 3.0),
        )
        torch.testing.assert_close(
            cached_v0[0, :, 10, :],
            torch.full((NUM_KV_HEADS, HEAD_DIM), 4.0),
        )
        # Layer 1: should be 5.0
        torch.testing.assert_close(
            cached_k1[0, :, 10, :],
            torch.full((NUM_KV_HEADS, HEAD_DIM), 5.0),
        )
        torch.testing.assert_close(
            cached_v1[0, :, 10, :],
            torch.full((NUM_KV_HEADS, HEAD_DIM), 6.0),
        )

    def test_multi_layer_multi_sequence(self, pool: PagedKVCachePool) -> None:
        """Multi-layer decode with two sequences at different lengths."""
        s0 = _prefill_sequence(pool, prompt_len=8)
        s1 = _prefill_sequence(pool, prompt_len=20)
        decode = pool.decode_view([s0, s1])

        k0 = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        v0 = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        k1 = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        v1 = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)

        decode.update(layer_idx=0, k=k0, v=v0)
        cached_k1, _cached_v1 = decode.update(layer_idx=1, k=k1, v=v1)

        # Verify layer 1 writes went to the correct pool positions.
        # s0: position 8 → block 0, offset 8
        block_id_s0 = pool.page_tables[s0][8 // BLOCK_SIZE]
        torch.testing.assert_close(
            pool.k[1, block_id_s0, :, 8 % BLOCK_SIZE, :],
            k1[0, :, 0, :],
        )
        torch.testing.assert_close(
            pool.v[1, block_id_s0, :, 8 % BLOCK_SIZE, :],
            v1[0, :, 0, :],
        )
        # s1: position 20 → block 1, offset 4
        block_id_s1 = pool.page_tables[s1][20 // BLOCK_SIZE]
        torch.testing.assert_close(
            pool.k[1, block_id_s1, :, 20 % BLOCK_SIZE, :],
            k1[1, :, 0, :],
        )

        # Gathered cache on layer 1 should have the new tokens.
        torch.testing.assert_close(
            cached_k1[0, :, 8, :],
            k1[0, :, 0, :],
        )
        torch.testing.assert_close(
            cached_k1[1, :, 20, :],
            k1[1, :, 0, :],
        )


# ---------------------------------------------------------------------------
# V (value) tensor verification
# ---------------------------------------------------------------------------


class TestValueTensorCorrectness:
    """Verify that V tensors are written and gathered correctly (not just K)."""

    def test_prefill_writes_v(self, pool: PagedKVCachePool) -> None:
        seq_id = pool.allocate_slot(initial_tokens=10)
        view = pool.prefill_view(seq_id)

        k = torch.randn(1, NUM_KV_HEADS, 10, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 10, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)

        block_id = pool.page_tables[seq_id][0]
        for pos in range(10):
            torch.testing.assert_close(
                pool.v[0, block_id, :, pos, :],
                v[0, :, pos, :],
            )

    def test_decode_gather_v_vs_naive(self, pool: PagedKVCachePool) -> None:
        """Gathered V matches naive per-position lookup."""
        s0 = _prefill_sequence(pool, prompt_len=20)
        decode = pool.decode_view([s0])

        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        _cached_k, cached_v = decode.update(layer_idx=0, k=k, v=v)

        for pos in range(21):
            blk_idx = pos // BLOCK_SIZE
            off = pos % BLOCK_SIZE
            blk_id = pool.page_tables[s0][blk_idx]
            torch.testing.assert_close(
                cached_v[0, :, pos, :],
                pool.v[0, blk_id, :, off, :],
            )

    def test_batched_prefill_writes_v(self, pool: PagedKVCachePool) -> None:
        s0 = pool.allocate_slot(initial_tokens=8)
        s1 = pool.allocate_slot(initial_tokens=12)
        view = pool.batched_prefill_view([s0, s1], [8, 12])

        k = torch.randn(2, NUM_KV_HEADS, 12, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, 12, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)

        # Verify V for s0 (8 tokens in block 0).
        block_id = pool.page_tables[s0][0]
        for pos in range(8):
            torch.testing.assert_close(
                pool.v[0, block_id, :, pos, :],
                v[0, :, pos, :],
            )
