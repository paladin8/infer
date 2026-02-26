"""Verify both SlottedKVCache and PagedKVCachePool satisfy CachePoolProtocol."""

from __future__ import annotations

import pytest
import torch

from infer.cache.paged import PagedKVCachePool
from infer.cache.slotted import SlottedKVCache

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
DTYPE = torch.float32
DEVICE = "cpu"


@pytest.fixture()
def slotted() -> SlottedKVCache:
    shape = (NUM_LAYERS, 4, NUM_KV_HEADS, 32, HEAD_DIM)
    k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    return SlottedKVCache(k, v, max_batch_size=4)


@pytest.fixture()
def paged() -> PagedKVCachePool:
    shape = (NUM_LAYERS, 16, NUM_KV_HEADS, 8, HEAD_DIM)
    k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    return PagedKVCachePool(k, v, total_blocks=16, block_size=8)


# ---------------------------------------------------------------------------
# CachePoolProtocol method signatures
# ---------------------------------------------------------------------------


class TestSlottedSatisfiesProtocol:
    def test_allocate_slot(self, slotted: SlottedKVCache) -> None:
        slot = slotted.allocate_slot(initial_tokens=100)
        assert isinstance(slot, int)

    def test_free_slot(self, slotted: SlottedKVCache) -> None:
        slot = slotted.allocate_slot()
        slotted.free_slot(slot)
        assert slotted.free_slot_count() == 4

    def test_get_seq_len(self, slotted: SlottedKVCache) -> None:
        slot = slotted.allocate_slot()
        assert slotted.get_seq_len(slot) == 0

    def test_free_slot_count(self, slotted: SlottedKVCache) -> None:
        assert slotted.free_slot_count() == 4

    def test_free_token_capacity(self, slotted: SlottedKVCache) -> None:
        assert slotted.free_token_capacity() is None

    def test_is_paged(self, slotted: SlottedKVCache) -> None:
        assert slotted.is_paged() is False

    def test_prefill_view(self, slotted: SlottedKVCache) -> None:
        slot = slotted.allocate_slot()
        view = slotted.prefill_view(slot)
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is False

    def test_decode_view(self, slotted: SlottedKVCache) -> None:
        view = slotted.decode_view([])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is False

    def test_batched_prefill_view(self, slotted: SlottedKVCache) -> None:
        slot = slotted.allocate_slot()
        view = slotted.batched_prefill_view([slot], [4])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is False

    def test_batched_chunked_prefill_view(self, slotted: SlottedKVCache) -> None:
        slot = slotted.allocate_slot()
        view = slotted.batched_chunked_prefill_view([slot], [0], [4])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is False


class TestPagedSatisfiesProtocol:
    def test_allocate_slot(self, paged: PagedKVCachePool) -> None:
        slot = paged.allocate_slot(initial_tokens=10)
        assert isinstance(slot, int)

    def test_free_slot(self, paged: PagedKVCachePool) -> None:
        slot = paged.allocate_slot(initial_tokens=8)
        paged.free_slot(slot)

    def test_get_seq_len(self, paged: PagedKVCachePool) -> None:
        slot = paged.allocate_slot()
        assert paged.get_seq_len(slot) == 0

    def test_free_slot_count(self, paged: PagedKVCachePool) -> None:
        assert paged.free_slot_count() == 16

    def test_free_token_capacity(self, paged: PagedKVCachePool) -> None:
        cap = paged.free_token_capacity()
        assert isinstance(cap, int)
        assert cap == 16 * 8

    def test_is_paged(self, paged: PagedKVCachePool) -> None:
        assert paged.is_paged() is True

    def test_prefill_view(self, paged: PagedKVCachePool) -> None:
        slot = paged.allocate_slot(initial_tokens=8)
        view = paged.prefill_view(slot)
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is True

    def test_decode_view(self, paged: PagedKVCachePool) -> None:
        view = paged.decode_view([])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is True

    def test_batched_prefill_view(self, paged: PagedKVCachePool) -> None:
        slot = paged.allocate_slot(initial_tokens=8)
        view = paged.batched_prefill_view([slot], [8])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is True

    def test_batched_chunked_prefill_view(self, paged: PagedKVCachePool) -> None:
        slot = paged.allocate_slot(initial_tokens=8)
        view = paged.batched_chunked_prefill_view([slot], [0], [4])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert callable(view.is_paged)
        assert view.is_paged() is True


# ---------------------------------------------------------------------------
# Cross-backend parity: same operations produce same seq_len tracking
# ---------------------------------------------------------------------------


class TestCrossBackendParity:
    def test_allocate_free_cycle(self, slotted: SlottedKVCache, paged: PagedKVCachePool) -> None:
        """Both backends support the same allocate → get_seq_len → free cycle."""
        for pool in (slotted, paged):
            slot = pool.allocate_slot(initial_tokens=10)
            assert pool.get_seq_len(slot) == 0
            pool.free_slot(slot)

    def test_prefill_view_seq_len_tracking(
        self, slotted: SlottedKVCache, paged: PagedKVCachePool
    ) -> None:
        """Both backends' prefill views track seq_len the same way."""
        for pool in (slotted, paged):
            slot = pool.allocate_slot(initial_tokens=8)
            view = pool.prefill_view(slot)
            assert view.seq_len == 0
            k = torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM)
            v = torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM)
            view.update(layer_idx=0, k=k, v=v)
            view.advance(8)
            assert view.seq_len == 8
            assert pool.get_seq_len(slot) == 8
