"""Unit tests for the SlottedKVCache, PrefillCacheView, and DecodeCacheView."""

from __future__ import annotations

import pytest
import torch

from infer.cache.simple import KVCache
from infer.cache.slotted import (
    BatchedPrefillCacheView,
    DecodeCacheView,
    PrefillCacheView,
    SlottedKVCache,
)
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
MAX_SEQ_LEN = 32
MAX_BATCH_SIZE = 4
DTYPE = torch.float32
DEVICE = "cpu"


@pytest.fixture()
def pool() -> SlottedKVCache:
    """A small SlottedKVCache pool for testing."""
    shape = (NUM_LAYERS, MAX_BATCH_SIZE, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
    k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    return SlottedKVCache(k, v, MAX_BATCH_SIZE)


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
# SlottedKVCache — allocation and slot management
# ---------------------------------------------------------------------------


class TestSlotAllocation:
    def test_initial_free_count(self, pool: SlottedKVCache) -> None:
        assert pool.free_slot_count() == MAX_BATCH_SIZE

    def test_allocate_all_slots(self, pool: SlottedKVCache) -> None:
        slots = [pool.allocate_slot() for _ in range(MAX_BATCH_SIZE)]
        assert pool.free_slot_count() == 0
        assert len(set(slots)) == MAX_BATCH_SIZE

    def test_allocate_when_full_raises(self, pool: SlottedKVCache) -> None:
        for _ in range(MAX_BATCH_SIZE):
            pool.allocate_slot()
        with pytest.raises(RuntimeError, match="No free cache slots"):
            pool.allocate_slot()

    def test_free_then_reallocate(self, pool: SlottedKVCache) -> None:
        slots = [pool.allocate_slot() for _ in range(MAX_BATCH_SIZE)]
        assert pool.free_slot_count() == 0
        pool.free_slot(slots[0])
        assert pool.free_slot_count() == 1
        new_slot = pool.allocate_slot()
        assert new_slot == slots[0]
        assert pool.free_slot_count() == 0

    def test_free_resets_seq_len(self, pool: SlottedKVCache) -> None:
        slot = pool.allocate_slot()
        pool.seq_lens[slot] = 42
        pool.free_slot(slot)
        assert pool.seq_lens[slot] == 0

    def test_initial_seq_lens(self, pool: SlottedKVCache) -> None:
        assert pool.seq_lens == [0] * MAX_BATCH_SIZE


class TestFromModelConfig:
    def test_shape_and_free_count(self, config: ModelConfig) -> None:
        pool = SlottedKVCache.from_model_config(
            config,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            dtype=DTYPE,
            device=DEVICE,
        )
        expected = (NUM_LAYERS, MAX_BATCH_SIZE, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        assert pool.k.shape == expected
        assert pool.v.shape == expected
        assert pool.free_slot_count() == MAX_BATCH_SIZE

    def test_uses_computed_head_dim(self) -> None:
        config = ModelConfig(
            model_type="llama",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=100,
            max_position_embeddings=128,
            head_dim=None,
        )
        pool = SlottedKVCache.from_model_config(
            config, max_seq_len=16, max_batch_size=2, dtype=DTYPE, device=DEVICE
        )
        # head_dim = 64 // 8 = 8
        assert pool.k.shape == (2, 2, 4, 16, 8)


# ---------------------------------------------------------------------------
# PrefillCacheView
# ---------------------------------------------------------------------------


class TestPrefillCacheView:
    def test_update_writes_to_pool(self, pool: SlottedKVCache) -> None:
        slot = pool.allocate_slot()
        view = pool.prefill_view(slot)
        prompt_len = 5
        k_new = torch.randn(1, NUM_KV_HEADS, prompt_len, HEAD_DIM)
        v_new = torch.randn(1, NUM_KV_HEADS, prompt_len, HEAD_DIM)

        cached_k, cached_v = view.update(layer_idx=0, k=k_new, v=v_new)

        # Returned tensors cover [0 : prompt_len]
        assert cached_k.shape == (1, NUM_KV_HEADS, prompt_len, HEAD_DIM)
        assert cached_v.shape == (1, NUM_KV_HEADS, prompt_len, HEAD_DIM)

        # Content matches what we wrote
        torch.testing.assert_close(cached_k, k_new)
        torch.testing.assert_close(cached_v, v_new)

        # Data is in the pool at the correct slot
        torch.testing.assert_close(pool.k[0, slot : slot + 1, :, :prompt_len, :], k_new)

    def test_advance_updates_view_and_pool(self, pool: SlottedKVCache) -> None:
        slot = pool.allocate_slot()
        view = pool.prefill_view(slot)
        prompt_len = 7

        k_new = torch.randn(1, NUM_KV_HEADS, prompt_len, HEAD_DIM)
        v_new = torch.randn(1, NUM_KV_HEADS, prompt_len, HEAD_DIM)
        view.update(layer_idx=0, k=k_new, v=v_new)
        view.advance(prompt_len)

        assert view.seq_len == prompt_len
        assert pool.seq_lens[slot] == prompt_len

    def test_decode_after_prefill(self, pool: SlottedKVCache) -> None:
        """Prefill then one decode step via PrefillCacheView (single-request path)."""
        slot = pool.allocate_slot()
        view = pool.prefill_view(slot)

        # Prefill 3 tokens
        k_prefill = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM)
        v_prefill = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM)
        view.update(layer_idx=0, k=k_prefill, v=v_prefill)
        view.advance(3)

        # Decode 1 token
        k_decode = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v_decode = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, _cached_v = view.update(layer_idx=0, k=k_decode, v=v_decode)

        assert cached_k.shape == (1, NUM_KV_HEADS, 4, HEAD_DIM)
        torch.testing.assert_close(cached_k[:, :, :3, :], k_prefill)
        torch.testing.assert_close(cached_k[:, :, 3:4, :], k_decode)

    def test_multiple_layers(self, pool: SlottedKVCache) -> None:
        slot = pool.allocate_slot()
        view = pool.prefill_view(slot)

        k0 = torch.ones(1, NUM_KV_HEADS, 2, HEAD_DIM)
        v0 = torch.ones(1, NUM_KV_HEADS, 2, HEAD_DIM)
        k1 = torch.ones(1, NUM_KV_HEADS, 2, HEAD_DIM) * 2
        v1 = torch.ones(1, NUM_KV_HEADS, 2, HEAD_DIM) * 2

        view.update(layer_idx=0, k=k0, v=v0)
        view.update(layer_idx=1, k=k1, v=v1)

        torch.testing.assert_close(pool.k[0, slot : slot + 1, :, :2, :], k0)
        torch.testing.assert_close(pool.k[1, slot : slot + 1, :, :2, :], k1)

    def test_returns_views_not_copies(self, pool: SlottedKVCache) -> None:
        slot = pool.allocate_slot()
        view = pool.prefill_view(slot)
        k_new = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM)
        v_new = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM)
        cached_k, _ = view.update(layer_idx=0, k=k_new, v=v_new)
        # Should be a view into the pool
        assert cached_k.data_ptr() == pool.k[0, slot : slot + 1, :, :3, :].data_ptr()


# ---------------------------------------------------------------------------
# DecodeCacheView
# ---------------------------------------------------------------------------


class TestDecodeCacheView:
    def _setup_slots(self, pool: SlottedKVCache, seq_lens: list[int]) -> list[int]:
        """Allocate slots and prefill them to given seq_lens."""
        slots = []
        for slen in seq_lens:
            slot = pool.allocate_slot()
            slots.append(slot)
            # Write random data for each layer at positions [0:slen]
            for layer in range(NUM_LAYERS):
                k = torch.randn(1, NUM_KV_HEADS, slen, HEAD_DIM)
                v = torch.randn(1, NUM_KV_HEADS, slen, HEAD_DIM)
                pool.k[layer, slot : slot + 1, :, :slen, :] = k
                pool.v[layer, slot : slot + 1, :, :slen, :] = v
            pool.seq_lens[slot] = slen
        return slots

    def test_seq_len_is_max(self, pool: SlottedKVCache) -> None:
        slots = self._setup_slots(pool, [10, 5, 20])
        view = pool.decode_view(slots)
        assert view.seq_len == 20

    def test_empty_slots(self, pool: SlottedKVCache) -> None:
        view = pool.decode_view([])
        assert view.seq_len == 0

    def test_update_writes_per_slot_positions(self, pool: SlottedKVCache) -> None:
        slots = self._setup_slots(pool, [10, 5])
        view = pool.decode_view(slots)

        batch = len(slots)
        k_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)
        v_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)

        view.update(layer_idx=0, k=k_new, v=v_new)

        # Verify per-slot writes: slot 0 at pos 10, slot 1 at pos 5
        torch.testing.assert_close(pool.k[0, slots[0], :, 10, :], k_new[0, :, 0, :])
        torch.testing.assert_close(pool.k[0, slots[1], :, 5, :], k_new[1, :, 0, :])

    def test_update_returns_gathered_cache(self, pool: SlottedKVCache) -> None:
        slots = self._setup_slots(pool, [10, 5])
        view = pool.decode_view(slots)

        batch = len(slots)
        k_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)
        v_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)

        cached_k, cached_v = view.update(layer_idx=0, k=k_new, v=v_new)

        # max_len = 10 + 1 = 11
        assert cached_k.shape == (batch, NUM_KV_HEADS, 11, HEAD_DIM)
        assert cached_v.shape == (batch, NUM_KV_HEADS, 11, HEAD_DIM)

    def test_advance_updates_all_slots(self, pool: SlottedKVCache) -> None:
        slots = self._setup_slots(pool, [10, 5, 20])
        view = pool.decode_view(slots)

        view.advance(1)

        assert pool.seq_lens[slots[0]] == 11
        assert pool.seq_lens[slots[1]] == 6
        assert pool.seq_lens[slots[2]] == 21
        assert view.seq_len == 21

    def test_gathered_cache_matches_pool(self, pool: SlottedKVCache) -> None:
        """Gathered K/V for each batch element matches the pool's slot data."""
        slots = self._setup_slots(pool, [8, 3])
        view = pool.decode_view(slots)

        batch = len(slots)
        k_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)
        v_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)

        cached_k, _ = view.update(layer_idx=0, k=k_new, v=v_new)

        # Batch element 0 should match pool slot data for positions [0:9]
        max_len = 8 + 1  # max seq_len + 1
        torch.testing.assert_close(
            cached_k[0, :, :max_len, :],
            pool.k[0, slots[0], :, :max_len, :],
        )


# ---------------------------------------------------------------------------
# Slot reuse
# ---------------------------------------------------------------------------


class TestSlotReuse:
    def test_reuse_after_free(self, pool: SlottedKVCache) -> None:
        """Freeing and reusing a slot resets position and overwrites old data."""
        # Allocate all slots so the freed one is the only option
        slots = [pool.allocate_slot() for _ in range(MAX_BATCH_SIZE)]
        slot = slots[0]

        # First request: prefill 10 tokens
        view1 = pool.prefill_view(slot)
        k1 = torch.ones(1, NUM_KV_HEADS, 10, HEAD_DIM)
        v1 = torch.ones(1, NUM_KV_HEADS, 10, HEAD_DIM)
        view1.update(layer_idx=0, k=k1, v=v1)
        view1.advance(10)
        assert pool.seq_lens[slot] == 10

        # Free only this slot
        pool.free_slot(slot)
        assert pool.seq_lens[slot] == 0
        assert pool.free_slot_count() == 1

        # Second request: must get the same slot back
        slot2 = pool.allocate_slot()
        assert slot2 == slot

        view2 = pool.prefill_view(slot2)
        k2 = torch.ones(1, NUM_KV_HEADS, 5, HEAD_DIM) * 2
        v2 = torch.ones(1, NUM_KV_HEADS, 5, HEAD_DIM) * 2
        view2.update(layer_idx=0, k=k2, v=v2)
        view2.advance(5)

        assert pool.seq_lens[slot2] == 5
        # Positions 0-4 should have new data (2s, not 1s)
        torch.testing.assert_close(pool.k[0, slot2 : slot2 + 1, :, :5, :], k2)


# ---------------------------------------------------------------------------
# Protocol compatibility
# ---------------------------------------------------------------------------


class TestProtocolCompatibility:
    def test_prefill_view_satisfies_protocol(self, pool: SlottedKVCache) -> None:
        """PrefillCacheView has seq_len, update(), advance() as required by KVCacheProtocol."""
        slot = pool.allocate_slot()
        view: PrefillCacheView = pool.prefill_view(slot)
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert view.seq_len == 0

    def test_decode_view_satisfies_protocol(self, pool: SlottedKVCache) -> None:
        """DecodeCacheView has seq_len, update(), advance() as required by KVCacheProtocol."""
        view: DecodeCacheView = pool.decode_view([])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert view.seq_len == 0

    def test_kv_cache_satisfies_protocol(self) -> None:
        """The original KVCache also satisfies the KVCacheProtocol interface."""
        cache = KVCache.allocate(
            num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=16, dtype=DTYPE, device=DEVICE
        )
        assert hasattr(cache, "seq_len")
        assert callable(cache.update)
        assert callable(cache.advance)
        assert cache.seq_len == 0


# ---------------------------------------------------------------------------
# PrefillCacheView overflow
# ---------------------------------------------------------------------------


class TestPrefillOverflow:
    def test_overflow_raises(self, pool: SlottedKVCache) -> None:
        """Writing past max_seq_len raises AssertionError."""
        slot = pool.allocate_slot()
        view = pool.prefill_view(slot)
        k_too_big = torch.randn(1, NUM_KV_HEADS, MAX_SEQ_LEN + 1, HEAD_DIM)
        v_too_big = torch.randn(1, NUM_KV_HEADS, MAX_SEQ_LEN + 1, HEAD_DIM)
        with pytest.raises(AssertionError, match="KV cache overflow"):
            view.update(layer_idx=0, k=k_too_big, v=v_too_big)

    def test_overflow_after_advance(self, pool: SlottedKVCache) -> None:
        """Overflow detected after advancing past available space."""
        slot = pool.allocate_slot()
        view = pool.prefill_view(slot)
        # Fill to capacity
        k_full = torch.randn(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        v_full = torch.randn(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        view.update(layer_idx=0, k=k_full, v=v_full)
        view.advance(MAX_SEQ_LEN)

        # One more token should overflow
        k_one = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v_one = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        with pytest.raises(AssertionError, match="KV cache overflow"):
            view.update(layer_idx=0, k=k_one, v=v_one)


# ---------------------------------------------------------------------------
# DecodeCacheView — additional coverage
# ---------------------------------------------------------------------------


class TestDecodeCacheViewExtra:
    def _setup_slots(self, pool: SlottedKVCache, seq_lens: list[int]) -> list[int]:
        """Allocate slots and prefill them to given seq_lens."""
        slots = []
        for slen in seq_lens:
            slot = pool.allocate_slot()
            slots.append(slot)
            for layer in range(NUM_LAYERS):
                k = torch.randn(1, NUM_KV_HEADS, slen, HEAD_DIM)
                v = torch.randn(1, NUM_KV_HEADS, slen, HEAD_DIM)
                pool.k[layer, slot : slot + 1, :, :slen, :] = k
                pool.v[layer, slot : slot + 1, :, :slen, :] = v
            pool.seq_lens[slot] = slen
        return slots

    def test_single_slot(self, pool: SlottedKVCache) -> None:
        """DecodeCacheView with a single active slot."""
        slots = self._setup_slots(pool, [15])
        view = pool.decode_view(slots)

        assert view.seq_len == 15

        k_new = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v_new = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, cached_v = view.update(layer_idx=0, k=k_new, v=v_new)

        # max_len = 15 + 1 = 16
        assert cached_k.shape == (1, NUM_KV_HEADS, 16, HEAD_DIM)
        assert cached_v.shape == (1, NUM_KV_HEADS, 16, HEAD_DIM)

        # Verify the write went to position 15
        torch.testing.assert_close(pool.k[0, slots[0], :, 15, :], k_new[0, :, 0, :])

        view.advance(1)
        assert pool.seq_lens[slots[0]] == 16
        assert view.seq_len == 16

    def test_multiple_layers(self, pool: SlottedKVCache) -> None:
        """DecodeCacheView writes correctly across multiple layers."""
        slots = self._setup_slots(pool, [8, 4])
        view = pool.decode_view(slots)

        batch = len(slots)
        k0 = torch.ones(batch, NUM_KV_HEADS, 1, HEAD_DIM)
        v0 = torch.ones(batch, NUM_KV_HEADS, 1, HEAD_DIM)
        k1 = torch.ones(batch, NUM_KV_HEADS, 1, HEAD_DIM) * 2
        v1 = torch.ones(batch, NUM_KV_HEADS, 1, HEAD_DIM) * 2

        view.update(layer_idx=0, k=k0, v=v0)
        view.update(layer_idx=1, k=k1, v=v1)

        # Layer 0: 1s at each slot's position
        torch.testing.assert_close(pool.k[0, slots[0], :, 8, :], k0[0, :, 0, :])
        torch.testing.assert_close(pool.k[0, slots[1], :, 4, :], k0[1, :, 0, :])
        # Layer 1: 2s at each slot's position
        torch.testing.assert_close(pool.k[1, slots[0], :, 8, :], k1[0, :, 0, :])
        torch.testing.assert_close(pool.k[1, slots[1], :, 4, :], k1[1, :, 0, :])

    def test_equal_seq_lens(self, pool: SlottedKVCache) -> None:
        """All slots at the same seq_len — no padding needed."""
        slots = self._setup_slots(pool, [10, 10, 10])
        view = pool.decode_view(slots)

        assert view.seq_len == 10

        batch = len(slots)
        k_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)
        v_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, _ = view.update(layer_idx=0, k=k_new, v=v_new)

        # max_len = 10 + 1 = 11, all batch elements have valid data through pos 10
        assert cached_k.shape == (batch, NUM_KV_HEADS, 11, HEAD_DIM)

        view.advance(1)
        assert all(pool.seq_lens[s] == 11 for s in slots)
        assert view.seq_len == 11


# ---------------------------------------------------------------------------
# BatchedPrefillCacheView
# ---------------------------------------------------------------------------


class TestBatchedPrefillCacheView:
    def test_scatter_write_to_pool(self, pool: SlottedKVCache) -> None:
        """Each batch element's K/V is written to its assigned slot."""
        slots = [pool.allocate_slot() for _ in range(3)]
        prompt_lens = [5, 3, 7]
        view = pool.batched_prefill_view(slots, prompt_lens)

        padded_len = max(prompt_lens)  # 7
        k = torch.randn(3, NUM_KV_HEADS, padded_len, HEAD_DIM)
        v = torch.randn(3, NUM_KV_HEADS, padded_len, HEAD_DIM)

        view.update(layer_idx=0, k=k, v=v)

        # Verify each slot got its batch element's data.
        for i, slot in enumerate(slots):
            torch.testing.assert_close(pool.k[0, slot, :, :padded_len, :], k[i])
            torch.testing.assert_close(pool.v[0, slot, :, :padded_len, :], v[i])

    def test_returns_input_kv(self, pool: SlottedKVCache) -> None:
        """update() returns the input K/V directly (not a pool gather)."""
        slots = [pool.allocate_slot() for _ in range(2)]
        view = pool.batched_prefill_view(slots, [4, 6])

        k = torch.randn(2, NUM_KV_HEADS, 6, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, 6, HEAD_DIM)

        cached_k, cached_v = view.update(layer_idx=0, k=k, v=v)

        # Returns the exact same tensors.
        assert cached_k is k
        assert cached_v is v

    def test_advance_sets_actual_prompt_lens(self, pool: SlottedKVCache) -> None:
        """advance() sets per-slot seq_lens to actual prompt lengths, not padded."""
        slots = [pool.allocate_slot() for _ in range(3)]
        prompt_lens = [5, 3, 7]
        view = pool.batched_prefill_view(slots, prompt_lens)

        padded_len = max(prompt_lens)
        k = torch.randn(3, NUM_KV_HEADS, padded_len, HEAD_DIM)
        v = torch.randn(3, NUM_KV_HEADS, padded_len, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)

        # Model calls advance(padded_len).
        view.advance(padded_len)

        # Pool seq_lens should be actual lengths, not padded.
        assert pool.seq_lens[slots[0]] == 5
        assert pool.seq_lens[slots[1]] == 3
        assert pool.seq_lens[slots[2]] == 7
        # View's internal seq_len tracks padded length (unused after prefill).
        assert view.seq_len == padded_len

    def test_multiple_layers(self, pool: SlottedKVCache) -> None:
        """Scatter-write works correctly across layers."""
        slots = [pool.allocate_slot() for _ in range(2)]
        view = pool.batched_prefill_view(slots, [4, 4])

        k0 = torch.ones(2, NUM_KV_HEADS, 4, HEAD_DIM)
        v0 = torch.ones(2, NUM_KV_HEADS, 4, HEAD_DIM)
        k1 = torch.ones(2, NUM_KV_HEADS, 4, HEAD_DIM) * 2
        v1 = torch.ones(2, NUM_KV_HEADS, 4, HEAD_DIM) * 2

        view.update(layer_idx=0, k=k0, v=v0)
        view.update(layer_idx=1, k=k1, v=v1)

        torch.testing.assert_close(pool.k[0, slots[0], :, :4, :], k0[0])
        torch.testing.assert_close(pool.k[1, slots[0], :, :4, :], k1[0])

    def test_overflow_raises(self, pool: SlottedKVCache) -> None:
        """Writing past max_seq_len raises AssertionError."""
        slots = [pool.allocate_slot()]
        view = pool.batched_prefill_view(slots, [MAX_SEQ_LEN + 1])

        k = torch.randn(1, NUM_KV_HEADS, MAX_SEQ_LEN + 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, MAX_SEQ_LEN + 1, HEAD_DIM)
        with pytest.raises(AssertionError, match="KV cache overflow"):
            view.update(layer_idx=0, k=k, v=v)

    def test_satisfies_protocol(self, pool: SlottedKVCache) -> None:
        """BatchedPrefillCacheView has seq_len, update(), advance()."""
        slots = [pool.allocate_slot()]
        view: BatchedPrefillCacheView = pool.batched_prefill_view(slots, [4])
        assert hasattr(view, "seq_len")
        assert callable(view.update)
        assert callable(view.advance)
        assert view.seq_len == 0

    def test_decode_after_batched_prefill(self, pool: SlottedKVCache) -> None:
        """Slots prefilled via batched view work correctly with DecodeCacheView."""
        slots = [pool.allocate_slot() for _ in range(2)]
        prompt_lens = [5, 3]
        view = pool.batched_prefill_view(slots, prompt_lens)

        padded_len = max(prompt_lens)
        k = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, padded_len, HEAD_DIM)
        view.update(layer_idx=0, k=k, v=v)
        view.advance(padded_len)

        # Now create a decode view over the same slots.
        decode_view = pool.decode_view(slots)

        # seq_len should be max of actual prompt lens (5), not padded (5 here too).
        assert decode_view.seq_len == 5

        # Decode step should work.
        k_dec = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        v_dec = torch.randn(2, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, _ = decode_view.update(layer_idx=0, k=k_dec, v=v_dec)

        # Gathered cache covers [0 : max_seq_len + 1] = [0:6].
        assert cached_k.shape == (2, NUM_KV_HEADS, 6, HEAD_DIM)
