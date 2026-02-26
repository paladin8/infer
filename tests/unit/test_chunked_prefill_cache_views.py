"""Tests for batched chunked prefill cache views (Phase 7, Deliverable 5).

Verifies that both the slotted and paged backends correctly:
- Write chunk KV at the right positions and gather full KV.
- Handle multi-chunk prefill (chunk 1, then chunk 2 with different start positions).
- Batch sequences at different prefill progress levels.
- Track seq_lens correctly after advance().
- Produce correct cache state for subsequent decode.
- Report correct seq_len property for the model's kv_len computation.
- Reuse gather indices across layers.
"""

from __future__ import annotations

import torch
from torch import Tensor

from infer.cache.paged import PagedKVCachePool
from infer.cache.slotted import SlottedKVCache

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
DTYPE = torch.float32
DEVICE = "cpu"


def _make_slotted() -> SlottedKVCache:
    shape = (NUM_LAYERS, 4, NUM_KV_HEADS, 64, HEAD_DIM)
    k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    return SlottedKVCache(k, v, max_batch_size=4)


def _make_paged(total_blocks: int = 32, block_size: int = 4) -> PagedKVCachePool:
    shape = (NUM_LAYERS, total_blocks, NUM_KV_HEADS, block_size, HEAD_DIM)
    k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    return PagedKVCachePool(k, v, total_blocks=total_blocks, block_size=block_size)


def _rand_kv(batch: int, seq_len: int) -> tuple[Tensor, Tensor]:
    k = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    return k, v


# ---------------------------------------------------------------------------
# Slotted backend
# ---------------------------------------------------------------------------


class TestSlottedChunkedPrefillView:
    def test_single_chunk_write_and_gather(self) -> None:
        """Write a single chunk at start_pos=0, verify gathered KV matches."""
        pool = _make_slotted()
        slot = pool.allocate_slot()
        chunk_len = 4
        k, v = _rand_kv(1, chunk_len)

        view = pool.batched_chunked_prefill_view([slot], [0], [chunk_len])
        for layer in range(NUM_LAYERS):
            cached_k, cached_v = view.update(layer, k, v)
            assert cached_k.shape == (1, NUM_KV_HEADS, chunk_len, HEAD_DIM)
            assert cached_v.shape == (1, NUM_KV_HEADS, chunk_len, HEAD_DIM)
            # The gathered KV should match the input (first chunk, no prior data).
            torch.testing.assert_close(cached_k[0, :, :chunk_len, :], k[0])
            torch.testing.assert_close(cached_v[0, :, :chunk_len, :], v[0])
        view.advance(chunk_len)
        assert pool.get_seq_len(slot) == chunk_len

    def test_two_chunks_sequential(self) -> None:
        """Write chunk 1 (start=0), then chunk 2 (start=4), verify full KV."""
        pool = _make_slotted()
        slot = pool.allocate_slot()

        # Chunk 1: positions 0-3.
        k1, v1 = _rand_kv(1, 4)
        view1 = pool.batched_chunked_prefill_view([slot], [0], [4])
        for layer in range(NUM_LAYERS):
            view1.update(layer, k1, v1)
        view1.advance(4)
        assert pool.get_seq_len(slot) == 4

        # Chunk 2: positions 4-7.
        k2, v2 = _rand_kv(1, 4)
        view2 = pool.batched_chunked_prefill_view([slot], [4], [4])
        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = view2.update(layer, k2, v2)
            # Should contain both chunks: 8 tokens total.
            assert cached_k.shape == (1, NUM_KV_HEADS, 8, HEAD_DIM)
            # Verify chunk 1 data in positions 0-3.
            torch.testing.assert_close(cached_k[0, :, :4, :], k1[0])
            # Verify chunk 2 data in positions 4-7.
            torch.testing.assert_close(cached_k[0, :, 4:8, :], k2[0])
        view2.advance(4)
        assert pool.get_seq_len(slot) == 8

    def test_batched_mixed_start_positions(self) -> None:
        """Batch two sequences at different progress levels."""
        pool = _make_slotted()
        slot_a = pool.allocate_slot()
        slot_b = pool.allocate_slot()

        # Pre-fill slot_a with 4 tokens using chunk 1.
        k_a1, v_a1 = _rand_kv(1, 4)
        view_a1 = pool.batched_chunked_prefill_view([slot_a], [0], [4])
        for layer in range(NUM_LAYERS):
            view_a1.update(layer, k_a1, v_a1)
        view_a1.advance(4)

        # Now batch: slot_a chunk 2 (start=4, len=4), slot_b chunk 1 (start=0, len=4).
        k_batch, v_batch = _rand_kv(2, 4)
        view = pool.batched_chunked_prefill_view([slot_a, slot_b], [4, 0], [4, 4])
        assert view.seq_len == 4  # max_kv_len=8, max_chunk_len=4, pos=8-4=4

        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = view.update(layer, k_batch, v_batch)
            # max_kv_len = max(4+4, 0+4) = 8
            assert cached_k.shape == (2, NUM_KV_HEADS, 8, HEAD_DIM)
            # Slot A: positions 0-3 from chunk 1, positions 4-7 from batch.
            torch.testing.assert_close(cached_k[0, :, :4, :], k_a1[0])
            torch.testing.assert_close(cached_k[0, :, 4:8, :], k_batch[0])
            # Slot B: positions 0-3 from batch.
            torch.testing.assert_close(cached_k[1, :, :4, :], k_batch[1])
        view.advance(4)
        assert pool.get_seq_len(slot_a) == 8
        assert pool.get_seq_len(slot_b) == 4

    def test_seq_len_property(self) -> None:
        """seq_len = max_kv_len - max_chunk_len for model kv_len computation."""
        pool = _make_slotted()
        s1 = pool.allocate_slot()
        s2 = pool.allocate_slot()
        # s1: start=8, chunk=4 -> kv_len=12.
        # s2: start=0, chunk=4 -> kv_len=4.
        # max_kv_len=12, max_chunk_len=4, pos=12-4=8.
        view = pool.batched_chunked_prefill_view([s1, s2], [8, 0], [4, 4])
        assert view.seq_len == 8
        # Model computes: kv_len = pos + seq_len = 8 + 4 = 12 = max_kv_len. Correct.

    def test_decode_after_chunked_prefill(self) -> None:
        """Decode view gathers correct KV after chunked prefill."""
        pool = _make_slotted()
        slot = pool.allocate_slot()

        # Chunk 1: 4 tokens.
        k1, v1 = _rand_kv(1, 4)
        view1 = pool.batched_chunked_prefill_view([slot], [0], [4])
        for layer in range(NUM_LAYERS):
            view1.update(layer, k1, v1)
        view1.advance(4)

        # Chunk 2: 4 tokens.
        k2, v2 = _rand_kv(1, 4)
        view2 = pool.batched_chunked_prefill_view([slot], [4], [4])
        for layer in range(NUM_LAYERS):
            view2.update(layer, k2, v2)
        view2.advance(4)

        # Decode step: write 1 token, gather all 9.
        decode_view = pool.decode_view([slot])
        k_dec, v_dec = _rand_kv(1, 1)
        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = decode_view.update(layer, k_dec, v_dec)
            # seq_len was 8, decode gathers 8+1=9 tokens.
            assert cached_k.shape == (1, NUM_KV_HEADS, 9, HEAD_DIM)
            # Verify prefill data is intact.
            torch.testing.assert_close(cached_k[0, :, :4, :], k1[0])
            torch.testing.assert_close(cached_k[0, :, 4:8, :], k2[0])

    def test_unequal_chunk_lens(self) -> None:
        """Batch elements with different chunk lengths (right-padded to max)."""
        pool = _make_slotted()
        s1 = pool.allocate_slot()
        s2 = pool.allocate_slot()

        # s1: start=0, chunk=6. s2: start=0, chunk=3.
        # max_chunk_len=6, input KV padded to 6.
        k, v = _rand_kv(2, 6)
        view = pool.batched_chunked_prefill_view([s1, s2], [0, 0], [6, 3])
        assert view.max_kv_len == 6
        assert view.max_chunk_len == 6

        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = view.update(layer, k, v)
            assert cached_k.shape == (2, NUM_KV_HEADS, 6, HEAD_DIM)
            # s1: all 6 tokens written.
            torch.testing.assert_close(cached_k[0, :, :6, :], k[0])
            # s2: first 3 tokens written, positions 3-5 are stale/zero.
            torch.testing.assert_close(cached_k[1, :, :3, :], k[1, :, :3, :])
        view.advance(6)
        assert pool.get_seq_len(s1) == 6
        assert pool.get_seq_len(s2) == 3


# ---------------------------------------------------------------------------
# Paged backend
# ---------------------------------------------------------------------------


class TestPagedChunkedPrefillView:
    def test_single_chunk_write_and_gather(self) -> None:
        """Write a single chunk at start_pos=0, verify gathered KV matches."""
        pool = _make_paged()
        seq_id = pool.allocate_slot(initial_tokens=4)
        chunk_len = 4
        k, v = _rand_kv(1, chunk_len)

        view = pool.batched_chunked_prefill_view([seq_id], [0], [chunk_len])
        for layer in range(NUM_LAYERS):
            cached_k, cached_v = view.update(layer, k, v)
            assert cached_k.shape == (1, NUM_KV_HEADS, chunk_len, HEAD_DIM)
            assert cached_v.shape == (1, NUM_KV_HEADS, chunk_len, HEAD_DIM)
            torch.testing.assert_close(cached_k[0, :, :chunk_len, :], k[0])
            torch.testing.assert_close(cached_v[0, :, :chunk_len, :], v[0])
        view.advance(chunk_len)
        assert pool.get_seq_len(seq_id) == chunk_len

    def test_two_chunks_sequential(self) -> None:
        """Write chunk 1, then chunk 2, verify full KV is gathered correctly."""
        pool = _make_paged()
        seq_id = pool.allocate_slot(initial_tokens=8)

        # Chunk 1: positions 0-3.
        k1, v1 = _rand_kv(1, 4)
        view1 = pool.batched_chunked_prefill_view([seq_id], [0], [4])
        for layer in range(NUM_LAYERS):
            view1.update(layer, k1, v1)
        view1.advance(4)
        assert pool.get_seq_len(seq_id) == 4

        # Chunk 2: positions 4-7.
        k2, v2 = _rand_kv(1, 4)
        view2 = pool.batched_chunked_prefill_view([seq_id], [4], [4])
        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = view2.update(layer, k2, v2)
            assert cached_k.shape == (1, NUM_KV_HEADS, 8, HEAD_DIM)
            # Verify chunk 1 data was gathered from blocks correctly.
            torch.testing.assert_close(cached_k[0, :, :4, :], k1[0])
            # Verify chunk 2 data.
            torch.testing.assert_close(cached_k[0, :, 4:8, :], k2[0])
        view2.advance(4)
        assert pool.get_seq_len(seq_id) == 8

    def test_batched_mixed_start_positions(self) -> None:
        """Batch two sequences at different progress levels."""
        pool = _make_paged()
        seq_a = pool.allocate_slot(initial_tokens=8)
        seq_b = pool.allocate_slot(initial_tokens=4)

        # Pre-fill seq_a with 4 tokens using chunk 1.
        k_a1, v_a1 = _rand_kv(1, 4)
        view_a1 = pool.batched_chunked_prefill_view([seq_a], [0], [4])
        for layer in range(NUM_LAYERS):
            view_a1.update(layer, k_a1, v_a1)
        view_a1.advance(4)

        # Now batch: seq_a chunk 2 (start=4, len=4), seq_b chunk 1 (start=0, len=4).
        k_batch, v_batch = _rand_kv(2, 4)
        view = pool.batched_chunked_prefill_view([seq_a, seq_b], [4, 0], [4, 4])
        assert view.seq_len == 4  # max_kv_len=8, max_chunk_len=4, pos=4

        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = view.update(layer, k_batch, v_batch)
            assert cached_k.shape == (2, NUM_KV_HEADS, 8, HEAD_DIM)
            # seq_a: positions 0-3 from chunk 1, positions 4-7 from batch.
            torch.testing.assert_close(cached_k[0, :, :4, :], k_a1[0])
            torch.testing.assert_close(cached_k[0, :, 4:8, :], k_batch[0])
            # seq_b: positions 0-3 from batch.
            torch.testing.assert_close(cached_k[1, :, :4, :], k_batch[1])
        view.advance(4)
        assert pool.get_seq_len(seq_a) == 8
        assert pool.get_seq_len(seq_b) == 4

    def test_seq_len_property(self) -> None:
        """seq_len = max_kv_len - max_chunk_len for model kv_len computation."""
        pool = _make_paged()
        s1 = pool.allocate_slot(initial_tokens=12)
        s2 = pool.allocate_slot(initial_tokens=4)
        view = pool.batched_chunked_prefill_view([s1, s2], [8, 0], [4, 4])
        assert view.seq_len == 8

    def test_decode_after_chunked_prefill(self) -> None:
        """Decode view gathers correct KV after chunked prefill."""
        pool = _make_paged()
        seq_id = pool.allocate_slot(initial_tokens=8)

        # Chunk 1.
        k1, v1 = _rand_kv(1, 4)
        view1 = pool.batched_chunked_prefill_view([seq_id], [0], [4])
        for layer in range(NUM_LAYERS):
            view1.update(layer, k1, v1)
        view1.advance(4)

        # Chunk 2.
        k2, v2 = _rand_kv(1, 4)
        view2 = pool.batched_chunked_prefill_view([seq_id], [4], [4])
        for layer in range(NUM_LAYERS):
            view2.update(layer, k2, v2)
        view2.advance(4)

        # Decode: write 1 token, gather all 9.
        decode_view = pool.decode_view([seq_id])
        k_dec, v_dec = _rand_kv(1, 1)
        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = decode_view.update(layer, k_dec, v_dec)
            assert cached_k.shape == (1, NUM_KV_HEADS, 9, HEAD_DIM)
            # Verify prefill data is intact.
            torch.testing.assert_close(cached_k[0, :, :4, :], k1[0])
            torch.testing.assert_close(cached_k[0, :, 4:8, :], k2[0])

    def test_gather_indices_reused_across_layers(self) -> None:
        """Verify that _build_indices is called once and reused."""
        pool = _make_paged()
        seq_id = pool.allocate_slot(initial_tokens=4)
        k, v = _rand_kv(1, 4)

        view = pool.batched_chunked_prefill_view([seq_id], [0], [4])
        assert not view._indices_built

        view.update(0, k, v)
        assert view._indices_built
        gather_bids_ref = view._gather_block_ids

        view.update(1, k, v)
        # Same object, not rebuilt.
        assert view._gather_block_ids is gather_bids_ref

    def test_unequal_chunk_lens(self) -> None:
        """Batch elements with different chunk lengths."""
        pool = _make_paged()
        s1 = pool.allocate_slot(initial_tokens=8)
        s2 = pool.allocate_slot(initial_tokens=4)

        # s1: start=0, chunk=6. s2: start=0, chunk=3.
        k, v = _rand_kv(2, 6)
        view = pool.batched_chunked_prefill_view([s1, s2], [0, 0], [6, 3])
        assert view.max_kv_len == 6
        assert view.max_chunk_len == 6

        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = view.update(layer, k, v)
            assert cached_k.shape == (2, NUM_KV_HEADS, 6, HEAD_DIM)
            # s1: all 6 tokens written and gathered.
            torch.testing.assert_close(cached_k[0, :, :6, :], k[0])
            # s2: first 3 tokens written and gathered.
            torch.testing.assert_close(cached_k[1, :, :3, :], k[1, :, :3, :])
        view.advance(6)
        assert pool.get_seq_len(s1) == 6
        assert pool.get_seq_len(s2) == 3


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------


class TestCrossBackendChunkedPrefillParity:
    def test_seq_lens_match_after_two_chunks(self) -> None:
        """Both backends produce the same seq_lens after a 2-chunk prefill."""
        slotted = _make_slotted()
        paged = _make_paged()

        s_slot = slotted.allocate_slot()
        p_seq = paged.allocate_slot(initial_tokens=8)

        pools: list[tuple[SlottedKVCache | PagedKVCachePool, int]] = [
            (slotted, s_slot),
            (paged, p_seq),
        ]
        for pool, slot in pools:
            k1, v1 = _rand_kv(1, 4)
            view1 = pool.batched_chunked_prefill_view([slot], [0], [4])
            for layer in range(NUM_LAYERS):
                view1.update(layer, k1, v1)
            view1.advance(4)

            k2, v2 = _rand_kv(1, 4)
            view2 = pool.batched_chunked_prefill_view([slot], [4], [4])
            for layer in range(NUM_LAYERS):
                view2.update(layer, k2, v2)
            view2.advance(4)

        assert slotted.get_seq_len(s_slot) == 8
        assert paged.get_seq_len(p_seq) == 8

    def test_seq_len_property_matches(self) -> None:
        """Both backends compute the same seq_len property."""
        slotted = _make_slotted()
        paged = _make_paged()

        s1 = slotted.allocate_slot()
        s2 = slotted.allocate_slot()
        p1 = paged.allocate_slot(initial_tokens=12)
        p2 = paged.allocate_slot(initial_tokens=4)

        sv = slotted.batched_chunked_prefill_view([s1, s2], [8, 0], [4, 4])
        pv = paged.batched_chunked_prefill_view([p1, p2], [8, 0], [4, 4])
        assert sv.seq_len == pv.seq_len == 8

    def test_gathered_kv_data_matches(self) -> None:
        """Same input data produces identical gathered KV from both backends."""
        slotted = _make_slotted()
        paged = _make_paged()

        s_slot = slotted.allocate_slot()
        p_seq = paged.allocate_slot(initial_tokens=8)

        # Use the same random data for both backends.
        k1, v1 = _rand_kv(1, 4)
        k2, v2 = _rand_kv(1, 4)

        results: dict[str, list[tuple[Tensor, Tensor]]] = {"slotted": [], "paged": []}
        items: list[tuple[str, SlottedKVCache | PagedKVCachePool, int]] = [
            ("slotted", slotted, s_slot),
            ("paged", paged, p_seq),
        ]
        for name, pool, slot in items:
            view1 = pool.batched_chunked_prefill_view([slot], [0], [4])
            for layer in range(NUM_LAYERS):
                view1.update(layer, k1, v1)
            view1.advance(4)

            view2 = pool.batched_chunked_prefill_view([slot], [4], [4])
            for layer in range(NUM_LAYERS):
                ck, cv = view2.update(layer, k2, v2)
                results[name].append((ck.clone(), cv.clone()))
            view2.advance(4)

        for layer in range(NUM_LAYERS):
            torch.testing.assert_close(results["slotted"][layer][0], results["paged"][layer][0])
            torch.testing.assert_close(results["slotted"][layer][1], results["paged"][layer][1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestChunkedPrefillEdgeCases:
    def test_paged_chunk_spanning_block_boundary(self) -> None:
        """Chunk that straddles a block boundary in the paged backend.

        block_size=4, start_pos=2, chunk_len=6 writes tokens at positions 2-7,
        spanning blocks 0 (offsets 2,3) and 1 (offsets 0,1,2,3).
        """
        pool = _make_paged(total_blocks=32, block_size=4)
        seq_id = pool.allocate_slot(initial_tokens=8)  # allocates 2 blocks

        k, v = _rand_kv(1, 6)
        view = pool.batched_chunked_prefill_view([seq_id], [2], [6])
        for layer in range(NUM_LAYERS):
            cached_k, _cached_v = view.update(layer, k, v)
            # max_kv_len = 2 + 6 = 8.
            assert cached_k.shape == (1, NUM_KV_HEADS, 8, HEAD_DIM)
            # Tokens at positions 2-7 should match input.
            torch.testing.assert_close(cached_k[0, :, 2:8, :], k[0])
        view.advance(6)
        assert pool.get_seq_len(seq_id) == 8

    def test_advance_ignores_n_parameter_slotted(self) -> None:
        """advance(n) sets seq_lens to kv_lens regardless of n."""
        pool = _make_slotted()
        slot = pool.allocate_slot()

        k, v = _rand_kv(1, 4)
        view = pool.batched_chunked_prefill_view([slot], [0], [4])
        for layer in range(NUM_LAYERS):
            view.update(layer, k, v)

        # Call advance with a deliberately wrong n.
        view.advance(999)
        assert pool.get_seq_len(slot) == 4  # kv_lens[0] = 0 + 4 = 4

    def test_advance_ignores_n_parameter_paged(self) -> None:
        """advance(n) sets seq_lens to kv_lens regardless of n."""
        pool = _make_paged()
        seq_id = pool.allocate_slot(initial_tokens=4)

        k, v = _rand_kv(1, 4)
        view = pool.batched_chunked_prefill_view([seq_id], [0], [4])
        for layer in range(NUM_LAYERS):
            view.update(layer, k, v)

        view.advance(999)
        assert pool.get_seq_len(seq_id) == 4

    def test_slotted_slot_idx_cached_across_layers(self) -> None:
        """Verify that _slot_idx tensor is created once and reused."""
        pool = _make_slotted()
        slot = pool.allocate_slot()
        k, v = _rand_kv(1, 4)

        view = pool.batched_chunked_prefill_view([slot], [0], [4])
        assert view._slot_idx is None

        view.update(0, k, v)
        assert view._slot_idx is not None
        ref = view._slot_idx

        view.update(1, k, v)
        assert view._slot_idx is ref  # Same object, not rebuilt
