"""Tests for position_ids-based attention mask construction.

Verifies that the per-element causal masks derived from position_ids
(used for chunked prefill) match expected patterns and are equivalent
to the standard causal_mask for the non-chunked case.
"""

from __future__ import annotations

import torch

from infer.models.common import causal_mask, sliding_window_causal_mask


def _build_position_ids_mask(
    position_ids: torch.Tensor,
    kv_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reproduce the position_ids-based mask construction from model forward.

    Args:
        position_ids: [batch, q_len] absolute positions.
        kv_len: Total KV length (pos + seq_len in model code).
        dtype: Output dtype.

    Returns:
        Mask [batch, 1, q_len, kv_len] with 0.0 for attend, -inf for mask.
    """
    kv_positions = torch.arange(kv_len, device=position_ids.device)
    return torch.where(
        kv_positions[None, None, :] <= position_ids[:, :, None],
        torch.tensor(0.0, dtype=dtype, device=position_ids.device),
        torch.tensor(float("-inf"), dtype=dtype, device=position_ids.device),
    ).unsqueeze(1)


def _build_position_ids_dual_masks(
    position_ids: torch.Tensor,
    kv_len: int,
    sliding_window: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the Gemma 3 dual mask construction from model forward.

    Returns:
        (global_mask, local_mask) each [batch, 1, q_len, kv_len].
    """
    kv_positions = torch.arange(kv_len, device=position_ids.device)
    causal = kv_positions[None, None, :] <= position_ids[:, :, None]
    in_window = (position_ids[:, :, None] - kv_positions[None, None, :]) < sliding_window
    global_mask = torch.where(
        causal,
        torch.tensor(0.0, dtype=dtype, device=position_ids.device),
        torch.tensor(float("-inf"), dtype=dtype, device=position_ids.device),
    ).unsqueeze(1)
    local_mask = torch.where(
        causal & in_window,
        torch.tensor(0.0, dtype=dtype, device=position_ids.device),
        torch.tensor(float("-inf"), dtype=dtype, device=position_ids.device),
    ).unsqueeze(1)
    return global_mask, local_mask


class TestPositionIdsMaskSquareCase:
    """Square case (start_pos=0): should match causal_mask(seq_len)."""

    def test_matches_causal_mask_single_element(self) -> None:
        seq_len = 6
        position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, 6]
        mask = _build_position_ids_mask(position_ids, kv_len=seq_len)
        expected = causal_mask(seq_len)
        torch.testing.assert_close(mask, expected)

    def test_matches_causal_mask_batch(self) -> None:
        """Batch of identical sequences all starting at 0."""
        seq_len = 4
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(3, -1)  # [3, 4]
        mask = _build_position_ids_mask(position_ids, kv_len=seq_len)
        expected = causal_mask(seq_len).expand(3, -1, -1, -1)
        torch.testing.assert_close(mask, expected)

    def test_seq_len_one(self) -> None:
        """Single token at position 0."""
        position_ids = torch.tensor([[0]])
        mask = _build_position_ids_mask(position_ids, kv_len=1)
        assert mask.shape == (1, 1, 1, 1)
        assert mask[0, 0, 0, 0].item() == 0.0


class TestPositionIdsMaskRectangularCase:
    """Rectangular case (start_pos > 0): chunked prefill."""

    def test_second_chunk(self) -> None:
        """Second chunk of a 2-chunk prefill: positions 4-7, kv_len=8."""
        chunk_len = 4
        start_pos = 4
        kv_len = 8
        position_ids = torch.arange(start_pos, start_pos + chunk_len).unsqueeze(0)  # [1, 4]
        mask = _build_position_ids_mask(position_ids, kv_len=kv_len)

        assert mask.shape == (1, 1, chunk_len, kv_len)

        # Each query position should attend to all KV positions <= its own.
        for qi in range(chunk_len):
            q_pos = start_pos + qi
            for ki in range(kv_len):
                val = mask[0, 0, qi, ki].item()
                if ki <= q_pos:
                    assert val == 0.0, f"q_pos={q_pos}, kv_pos={ki} should be 0.0"
                else:
                    assert val == float("-inf"), f"q_pos={q_pos}, kv_pos={ki} should be -inf"

    def test_third_chunk_large_offset(self) -> None:
        """Third chunk: positions 1024-1535, kv_len=1536."""
        chunk_len = 512
        start_pos = 1024
        kv_len = 1536
        position_ids = torch.arange(start_pos, start_pos + chunk_len).unsqueeze(0)

        mask = _build_position_ids_mask(position_ids, kv_len=kv_len)
        assert mask.shape == (1, 1, chunk_len, kv_len)

        # First query (pos=1024) should attend to KV positions 0..1024.
        assert mask[0, 0, 0, 1024].item() == 0.0
        assert mask[0, 0, 0, 1025].item() == float("-inf")

        # Last query (pos=1535) should attend to all KV positions 0..1535.
        assert mask[0, 0, -1, 1535].item() == 0.0

    def test_first_chunk_is_square(self) -> None:
        """First chunk (start_pos=0) should be equivalent to causal_mask."""
        chunk_len = 4
        position_ids = torch.arange(chunk_len).unsqueeze(0)
        mask = _build_position_ids_mask(position_ids, kv_len=chunk_len)
        expected = causal_mask(chunk_len)
        torch.testing.assert_close(mask, expected)


class TestPositionIdsMaskBatchedCase:
    """Batched chunks at different progress levels."""

    def test_two_chunks_different_offsets(self) -> None:
        """Batch: element 0 at start_pos=0 (chunk_len=4), element 1 at start_pos=4 (chunk_len=4)."""
        max_chunk_len = 4
        kv_len = 8  # max(0+4, 4+4) = 8
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3],  # Element 0: first chunk
                [4, 5, 6, 7],  # Element 1: second chunk
            ]
        )  # [2, 4]

        mask = _build_position_ids_mask(position_ids, kv_len=kv_len)
        assert mask.shape == (2, 1, max_chunk_len, kv_len)

        # Element 0: positions 0-3, square causal mask in top-left 4x4.
        for qi in range(4):
            for ki in range(kv_len):
                val = mask[0, 0, qi, ki].item()
                if ki <= qi:
                    assert val == 0.0
                else:
                    assert val == float("-inf")

        # Element 1: positions 4-7, all kv_pos 0-7 visible to last query.
        for qi in range(4):
            q_pos = 4 + qi
            for ki in range(kv_len):
                val = mask[1, 0, qi, ki].item()
                if ki <= q_pos:
                    assert val == 0.0
                else:
                    assert val == float("-inf")

    def test_three_elements_mixed_progress(self) -> None:
        """Three elements at different stages, padded to max_chunk_len=4."""
        # Element 0: start=0, chunk=4, kv_len=4
        # Element 1: start=4, chunk=3 (padded to 4), kv_len=7
        # Element 2: start=8, chunk=2 (padded to 4), kv_len=10
        # Overall max_kv_len = 10
        # Padded positions use 0 (padding_mask would handle masking those out)
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3],  # Element 0
                [4, 5, 6, 0],  # Element 1 (padded last position)
                [8, 9, 0, 0],  # Element 2 (padded last two positions)
            ]
        )
        kv_len = 10
        mask = _build_position_ids_mask(position_ids, kv_len=kv_len)
        assert mask.shape == (3, 1, 4, 10)

        # Element 2, first query (pos=8): attends to kv 0..8.
        assert mask[2, 0, 0, 8].item() == 0.0
        assert mask[2, 0, 0, 9].item() == float("-inf")

        # Element 2, second query (pos=9): attends to kv 0..9.
        assert mask[2, 0, 1, 9].item() == 0.0


class TestPositionIdsMaskWithPadding:
    """Position_ids mask composed with padding_mask."""

    def test_padding_masks_out_invalid_kv(self) -> None:
        """Padding mask should block padded KV positions on top of causal mask."""
        # Two sequences: element 0 has kv_len=4, element 1 has kv_len=8.
        # Padded to max_kv_len=8, so element 0 has 4 padded KV positions.
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        )
        kv_len = 8
        mask = _build_position_ids_mask(position_ids, kv_len=kv_len)

        # Simulate padding_mask: True for valid, False for padding.
        padding_mask = torch.tensor(
            [
                [True, True, True, True, False, False, False, False],  # element 0: only 4 valid
                [True, True, True, True, True, True, True, True],  # element 1: all valid
            ]
        )
        pad_mask = ~padding_mask[:, None, None, :kv_len]
        mask.masked_fill_(pad_mask, float("-inf"))

        # Element 0: query at pos=3 should attend to kv 0..3 only.
        assert mask[0, 0, 3, 3].item() == 0.0
        assert mask[0, 0, 3, 4].item() == float("-inf")  # Padded

        # Element 0: query at pos=0 should attend to kv 0 only.
        assert mask[0, 0, 0, 0].item() == 0.0
        assert mask[0, 0, 0, 1].item() == float("-inf")  # Causal

        # Element 1: query at pos=7 should attend to all 8 positions.
        for ki in range(8):
            assert mask[1, 0, 3, ki].item() == 0.0

    def test_chunked_with_padding(self) -> None:
        """Second chunk with padding_mask masking past the actual KV length."""
        # Element at start_pos=4, chunk_len=4, real kv_len=8 in max_kv_len=10.
        position_ids = torch.tensor([[4, 5, 6, 7]])
        kv_len = 10
        mask = _build_position_ids_mask(position_ids, kv_len=kv_len)

        padding_mask = torch.zeros(1, kv_len, dtype=torch.bool)
        padding_mask[0, :8] = True  # Only first 8 are valid
        pad_mask = ~padding_mask[:, None, None, :kv_len]
        mask.masked_fill_(pad_mask, float("-inf"))

        # Query at pos=7: attends to kv 0..7, not 8..9.
        assert mask[0, 0, 3, 7].item() == 0.0
        assert mask[0, 0, 3, 8].item() == float("-inf")
        assert mask[0, 0, 3, 9].item() == float("-inf")


class TestPositionIdsMaskDecodeSingleToken:
    """When seq_len=1 and position_ids is provided, the model uses a zero mask."""

    def test_single_token_mask_is_all_attend(self) -> None:
        """Decode path: seq_len=1 creates zeros regardless of position_ids."""
        # This mirrors what the model actually does: the seq_len==1 branch
        # creates a zero mask, not a position_ids-based mask.
        kv_len = 10
        batch_size = 2
        mask = torch.zeros(batch_size, 1, 1, kv_len)
        # All positions are attended to (decode attends to full cache).
        for b in range(batch_size):
            for ki in range(kv_len):
                assert mask[b, 0, 0, ki].item() == 0.0


class TestPositionIdsDualMasks:
    """Gemma 3 dual masks: global causal + local sliding window."""

    def test_square_matches_standard_masks(self) -> None:
        """Start_pos=0 should match standard causal_mask and sliding_window_causal_mask."""
        seq_len = 8
        window = 3
        position_ids = torch.arange(seq_len).unsqueeze(0)
        global_mask, local_mask = _build_position_ids_dual_masks(
            position_ids,
            kv_len=seq_len,
            sliding_window=window,
        )

        expected_global = causal_mask(seq_len)
        expected_local = sliding_window_causal_mask(seq_len, window)

        torch.testing.assert_close(global_mask, expected_global)
        torch.testing.assert_close(local_mask, expected_local)

    def test_chunked_global_mask_correct(self) -> None:
        """Second chunk: global mask should be causal only."""
        chunk_len = 4
        start_pos = 4
        kv_len = 8
        position_ids = torch.arange(start_pos, start_pos + chunk_len).unsqueeze(0)
        global_mask, _ = _build_position_ids_dual_masks(
            position_ids,
            kv_len=kv_len,
            sliding_window=3,
        )

        # Same as non-windowed mask: kv_pos <= q_pos.
        for qi in range(chunk_len):
            q_pos = start_pos + qi
            for ki in range(kv_len):
                val = global_mask[0, 0, qi, ki].item()
                if ki <= q_pos:
                    assert val == 0.0
                else:
                    assert val == float("-inf")

    def test_chunked_local_mask_window(self) -> None:
        """Second chunk: local mask should respect sliding window."""
        chunk_len = 4
        start_pos = 4
        kv_len = 8
        window = 3
        position_ids = torch.arange(start_pos, start_pos + chunk_len).unsqueeze(0)
        _, local_mask = _build_position_ids_dual_masks(
            position_ids,
            kv_len=kv_len,
            sliding_window=window,
        )

        for qi in range(chunk_len):
            q_pos = start_pos + qi
            for ki in range(kv_len):
                val = local_mask[0, 0, qi, ki].item()
                is_causal = ki <= q_pos
                is_in_window = (q_pos - ki) < window
                if is_causal and is_in_window:
                    assert val == 0.0, f"q_pos={q_pos}, kv={ki} should be 0.0"
                else:
                    assert val == float("-inf"), f"q_pos={q_pos}, kv={ki} should be -inf"

    def test_batched_dual_masks_different_offsets(self) -> None:
        """Batch with different start positions."""
        window = 4
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [8, 9, 10, 11],
            ]
        )
        kv_len = 12
        global_mask, local_mask = _build_position_ids_dual_masks(
            position_ids,
            kv_len=kv_len,
            sliding_window=window,
        )

        # Element 1, query at pos=8: global allows kv 0..8, local allows kv 5..8.
        assert global_mask[1, 0, 0, 8].item() == 0.0
        assert global_mask[1, 0, 0, 0].item() == 0.0  # Global sees everything causal
        assert local_mask[1, 0, 0, 8].item() == 0.0  # In window
        assert local_mask[1, 0, 0, 4].item() == float("-inf")  # 8-4=4, not < 4
        assert local_mask[1, 0, 0, 5].item() == 0.0  # 8-5=3 < 4, in window
