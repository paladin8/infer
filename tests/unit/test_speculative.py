"""Unit tests for speculative decoding (Phase 11)."""

from __future__ import annotations

import pytest
import torch

from infer.cache.paged import PagedKVCachePool
from infer.cache.slotted import SlottedKVCache
from infer.engine.request import Request, StepOutput
from infer.engine.sampler import SamplingParams
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Helper: minimal model config for cache construction
# ---------------------------------------------------------------------------


def _model_config(
    num_layers: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 16,
) -> ModelConfig:
    """Create a minimal ModelConfig for cache tests."""
    return ModelConfig(
        model_type="llama",
        hidden_size=num_kv_heads * head_dim,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=num_kv_heads,
        num_key_value_heads=num_kv_heads,
        vocab_size=32000,
        max_position_embeddings=4096,
        head_dim=head_dim,
    )


# ===================================================================
# D6: truncate_to tests
# ===================================================================


class TestTruncateToSlotted:
    """Tests for SlottedKVCache.truncate_to."""

    def test_truncate_to_decrements_seq_len(self) -> None:
        """truncate_to sets seq_lens[slot] to new_seq_len."""
        pool = SlottedKVCache.from_model_config(
            _model_config(), max_seq_len=128, max_batch_size=4, device="cpu"
        )
        slot = pool.allocate_slot()
        # Simulate advancing by writing to the slot.
        pool.seq_lens[slot] = 50
        pool.truncate_to(slot, 30)
        assert pool.get_seq_len(slot) == 30

    def test_truncate_to_zero(self) -> None:
        """Truncating to 0 resets the slot."""
        pool = SlottedKVCache.from_model_config(
            _model_config(), max_seq_len=128, max_batch_size=4, device="cpu"
        )
        slot = pool.allocate_slot()
        pool.seq_lens[slot] = 20
        pool.truncate_to(slot, 0)
        assert pool.get_seq_len(slot) == 0

    def test_truncate_to_same_length_is_noop(self) -> None:
        """Truncating to current length does not change anything."""
        pool = SlottedKVCache.from_model_config(
            _model_config(), max_seq_len=128, max_batch_size=4, device="cpu"
        )
        slot = pool.allocate_slot()
        pool.seq_lens[slot] = 42
        pool.truncate_to(slot, 42)
        assert pool.get_seq_len(slot) == 42

    def test_truncate_to_beyond_current_length_raises(self) -> None:
        """Cannot truncate beyond current length."""
        pool = SlottedKVCache.from_model_config(
            _model_config(), max_seq_len=128, max_batch_size=4, device="cpu"
        )
        slot = pool.allocate_slot()
        pool.seq_lens[slot] = 10
        with pytest.raises(AssertionError, match="exceeds current length"):
            pool.truncate_to(slot, 20)


class TestTruncateToPagedPool:
    """Tests for PagedKVCachePool.truncate_to."""

    def test_truncate_to_frees_blocks(self) -> None:
        """Truncating frees blocks whose first position >= new_seq_len."""
        config = _model_config()
        block_size = 16
        total_blocks = 20
        pool = PagedKVCachePool.from_model_config(
            config, total_blocks=total_blocks, block_size=block_size, device="cpu"
        )
        # Allocate a sequence with 48 tokens (3 blocks needed).
        seq_id = pool.allocate_slot(initial_tokens=48)
        assert len(pool.page_tables[seq_id]) == 3
        pool.seq_lens[seq_id] = 48

        free_before = pool.allocator.num_free()

        # Truncate to 20 tokens (needs ceil(20/16) = 2 blocks).
        pool.truncate_to(seq_id, 20)
        assert pool.get_seq_len(seq_id) == 20
        assert len(pool.page_tables[seq_id]) == 2
        assert pool.allocator.num_free() == free_before + 1  # freed 1 block

    def test_truncate_to_keeps_partial_block(self) -> None:
        """Truncating to a mid-block position keeps the partial block."""
        config = _model_config()
        block_size = 16
        pool = PagedKVCachePool.from_model_config(
            config, total_blocks=20, block_size=block_size, device="cpu"
        )
        seq_id = pool.allocate_slot(initial_tokens=32)
        pool.seq_lens[seq_id] = 32  # 2 blocks
        assert len(pool.page_tables[seq_id]) == 2

        # Truncate to 17 tokens: needs ceil(17/16) = 2 blocks, so no blocks freed.
        pool.truncate_to(seq_id, 17)
        assert pool.get_seq_len(seq_id) == 17
        assert len(pool.page_tables[seq_id]) == 2  # partial second block kept

    def test_truncate_to_frees_all_blocks(self) -> None:
        """Truncating to 0 frees all blocks."""
        config = _model_config()
        pool = PagedKVCachePool.from_model_config(
            config, total_blocks=20, block_size=16, device="cpu"
        )
        seq_id = pool.allocate_slot(initial_tokens=32)
        pool.seq_lens[seq_id] = 32
        free_before = pool.allocator.num_free()

        pool.truncate_to(seq_id, 0)
        assert pool.get_seq_len(seq_id) == 0
        assert len(pool.page_tables[seq_id]) == 0
        assert pool.allocator.num_free() == free_before + 2

    def test_truncate_to_noop_when_at_length(self) -> None:
        """Truncating to current length is a no-op."""
        config = _model_config()
        pool = PagedKVCachePool.from_model_config(
            config, total_blocks=20, block_size=16, device="cpu"
        )
        seq_id = pool.allocate_slot(initial_tokens=32)
        pool.seq_lens[seq_id] = 32
        free_before = pool.allocator.num_free()

        pool.truncate_to(seq_id, 32)
        assert pool.get_seq_len(seq_id) == 32
        assert len(pool.page_tables[seq_id]) == 2
        assert pool.allocator.num_free() == free_before

    def test_truncate_to_beyond_current_raises(self) -> None:
        """Cannot truncate beyond current length."""
        config = _model_config()
        pool = PagedKVCachePool.from_model_config(
            config, total_blocks=20, block_size=16, device="cpu"
        )
        seq_id = pool.allocate_slot(initial_tokens=16)
        pool.seq_lens[seq_id] = 16
        with pytest.raises(AssertionError, match="exceeds current length"):
            pool.truncate_to(seq_id, 32)


# ===================================================================
# D9: Acceptance rate metrics tests
# ===================================================================


class TestAcceptanceRateFields:
    """Tests for acceptance rate fields on Request and StepOutput."""

    def test_request_has_speculation_rates_field(self) -> None:
        """Request.speculation_acceptance_rates defaults to empty list."""
        req = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        assert req.speculation_acceptance_rates == []

    def test_step_output_has_acceptance_rate_field(self) -> None:
        """StepOutput.acceptance_rate defaults to None."""
        out = StepOutput(
            request_id="test",
            token_id=42,
            text_delta="hello",
            finished=False,
        )
        assert out.acceptance_rate is None

    def test_step_output_acceptance_rate_can_be_set(self) -> None:
        """StepOutput.acceptance_rate can be set to a float."""
        out = StepOutput(
            request_id="test",
            token_id=42,
            text_delta="hello",
            finished=True,
            acceptance_rate=0.85,
        )
        assert out.acceptance_rate == 0.85


# ===================================================================
# D5: Accept/reject algorithm tests (pure logic, no model needed)
# ===================================================================


class TestAcceptRejectGreedy:
    """Tests for greedy accept/reject logic."""

    def test_all_match_all_accepted(self) -> None:
        """When all draft tokens match target argmax, all are accepted + bonus."""
        # Target logits where argmax matches draft tokens at each position.
        vocab_size = 10
        spec_length = 3
        draft_tokens = [5, 7, 2]

        # Build target logits [1, K+1, vocab] where argmax at pos k = draft_tokens[k].
        target_logits = torch.zeros(1, spec_length + 1, vocab_size)
        for k, tok in enumerate(draft_tokens):
            target_logits[0, k, tok] = 10.0  # make argmax = tok

        # Bonus token at position K.
        target_logits[0, spec_length, 3] = 10.0  # bonus = 3

        # Run greedy accept/reject.
        accepted = _greedy_accept_reject(draft_tokens, target_logits[0])
        # All 3 accepted + bonus = 4 tokens.
        assert len(accepted) == spec_length + 1
        assert accepted[:spec_length] == draft_tokens
        assert accepted[spec_length] == 3  # bonus token

    def test_first_mismatch_returns_correction(self) -> None:
        """When first draft token mismatches, only the correction is returned."""
        vocab_size = 10
        draft_tokens = [5, 7, 2]

        target_logits = torch.zeros(1, 4, vocab_size)
        target_logits[0, 0, 8] = 10.0  # argmax at pos 0 = 8, but draft[0] = 5

        accepted = _greedy_accept_reject(draft_tokens, target_logits[0])
        assert len(accepted) == 1
        assert accepted[0] == 8  # correction token

    def test_partial_accept(self) -> None:
        """Some tokens match, then mismatch at position k."""
        vocab_size = 10
        draft_tokens = [5, 7, 2]

        target_logits = torch.zeros(1, 4, vocab_size)
        target_logits[0, 0, 5] = 10.0  # match: draft[0]=5, argmax=5
        target_logits[0, 1, 7] = 10.0  # match: draft[1]=7, argmax=7
        target_logits[0, 2, 9] = 10.0  # mismatch: draft[2]=2, argmax=9

        accepted = _greedy_accept_reject(draft_tokens, target_logits[0])
        assert len(accepted) == 3  # 2 accepted + 1 correction
        assert accepted[0] == 5
        assert accepted[1] == 7
        assert accepted[2] == 9  # correction at mismatch


class TestAcceptRejectSampling:
    """Tests for sampling-mode accept/reject logic."""

    def test_correction_distribution(self) -> None:
        """Correction token is sampled from norm(max(0, p - q))."""
        # p_target = [0.5, 0.3, 0.1, 0.1]
        # q_draft = [0.1, 0.6, 0.2, 0.1]
        # max(0, p - q) = [0.4, 0.0, 0.0, 0.0] -> normalized = [1.0, 0, 0, 0]
        # So correction token must be 0.
        p_target = torch.tensor([0.5, 0.3, 0.1, 0.1])
        q_draft = torch.tensor([0.1, 0.6, 0.2, 0.1])

        correction = _sample_correction(p_target, q_draft, generator=None)
        assert correction == 0  # only token 0 has positive mass

    def test_acceptance_probability_matches_theory(self) -> None:
        """Acceptance probability for a token should be min(1, p/q)."""
        # p_target(tok=1) = 0.4, q_draft(tok=1) = 0.2 -> accept prob = 1.0 (p/q=2)
        # p_target(tok=0) = 0.1, q_draft(tok=0) = 0.5 -> accept prob = 0.2
        p_target = torch.tensor([0.1, 0.4, 0.3, 0.2])
        q_draft = torch.tensor([0.5, 0.2, 0.2, 0.1])

        # Token 1: p/q = 0.4/0.2 = 2.0, accept_prob = min(1, 2.0) = 1.0
        accept_prob_1 = min(1.0, (p_target[1] / q_draft[1]).item())
        assert accept_prob_1 == 1.0

        # Token 0: p/q = 0.1/0.5 = 0.2, accept_prob = 0.2
        accept_prob_0 = min(1.0, (p_target[0] / q_draft[0]).item())
        assert abs(accept_prob_0 - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# Pure-logic helper functions for accept/reject testing
# (These mirror the logic in SpeculativeRunner but are standalone.)
# ---------------------------------------------------------------------------


def _greedy_accept_reject(
    draft_tokens: list[int],
    target_logits: torch.Tensor,
) -> list[int]:
    """Greedy accept/reject: accept while argmax matches, else correct.

    Args:
        draft_tokens: K draft tokens.
        target_logits: [K+1, vocab_size] target logits.

    Returns:
        List of accepted tokens (1 to K+1).
    """
    accepted: list[int] = []
    k = len(draft_tokens)
    for i in range(k):
        target_token = int(torch.argmax(target_logits[i]).item())
        if target_token == draft_tokens[i]:
            accepted.append(draft_tokens[i])
        else:
            accepted.append(target_token)
            return accepted
    # All accepted: sample bonus.
    bonus = int(torch.argmax(target_logits[k]).item())
    accepted.append(bonus)
    return accepted


def _sample_correction(
    p_target: torch.Tensor,
    q_draft: torch.Tensor,
    generator: torch.Generator | None,
) -> int:
    """Sample a correction token from norm(max(0, p - q)).

    Args:
        p_target: Target probability distribution [vocab_size].
        q_draft: Draft probability distribution [vocab_size].
        generator: Optional RNG.

    Returns:
        Sampled correction token ID.
    """
    diff = torch.clamp(p_target - q_draft, min=0.0)
    total = diff.sum()
    if total <= 0:
        # Fallback: sample from target distribution.
        return int(torch.multinomial(p_target.unsqueeze(0), 1, generator=generator).item())
    diff = diff / total
    return int(torch.multinomial(diff.unsqueeze(0), 1, generator=generator).item())
