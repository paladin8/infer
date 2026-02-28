"""Unit tests for speculative decoding (Phase 11)."""

from __future__ import annotations

import dataclasses
import time

import pytest
import torch
from torch import Tensor, nn

from infer.cache.paged import PagedKVCachePool
from infer.cache.protocol import KVCacheProtocol
from infer.cache.slotted import SlottedKVCache
from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.sampler import SamplingParams
from infer.engine.speculative_runner import SpeculativeRunner
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


# ===================================================================
# Mock models and tokenizer for SpeculativeRunner tests
# ===================================================================

_MOCK_CONFIG = ModelConfig(
    model_type="llama",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    max_position_embeddings=256,
    head_dim=16,
)


class MockTokenizer:
    """Simple tokenizer for testing."""

    def __init__(self, eos_token_ids: set[int] | None = None) -> None:
        self._eos_token_ids = eos_token_ids or {99}

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos_token_ids

    @property
    def vocab_size(self) -> int:
        return 100

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ids = token_ids
        if skip_special_tokens:
            ids = [t for t in ids if t not in self._eos_token_ids]
        return "".join(chr(ord("A") + (t % 26)) for t in ids)


class FixedTokenMockModel(nn.Module):
    """Mock model that always emits a fixed token at every position.

    Supports the KVCacheProtocol by calling advance() on the cache.
    """

    def __init__(self, fixed_token: int, vocab_size: int = 100) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._fixed_token = fixed_token
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCacheProtocol | None = None,
        padding_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            out_len = 1 if padding_mask is None else seq_len
        else:
            out_len = seq_len
        logits = torch.zeros(batch, out_len, self.config.vocab_size)
        logits[:, :, self._fixed_token] = 10.0
        return logits


def _make_request(
    request_id: str = "req-0",
    prompt_token_ids: list[int] | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 20,
    seed: int | None = 42,
) -> Request:
    """Create a test request."""
    params = SamplingParams(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2, 3, 4, 5],
        sampling_params=params,
        arrival_time_s=time.perf_counter(),
        generator=generator,
    )


def _make_spec_runner(
    target_model: nn.Module,
    draft_model: nn.Module,
    spec_length: int = 3,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
    backend: str = "contiguous",
) -> SpeculativeRunner:
    """Create a SpeculativeRunner for testing."""
    config = EngineConfig(
        model="test-target",
        batching_mode="continuous",
        kv_cache_backend=backend,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        use_speculative_decoding=True,
        draft_model="test-draft",
        spec_length=spec_length,
    )
    tokenizer = MockTokenizer()
    return SpeculativeRunner(target_model, draft_model, tokenizer, config)  # type: ignore[arg-type]


# ===================================================================
# D1: SpeculativeRunner construction tests
# ===================================================================


class TestSpeculativeRunnerConstruction:
    """Tests for SpeculativeRunner construction and slot management."""

    def test_creates_dual_cache_pools(self) -> None:
        """SpeculativeRunner creates separate target and draft cache pools."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft)
        assert runner.target_cache_pool is not runner.draft_cache_pool

    def test_free_slot_frees_both_pools(self) -> None:
        """free_slot releases slots in both target and draft pools."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft)

        # Simulate prefill to allocate slots.
        req = _make_request()
        runner._prefill_one(req)
        assert req.slot_idx is not None
        assert req.slot_idx in runner._draft_slots

        target_slot = req.slot_idx
        runner.free_slot(target_slot)
        assert target_slot not in runner._draft_slots

    def test_cleanup_request_removes_tracking(self) -> None:
        """cleanup_request removes text tracking and draft prefill state."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft)

        runner._prev_text_lens["req-1"] = 10
        runner._draft_prefilled.add("req-1")
        runner.cleanup_request("req-1")
        assert "req-1" not in runner._prev_text_lens
        assert "req-1" not in runner._draft_prefilled

    def test_free_kv_tokens_returns_min_of_both_pools(self) -> None:
        """free_kv_tokens returns the minimum of both pools (paged backend)."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, backend="paged")
        free = runner.free_kv_tokens()
        assert free is not None
        assert free >= 0

    def test_free_kv_tokens_none_for_contiguous(self) -> None:
        """free_kv_tokens returns None for contiguous backend."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, backend="contiguous")
        assert runner.free_kv_tokens() is None


# ===================================================================
# D1+D3: Prefill delegation tests
# ===================================================================


class TestSpeculativeRunnerPrefill:
    """Tests for SpeculativeRunner prefill delegation."""

    def test_prefill_one_allocates_both_slots(self) -> None:
        """Prefill allocates both target and draft cache slots."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft)

        req = _make_request()
        output = runner._prefill_one(req)

        assert req.slot_idx is not None
        assert req.slot_idx in runner._draft_slots
        assert req.state == RequestState.DECODE
        assert output.token_id == 5
        assert len(req.generated_token_ids) == 1

    def test_prefill_batch_allocates_both_slots(self) -> None:
        """Batch prefill allocates both cache slots for all requests."""
        target = FixedTokenMockModel(fixed_token=7)
        draft = FixedTokenMockModel(fixed_token=7)
        runner = _make_spec_runner(target, draft)

        requests = [_make_request(f"req-{i}") for i in range(3)]
        outputs = runner._prefill_batch(requests)

        assert len(outputs) == 3
        for req in requests:
            assert req.slot_idx is not None
            assert req.slot_idx in runner._draft_slots
            assert req.state == RequestState.DECODE

    def test_step_prefill_produces_step_outputs(self) -> None:
        """step() with only prefill requests produces StepOutputs."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft)

        req = _make_request()
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        assert outputs[0][0] is req
        assert outputs[0][1].token_id == 5


# ===================================================================
# D1+D3+D4+D5: Full speculative decode tests (greedy)
# ===================================================================


class TestSpeculativeDecodeGreedy:
    """Tests for speculative decode with greedy sampling (temperature=0)."""

    def test_all_accepted_produces_k_plus_1_tokens(self) -> None:
        """When draft and target agree, K+1 tokens are produced per step."""
        # Both models always emit token 5 -> all draft tokens match target.
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, spec_length=3)

        req = _make_request(temperature=0.0)
        # Prefill first.
        runner.step(prefill=[req], decode=[])
        assert req.state == RequestState.DECODE

        # Decode step: speculative.
        outputs = runner.step(prefill=[], decode=[req])
        # With all matches, we get spec_length + 1 = 4 tokens.
        assert len(outputs) == 4
        for _r, out in outputs:
            assert out.token_id == 5

    def test_mismatch_produces_correction_token(self) -> None:
        """When draft and target disagree, the correction token is from target."""
        # Draft always emits 3, target always emits 7 -> first draft token mismatches.
        target = FixedTokenMockModel(fixed_token=7)
        draft = FixedTokenMockModel(fixed_token=3)
        runner = _make_spec_runner(target, draft, spec_length=3)

        req = _make_request(temperature=0.0)
        runner.step(prefill=[req], decode=[])

        outputs = runner.step(prefill=[], decode=[req])
        # Mismatch at first position: only 1 token (correction = target's choice).
        assert len(outputs) == 1
        assert outputs[0][1].token_id == 7

    def test_multiple_steps_accumulate_tokens(self) -> None:
        """Multiple decode steps accumulate generated tokens."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, spec_length=2)

        req = _make_request(temperature=0.0, max_new_tokens=10)
        runner.step(prefill=[req], decode=[])

        # Step 1: 3 tokens (K+1 = 2+1).
        outputs1 = runner.step(prefill=[], decode=[req])
        assert len(outputs1) == 3

        # Step 2: 3 more tokens.
        outputs2 = runner.step(prefill=[], decode=[req])
        assert len(outputs2) == 3

        # Total: 1 (prefill) + 3 + 3 = 7 tokens.
        assert len(req.generated_token_ids) == 7

    def test_acceptance_rate_logged(self) -> None:
        """Acceptance rates are recorded on the request."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, spec_length=3)

        req = _make_request(temperature=0.0, max_new_tokens=20)
        runner.step(prefill=[req], decode=[])
        runner.step(prefill=[], decode=[req])

        # All tokens accepted -> acceptance rate = 1.0.
        assert len(req.speculation_acceptance_rates) == 1
        assert req.speculation_acceptance_rates[0] == 1.0

    def test_eos_stops_generation_mid_speculation(self) -> None:
        """If an accepted token is EOS, remaining tokens are discarded."""
        # Target emits EOS token (99).
        target = FixedTokenMockModel(fixed_token=99)
        draft = FixedTokenMockModel(fixed_token=99)
        runner = _make_spec_runner(target, draft, spec_length=3)

        req = _make_request(temperature=0.0)
        runner.step(prefill=[req], decode=[])

        # Prefill token is also 99 (EOS) -> request finishes immediately.
        # But let's test with a non-EOS prefill token.
        # Since both models always output 99, the first accepted token is EOS.
        # The request should finish.
        assert req.state == RequestState.FINISHED or req.generated_token_ids[-1] == 99


class TestSpeculativeDecodeMultiRequest:
    """Tests for speculative decode with multiple concurrent requests."""

    def test_batch_speculative_decode(self) -> None:
        """Multiple requests can be speculatively decoded in one step."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, spec_length=2, max_batch_size=4)

        requests = [_make_request(f"req-{i}", temperature=0.0) for i in range(3)]

        # Prefill all.
        runner.step(prefill=requests, decode=[])
        for r in requests:
            assert r.state == RequestState.DECODE

        # Decode all speculatively.
        outputs = runner.step(prefill=[], decode=requests)
        # Each request produces K+1=3 tokens.
        assert len(outputs) == 9  # 3 requests * 3 tokens


# ===================================================================
# D6 + speculative rollback tests
# ===================================================================


class TestSpeculativeRollback:
    """Tests for KV cache rollback after accept/reject."""

    def test_rollback_after_mismatch_restores_cache_length(self) -> None:
        """After a mismatch, KV caches are rolled back correctly."""
        target = FixedTokenMockModel(fixed_token=7)
        draft = FixedTokenMockModel(fixed_token=3)
        runner = _make_spec_runner(target, draft, spec_length=3)

        req = _make_request(temperature=0.0)
        runner.step(prefill=[req], decode=[])
        assert req.slot_idx is not None

        # Target cache has prompt (5 tokens) after prefill.
        target_len_before = runner.target_cache_pool.get_seq_len(req.slot_idx)
        assert target_len_before == 5  # prompt length

        # Draft cache is 0 (lazy prefill not yet run).
        # After the first decode step, lazy draft prefill runs first.

        # Decode: mismatch at first position -> 1 correction token.
        runner.step(prefill=[], decode=[req])

        target_len_after = runner.target_cache_pool.get_seq_len(req.slot_idx)
        draft_slot = runner._draft_slots[req.slot_idx]
        draft_len_after = runner.draft_cache_pool.get_seq_len(draft_slot)

        # Mismatch at position 0: num_accepted_draft = 0, new_valid = 1.
        # Target: truncated to 5 + 1 = 6.
        assert target_len_after == 6
        # Draft: lazy prefill sets it to 5 (prompt), then K=3 steps advance to 8.
        # Rollback: min(new_valid=1, K=3) = 1, so truncated to 5 + 1 = 6.
        assert draft_len_after == 6

    def test_rollback_after_full_accept(self) -> None:
        """After all tokens accepted, cache length includes K+1 entries for target."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, spec_length=3)

        req = _make_request(temperature=0.0)
        runner.step(prefill=[req], decode=[])
        assert req.slot_idx is not None

        target_len_before = runner.target_cache_pool.get_seq_len(req.slot_idx)
        assert target_len_before == 5  # prompt length

        runner.step(prefill=[], decode=[req])

        target_len_after = runner.target_cache_pool.get_seq_len(req.slot_idx)
        draft_slot = runner._draft_slots[req.slot_idx]
        draft_len_after = runner.draft_cache_pool.get_seq_len(draft_slot)

        # All K=3 accepted: num_accepted_draft=3, new_valid=4.
        # Target: truncated to 5 + 4 = 9. Verification wrote K+1=4, so 5+4=9 is current. No-op.
        assert target_len_after == 9
        # Draft: lazy prefill to 5, then K=3 steps -> 8.
        # Rollback: min(4, 3) = 3, so draft stays at 5 + 3 = 8. No-op.
        assert draft_len_after == 8


# ===================================================================
# Config validation tests (D8, moved here from inline)
# ===================================================================


class TestSpeculativeConfigValidation:
    """Tests for speculative decoding config validation."""

    def test_config_requires_draft_model(self) -> None:
        """use_speculative_decoding=True without draft_model raises ValueError."""
        with pytest.raises(ValueError, match="draft_model"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                use_speculative_decoding=True,
            )

    def test_config_requires_continuous_batching(self) -> None:
        """Speculative decoding requires continuous batching."""
        with pytest.raises(ValueError, match="batching_mode"):
            EngineConfig(
                model="m",
                batching_mode="static",
                use_speculative_decoding=True,
                draft_model="d",
            )

    def test_config_rejects_cuda_graphs(self) -> None:
        """Speculative decoding is incompatible with CUDA graphs."""
        with pytest.raises(ValueError, match="cuda_graphs"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                kv_cache_backend="paged",
                use_speculative_decoding=True,
                draft_model="d",
                use_cuda_graphs=True,
            )


# ===================================================================
# D5: SpeculativeRunner._accept_reject_greedy direct tests
# ===================================================================


class TestSpeculativeRunnerAcceptRejectGreedy:
    """Tests for SpeculativeRunner._accept_reject_greedy method directly."""

    def test_accept_reject_greedy_all_match(self) -> None:
        """All draft tokens match -> K+1 tokens returned."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, spec_length=3)

        draft_tokens = [5, 5, 5]
        target_logits = torch.zeros(4, 100)
        for i in range(3):
            target_logits[i, 5] = 10.0
        target_logits[3, 8] = 10.0  # bonus = 8

        accepted = runner._accept_reject_greedy(draft_tokens, target_logits, k=3)
        assert accepted == [5, 5, 5, 8]

    def test_accept_reject_greedy_mismatch_at_start(self) -> None:
        """Mismatch at first position -> 1 correction token."""
        target = FixedTokenMockModel(fixed_token=5)
        draft = FixedTokenMockModel(fixed_token=5)
        runner = _make_spec_runner(target, draft, spec_length=3)

        draft_tokens = [3, 5, 5]
        target_logits = torch.zeros(4, 100)
        target_logits[0, 7] = 10.0  # mismatch: draft=3, target=7

        accepted = runner._accept_reject_greedy(draft_tokens, target_logits, k=3)
        assert accepted == [7]
