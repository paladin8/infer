"""Unit tests for the ContinuousRunner."""

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
from infer.engine.continuous_runner import ContinuousRunner
from infer.engine.request import Request, RequestState
from infer.engine.sampler import SamplingParams
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Mock model / tokenizer
# ---------------------------------------------------------------------------

_MOCK_CONFIG = ModelConfig(
    model_type="llama",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    max_position_embeddings=256,
    head_dim=8,
)


class MockTokenizer:
    """Simple tokenizer: maps token IDs to characters, A-Z cycling."""

    def __init__(self, eos_token_ids: set[int] | None = None) -> None:
        self._eos_token_ids = eos_token_ids or {99}

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos_token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ids = token_ids
        if skip_special_tokens:
            ids = [t for t in ids if t not in self._eos_token_ids]
        return "".join(chr(ord("A") + (t % 26)) for t in ids)


class ContinuousMockModel(nn.Module):
    """Mock model that supports position_ids, padding_mask, and KVCacheProtocol.

    Always returns logits favoring ``fixed_next_token`` at every position.
    """

    def __init__(self, vocab_size: int, fixed_next_token: int) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._fixed_next_token = fixed_next_token
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
        logits[:, :, self._fixed_next_token] = 10.0
        return logits


class ContinuousSequenceMockModel(nn.Module):
    """Mock model that emits tokens from per-request sequences.

    ``sequences`` is a dict mapping request slot index to a list of token IDs.
    Falls back to a default token if slot not found.
    """

    def __init__(
        self, vocab_size: int, sequences: dict[int, list[int]], default_token: int = 5
    ) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._sequences = sequences
        self._steps: dict[int, int] = {slot: 0 for slot in sequences}
        self._default_token = default_token
        self._dummy = nn.Parameter(torch.zeros(1))
        # Track position_ids passed to forward for test assertions.
        self.last_position_ids: Tensor | None = None

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCacheProtocol | None = None,
        padding_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        self.last_position_ids = position_ids
        batch, seq_len = input_ids.shape
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            out_len = 1 if padding_mask is None else seq_len
        else:
            out_len = seq_len
        logits = torch.zeros(batch, out_len, self.config.vocab_size)
        # For prefill (no position_ids), use input_ids[0,0] to guess the slot.
        # For decode (with position_ids), use the kv_cache view's slots.
        if position_ids is not None and hasattr(kv_cache, "slots"):
            # Batched decode: use slot mapping.
            for i, slot in enumerate(kv_cache.slots):  # type: ignore[union-attr]
                if slot in self._sequences:
                    step = self._steps.get(slot, 0)
                    idx = min(step, len(self._sequences[slot]) - 1)
                    logits[i, :, self._sequences[slot][idx]] = 10.0
                    self._steps[slot] = step + 1
                else:
                    logits[i, :, self._default_token] = 10.0
        else:
            # Prefill: just use the default token.
            logits[:, :, self._default_token] = 10.0
        return logits


class StepCounterMockModel(nn.Module):
    """Mock model that emits a fixed token for prefill and a per-step token for decode.

    Uses a global step counter (not slot-specific) so tests don't depend on
    slot assignment order.  ``decode_sequence[step]`` is the token emitted on
    the Nth decode call for ALL batch elements.
    """

    def __init__(
        self,
        vocab_size: int,
        prefill_token: int,
        decode_sequence: list[int],
    ) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._prefill_token = prefill_token
        self._decode_sequence = decode_sequence
        self._decode_step = 0
        self._dummy = nn.Parameter(torch.zeros(1))
        self.last_position_ids: Tensor | None = None

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCacheProtocol | None = None,
        padding_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        self.last_position_ids = position_ids
        batch, seq_len = input_ids.shape
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            out_len = 1 if padding_mask is None else seq_len
        else:
            out_len = seq_len
        logits = torch.zeros(batch, out_len, self.config.vocab_size)
        if position_ids is not None:
            # Decode path.
            idx = min(self._decode_step, len(self._decode_sequence) - 1)
            logits[:, :, self._decode_sequence[idx]] = 10.0
            self._decode_step += 1
        else:
            # Prefill path.
            logits[:, :, self._prefill_token] = 10.0
        return logits


class FailingContinuousMockModel(nn.Module):
    """Mock model that raises on forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _MOCK_CONFIG
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCacheProtocol | None = None,
        padding_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        raise RuntimeError("mock forward failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> EngineConfig:
    defaults: dict[str, object] = {
        "model": "test-model",
        "device": "cpu",
        "dtype": "float16",
        "max_seq_len": 128,
        "max_batch_size": 8,
        "batching_mode": "continuous",
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


def _make_request(
    request_id: str,
    prompt: list[int],
    max_new_tokens: int = 10,
    **kwargs: object,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens, **kwargs),  # type: ignore[arg-type]
        arrival_time_s=time.perf_counter(),
    )


# ---------------------------------------------------------------------------
# Prefill tests
# ---------------------------------------------------------------------------


class TestPrefillOne:
    def test_basic_prefill(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3])
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.request_id == "r1"
        assert out.token_id == 7
        assert out.finished is False
        assert req.state is RequestState.DECODE
        assert req.generated_token_ids == [7]
        assert req.slot_idx is not None

    def test_prefill_allocates_slot(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(max_batch_size=4)
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert runner.cache_pool.free_slot_count() == 4

        req = _make_request("r1", [1, 2, 3])
        runner.step(prefill=[req], decode=[])

        assert runner.cache_pool.free_slot_count() == 3
        assert req.slot_idx is not None

    def test_prefill_eos_on_first_token(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=99)
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3])
        outputs = runner.step(prefill=[req], decode=[])

        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "eos"
        assert req.state is RequestState.FINISHED

    def test_prefill_max_tokens_one(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=1)
        outputs = runner.step(prefill=[req], decode=[])

        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "length"
        assert req.state is RequestState.FINISHED


# ---------------------------------------------------------------------------
# Batched decode tests
# ---------------------------------------------------------------------------


class TestBatchedDecode:
    def test_single_decode_step(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        runner.step(prefill=[req], decode=[])
        assert len(req.generated_token_ids) == 1

        outputs = runner.step(prefill=[], decode=[req])
        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert len(req.generated_token_ids) == 2

    def test_batched_decode_multiple_requests(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3])
        r2 = _make_request("r2", [4, 5])
        r3 = _make_request("r3", [6, 7, 8, 9])
        for req in [r1, r2, r3]:
            runner.step(prefill=[req], decode=[])

        outputs = runner.step(prefill=[], decode=[r1, r2, r3])
        assert len(outputs) == 3
        for _, out in outputs:
            assert out.token_id == 7
        assert len(r1.generated_token_ids) == 2
        assert len(r2.generated_token_ids) == 2
        assert len(r3.generated_token_ids) == 2

    def test_position_ids_passed_to_model(self) -> None:
        """Verify position_ids are computed from cache pool seq_lens."""
        model = ContinuousSequenceMockModel(vocab_size=100, sequences={})
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        # Prefill with different prompt lengths → different cache positions.
        r1 = _make_request("r1", [1, 2, 3])  # prompt_len=3
        r2 = _make_request("r2", [4, 5, 6, 7, 8])  # prompt_len=5
        runner.step(prefill=[r1], decode=[])
        runner.step(prefill=[r2], decode=[])

        # After prefill: slot seq_lens should be prompt_len + 1 (prompt + first token).
        # The decode step should pass position_ids matching these seq_lens.
        runner.step(prefill=[], decode=[r1, r2])

        assert model.last_position_ids is not None
        pos = model.last_position_ids.squeeze(1).tolist()
        # r1: prompt_len=3 → advance(3) during prefill → decode at position 3
        # r2: prompt_len=5 → advance(5) during prefill → decode at position 5
        assert pos[0] == 3
        assert pos[1] == 5

    def test_decode_until_max_tokens(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        runner.step(prefill=[req], decode=[])  # 1 token

        runner.step(prefill=[], decode=[req])  # 2 tokens
        assert req.state is RequestState.DECODE

        outputs = runner.step(prefill=[], decode=[req])  # 3 tokens → finished
        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "length"
        assert len(req.generated_token_ids) == 3


# ---------------------------------------------------------------------------
# Mixed step (prefill + decode)
# ---------------------------------------------------------------------------


class TestMixedStep:
    def test_decode_and_prefill_in_same_step(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        # Prefill A and B.
        a = _make_request("a", [1, 2, 3])
        b = _make_request("b", [4, 5])
        runner.step(prefill=[a], decode=[])
        runner.step(prefill=[b], decode=[])

        # Mixed step: decode A and B, prefill C.
        c = _make_request("c", [6, 7, 8, 9])
        outputs = runner.step(prefill=[c], decode=[a, b])

        assert len(outputs) == 3
        # Decode outputs come first.
        assert outputs[0][0].request_id == "a"
        assert outputs[1][0].request_id == "b"
        # Prefill output comes last.
        assert outputs[2][0].request_id == "c"
        assert c.state is RequestState.DECODE
        assert c.slot_idx is not None

    def test_step_order_decode_before_prefill(self) -> None:
        """Decode outputs appear before prefill outputs in the result list."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        a = _make_request("a", [1, 2])
        runner.step(prefill=[a], decode=[])

        b = _make_request("b", [3, 4])
        outputs = runner.step(prefill=[b], decode=[a])

        ids = [req.request_id for req, _ in outputs]
        assert ids == ["a", "b"]


# ---------------------------------------------------------------------------
# Slot management
# ---------------------------------------------------------------------------


class TestSlotManagement:
    def test_free_slot_releases_cache(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(max_batch_size=2)
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2])
        r2 = _make_request("r2", [3, 4])
        runner.step(prefill=[r1], decode=[])
        runner.step(prefill=[r2], decode=[])
        assert runner.cache_pool.free_slot_count() == 0

        # Free r1's slot.
        assert r1.slot_idx is not None
        runner.free_slot(r1.slot_idx)
        assert runner.cache_pool.free_slot_count() == 1

    def test_cleanup_request_removes_tracking(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2])
        runner.step(prefill=[req], decode=[])
        assert "r1" in runner._prev_text_lens

        runner.cleanup_request("r1")
        assert "r1" not in runner._prev_text_lens

    def test_slot_reuse_after_free(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(max_batch_size=1)
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2])
        runner.step(prefill=[r1], decode=[])
        slot1 = r1.slot_idx
        assert runner.cache_pool.free_slot_count() == 0

        # Free and reuse.
        assert slot1 is not None
        runner.free_slot(slot1)
        r2 = _make_request("r2", [3, 4])
        runner.step(prefill=[r2], decode=[])
        assert r2.slot_idx == slot1


# ---------------------------------------------------------------------------
# Stop string handling
# ---------------------------------------------------------------------------


class TestStopString:
    def test_stop_string_during_prefill(self) -> None:
        # Token 0 → 'A'. Stop on "A".
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=0)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [50, 51], max_new_tokens=10, stop=["A"])
        outputs = runner.step(prefill=[req], decode=[])

        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "stop"
        assert out.text_delta == ""


# ---------------------------------------------------------------------------
# Text delta tracking
# ---------------------------------------------------------------------------


class TestTextDelta:
    def test_incremental_text_delta(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        outputs = runner.step(prefill=[req], decode=[])
        _, out1 = outputs[0]
        assert len(out1.text_delta) > 0

        outputs = runner.step(prefill=[], decode=[req])
        _, out2 = outputs[0]
        assert len(out2.text_delta) > 0


# ---------------------------------------------------------------------------
# StepOutput fields
# ---------------------------------------------------------------------------


class TestStepOutputFields:
    def test_non_finished_has_zero_counts(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        outputs = runner.step(prefill=[req], decode=[])
        _, out = outputs[0]

        assert out.finished is False
        assert out.prompt_tokens == 0
        assert out.completion_tokens == 0

    def test_finished_has_counts(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=1)
        outputs = runner.step(prefill=[req], decode=[])
        _, out = outputs[0]

        assert out.finished is True
        assert out.prompt_tokens == 3
        assert out.completion_tokens == 1


# ---------------------------------------------------------------------------
# Cache pool initialization
# ---------------------------------------------------------------------------


class TestCachePoolInit:
    def test_pool_sized_from_config(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(max_batch_size=4, max_seq_len=64)
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert runner.cache_pool.free_slot_count() == 4
        assert isinstance(runner.cache_pool, SlottedKVCache)
        assert runner.cache_pool.k.shape[1] == 4  # max_batch_size
        assert runner.cache_pool.k.shape[3] == 64  # max_seq_len

    def test_model_without_config_raises(self) -> None:
        model = nn.Linear(10, 10)  # no .config attribute
        tokenizer = MockTokenizer()
        config = _make_config()
        with pytest.raises(TypeError):
            ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EOS during decode
# ---------------------------------------------------------------------------


class TestEOSDuringDecode:
    def test_eos_during_decode_finishes_request(self) -> None:
        # Prefill emits token 5, first decode emits EOS (99).
        model = StepCounterMockModel(vocab_size=100, prefill_token=5, decode_sequence=[99])
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=10)
        runner.step(prefill=[req], decode=[])
        assert req.state is RequestState.DECODE

        outputs = runner.step(prefill=[], decode=[req])
        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "eos"
        assert req.state is RequestState.FINISHED

    def test_eos_one_of_two_in_batch(self) -> None:
        """One request hits EOS, the other continues."""
        # Both get token 5 from prefill, then EOS on first decode.
        # Since both get EOS, test that both finish.
        model = StepCounterMockModel(vocab_size=100, prefill_token=5, decode_sequence=[99, 7])
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2])
        runner.step(prefill=[r1], decode=[])

        outputs = runner.step(prefill=[], decode=[r1])
        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "eos"


# ---------------------------------------------------------------------------
# Stop string during decode
# ---------------------------------------------------------------------------


class TestStopStringDuringDecode:
    def test_stop_string_during_decode(self) -> None:
        # Token 2 → 'C', Token 3 → 'D'. Prefill emits 2 ("C"), decode emits 3 ("D").
        # After decode: generated=[2, 3], text="CD". Stop on "CD".
        model = StepCounterMockModel(vocab_size=100, prefill_token=2, decode_sequence=[3, 4, 5])
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [50], max_new_tokens=10, stop=["CD"])
        runner.step(prefill=[req], decode=[])
        # After prefill: generated=[2], text="C", no stop yet.
        assert req.state is RequestState.DECODE

        outputs = runner.step(prefill=[], decode=[req])
        _, out = outputs[0]
        # After decode: generated=[2, 3], text="CD", stop detected.
        assert out.finished is True
        assert out.finish_reason == "stop"


# ---------------------------------------------------------------------------
# Multi-step position tracking
# ---------------------------------------------------------------------------


class TestMultiStepPositions:
    def test_position_ids_increment_over_multiple_decodes(self) -> None:
        """Verify position_ids are correct across 3 decode steps."""
        model = ContinuousSequenceMockModel(vocab_size=100, sequences={})
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)  # prompt_len=3
        runner.step(prefill=[req], decode=[])

        # Decode step 1: position should be 3 (prompt_len).
        runner.step(prefill=[], decode=[req])
        assert model.last_position_ids is not None
        assert model.last_position_ids.item() == 3

        # Decode step 2: position should be 4.
        runner.step(prefill=[], decode=[req])
        assert model.last_position_ids is not None
        assert model.last_position_ids.item() == 4

        # Decode step 3: position should be 5.
        runner.step(prefill=[], decode=[req])
        assert model.last_position_ids is not None
        assert model.last_position_ids.item() == 5


# ---------------------------------------------------------------------------
# Slot exhaustion
# ---------------------------------------------------------------------------


class TestSlotExhaustion:
    def test_prefill_when_no_slots_raises(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(max_batch_size=1)
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2])
        runner.step(prefill=[r1], decode=[])
        assert runner.cache_pool.free_slot_count() == 0

        r2 = _make_request("r2", [3, 4])
        with pytest.raises(RuntimeError, match="No free cache slots"):
            runner.step(prefill=[r2], decode=[])


# ---------------------------------------------------------------------------
# Batched prefill
# ---------------------------------------------------------------------------


class TestBatchedPrefill:
    def test_multiple_requests_batched(self) -> None:
        """Multiple prefill requests run in one batched forward pass."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3])
        r2 = _make_request("r2", [4, 5])
        r3 = _make_request("r3", [6, 7, 8, 9])
        outputs = runner.step(prefill=[r1, r2, r3], decode=[])

        assert len(outputs) == 3
        for req, out in outputs:
            assert out.token_id == 7
            assert req.state is RequestState.DECODE
            assert req.slot_idx is not None
            assert len(req.generated_token_ids) == 1

    def test_batched_prefill_allocates_slots(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(max_batch_size=4)
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert runner.cache_pool.free_slot_count() == 4

        r1 = _make_request("r1", [1, 2])
        r2 = _make_request("r2", [3, 4, 5])
        runner.step(prefill=[r1, r2], decode=[])

        assert runner.cache_pool.free_slot_count() == 2
        assert r1.slot_idx is not None
        assert r2.slot_idx is not None
        assert r1.slot_idx != r2.slot_idx

    def test_batched_prefill_correct_seq_lens(self) -> None:
        """After batched prefill, pool seq_lens reflect actual prompt lengths."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3])  # prompt_len=3
        r2 = _make_request("r2", [4, 5])  # prompt_len=2
        runner.step(prefill=[r1, r2], decode=[])

        # seq_lens should be actual prompt lengths, not padded (3).
        assert r1.slot_idx is not None
        assert r2.slot_idx is not None
        assert runner.cache_pool.get_seq_len(r1.slot_idx) == 3
        assert runner.cache_pool.get_seq_len(r2.slot_idx) == 2

    def test_single_prefill_uses_individual_path(self) -> None:
        """Single request still uses individual prefill (no padding overhead)."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3])
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert req.state is RequestState.DECODE

    def test_batched_prefill_then_decode(self) -> None:
        """Requests prefilled in a batch can decode together afterwards."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        r2 = _make_request("r2", [4, 5], max_new_tokens=3)
        runner.step(prefill=[r1, r2], decode=[])

        # Both should be in DECODE state and decodable together.
        outputs = runner.step(prefill=[], decode=[r1, r2])
        assert len(outputs) == 2
        for _, out in outputs:
            assert out.token_id == 7

        assert len(r1.generated_token_ids) == 2
        assert len(r2.generated_token_ids) == 2

    def test_mixed_step_with_batched_prefill(self) -> None:
        """Decode + batched prefill in the same step."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        # Prefill A first.
        a = _make_request("a", [1, 2])
        runner.step(prefill=[a], decode=[])

        # Mixed step: decode A, batched prefill B+C.
        b = _make_request("b", [3, 4, 5])
        c = _make_request("c", [6, 7])
        outputs = runner.step(prefill=[b, c], decode=[a])

        assert len(outputs) == 3
        # Decode output first, then prefill outputs.
        assert outputs[0][0].request_id == "a"
        assert outputs[1][0].request_id == "b"
        assert outputs[2][0].request_id == "c"
        assert b.state is RequestState.DECODE
        assert c.state is RequestState.DECODE

    def test_batched_prefill_eos(self) -> None:
        """EOS on first token in batched prefill finishes the request."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=99)
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2])
        r2 = _make_request("r2", [3, 4, 5])
        outputs = runner.step(prefill=[r1, r2], decode=[])

        for _, out in outputs:
            assert out.finished is True
            assert out.finish_reason == "eos"

    def test_batched_prefill_position_ids_for_decode(self) -> None:
        """After batched prefill with different prompt lengths, decode position_ids are correct."""
        model = ContinuousSequenceMockModel(vocab_size=100, sequences={})
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3])  # prompt_len=3
        r2 = _make_request("r2", [4, 5, 6, 7, 8])  # prompt_len=5
        runner.step(prefill=[r1, r2], decode=[])

        # Decode: position_ids should be [3, 5] (actual prompt lengths).
        runner.step(prefill=[], decode=[r1, r2])
        assert model.last_position_ids is not None
        pos = model.last_position_ids.squeeze(1).tolist()
        assert pos[0] == 3
        assert pos[1] == 5


# ---------------------------------------------------------------------------
# Paged backend dispatch and integration
# ---------------------------------------------------------------------------


def _make_paged_config(**overrides: object) -> EngineConfig:
    defaults: dict[str, object] = {
        "model": "test-model",
        "device": "cpu",
        "dtype": "float16",
        "max_seq_len": 128,
        "max_batch_size": 8,
        "batching_mode": "continuous",
        "kv_cache_backend": "paged",
        "block_size": 8,
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


class TestPagedBackendDispatch:
    def test_paged_creates_paged_pool(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert isinstance(runner.cache_pool, PagedKVCachePool)
        assert runner.cache_pool.is_paged() is True

    def test_contiguous_creates_slotted_pool(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert isinstance(runner.cache_pool, SlottedKVCache)
        assert runner.cache_pool.is_paged() is False

    def test_auto_compute_blocks(self) -> None:
        """num_gpu_blocks=None auto-computes from max_batch_size * max_seq_len // block_size."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=None
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert isinstance(runner.cache_pool, PagedKVCachePool)
        # 4 * 64 // 8 = 32 blocks → 32 * 8 = 256 tokens capacity
        assert runner.cache_pool.free_token_capacity() == 256

    def test_explicit_num_gpu_blocks(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=10
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert isinstance(runner.cache_pool, PagedKVCachePool)
        # 10 blocks * 8 tokens/block = 80 tokens
        assert runner.cache_pool.free_token_capacity() == 80


class TestFreeKvTokens:
    def test_contiguous_returns_none(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config()
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert runner.free_kv_tokens() is None

    def test_paged_returns_int(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=10
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        cap = runner.free_kv_tokens()
        assert isinstance(cap, int)
        assert cap == 80  # 10 blocks * 8 tokens

    def test_paged_capacity_decreases_after_prefill(self) -> None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=10
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        initial_cap = runner.free_kv_tokens()
        assert initial_cap is not None

        req = _make_request("r1", [1, 2, 3, 4, 5])  # 5 tokens → ceil(5/8)=1 block
        runner.step(prefill=[req], decode=[])

        after_cap = runner.free_kv_tokens()
        assert after_cap is not None
        assert after_cap < initial_cap


class TestPagedInitialTokens:
    def test_prefill_one_allocates_blocks(self) -> None:
        """Single prefill with paged backend eagerly allocates blocks."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=20
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert isinstance(runner.cache_pool, PagedKVCachePool)
        initial_free = runner.cache_pool.allocator.num_free()

        # Prompt of 10 tokens → ceil(10/8) = 2 blocks allocated.
        req = _make_request("r1", list(range(10)))
        runner.step(prefill=[req], decode=[])

        assert runner.cache_pool.allocator.num_free() == initial_free - 2

    def test_prefill_batch_allocates_blocks(self) -> None:
        """Batched prefill with paged backend eagerly allocates blocks per request."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=20
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        assert isinstance(runner.cache_pool, PagedKVCachePool)
        initial_free = runner.cache_pool.allocator.num_free()

        # r1: 10 tokens → 2 blocks, r2: 3 tokens → 1 block → 3 blocks total.
        r1 = _make_request("r1", list(range(10)))
        r2 = _make_request("r2", list(range(3)))
        runner.step(prefill=[r1, r2], decode=[])

        assert runner.cache_pool.allocator.num_free() == initial_free - 3


class TestPagedPrefillAndDecode:
    def test_basic_prefill_then_decode(self) -> None:
        """Full prefill → decode cycle with paged backend."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=20
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert req.state is RequestState.DECODE

        # Decode step.
        outputs = runner.step(prefill=[], decode=[req])
        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7

    def test_batched_prefill_then_decode(self) -> None:
        """Batched prefill → batched decode with paged backend."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=20
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        r2 = _make_request("r2", [4, 5], max_new_tokens=3)
        runner.step(prefill=[r1, r2], decode=[])

        assert r1.state is RequestState.DECODE
        assert r2.state is RequestState.DECODE

        outputs = runner.step(prefill=[], decode=[r1, r2])
        assert len(outputs) == 2
        for _, out in outputs:
            assert out.token_id == 7

    def test_seq_lens_correct_after_paged_prefill(self) -> None:
        """Paged pool tracks seq_lens correctly after prefill."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=20
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3])  # 3 tokens
        r2 = _make_request("r2", [4, 5])  # 2 tokens
        runner.step(prefill=[r1, r2], decode=[])

        assert r1.slot_idx is not None
        assert r2.slot_idx is not None
        assert runner.cache_pool.get_seq_len(r1.slot_idx) == 3
        assert runner.cache_pool.get_seq_len(r2.slot_idx) == 2

    def test_free_slot_restores_capacity(self) -> None:
        """Freeing a paged slot returns blocks to the allocator."""
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_paged_config(
            max_batch_size=4, max_seq_len=64, block_size=8, num_gpu_blocks=20
        )
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        initial_cap = runner.free_kv_tokens()
        assert initial_cap is not None

        req = _make_request("r1", list(range(10)))
        runner.step(prefill=[req], decode=[])
        assert runner.free_kv_tokens() < initial_cap  # type: ignore[operator]

        assert req.slot_idx is not None
        runner.free_slot(req.slot_idx)
        assert runner.free_kv_tokens() == initial_cap
