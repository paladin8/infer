"""Tests for Phase 7 chunked prefill: scheduler, runner, and engine integration."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import torch
from torch import Tensor, nn

from infer.cache.paged import PagedKVCachePool
from infer.cache.protocol import KVCacheProtocol
from infer.engine.config import EngineConfig
from infer.engine.continuous_runner import ContinuousRunner
from infer.engine.engine import Engine
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.sampler import SamplingParams
from infer.engine.scheduler import ContinuousScheduler
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Mock model / tokenizer (shared with test_continuous_runner.py patterns)
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


class ChunkedMockModel(nn.Module):
    """Mock model supporting chunked prefill.

    Always returns logits favoring ``fixed_next_token`` at every position.
    Properly advances the KV cache and returns full-sequence logits when
    padding_mask is present.
    """

    def __init__(self, vocab_size: int, fixed_next_token: int) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._fixed_next_token = fixed_next_token
        self._dummy = nn.Parameter(torch.zeros(1))
        self.last_position_ids: Tensor | None = None
        self.last_padding_mask: Tensor | None = None
        self.forward_count: int = 0

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCacheProtocol | None = None,
        padding_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        self.last_position_ids = position_ids
        self.last_padding_mask = padding_mask
        self.forward_count += 1
        batch, seq_len = input_ids.shape
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            out_len = 1 if padding_mask is None else seq_len
        else:
            out_len = seq_len
        logits = torch.zeros(batch, out_len, self.config.vocab_size)
        logits[:, :, self._fixed_next_token] = 10.0
        return logits


class EosMockModel(nn.Module):
    """Mock model that emits EOS token."""

    def __init__(self, vocab_size: int, eos_token: int) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._eos_token = eos_token
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
        logits[:, :, self._eos_token] = 10.0
        return logits


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
        "use_chunked_prefill": True,
        "prefill_chunk_size": 4,
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


def _make_paged_config(**overrides: object) -> EngineConfig:
    defaults: dict[str, object] = {
        "model": "test-model",
        "device": "cpu",
        "dtype": "float16",
        "max_seq_len": 128,
        "max_batch_size": 8,
        "batching_mode": "continuous",
        "kv_cache_backend": "paged",
        "block_size": 4,
        "use_chunked_prefill": True,
        "prefill_chunk_size": 4,
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


def _make_sched_config(**overrides: object) -> EngineConfig:
    """Config for scheduler-only tests (no model needed)."""
    defaults: dict[str, object] = {
        "model": "test-model",
        "batching_mode": "continuous",
        "use_chunked_prefill": True,
        "prefill_chunk_size": 4,
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


def _make_runner(model: nn.Module, config: EngineConfig | None = None) -> ContinuousRunner:
    tokenizer = MockTokenizer()
    if config is None:
        config = _make_config()
    return ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]


# ===========================================================================
# Scheduler tests (D6)
# ===========================================================================


class TestSchedulerPrefillRequests:
    def test_returns_waiting_requests(self) -> None:
        """WAITING requests are returned by prefill_requests()."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=4))
        a, b = _make_request("a", [1, 2, 3]), _make_request("b", [4, 5])
        for r in [a, b]:
            sched.add_request(r)
        sched.admit()

        result = sched.prefill_requests()
        assert [r.request_id for r in result] == ["a", "b"]

    def test_returns_continuing_prefill_before_new(self) -> None:
        """PREFILL requests (continuing) appear before WAITING (new)."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=4))
        a = _make_request("a", list(range(10)))
        b = _make_request("b", [1, 2])
        sched.add_request(a)
        sched.add_request(b)
        sched.admit()

        # Simulate a in mid-prefill.
        a.state = RequestState.PREFILL
        a.prefill_progress = 4  # partial

        result = sched.prefill_requests()
        assert result[0].request_id == "a"  # continuing first
        assert result[1].request_id == "b"  # new second

    def test_max_chunks_cap(self) -> None:
        """max_chunks limits the number of returned requests."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=8))
        for i in range(5):
            sched.add_request(_make_request(f"r{i}", [1, 2]))
        sched.admit()

        result = sched.prefill_requests(max_chunks=2)
        assert len(result) == 2

    def test_max_chunks_none_returns_all(self) -> None:
        """max_chunks=None returns all prefill-needing requests."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=8))
        for i in range(5):
            sched.add_request(_make_request(f"r{i}", [1, 2]))
        sched.admit()

        result = sched.prefill_requests(max_chunks=None)
        assert len(result) == 5

    def test_excludes_decode_requests(self) -> None:
        """DECODE requests are not in prefill_requests()."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=4))
        a = _make_request("a", [1, 2])
        b = _make_request("b", [3, 4])
        for r in [a, b]:
            sched.add_request(r)
        sched.admit()

        a.state = RequestState.DECODE
        result = sched.prefill_requests()
        assert [r.request_id for r in result] == ["b"]

    def test_excludes_completed_prefill(self) -> None:
        """PREFILL requests with full progress are excluded."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=4))
        a = _make_request("a", [1, 2, 3])
        sched.add_request(a)
        sched.admit()

        a.state = RequestState.PREFILL
        a.prefill_progress = 3  # == len(prompt_token_ids)
        result = sched.prefill_requests()
        assert result == []


class TestSchedulerPrefillDecodeDisjoint:
    def test_prefill_and_decode_disjoint(self) -> None:
        """prefill_requests() and decode_requests() return disjoint sets."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=4))
        a = _make_request("a", [1, 2])
        b = _make_request("b", [3, 4])
        c = _make_request("c", list(range(10)))
        for r in [a, b, c]:
            sched.add_request(r)
        sched.admit()

        a.state = RequestState.DECODE
        b.state = RequestState.DECODE
        c.state = RequestState.PREFILL
        c.prefill_progress = 4

        prefill = sched.prefill_requests()
        decode = sched.decode_requests()
        prefill_ids = {r.request_id for r in prefill}
        decode_ids = {r.request_id for r in decode}
        assert prefill_ids & decode_ids == set()
        assert prefill_ids == {"c"}
        assert decode_ids == {"a", "b"}


class TestSchedulerHasWorkDuringChunkedPrefill:
    def test_has_work_during_multi_step_prefill(self) -> None:
        """has_work() returns True while a request is mid-prefill."""
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=4))
        a = _make_request("a", list(range(10)))
        sched.add_request(a)
        sched.admit()

        a.state = RequestState.PREFILL
        a.prefill_progress = 4
        assert sched.has_work() is True

    def test_empty_after_all_retired(self) -> None:
        sched = ContinuousScheduler(_make_sched_config(max_batch_size=4))
        a = _make_request("a", [1, 2])
        sched.add_request(a)
        sched.admit()
        a.state = RequestState.FINISHED
        sched.retire()
        assert sched.has_work() is False


# ===========================================================================
# Runner tests (D7)
# ===========================================================================


class TestChunkedPrefillBasic:
    def test_single_chunk_completes_prefill(self) -> None:
        """Prompt shorter than chunk_size completes in one chunk."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=8)
        runner = _make_runner(model, config)

        req = _make_request("r1", [1, 2, 3])  # 3 < chunk_size=8
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert req.state is RequestState.DECODE
        assert req.prefill_progress == 3
        assert req.generated_token_ids == [7]

    def test_multi_chunk_prefill(self) -> None:
        """Prompt longer than chunk_size takes multiple chunks."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(10)))  # 10 tokens, chunk_size=4

        # Chunk 1: tokens [0..3], progress=4. Intermediate → no output.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 0
        assert req.state is RequestState.PREFILL
        assert req.prefill_progress == 4

        # Chunk 2: tokens [4..7], progress=8. Intermediate → no output.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 0
        assert req.prefill_progress == 8

        # Chunk 3: tokens [8..9], progress=10. Final → output.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert req.state is RequestState.DECODE
        assert req.prefill_progress == 10

    def test_prompt_len_equals_chunk_size(self) -> None:
        """Prompt exactly equal to chunk_size completes in one chunk."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", [1, 2, 3, 4])
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        assert req.state is RequestState.DECODE
        assert req.prefill_progress == 4

    def test_prompt_len_one(self) -> None:
        """Single-token prompt completes immediately."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", [42])
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        assert req.state is RequestState.DECODE
        assert req.prefill_progress == 1

    def test_one_token_final_chunk(self) -> None:
        """prompt_len % chunk_size == 1 → 1-token final chunk."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(5)))  # 5 tokens → chunks of 4 + 1

        # Chunk 1: 4 tokens → intermediate.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 0
        assert req.prefill_progress == 4

        # Chunk 2: 1 token → final.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 1
        assert req.state is RequestState.DECODE


class TestChunkedPrefillOutputSilence:
    def test_no_output_during_intermediate_chunks(self) -> None:
        """Output queue receives nothing during intermediate prefill chunks."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(10)))

        # Chunk 1: intermediate.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 0

        # Chunk 2: intermediate.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 0

        # Chunk 3: final.
        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 1


class TestChunkedPrefillSlotAllocation:
    def test_slot_allocated_on_first_chunk(self) -> None:
        """Slot is allocated only on the first chunk, not subsequent ones."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4, max_batch_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(10)))

        # Before: 4 free slots.
        assert runner.cache_pool.free_slot_count() == 4

        # Chunk 1: allocates slot.
        runner.step(prefill=[req], decode=[])
        assert runner.cache_pool.free_slot_count() == 3
        slot_after_1 = req.slot_idx

        # Chunk 2: no additional allocation.
        runner.step(prefill=[req], decode=[])
        assert runner.cache_pool.free_slot_count() == 3
        assert req.slot_idx == slot_after_1

    def test_state_transitions(self) -> None:
        """WAITING → PREFILL (first chunk) → DECODE (last chunk)."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(8)))
        assert req.state is RequestState.WAITING

        # Chunk 1.
        runner.step(prefill=[req], decode=[])
        assert req.state is RequestState.PREFILL

        # Chunk 2 (final).
        runner.step(prefill=[req], decode=[])
        assert req.state is RequestState.DECODE


class TestChunkedPrefillSeqLens:
    def test_seq_lens_after_each_chunk(self) -> None:
        """pool.seq_lens[slot] == prefill_progress after every chunk."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(10)))

        runner.step(prefill=[req], decode=[])
        assert req.slot_idx is not None
        assert runner.cache_pool.get_seq_len(req.slot_idx) == 4
        assert req.prefill_progress == 4

        runner.step(prefill=[req], decode=[])
        assert runner.cache_pool.get_seq_len(req.slot_idx) == 8
        assert req.prefill_progress == 8

        runner.step(prefill=[req], decode=[])
        assert runner.cache_pool.get_seq_len(req.slot_idx) == 10
        assert req.prefill_progress == 10

    def test_seq_lens_equals_prompt_len_after_final_chunk(self) -> None:
        """pool.seq_lens[slot] == len(prompt_token_ids) after final chunk."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(7)))

        # Chunk 1.
        runner.step(prefill=[req], decode=[])
        # Chunk 2 (final).
        runner.step(prefill=[req], decode=[])

        assert req.slot_idx is not None
        assert runner.cache_pool.get_seq_len(req.slot_idx) == 7


class TestChunkedPrefillDecodeAfter:
    def test_decode_after_chunked_prefill(self) -> None:
        """Decode works correctly after chunked prefill completes."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(8)), max_new_tokens=3)

        # Chunk 1 + 2 (complete prefill).
        runner.step(prefill=[req], decode=[])
        runner.step(prefill=[req], decode=[])
        assert req.state is RequestState.DECODE
        assert len(req.generated_token_ids) == 1

        # Decode step 1.
        outputs = runner.step(prefill=[], decode=[req])
        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert len(req.generated_token_ids) == 2

        # Decode step 2 → finishes (max_new_tokens=3).
        outputs = runner.step(prefill=[], decode=[req])
        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "length"

    def test_position_ids_correct_after_chunked_prefill(self) -> None:
        """Decode position_ids start at prompt_len after chunked prefill."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(8)), max_new_tokens=5)

        # Complete prefill: 2 chunks of 4.
        runner.step(prefill=[req], decode=[])
        runner.step(prefill=[req], decode=[])

        # Decode: position should be 8 (prompt_len).
        runner.step(prefill=[], decode=[req])
        assert model.last_position_ids is not None
        assert model.last_position_ids.item() == 8


class TestChunkedPrefillMultiTokenGeneration:
    def test_10_token_generation_after_chunked_prefill(self) -> None:
        """Generate 10+ tokens after chunked prefill, verifying KV cache state.

        Exit criterion 3: "at least 10 tokens, greedy decode" to confirm
        the cache state is fully correct across multiple decode steps.
        """
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(12)), max_new_tokens=12)

        # Chunked prefill: 3 chunks of 4.
        runner.step(prefill=[req], decode=[])
        runner.step(prefill=[req], decode=[])
        runner.step(prefill=[req], decode=[])
        assert req.state is RequestState.DECODE
        assert len(req.generated_token_ids) == 1

        # Generate 11 more tokens (total 12 = max_new_tokens).
        for i in range(11):
            outputs = runner.step(prefill=[], decode=[req])
            assert len(outputs) == 1
            _, out = outputs[0]
            assert out.token_id == 7
            if i < 10:
                assert out.finished is False
            else:
                assert out.finished is True
                assert out.finish_reason == "length"

        assert len(req.generated_token_ids) == 12

    def test_10_token_generation_paged_backend(self) -> None:
        """Same as above but with paged backend."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_paged_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(12)), max_new_tokens=12)

        # Chunked prefill: 3 chunks.
        for _ in range(3):
            runner.step(prefill=[req], decode=[])
        assert req.state is RequestState.DECODE

        # Generate 11 more tokens.
        for _ in range(11):
            runner.step(prefill=[], decode=[req])

        assert len(req.generated_token_ids) == 12
        assert req.state is RequestState.FINISHED


class TestChunkedPrefillEOS:
    def test_eos_on_last_chunk(self) -> None:
        """EOS on first token of last chunk finishes the request."""
        model = EosMockModel(vocab_size=100, eos_token=99)
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_config(prefill_chunk_size=4)
        runner = ContinuousRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", list(range(8)))

        runner.step(prefill=[req], decode=[])
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "eos"
        assert req.state is RequestState.FINISHED


class TestChunkedPrefillBatched:
    def test_batched_mixed_progress(self) -> None:
        """Multiple requests at different prefill progress in same batch."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        a = _make_request("a", list(range(8)))  # 2 chunks
        b = _make_request("b", list(range(12)))  # 3 chunks

        # Chunk 1 for both.
        outputs = runner.step(prefill=[a, b], decode=[])
        assert len(outputs) == 0
        assert a.prefill_progress == 4
        assert b.prefill_progress == 4

        # Chunk 2: a finishes, b continues.
        outputs = runner.step(prefill=[a, b], decode=[])
        assert len(outputs) == 1
        assert outputs[0][0].request_id == "a"
        assert a.state is RequestState.DECODE
        assert b.prefill_progress == 8

        # Chunk 3: b finishes.
        outputs = runner.step(prefill=[b], decode=[])
        assert len(outputs) == 1
        assert outputs[0][0].request_id == "b"
        assert b.state is RequestState.DECODE

    def test_concurrent_chunked_prefill_and_decode(self) -> None:
        """Decode runs alongside chunked prefill in the same step."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        # Prefill A fully first.
        a = _make_request("a", [1, 2, 3])  # short, completes in 1 chunk
        outputs = runner.step(prefill=[a], decode=[])
        assert a.state is RequestState.DECODE

        # Now chunked prefill B while decoding A.
        b = _make_request("b", list(range(8)))
        outputs = runner.step(prefill=[b], decode=[a])
        # Decode output for A, no output for B (intermediate chunk).
        assert len(outputs) == 1
        assert outputs[0][0].request_id == "a"
        assert b.prefill_progress == 4


class TestChunkedPrefillMaxTokensOne:
    def test_max_tokens_one(self) -> None:
        """max_new_tokens=1 with chunked prefill."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", [1, 2], max_new_tokens=1)
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.finished is True
        assert out.finish_reason == "length"


class TestChunkedPrefillPagedBackend:
    def test_paged_chunked_prefill(self) -> None:
        """Chunked prefill works with paged backend."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_paged_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        assert isinstance(runner.cache_pool, PagedKVCachePool)

        req = _make_request("r1", list(range(8)))

        runner.step(prefill=[req], decode=[])
        assert req.prefill_progress == 4

        outputs = runner.step(prefill=[req], decode=[])
        assert len(outputs) == 1
        assert req.state is RequestState.DECODE

    def test_paged_seq_lens_after_chunks(self) -> None:
        """Paged backend tracks seq_lens correctly through chunks."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_paged_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(10)))

        runner.step(prefill=[req], decode=[])
        assert req.slot_idx is not None
        assert runner.cache_pool.get_seq_len(req.slot_idx) == 4

        runner.step(prefill=[req], decode=[])
        assert runner.cache_pool.get_seq_len(req.slot_idx) == 8

        runner.step(prefill=[req], decode=[])
        assert runner.cache_pool.get_seq_len(req.slot_idx) == 10

    def test_paged_short_prompt_single_chunk(self) -> None:
        """Paged backend with chunk_size > prompt_len completes in one chunk."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_paged_config(prefill_chunk_size=8)  # chunk_size=8 > prompt_len=2
        runner = _make_runner(model, config)

        assert isinstance(runner.cache_pool, PagedKVCachePool)

        req = _make_request("r1", [1, 2])
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert req.state is RequestState.DECODE
        assert req.prefill_progress == 2

    def test_paged_decode_after_chunked_prefill(self) -> None:
        """Decode works after chunked prefill with paged backend."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_paged_config(prefill_chunk_size=4)
        runner = _make_runner(model, config)

        req = _make_request("r1", list(range(8)), max_new_tokens=3)

        runner.step(prefill=[req], decode=[])
        runner.step(prefill=[req], decode=[])
        assert req.state is RequestState.DECODE

        outputs = runner.step(prefill=[], decode=[req])
        assert len(outputs) == 1
        assert outputs[0][1].token_id == 7


class TestChunkedPrefillDisabled:
    def test_disabled_uses_old_path(self) -> None:
        """use_chunked_prefill=False uses the old prefill path."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        config = _make_config(use_chunked_prefill=False)
        runner = _make_runner(model, config)

        req = _make_request("r1", [1, 2, 3])
        outputs = runner.step(prefill=[req], decode=[])

        assert len(outputs) == 1
        _, out = outputs[0]
        assert out.token_id == 7
        assert req.state is RequestState.DECODE
        # Old path: no prefill_progress tracking.
        assert req.prefill_progress == 0


# ===========================================================================
# Engine integration tests (D8)
# ===========================================================================


class TestEngineChunkedPrefill:
    def test_engine_chunked_prefill_completes(self) -> None:
        """Engine drives chunked prefill to completion over multiple steps."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(
            prefill_chunk_size=4,
            max_batch_size=4,
        )
        engine = Engine.from_components(config, model, tokenizer)

        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        engine.add_request(
            "r1", list(range(10)), SamplingParams(temperature=0.0, max_new_tokens=3), queue
        )

        # Step 1: admits request, processes chunk 1.
        engine.step()
        assert queue.empty()  # intermediate chunk → no output

        # Step 2: chunk 2.
        engine.step()
        assert queue.empty()

        # Step 3: chunk 3 (final) → output.
        engine.step()
        assert not queue.empty()
        out = queue.get_nowait()
        assert out.token_id == 7
        assert out.finished is False

    def test_engine_concurrent_prefill_and_decode(self) -> None:
        """Decode gets output every step while another request is chunked-prefilling."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(
            prefill_chunk_size=4,
            max_batch_size=4,
        )
        engine = Engine.from_components(config, model, tokenizer)

        q1: asyncio.Queue[StepOutput] = asyncio.Queue()
        q2: asyncio.Queue[StepOutput] = asyncio.Queue()

        # Short prompt: completes prefill in 1 chunk.
        engine.add_request("r1", [1, 2, 3], SamplingParams(temperature=0.0, max_new_tokens=5), q1)
        engine.step()  # r1 prefill completes
        out1 = q1.get_nowait()
        assert out1.token_id == 7

        # Long prompt: needs multiple chunks.
        engine.add_request(
            "r2", list(range(10)), SamplingParams(temperature=0.0, max_new_tokens=5), q2
        )

        # Next steps: r1 decodes, r2 chunked-prefills.
        engine.step()  # r1 decode, r2 chunk 1
        assert not q1.empty()  # r1 got decode output
        assert q2.empty()  # r2 intermediate, no output

        engine.step()  # r1 decode, r2 chunk 2
        assert not q1.empty()
        assert q2.empty()

        engine.step()  # r1 decode, r2 chunk 3 (final)
        assert not q1.empty()
        assert not q2.empty()
        out2 = q2.get_nowait()
        assert out2.token_id == 7

    def test_engine_max_prefill_chunks_per_step(self) -> None:
        """max_prefill_chunks_per_step limits chunks per step."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(
            prefill_chunk_size=4,
            max_batch_size=8,
            max_prefill_chunks_per_step=2,
        )
        engine = Engine.from_components(config, model, tokenizer)

        queues = []
        for i in range(4):
            q: asyncio.Queue[StepOutput] = asyncio.Queue()
            queues.append(q)
            engine.add_request(
                f"r{i}", [1, 2, 3], SamplingParams(temperature=0.0, max_new_tokens=5), q
            )

        # Step 1: only 2 of the 4 get chunked (due to cap).
        engine.step()
        # Count how many queues got output (short prompts complete in 1 chunk).
        outputs_received = sum(1 for q in queues if not q.empty())
        assert outputs_received == 2

    def test_engine_request_arriving_mid_chunk(self) -> None:
        """Request arriving while another is mid-prefill gets batched."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(
            prefill_chunk_size=4,
            max_batch_size=4,
        )
        engine = Engine.from_components(config, model, tokenizer)

        q1: asyncio.Queue[StepOutput] = asyncio.Queue()
        q2: asyncio.Queue[StepOutput] = asyncio.Queue()

        # Start long-prompt request.
        engine.add_request(
            "r1", list(range(10)), SamplingParams(temperature=0.0, max_new_tokens=5), q1
        )
        engine.step()  # r1 chunk 1
        assert q1.empty()

        # Add second request while r1 is mid-prefill.
        engine.add_request("r2", [1, 2, 3], SamplingParams(temperature=0.0, max_new_tokens=5), q2)
        engine.step()  # r1 chunk 2, r2 chunk 1 (completes)
        assert q1.empty()  # r1 still mid-prefill
        assert not q2.empty()  # r2 completed

    def test_engine_disabled_chunked_prefill_regression(self) -> None:
        """use_chunked_prefill=False produces same results as before."""
        model = ChunkedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_config(
            use_chunked_prefill=False,
            max_batch_size=4,
        )
        engine = Engine.from_components(config, model, tokenizer)

        q: asyncio.Queue[StepOutput] = asyncio.Queue()
        engine.add_request("r1", [1, 2, 3], SamplingParams(temperature=0.0, max_new_tokens=3), q)

        engine.step()  # prefill (old path) → first token
        assert not q.empty()
        out = q.get_nowait()
        assert out.token_id == 7

    def test_engine_chunked_failure_cleanup(self) -> None:
        """Exception during chunked prefill marks requests as FAILED."""

        class FailOnChunkModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = _MOCK_CONFIG
                self._dummy = nn.Parameter(torch.zeros(1))
                self.call_count = 0

            def forward(
                self,
                input_ids: Tensor,
                kv_cache: KVCacheProtocol | None = None,
                padding_mask: Tensor | None = None,
                position_ids: Tensor | None = None,
            ) -> Tensor:
                self.call_count += 1
                if self.call_count >= 2:
                    raise RuntimeError("chunk failure")
                batch, seq_len = input_ids.shape
                if kv_cache is not None:
                    kv_cache.advance(seq_len)
                    out_len = 1 if padding_mask is None else seq_len
                else:
                    out_len = seq_len
                logits = torch.zeros(batch, out_len, self.config.vocab_size)
                logits[:, :, 7] = 10.0
                return logits

        model = FailOnChunkModel()
        tokenizer = MockTokenizer()
        config = _make_config(prefill_chunk_size=4, max_batch_size=4)
        engine = Engine.from_components(config, model, tokenizer)

        q: asyncio.Queue[StepOutput] = asyncio.Queue()
        engine.add_request(
            "r1", list(range(10)), SamplingParams(temperature=0.0, max_new_tokens=5), q
        )

        # Step 1: chunk 1 succeeds.
        engine.step()
        assert q.empty()

        # Step 2: chunk 2 fails.
        engine.step()
        assert not q.empty()
        out = q.get_nowait()
        assert out.finished is True
        assert out.error is not None
        assert "chunk failure" in out.error

        # Step 3: retire cycle frees the slot — verify no block leak.
        assert isinstance(engine.runner, ContinuousRunner)
        initial_free = engine.runner.cache_pool.free_slot_count()
        engine.step()  # triggers retire → free_slot
        assert engine.runner.cache_pool.free_slot_count() == initial_free + 1
