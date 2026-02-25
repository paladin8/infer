"""Unit tests for the Engine class."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
import torch
from torch import Tensor, nn

from infer.cache.protocol import KVCacheProtocol
from infer.cache.simple import KVCache
from infer.engine.config import EngineConfig
from infer.engine.continuous_runner import ContinuousRunner
from infer.engine.engine import Engine
from infer.engine.request import RequestState, StepOutput
from infer.engine.runner import ModelRunner
from infer.engine.sampler import SamplingParams
from infer.engine.scheduler import ContinuousScheduler, StaticScheduler
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors test_runner.py patterns)
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

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 100 for c in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ids = token_ids
        if skip_special_tokens:
            ids = [t for t in ids if t not in self._eos_token_ids]
        return "".join(chr(ord("A") + (t % 26)) for t in ids)


class BatchedMockModel(nn.Module):
    """Mock model that always returns logits favoring a fixed token."""

    def __init__(self, vocab_size: int, fixed_next_token: int) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._fixed_next_token = fixed_next_token
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        padding_mask: Tensor | None = None,
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


class FailingMockModel(nn.Module):
    """Mock model that raises on forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _MOCK_CONFIG
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        raise RuntimeError("mock forward failure")


class BatchedSequenceMockModel(nn.Module):
    """Mock model that emits tokens from a per-batch-slot sequence."""

    def __init__(self, vocab_size: int, sequences: list[list[int]]) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._sequences = sequences
        self._steps: list[int] = [0] * len(sequences)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            out_len = 1 if padding_mask is None else seq_len
        else:
            out_len = seq_len
        logits = torch.zeros(batch, out_len, self.config.vocab_size)
        for i in range(batch):
            if i < len(self._sequences):
                idx = min(self._steps[i], len(self._sequences[i]) - 1)
                logits[i, :, self._sequences[i][idx]] = 10.0
                self._steps[i] += 1
        return logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_config(**overrides: object) -> EngineConfig:
    defaults: dict[str, object] = {
        "model": "test-model",
        "device": "cpu",
        "dtype": "float16",
        "max_seq_len": 128,
        "max_batch_size": 8,
        "batch_wait_timeout_s": 0.0,  # dispatch immediately
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


def _make_engine(
    model: nn.Module | None = None,
    tokenizer: MockTokenizer | None = None,
    **config_overrides: object,
) -> Engine:
    if model is None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
    if tokenizer is None:
        tokenizer = MockTokenizer()
    config = _make_engine_config(**config_overrides)
    return Engine.from_components(config, model, tokenizer)


# ---------------------------------------------------------------------------
# TestAddRequest
# ---------------------------------------------------------------------------


class TestAddRequest:
    def test_string_prompt_tokenized(self) -> None:
        engine = _make_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        result = engine.add_request("r1", "hello", params, queue)
        assert result is True
        # Scheduler should have a waiting request.
        assert engine.scheduler.has_work()
        # Check prompt was tokenized (not left as string).
        req = engine.scheduler.waiting[0]
        assert isinstance(req.prompt_token_ids, list)
        assert all(isinstance(t, int) for t in req.prompt_token_ids)

    def test_token_id_passthrough(self) -> None:
        engine = _make_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        engine.add_request("r1", [1, 2, 3], params, queue)
        req = engine.scheduler.waiting[0]
        assert req.prompt_token_ids == [1, 2, 3]

    def test_prompt_too_long_raises(self) -> None:
        engine = _make_engine(max_seq_len=10)
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            engine.add_request("r1", [1, 2, 3, 4, 5, 6], params, queue)

    def test_queue_full_returns_false(self) -> None:
        engine = _make_engine(max_waiting_requests=1)
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        assert engine.add_request("r1", [1, 2], params, queue) is True
        assert engine.add_request("r2", [3, 4], params, queue) is False

    def test_seed_creates_generator(self) -> None:
        engine = _make_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=1.0, max_new_tokens=5, seed=42)
        engine.add_request("r1", [1, 2, 3], params, queue)
        req = engine.scheduler.waiting[0]
        assert req.generator is not None

    def test_no_seed_no_generator(self) -> None:
        engine = _make_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=1.0, max_new_tokens=5)
        engine.add_request("r1", [1, 2, 3], params, queue)
        req = engine.scheduler.waiting[0]
        assert req.generator is None

    def test_global_seed_fallback(self) -> None:
        engine = _make_engine(seed=123)
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        # No per-request seed, but global seed is set.
        params = SamplingParams(temperature=1.0, max_new_tokens=5)
        engine.add_request("r1", [1, 2, 3], params, queue)
        req = engine.scheduler.waiting[0]
        assert req.generator is not None


# ---------------------------------------------------------------------------
# TestStep
# ---------------------------------------------------------------------------


class TestStep:
    def test_no_work_noop(self) -> None:
        engine = _make_engine()
        # step() with nothing in queue should be a no-op.
        engine.step()

    def test_prefill_then_decode(self) -> None:
        engine = _make_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        engine.add_request("r1", [1, 2, 3], params, queue)

        # First step: prefill.
        engine.step()
        out = queue.get_nowait()
        assert out.token_id == 7
        assert out.finished is False

        # Second step: decode.
        engine.step()
        out = queue.get_nowait()
        assert out.token_id == 7

    def test_completion_clears_batch(self) -> None:
        engine = _make_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=1)
        engine.add_request("r1", [1, 2, 3], params, queue)

        # Prefill: finishes immediately (max_new_tokens=1).
        engine.step()
        out = queue.get_nowait()
        assert out.finished is True

        # Runner batch state should be cleared.
        assert isinstance(engine.runner, ModelRunner)
        assert engine.runner._kv_cache is None

    def test_output_queue_delivery(self) -> None:
        engine = _make_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=2)
        engine.add_request("r1", [1, 2, 3], params, queue)

        engine.step()  # prefill
        engine.step()  # decode (finishes)

        outputs: list[StepOutput] = []
        while not queue.empty():
            outputs.append(queue.get_nowait())
        assert len(outputs) == 2
        assert outputs[-1].finished is True

    def test_multiple_requests_in_batch(self) -> None:
        model = BatchedSequenceMockModel(
            vocab_size=100,
            sequences=[
                [5, 5],  # r1
                [7, 7],  # r2
            ],
        )
        engine = _make_engine(model=model, max_batch_size=2)
        q1: asyncio.Queue[StepOutput] = asyncio.Queue()
        q2: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=2)
        engine.add_request("r1", [1, 2], params, q1)
        engine.add_request("r2", [3, 4], params, q2)

        # Prefill both.
        engine.step()
        out1 = q1.get_nowait()
        out2 = q2.get_nowait()
        assert out1.request_id == "r1"
        assert out2.request_id == "r2"


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_forward_exception_fails_all(self) -> None:
        model = FailingMockModel()
        engine = _make_engine(model=model)
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        engine.add_request("r1", [1, 2, 3], params, queue)

        engine.step()

        out = queue.get_nowait()
        assert out.finished is True
        assert out.error is not None
        assert "mock forward failure" in out.error

        # Batch should be cleared.
        assert isinstance(engine.runner, ModelRunner)
        assert engine.runner._kv_cache is None

        # Request should be FAILED.
        assert isinstance(engine.scheduler, StaticScheduler)
        req = engine.scheduler.active[0]
        assert req.state is RequestState.FAILED


# ---------------------------------------------------------------------------
# TestHasWork
# ---------------------------------------------------------------------------


class TestHasWork:
    def test_delegates_to_scheduler(self) -> None:
        engine = _make_engine()
        assert engine.has_work() is False

        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        engine.add_request("r1", [1, 2, 3], params, queue)
        assert engine.has_work() is True


# ===========================================================================
# Continuous batching engine tests
# ===========================================================================


class ContinuousMockModel(nn.Module):
    """Mock model that accepts position_ids for continuous batching."""

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


class FailingContinuousMockModel(nn.Module):
    """Mock model that raises on forward pass (continuous batching compatible)."""

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


def _make_continuous_engine(
    model: nn.Module | None = None,
    tokenizer: MockTokenizer | None = None,
    **config_overrides: object,
) -> Engine:
    if model is None:
        model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
    if tokenizer is None:
        tokenizer = MockTokenizer()
    config = _make_engine_config(batching_mode="continuous", **config_overrides)
    return Engine.from_components(config, model, tokenizer)


class TestContinuousEngineInit:
    def test_creates_continuous_scheduler_and_runner(self) -> None:
        engine = _make_continuous_engine()
        assert isinstance(engine.scheduler, ContinuousScheduler)
        assert isinstance(engine.runner, ContinuousRunner)

    def test_static_mode_creates_static_components(self) -> None:
        engine = _make_engine()
        assert isinstance(engine.scheduler, StaticScheduler)
        assert isinstance(engine.runner, ModelRunner)


class TestContinuousStep:
    def test_no_work_noop(self) -> None:
        engine = _make_continuous_engine()
        engine.step()  # should not raise

    def test_prefill_then_decode(self) -> None:
        engine = _make_continuous_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        engine.add_request("r1", [1, 2, 3], params, queue)

        # Step 1: schedule admits request, runner prefills it.
        engine.step()
        out = queue.get_nowait()
        assert out.token_id == 7
        assert out.finished is False

        # Step 2: request is now in DECODE state, runner decodes.
        engine.step()
        out = queue.get_nowait()
        assert out.token_id == 7

    def test_completion_frees_slot(self) -> None:
        engine = _make_continuous_engine(max_batch_size=2)
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=1)
        engine.add_request("r1", [1, 2, 3], params, queue)

        # Prefill: finishes immediately (max_new_tokens=1).
        engine.step()
        out = queue.get_nowait()
        assert out.finished is True

        # Next step: scheduler retires the request and frees the slot.
        engine.step()
        assert isinstance(engine.runner, ContinuousRunner)
        assert engine.runner.cache_pool.free_slot_count() == 2

    def test_output_queue_delivery(self) -> None:
        engine = _make_continuous_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=2)
        engine.add_request("r1", [1, 2, 3], params, queue)

        engine.step()  # prefill (1 token)
        engine.step()  # decode (2 tokens → finished)

        outputs: list[StepOutput] = []
        while not queue.empty():
            outputs.append(queue.get_nowait())
        assert len(outputs) == 2
        assert outputs[-1].finished is True

    def test_retire_and_admit_new_request(self) -> None:
        """When a request finishes, its slot is freed and a new request can be admitted."""
        engine = _make_continuous_engine(max_batch_size=1)
        q1: asyncio.Queue[StepOutput] = asyncio.Queue()
        q2: asyncio.Queue[StepOutput] = asyncio.Queue()
        params1 = SamplingParams(temperature=0.0, max_new_tokens=1)
        params2 = SamplingParams(temperature=0.0, max_new_tokens=1)

        engine.add_request("r1", [1, 2], params1, q1)
        engine.step()  # prefill r1 → finishes (max_new_tokens=1)

        out1 = q1.get_nowait()
        assert out1.finished is True

        # Add r2 while r1 is finished but not yet retired.
        engine.add_request("r2", [3, 4], params2, q2)

        # Next step: retires r1 (frees slot), admits and prefills r2.
        engine.step()
        out2 = q2.get_nowait()
        assert out2.request_id == "r2"
        assert out2.finished is True

    def test_has_work_continuous(self) -> None:
        engine = _make_continuous_engine()
        assert engine.has_work() is False

        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=1)
        engine.add_request("r1", [1, 2, 3], params, queue)
        assert engine.has_work() is True

        engine.step()  # prefill → finishes
        engine.step()  # retire
        assert engine.has_work() is False


class TestContinuousErrorHandling:
    def test_forward_exception_fails_all(self) -> None:
        model = FailingContinuousMockModel()
        engine = _make_continuous_engine(model=model)
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        engine.add_request("r1", [1, 2, 3], params, queue)

        engine.step()

        out = queue.get_nowait()
        assert out.finished is True
        assert out.error is not None
        assert "mock forward failure" in out.error

        # Request should be FAILED.
        assert isinstance(engine.scheduler, ContinuousScheduler)
        assert engine.scheduler.active[0].state is RequestState.FAILED

    def test_mixed_prefill_decode_error_fails_all(self) -> None:
        """When a forward pass fails with both prefill and decode requests,
        all requests are marked FAILED and get error output."""

        class FailOnSecondCallModel(nn.Module):
            """Succeeds on first forward (prefill r1), fails on second (which
            includes both decode r1 and prefill r2)."""

            def __init__(self) -> None:
                super().__init__()
                self.config = _MOCK_CONFIG
                self._dummy = nn.Parameter(torch.zeros(1))
                self._call_count = 0

            def forward(
                self,
                input_ids: Tensor,
                kv_cache: KVCacheProtocol | None = None,
                padding_mask: Tensor | None = None,
                position_ids: Tensor | None = None,
            ) -> Tensor:
                self._call_count += 1
                if self._call_count >= 3:
                    raise RuntimeError("mock failure on mixed step")
                batch, seq_len = input_ids.shape
                if kv_cache is not None:
                    kv_cache.advance(seq_len)
                    out_len = 1 if padding_mask is None else seq_len
                else:
                    out_len = seq_len
                logits = torch.zeros(batch, out_len, self.config.vocab_size)
                logits[:, :, 7] = 10.0
                return logits

        model = FailOnSecondCallModel()
        engine = _make_continuous_engine(model=model, max_batch_size=4)
        q1: asyncio.Queue[StepOutput] = asyncio.Queue()
        q2: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)

        engine.add_request("r1", [1, 2, 3], params, q1)
        engine.step()  # prefill r1 (call 1 succeeds)
        out1 = q1.get_nowait()
        assert out1.finished is False

        # Add r2 so next step has decode(r1) + prefill(r2).
        engine.add_request("r2", [4, 5], params, q2)
        engine.step()  # decode r1 (call 2), prefill r2 (call 3 fails)

        # Both should get error output.
        all_outputs: list[StepOutput] = []
        while not q1.empty():
            all_outputs.append(q1.get_nowait())
        while not q2.empty():
            all_outputs.append(q2.get_nowait())

        error_outputs = [o for o in all_outputs if o.error is not None]
        assert len(error_outputs) >= 1
        assert "mock failure on mixed step" in error_outputs[0].error  # type: ignore[operator]

        # Both requests should be FAILED.
        assert isinstance(engine.scheduler, ContinuousScheduler)
        for req in engine.scheduler.active:
            assert req.state is RequestState.FAILED


class TestContinuousPagedEngine:
    """Engine integration tests with paged KV cache backend."""

    def _make_paged_engine(
        self,
        model: nn.Module | None = None,
        tokenizer: MockTokenizer | None = None,
        **config_overrides: object,
    ) -> Engine:
        if model is None:
            model = ContinuousMockModel(vocab_size=100, fixed_next_token=7)
        if tokenizer is None:
            tokenizer = MockTokenizer()
        defaults: dict[str, object] = {
            "kv_cache_backend": "paged",
            "block_size": 8,
            "num_gpu_blocks": 20,
        }
        defaults.update(config_overrides)
        config = _make_engine_config(batching_mode="continuous", **defaults)
        return Engine.from_components(config, model, tokenizer)

    def test_paged_prefill_then_decode(self) -> None:
        engine = self._make_paged_engine()
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        engine.add_request("r1", [1, 2, 3], params, queue)

        engine.step()  # prefill
        out = queue.get_nowait()
        assert out.token_id == 7
        assert out.finished is False

        engine.step()  # decode
        out = queue.get_nowait()
        assert out.token_id == 7

    def test_paged_completion_frees_blocks(self) -> None:
        engine = self._make_paged_engine(num_gpu_blocks=20)
        assert isinstance(engine.runner, ContinuousRunner)
        initial_cap = engine.runner.free_kv_tokens()
        assert initial_cap is not None

        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=1)
        engine.add_request("r1", [1, 2, 3], params, queue)

        engine.step()  # prefill → finishes
        out = queue.get_nowait()
        assert out.finished is True

        # Blocks should be allocated (capacity decreased).
        mid_cap = engine.runner.free_kv_tokens()
        assert mid_cap is not None
        assert mid_cap < initial_cap

        engine.step()  # retire → free blocks
        after_cap = engine.runner.free_kv_tokens()
        assert after_cap == initial_cap

    def test_paged_retire_admit_cycle(self) -> None:
        """Paged backend: retire frees blocks, enabling admission of new request."""
        engine = self._make_paged_engine(max_batch_size=1, num_gpu_blocks=4, block_size=8)
        q1: asyncio.Queue[StepOutput] = asyncio.Queue()
        q2: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=1)

        engine.add_request("r1", [1, 2], params, q1)
        engine.step()  # prefill r1 → finishes
        assert q1.get_nowait().finished is True

        engine.add_request("r2", [3, 4], params, q2)
        engine.step()  # retire r1, admit r2, prefill r2
        out2 = q2.get_nowait()
        assert out2.request_id == "r2"
        assert out2.finished is True

    def test_paged_multiple_retire_frees_slots(self) -> None:
        engine = self._make_paged_engine(max_batch_size=2, num_gpu_blocks=20)
        q1: asyncio.Queue[StepOutput] = asyncio.Queue()
        q2: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=1)

        engine.add_request("r1", [1, 2], params, q1)
        engine.add_request("r2", [3, 4], params, q2)
        engine.step()  # prefill both → both finish

        assert q1.get_nowait().finished is True
        assert q2.get_nowait().finished is True

        assert isinstance(engine.runner, ContinuousRunner)
        initial_cap = engine.runner.free_kv_tokens()

        engine.step()  # retire both → free all blocks
        after_cap = engine.runner.free_kv_tokens()
        assert after_cap is not None
        assert initial_cap is not None
        assert after_cap > initial_cap
