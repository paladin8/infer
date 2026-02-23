"""Unit tests for the Engine class."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
import torch
from torch import Tensor, nn

from infer.cache.simple import KVCache
from infer.engine.config import EngineConfig
from infer.engine.engine import Engine
from infer.engine.request import RequestState, StepOutput
from infer.engine.sampler import SamplingParams
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
        assert engine.runner._kv_cache is None

        # Request should be FAILED.
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
