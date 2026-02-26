"""Unit tests for the request data model."""

from __future__ import annotations

import asyncio

import torch

from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.sampler import SamplingParams


class TestRequestState:
    def test_initial_state_is_waiting(self) -> None:
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        assert req.state is RequestState.WAITING

    def test_field_defaults(self) -> None:
        req = Request(
            request_id="r1",
            prompt_token_ids=[10, 20],
            sampling_params=SamplingParams(max_new_tokens=64),
            arrival_time_s=1.0,
        )
        assert req.generated_token_ids == []
        assert req.finish_reason is None
        assert req.error is None
        assert req.generator is None
        assert req.output_queue is None

    def test_generated_token_ids_independent(self) -> None:
        """Each request gets its own list (no shared default mutable)."""
        r1 = Request(
            request_id="r1",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        r2 = Request(
            request_id="r2",
            prompt_token_ids=[2],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        r1.generated_token_ids.append(99)
        assert r2.generated_token_ids == []

    def test_state_transitions(self) -> None:
        req = Request(
            request_id="r1",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        assert req.state is RequestState.WAITING

        req.state = RequestState.PREFILL
        assert req.state is RequestState.PREFILL

        req.state = RequestState.DECODE
        assert req.state is RequestState.DECODE

        req.state = RequestState.FINISHED
        assert req.state is RequestState.FINISHED

    def test_failed_transition(self) -> None:
        req = Request(
            request_id="r1",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        req.state = RequestState.FAILED
        req.error = "GPU OOM"
        assert req.state is RequestState.FAILED
        assert req.error == "GPU OOM"

    def test_generator_creation_from_seed(self) -> None:
        req = Request(
            request_id="r1",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(seed=42),
            arrival_time_s=0.0,
        )
        # Simulate what the engine does at enqueue time.
        req.generator = torch.Generator(device="cpu")
        req.generator.manual_seed(42)
        assert req.generator is not None

    def test_generator_determinism(self) -> None:
        """Two generators with the same seed produce identical random values."""
        g1 = torch.Generator(device="cpu")
        g1.manual_seed(42)
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(42)

        vals1 = torch.randn(5, generator=g1)
        vals2 = torch.randn(5, generator=g2)
        assert torch.equal(vals1, vals2)

    def test_output_queue_assignment(self) -> None:
        req = Request(
            request_id="r1",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        req.output_queue = queue
        assert req.output_queue is queue

    def test_prefill_progress_default(self) -> None:
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        assert req.prefill_progress == 0

    def test_prefill_progress_increments(self) -> None:
        """Simulate chunked prefill: progress increments each step."""
        req = Request(
            request_id="r1",
            prompt_token_ids=list(range(10)),
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        chunk_size = 4
        # First chunk: 0..3
        req.prefill_progress = chunk_size
        assert req.prefill_progress == 4

        # Second chunk: 4..7
        req.prefill_progress += chunk_size
        assert req.prefill_progress == 8

        # Final chunk: 8..9
        req.prefill_progress = len(req.prompt_token_ids)
        assert req.prefill_progress == 10


class TestStepOutput:
    def test_basic_construction(self) -> None:
        out = StepOutput(
            request_id="r1",
            token_id=42,
            text_delta="hello",
            finished=False,
        )
        assert out.request_id == "r1"
        assert out.token_id == 42
        assert out.text_delta == "hello"
        assert out.finished is False
        assert out.finish_reason is None
        assert out.error is None
        assert out.prompt_tokens == 0
        assert out.completion_tokens == 0

    def test_finished_with_reason(self) -> None:
        out = StepOutput(
            request_id="r1",
            token_id=2,
            text_delta="",
            finished=True,
            finish_reason="eos",
            prompt_tokens=10,
            completion_tokens=50,
        )
        assert out.finished is True
        assert out.finish_reason == "eos"
        assert out.prompt_tokens == 10
        assert out.completion_tokens == 50

    def test_error_output(self) -> None:
        out = StepOutput(
            request_id="r1",
            token_id=None,
            text_delta="",
            finished=True,
            error="KV cache overflow",
        )
        assert out.error == "KV cache overflow"
        assert out.token_id is None
