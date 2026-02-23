"""Integration tests for the FastAPI server and SSE endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import AsyncGenerator

import httpx
import pytest
import torch
from torch import Tensor, nn

from infer.cache.simple import KVCache
from infer.engine.config import EngineConfig
from infer.engine.engine import Engine
from infer.engine.request import StepOutput
from infer.engine.sampler import SamplingParams
from infer.loader.config import ModelConfig
from infer.server.api import create_app_with_engine, engine_loop

# ---------------------------------------------------------------------------
# Mock infrastructure
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    model: nn.Module | None = None,
    tokenizer: MockTokenizer | None = None,
    **overrides: object,
) -> Engine:
    defaults: dict[str, object] = {
        "model": "test-model",
        "device": "cpu",
        "dtype": "float16",
        "max_seq_len": 128,
        "max_batch_size": 8,
        "batch_wait_timeout_s": 0.0,
    }
    defaults.update(overrides)
    config = EngineConfig(**defaults)  # type: ignore[arg-type]
    if model is None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
    if tokenizer is None:
        tokenizer = MockTokenizer()
    return Engine.from_components(config, model, tokenizer)


def _parse_sse_events(text: str) -> list[dict[str, str]]:
    """Parse SSE text into a list of {event, data} dicts."""
    events: list[dict[str, str]] = []
    current_event = ""
    current_data = ""
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:") :].strip()
        elif line == "" and (current_event or current_data):
            events.append({"event": current_event, "data": current_data})
            current_event = ""
            current_data = ""
    # Capture final event if no trailing blank line.
    if current_event or current_data:
        events.append({"event": current_event, "data": current_data})
    return events


async def _make_client(
    engine: Engine,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Create an httpx client with a running engine loop."""
    app = create_app_with_engine(engine)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    loop_task = asyncio.create_task(engine_loop(engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    loop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await loop_task


# ---------------------------------------------------------------------------
# TestValidRequest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestValidRequest:
    async def test_sse_stream_tokens_and_done(self) -> None:
        engine = _make_engine()
        async for client in _make_client(engine):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "hi",
                    "max_tokens": 3,
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 200

            events = _parse_sse_events(resp.text)
            token_events = [e for e in events if e["event"] == "token"]
            done_events = [e for e in events if e["event"] == "done"]

            assert len(token_events) >= 1
            assert len(done_events) == 1

            # Verify token event structure.
            token_data = json.loads(token_events[0]["data"])
            assert "token" in token_data
            assert "token_id" in token_data

    async def test_string_prompt(self) -> None:
        engine = _make_engine()
        async for client in _make_client(engine):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "hello world",
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 200

    async def test_token_id_prompt(self) -> None:
        engine = _make_engine()
        async for client in _make_client(engine):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": [1, 2, 3],
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 200

    async def test_done_event_has_usage(self) -> None:
        engine = _make_engine()
        async for client in _make_client(engine):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": [1, 2, 3],
                    "max_tokens": 2,
                    "temperature": 0.0,
                },
            )
            events = _parse_sse_events(resp.text)
            done_events = [e for e in events if e["event"] == "done"]
            assert len(done_events) == 1
            done_data = json.loads(done_events[0]["data"])
            assert "usage" in done_data
            assert done_data["usage"]["prompt_tokens"] == 3
            assert done_data["usage"]["completion_tokens"] == 2
            assert done_data["usage"]["total_tokens"] == 5


# ---------------------------------------------------------------------------
# TestSSEContract — verify SSE event format matches the design doc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSSEContract:
    async def test_final_token_emitted_before_done(self) -> None:
        """When the last step finishes with text_delta, a token event precedes the done event."""
        engine = _make_engine()
        async for client in _make_client(engine):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": [1, 2, 3],
                    "max_tokens": 3,
                    "temperature": 0.0,
                },
            )
            events = _parse_sse_events(resp.text)
            token_events = [e for e in events if e["event"] == "token"]
            done_events = [e for e in events if e["event"] == "done"]

            # The done event should be last.
            assert done_events
            done_idx = events.index(done_events[0])
            # There should be a token event right before done (the final token).
            assert done_idx > 0
            assert events[done_idx - 1]["event"] == "token"

            # Concatenated token text should be non-empty.
            full_text = "".join(json.loads(e["data"])["token"] for e in token_events)
            assert len(full_text) > 0

            # All token events carry the "token" key (not "text_delta").
            for ev in token_events:
                data = json.loads(ev["data"])
                assert "token" in data
                assert "text_delta" not in data

    async def test_error_event_is_json(self) -> None:
        """A forward-pass failure emits an SSE error event with JSON data."""
        model = FailingMockModel()
        engine = _make_engine(model=model)
        async for client in _make_client(engine):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": [1, 2, 3],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 200  # SSE stream starts before error occurs
            events = _parse_sse_events(resp.text)
            error_events = [e for e in events if e["event"] == "error"]
            assert len(error_events) == 1

            # Data must be valid JSON with an "error" key.
            error_data = json.loads(error_events[0]["data"])
            assert "error" in error_data
            assert "mock forward failure" in error_data["error"]

    async def test_done_event_has_total_tokens(self) -> None:
        """The done event includes total_tokens = prompt_tokens + completion_tokens."""
        engine = _make_engine()
        async for client in _make_client(engine):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": [1, 2, 3, 4, 5],
                    "max_tokens": 3,
                    "temperature": 0.0,
                },
            )
            events = _parse_sse_events(resp.text)
            done_events = [e for e in events if e["event"] == "done"]
            assert len(done_events) == 1
            done_data = json.loads(done_events[0]["data"])
            usage = done_data["usage"]
            assert usage["prompt_tokens"] == 5
            assert usage["completion_tokens"] == 3
            assert usage["total_tokens"] == 8


# ---------------------------------------------------------------------------
# TestErrorResponses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorResponses:
    async def test_empty_string_prompt_400(self) -> None:
        engine = _make_engine()
        app = create_app_with_engine(engine)
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "", "temperature": 0.0},
            )
            assert resp.status_code == 400

    async def test_empty_list_prompt_400(self) -> None:
        engine = _make_engine()
        app = create_app_with_engine(engine)
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": [], "temperature": 0.0},
            )
            assert resp.status_code == 400

    async def test_wrong_model_422(self) -> None:
        engine = _make_engine()
        app = create_app_with_engine(engine)
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={"model": "wrong-model", "prompt": "hi", "temperature": 0.0},
            )
            assert resp.status_code == 422

    async def test_negative_temperature_422(self) -> None:
        engine = _make_engine()
        app = create_app_with_engine(engine)
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "hi", "temperature": -1.0},
            )
            assert resp.status_code == 422

    async def test_unknown_field_422(self) -> None:
        engine = _make_engine()
        app = create_app_with_engine(engine)
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "hi",
                    "temperature": 0.0,
                    "bogus_field": True,
                },
            )
            assert resp.status_code == 422

    async def test_prompt_too_long_422(self) -> None:
        engine = _make_engine(max_seq_len=10)
        app = create_app_with_engine(engine)
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": list(range(10)),
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 422

    async def test_queue_full_503(self) -> None:
        engine = _make_engine(max_waiting_requests=1, batch_wait_timeout_s=999.0)
        # Pre-fill the waiting queue directly so the HTTP request gets 503.
        dummy_queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        engine.add_request("dummy", [1, 2], params, dummy_queue)

        app = create_app_with_engine(engine)
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": [3, 4],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 503
