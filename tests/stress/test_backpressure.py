"""Stress tests for queue backpressure and concurrent disconnect handling.

These tests verify that the server behaves correctly under overload conditions:
excess requests get 503 responses, successful requests complete normally,
and client disconnects do not cause deadlocks.
"""

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
from infer.loader.config import ModelConfig
from infer.server.api import create_app_with_engine, engine_loop

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors test_api.py)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(**overrides: object) -> Engine:
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
    model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
    tokenizer = MockTokenizer()
    return Engine.from_components(config, model, tokenizer)


def _parse_sse_events(text: str) -> list[dict[str, str]]:
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
    if current_event or current_data:
        events.append({"event": current_event, "data": current_data})
    return events


async def _make_client(engine: Engine) -> AsyncGenerator[httpx.AsyncClient]:
    app = create_app_with_engine(engine)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    loop_task = asyncio.create_task(engine_loop(engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    loop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await loop_task


def _completion_payload(max_tokens: int = 3) -> dict[str, object]:
    return {
        "model": "test-model",
        "prompt": [1, 2, 3],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


# ---------------------------------------------------------------------------
# TestQueueGrowthUnderLoad
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
class TestQueueGrowthUnderLoad:
    async def test_excess_requests_get_503(self) -> None:
        """Send max_waiting_requests + overflow requests concurrently.

        Requests beyond the queue capacity should get 503 responses.
        The rest should complete successfully with a done event.
        """
        max_waiting = 4
        overflow = 6
        total = max_waiting + overflow
        # max_batch_size=2 so the engine processes requests in small batches,
        # keeping the queue populated while new requests arrive.
        engine = _make_engine(
            max_waiting_requests=max_waiting,
            max_batch_size=2,
        )
        async for client in _make_client(engine):
            tasks = [
                asyncio.create_task(
                    client.post("/v1/completions", json=_completion_payload(max_tokens=5))
                )
                for _ in range(total)
            ]
            responses = await asyncio.gather(*tasks)

            status_codes = [r.status_code for r in responses]
            num_503 = status_codes.count(503)
            num_200 = status_codes.count(200)

            # At least some requests should be rejected (503).
            assert num_503 >= 1, f"Expected at least 1 rejection, got status codes: {status_codes}"
            # All responses should be either 200 or 503.
            assert num_200 + num_503 == total, f"Unexpected status codes: {status_codes}"

            # All 200 responses should have a valid done event.
            for resp in responses:
                if resp.status_code == 200:
                    events = _parse_sse_events(resp.text)
                    done_events = [e for e in events if e["event"] == "done"]
                    assert len(done_events) == 1, f"Expected 1 done event, got {len(done_events)}"
                    done_data = json.loads(done_events[0]["data"])
                    assert "finish_reason" in done_data
                    assert "usage" in done_data

    async def test_no_unbounded_memory_growth(self) -> None:
        """Send requests in waves and verify each wave completes.

        This tests that the server properly clears batch state and the queue
        doesn't grow unboundedly across multiple waves of requests.
        """
        max_waiting = 4
        engine = _make_engine(max_waiting_requests=max_waiting, max_batch_size=4)
        async for client in _make_client(engine):
            for wave in range(3):
                tasks = [
                    asyncio.create_task(
                        client.post("/v1/completions", json=_completion_payload(max_tokens=3))
                    )
                    for _ in range(max_waiting)
                ]
                responses = await asyncio.gather(*tasks)
                for resp in responses:
                    assert resp.status_code == 200, (
                        f"Wave {wave}: expected 200, got {resp.status_code}"
                    )
                    events = _parse_sse_events(resp.text)
                    done_events = [e for e in events if e["event"] == "done"]
                    assert len(done_events) == 1

            # After all waves, scheduler should be idle.
            assert not engine.has_work()


# ---------------------------------------------------------------------------
# TestConcurrentDisconnect
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
class TestConcurrentDisconnect:
    async def test_unconsumed_queues_no_deadlock(self) -> None:
        """Enqueue requests but don't consume their output queues (simulating disconnect).

        In static batching, disconnected clients' output queues simply fill up.
        The engine must finish the batch without deadlocking, clear batch state,
        and be ready for the next batch.
        """
        from infer.engine.request import StepOutput
        from infer.engine.sampler import SamplingParams

        engine = _make_engine(max_batch_size=8)

        # Enqueue 8 requests — 4 with consumed queues, 4 "disconnected" (unconsumed).
        consumed_queues: list[asyncio.Queue[StepOutput]] = []
        for i in range(8):
            queue: asyncio.Queue[StepOutput] = asyncio.Queue()
            params = SamplingParams(temperature=0.0, max_new_tokens=5)
            added = engine.add_request(f"req-{i}", [1, 2, 3], params, queue)
            assert added
            if i >= 4:
                consumed_queues.append(queue)

        # Run the engine until the batch completes. The 4 unconsumed queues
        # fill up silently — the engine should not block on them.
        for _ in range(500):
            if not engine.has_work():
                break
            engine.step()

        assert not engine.has_work(), "Engine should be idle after batch completes"

        # The consumed queues should have received all outputs (tokens + done).
        for queue in consumed_queues:
            outputs: list[StepOutput] = []
            while not queue.empty():
                outputs.append(queue.get_nowait())
            assert any(o.finished for o in outputs), "Each request should have a done output"

    async def test_server_responsive_after_disconnect(self) -> None:
        """After clients disconnect, the server should handle new requests normally.

        Enqueue requests directly (simulating disconnected clients that won't
        consume output), let the engine drain, then send a normal HTTP request.
        """
        from infer.engine.request import StepOutput
        from infer.engine.sampler import SamplingParams

        engine = _make_engine(max_batch_size=4)
        async for client in _make_client(engine):
            # Enqueue 4 requests directly — no one will consume their queues.
            for i in range(4):
                queue: asyncio.Queue[StepOutput] = asyncio.Queue()
                params = SamplingParams(temperature=0.0, max_new_tokens=5)
                engine.add_request(f"disconnect-{i}", [1, 2, 3], params, queue)

            # Wait for the engine to finish processing the batch.
            for _ in range(200):
                if not engine.has_work():
                    break
                await asyncio.sleep(0.01)
            assert not engine.has_work()

            # Send a normal HTTP request — server should respond fine.
            resp = await client.post("/v1/completions", json=_completion_payload(max_tokens=3))
            assert resp.status_code == 200
            events = _parse_sse_events(resp.text)
            done_events = [e for e in events if e["event"] == "done"]
            assert len(done_events) == 1
            done_data = json.loads(done_events[0]["data"])
            assert done_data["finish_reason"] in ("length", "eos")
