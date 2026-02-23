"""FastAPI application with SSE streaming for the inference engine."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import orjson
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator
from sse_starlette.sse import EventSourceResponse

from infer.engine.config import EngineConfig
from infer.engine.engine import Engine
from infer.engine.request import StepOutput
from infer.engine.sampler import SamplingParams


class CompletionRequest(BaseModel):
    """Request body for the completions endpoint."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str | list[int]
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    stream: bool = True
    stop: str | list[str] | None = None
    seed: int | None = None

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_tokens must be >= 1")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("temperature must be >= 0")
        return v

    @field_validator("top_p")
    @classmethod
    def top_p_range(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        return v


async def engine_loop(engine: Engine) -> None:
    """Background loop that drives the engine."""
    while True:
        if engine.has_work():
            engine.step()
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.001)


def _build_completions_endpoint(app: FastAPI, engine: Engine) -> None:
    """Register the POST /v1/completions route."""

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest) -> EventSourceResponse:
        # Validate prompt is non-empty.
        if isinstance(body.prompt, str) and len(body.prompt) == 0:
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        if isinstance(body.prompt, list) and len(body.prompt) == 0:
            raise HTTPException(status_code=400, detail="prompt must not be empty")

        # Model mismatch.
        if body.model != engine.model_id:
            raise HTTPException(
                status_code=422,
                detail=f"model mismatch: expected {engine.model_id!r}, got {body.model!r}",
            )

        # Normalize stop field.
        stop: list[str] | None = None
        if isinstance(body.stop, str):
            stop = [body.stop]
        elif isinstance(body.stop, list):
            stop = body.stop

        params = SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            repetition_penalty=body.repetition_penalty,
            max_new_tokens=body.max_tokens,
            stop=stop,
            seed=body.seed,
        )

        request_id = str(uuid.uuid4())
        output_queue: asyncio.Queue[StepOutput] = asyncio.Queue()

        try:
            accepted = engine.add_request(request_id, body.prompt, params, output_queue)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

        if not accepted:
            raise HTTPException(status_code=503, detail="server is at capacity, try again later")

        async def event_generator() -> AsyncGenerator[dict[str, Any]]:
            while True:
                output = await output_queue.get()

                if output.error is not None:
                    yield {
                        "event": "error",
                        "data": orjson.dumps({"error": output.error}).decode(),
                    }
                    return

                if output.finished:
                    # Emit the final token if it carries text.
                    if output.text_delta:
                        yield {
                            "event": "token",
                            "data": orjson.dumps(
                                {"token": output.text_delta, "token_id": output.token_id}
                            ).decode(),
                        }
                    total = output.prompt_tokens + output.completion_tokens
                    yield {
                        "event": "done",
                        "data": orjson.dumps(
                            {
                                "finish_reason": output.finish_reason,
                                "usage": {
                                    "prompt_tokens": output.prompt_tokens,
                                    "completion_tokens": output.completion_tokens,
                                    "total_tokens": total,
                                },
                            }
                        ).decode(),
                    }
                    return

                yield {
                    "event": "token",
                    "data": orjson.dumps(
                        {"token": output.text_delta, "token_id": output.token_id}
                    ).decode(),
                }

        return EventSourceResponse(event_generator())


def create_app(config: EngineConfig) -> FastAPI:
    """Create a FastAPI app that loads a real model and starts the engine loop."""

    # Engine is created inside the lifespan but routes need the engine reference.
    # We use a mutable container so the route closure can capture it.
    engine_ref: list[Engine] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        engine = Engine(config)
        engine_ref.append(engine)
        app.state.engine = engine
        task = asyncio.create_task(engine_loop(engine))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan)

    # Deferred engine wrapper — routes are registered at app creation but the
    # engine is only available after lifespan startup.
    class _EngineProxy:
        """Proxy that forwards attribute access to the real engine once started."""

        def __getattr__(self, name: str) -> Any:
            return getattr(engine_ref[0], name)

    _build_completions_endpoint(app, _EngineProxy())  # type: ignore[arg-type]
    return app


def create_app_with_engine(engine: Engine) -> FastAPI:
    """Create a FastAPI app with a pre-built engine (for testing)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.engine = engine
        task = asyncio.create_task(engine_loop(engine))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan)
    _build_completions_endpoint(app, engine)
    return app
