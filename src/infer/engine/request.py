"""Request lifecycle types for the serving engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

import torch

from infer.engine.sampler import SamplingParams


class RequestState(Enum):
    """Lifecycle states for a completion request.

    Transitions::

        WAITING ──(added to batch)──> PREFILL ──(prefill done)──> DECODE ──> FINISHED
           │                            │                           │
           └──────────(error)──>      FAILED     <──(error)─────────┘
    """

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class StepOutput:
    """Per-step result pushed from the engine to the API layer.

    One ``StepOutput`` is produced per request per engine step.  The API
    layer reads these from the request's ``asyncio.Queue`` and emits SSE
    events.
    """

    request_id: str
    token_id: int | None
    text_delta: str
    finished: bool
    finish_reason: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Request:
    """A single completion request tracked by the engine.

    Created by ``Engine.add_request()`` when a new HTTP request arrives.
    The engine and scheduler mutate ``state`` and ``generated_token_ids``
    as the request progresses through its lifecycle.
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time_s: float

    # Mutable state
    state: RequestState = RequestState.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None

    # Per-request RNG — created from sampling_params.seed at enqueue time.
    # None means non-deterministic sampling.
    generator: torch.Generator | None = field(default=None, repr=False)

    # Cache pool slot — assigned during prefill, freed on retire (continuous batching).
    slot_idx: int | None = field(default=None, repr=False)

    # Output channel — set by the server when the request is enqueued.
    output_queue: asyncio.Queue[StepOutput] | None = field(default=None, repr=False)
