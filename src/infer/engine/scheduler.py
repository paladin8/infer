"""Scheduling policies for the serving engine."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState


class StaticScheduler:
    """FCFS scheduler that forms fixed batches with a bounded wait.

    When no batch is active and at least one request is waiting, the
    scheduler starts a timer.  Once the batch fills to ``max_batch_size``
    or the timer expires (``batch_wait_timeout_s``), the waiting requests
    are pulled into the active batch and dispatched.

    The active batch runs until all requests finish (or fail).  Then the
    next batch is formed from the waiting queue.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self._batch_wait_deadline: float | None = None

    def add_request(self, request: Request) -> bool:
        """Add a request to the waiting queue.

        Returns ``False`` if the queue is full (caller should return 503).
        """
        if len(self.waiting) >= self.config.max_waiting_requests:
            return False
        self.waiting.append(request)
        return True

    def schedule(self) -> list[Request]:
        """Return the current active batch, starting a new one if needed.

        Called at the start of each engine step.  Behavior:

        1. If ``self.active`` has unfinished requests, return it unchanged.
        2. If all active requests are finished/failed, clear the active list.
        3. If the waiting queue is non-empty and no batch is forming, start
           the wait timer.
        4. If the waiting queue has ``max_batch_size`` requests OR the wait
           timer has expired, pull requests into ``self.active`` and dispatch.
        5. Otherwise return empty (still waiting for the batch to fill).
        """
        # Step 1-2: check active batch status.
        if self.active:
            if self._batch_done():
                self.active = []
            else:
                return self.active

        # Step 3-5: form a new batch from waiting queue.
        if not self.waiting:
            self._batch_wait_deadline = None
            return []

        # Start timer on first request seen since last batch.
        if self._batch_wait_deadline is None:
            self._batch_wait_deadline = time.perf_counter() + self.config.batch_wait_timeout_s

        # Dispatch if batch is full or timer expired.
        batch_full = len(self.waiting) >= self.config.max_batch_size
        timer_expired = time.perf_counter() >= self._batch_wait_deadline

        if not batch_full and not timer_expired:
            return []

        # Pull up to max_batch_size from the waiting queue.
        count = min(len(self.waiting), self.config.max_batch_size)
        self.active = [self.waiting.popleft() for _ in range(count)]
        self._batch_wait_deadline = None
        return self.active

    def has_work(self) -> bool:
        """True if there are active or waiting requests."""
        return bool(self.active) or bool(self.waiting)

    def _batch_done(self) -> bool:
        """True if every request in the active batch is finished or failed."""
        return all(r.state in (RequestState.FINISHED, RequestState.FAILED) for r in self.active)


# ---------------------------------------------------------------------------
# Continuous batching scheduler
# ---------------------------------------------------------------------------

_TERMINAL_STATES = frozenset({RequestState.FINISHED, RequestState.FAILED})


@dataclass
class ScheduleOutput:
    """Result of one continuous scheduling step."""

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    retired: list[Request] = field(default_factory=list)


class ContinuousScheduler:
    """Per-step scheduler for continuous batching.

    Phase 6 splits the monolithic ``schedule()`` into ``retire()``,
    ``admit()``, and ``decode_requests()`` to support the
    retire → free-blocks → admit-with-budget ordering needed for
    paged allocation.  Both backends use the split interface.
    The combined ``schedule()`` is retained for backward compatibility
    in existing Phase 5 tests.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []

    def add_request(self, request: Request) -> bool:
        """Add a request to the waiting queue.

        Returns ``False`` if the queue is full (caller should return 503).
        """
        if len(self.waiting) >= self.config.max_waiting_requests:
            return False
        self.waiting.append(request)
        return True

    def retire(self) -> list[Request]:
        """Remove finished/failed requests from the active set.

        Returns the retired list so the engine can free their cache resources
        before admitting new requests.
        """
        retired = [r for r in self.active if r.state in _TERMINAL_STATES]
        self.active = [r for r in self.active if r.state not in _TERMINAL_STATES]
        return retired

    def admit(self, free_kv_tokens: int | None = None) -> list[Request]:
        """Admit new requests from the waiting queue.

        Checks two admission constraints:
        1. Compute budget: ``len(active) < max_batch_size``.
        2. Memory budget: cumulative prompt tokens of admitted requests
           must not exceed 80% of ``free_kv_tokens`` (when provided).

        When ``free_kv_tokens is None`` (contiguous backend), only the
        compute budget is checked — identical to Phase 5 behavior.

        Returns the list of newly admitted requests (need prefill).
        """
        capacity = self.config.max_batch_size - len(self.active)
        new: list[Request] = []
        remaining_tokens = int(free_kv_tokens * 0.8) if free_kv_tokens is not None else None

        while self.waiting and len(new) < capacity:
            req = self.waiting[0]
            if remaining_tokens is not None:
                prompt_len = len(req.prompt_token_ids)
                if prompt_len > remaining_tokens:
                    break
                remaining_tokens -= prompt_len
            new.append(self.waiting.popleft())

        self.active.extend(new)
        return new

    def decode_requests(self) -> list[Request]:
        """Return active requests in DECODE state."""
        return [r for r in self.active if r.state == RequestState.DECODE]

    def schedule(self) -> ScheduleOutput:
        """Combined retire + admit + decode (backward-compatible convenience).

        Equivalent to ``retire()`` then ``admit(free_kv_tokens=None)`` then
        ``decode_requests()``. Does NOT pass a block budget — only safe for
        the contiguous backend where token budget is not applicable.

        Retained for backward compatibility with existing Phase 5 tests.
        The engine always uses the split interface.
        """
        retired = self.retire()
        prefill = self.admit(free_kv_tokens=None)
        decode = self.decode_requests()
        return ScheduleOutput(prefill=prefill, decode=decode, retired=retired)

    def has_work(self) -> bool:
        """True if there are active or waiting requests."""
        return bool(self.active) or bool(self.waiting)
