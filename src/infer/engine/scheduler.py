"""Static-batching scheduler with FCFS ordering and bounded wait."""

from __future__ import annotations

import time
from collections import deque

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
