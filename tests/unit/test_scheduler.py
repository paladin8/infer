"""Unit tests for the StaticScheduler."""

from __future__ import annotations

import time
from unittest.mock import patch

from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState
from infer.engine.sampler import SamplingParams
from infer.engine.scheduler import StaticScheduler


def _make_request(request_id: str, prompt_len: int = 3) -> Request:
    """Helper to create a dummy request."""
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=10),
        arrival_time_s=time.perf_counter(),
    )


def _make_config(**overrides: object) -> EngineConfig:
    """Helper to create an EngineConfig with overrides."""
    defaults: dict[str, object] = {
        "model": "test-model",
        "batch_wait_timeout_s": 0.0,  # no wait by default in tests
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


class TestBasicBatchFormation:
    def test_empty_scheduler_returns_empty(self) -> None:
        sched = StaticScheduler(_make_config())
        assert sched.schedule() == []

    def test_single_request_dispatched(self) -> None:
        sched = StaticScheduler(_make_config())
        req = _make_request("r1")
        assert sched.add_request(req) is True
        batch = sched.schedule()
        assert batch == [req]

    def test_multiple_requests_dispatched(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=4))
        reqs = [_make_request(f"r{i}") for i in range(3)]
        for r in reqs:
            sched.add_request(r)
        batch = sched.schedule()
        assert batch == reqs

    def test_active_batch_returned_unchanged(self) -> None:
        """Calling schedule() while a batch is active returns the same batch."""
        sched = StaticScheduler(_make_config())
        req = _make_request("r1")
        sched.add_request(req)

        batch1 = sched.schedule()
        # Simulate the request being in DECODE state (not finished).
        batch1[0].state = RequestState.DECODE

        batch2 = sched.schedule()
        assert batch2 is batch1


class TestBatchCompletion:
    def test_batch_clears_when_all_finished(self) -> None:
        sched = StaticScheduler(_make_config())
        r1, r2 = _make_request("r1"), _make_request("r2")
        sched.add_request(r1)
        sched.add_request(r2)

        batch = sched.schedule()
        assert len(batch) == 2

        # Finish both requests.
        r1.state = RequestState.FINISHED
        r2.state = RequestState.FINISHED

        # Next schedule should clear the batch and return empty (no waiting).
        assert sched.schedule() == []

    def test_partial_completion_keeps_batch(self) -> None:
        sched = StaticScheduler(_make_config())
        r1, r2, r3 = _make_request("r1"), _make_request("r2"), _make_request("r3")
        for r in [r1, r2, r3]:
            sched.add_request(r)

        batch = sched.schedule()
        # Simulate realistic state: all went through prefill, now decoding.
        r1.state = RequestState.FINISHED
        r2.state = RequestState.DECODE
        r3.state = RequestState.DECODE

        # Batch is NOT done (r2, r3 still active).
        assert sched.schedule() is batch

    def test_failed_counts_as_done(self) -> None:
        sched = StaticScheduler(_make_config())
        req = _make_request("r1")
        sched.add_request(req)
        sched.schedule()
        req.state = RequestState.FAILED
        # Batch should clear.
        assert sched.schedule() == []

    def test_next_batch_after_completion(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=2))
        r1, r2, r3 = _make_request("r1"), _make_request("r2"), _make_request("r3")
        for r in [r1, r2, r3]:
            sched.add_request(r)

        # First batch: r1, r2.
        batch1 = sched.schedule()
        assert [r.request_id for r in batch1] == ["r1", "r2"]

        r1.state = RequestState.FINISHED
        r2.state = RequestState.FINISHED

        # Second batch: r3.
        batch2 = sched.schedule()
        assert [r.request_id for r in batch2] == ["r3"]


class TestQueueOverflow:
    def test_add_fails_when_queue_full(self) -> None:
        sched = StaticScheduler(_make_config(max_waiting_requests=2))
        assert sched.add_request(_make_request("r1")) is True
        assert sched.add_request(_make_request("r2")) is True
        assert sched.add_request(_make_request("r3")) is False

    def test_queue_frees_after_dispatch(self) -> None:
        sched = StaticScheduler(_make_config(max_waiting_requests=2))
        sched.add_request(_make_request("r1"))
        sched.add_request(_make_request("r2"))

        # Dispatch moves requests from waiting to active.
        sched.schedule()
        assert len(sched.waiting) == 0

        # Now we can add more.
        assert sched.add_request(_make_request("r3")) is True


class TestFCFSOrdering:
    def test_insertion_order_preserved(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=5))
        ids = ["c", "a", "b"]
        for rid in ids:
            sched.add_request(_make_request(rid))
        batch = sched.schedule()
        assert [r.request_id for r in batch] == ids


class TestBatchSizeLimit:
    def test_respects_max_batch_size(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=2))
        for i in range(5):
            sched.add_request(_make_request(f"r{i}"))

        batch1 = sched.schedule()
        assert len(batch1) == 2
        assert [r.request_id for r in batch1] == ["r0", "r1"]

        # Finish batch1.
        for r in batch1:
            r.state = RequestState.FINISHED

        batch2 = sched.schedule()
        assert len(batch2) == 2
        assert [r.request_id for r in batch2] == ["r2", "r3"]

        for r in batch2:
            r.state = RequestState.FINISHED

        batch3 = sched.schedule()
        assert len(batch3) == 1
        assert batch3[0].request_id == "r4"


class TestHasWork:
    def test_empty_no_work(self) -> None:
        sched = StaticScheduler(_make_config())
        assert sched.has_work() is False

    def test_waiting_has_work(self) -> None:
        sched = StaticScheduler(_make_config())
        sched.add_request(_make_request("r1"))
        assert sched.has_work() is True

    def test_active_has_work(self) -> None:
        sched = StaticScheduler(_make_config())
        sched.add_request(_make_request("r1"))
        sched.schedule()
        assert sched.has_work() is True

    def test_all_finished_no_work(self) -> None:
        sched = StaticScheduler(_make_config())
        req = _make_request("r1")
        sched.add_request(req)
        sched.schedule()
        req.state = RequestState.FINISHED
        # Active batch is done, waiting is empty.
        sched.schedule()  # clears the batch
        assert sched.has_work() is False


class TestRequestsDuringActiveBatch:
    def test_requests_enqueued_during_active_batch(self) -> None:
        """New requests added while a batch is active go to the next batch."""
        sched = StaticScheduler(_make_config(max_batch_size=2))
        r1 = _make_request("r1")
        sched.add_request(r1)
        batch1 = sched.schedule()
        assert batch1 == [r1]

        # Simulate r1 in decode.
        r1.state = RequestState.DECODE

        # New requests arrive while batch1 is active.
        r2 = _make_request("r2")
        r3 = _make_request("r3")
        sched.add_request(r2)
        sched.add_request(r3)

        # schedule() returns the active batch, not the new requests.
        assert sched.schedule() is batch1

        # Finish batch1.
        r1.state = RequestState.FINISHED

        # Next batch picks up the queued requests.
        batch2 = sched.schedule()
        assert [r.request_id for r in batch2] == ["r2", "r3"]


class TestBoundedWait:
    def test_dispatches_immediately_when_full(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=3, batch_wait_timeout_s=10.0))
        for i in range(3):
            sched.add_request(_make_request(f"r{i}"))
        # Batch is full — dispatches immediately despite long timeout.
        batch = sched.schedule()
        assert len(batch) == 3

    def test_waits_when_not_full(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=4, batch_wait_timeout_s=10.0))
        sched.add_request(_make_request("r1"))
        sched.add_request(_make_request("r2"))
        # Not full, timeout hasn't expired.
        assert sched.schedule() == []

    def test_dispatches_on_timeout(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=4, batch_wait_timeout_s=0.5))
        sched.add_request(_make_request("r1"))
        sched.add_request(_make_request("r2"))

        # First call at t=0.0 starts the timer (deadline = 0.5).
        with patch("infer.engine.scheduler.time.perf_counter", return_value=0.0):
            assert sched.schedule() == []

        # Second call at t=0.3 — timer not yet expired.
        with patch("infer.engine.scheduler.time.perf_counter", return_value=0.3):
            assert sched.schedule() == []

        # Third call at t=0.6 — timer expired.
        with patch("infer.engine.scheduler.time.perf_counter", return_value=0.6):
            batch = sched.schedule()
        assert len(batch) == 2

    def test_timer_resets_between_batches(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=4, batch_wait_timeout_s=0.5))
        sched.add_request(_make_request("r1"))

        # t=0.0: start timer.
        with patch("infer.engine.scheduler.time.perf_counter", return_value=0.0):
            sched.schedule()
        # t=0.6: dispatch.
        with patch("infer.engine.scheduler.time.perf_counter", return_value=0.6):
            batch = sched.schedule()
        assert len(batch) == 1

        batch[0].state = RequestState.FINISHED
        sched.schedule()  # clears batch

        # Add new request — timer should start fresh from t=1.0.
        sched.add_request(_make_request("r2"))
        with patch("infer.engine.scheduler.time.perf_counter", return_value=1.0):
            assert sched.schedule() == []  # new timer started at 1.0, deadline 1.5

        # t=1.3 — not expired yet.
        with patch("infer.engine.scheduler.time.perf_counter", return_value=1.3):
            assert sched.schedule() == []

        # t=1.6 — expired.
        with patch("infer.engine.scheduler.time.perf_counter", return_value=1.6):
            batch2 = sched.schedule()
        assert len(batch2) == 1
        assert batch2[0].request_id == "r2"

    def test_zero_timeout_dispatches_immediately(self) -> None:
        sched = StaticScheduler(_make_config(max_batch_size=4, batch_wait_timeout_s=0.0))
        sched.add_request(_make_request("r1"))
        # Zero timeout means dispatch right away.
        batch = sched.schedule()
        assert len(batch) == 1
