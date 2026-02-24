"""Unit tests for the ContinuousScheduler."""

from __future__ import annotations

import time

from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState
from infer.engine.sampler import SamplingParams
from infer.engine.scheduler import ContinuousScheduler


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
        "batching_mode": "continuous",
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


class TestPerStepAdmitRetire:
    def test_first_schedule_prefills_all(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)

        out = sched.schedule()
        assert [r.request_id for r in out.prefill] == ["a", "b", "c"]
        assert out.decode == []
        assert out.retired == []

    def test_decode_after_prefill(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)

        sched.schedule()
        # Simulate prefill completing — requests move to DECODE state.
        a.state = RequestState.DECODE
        b.state = RequestState.DECODE
        c.state = RequestState.DECODE

        out = sched.schedule()
        assert out.prefill == []
        assert [r.request_id for r in out.decode] == ["a", "b", "c"]
        assert out.retired == []

    def test_retire_and_admit_in_same_step(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)

        sched.schedule()
        a.state = RequestState.DECODE
        b.state = RequestState.DECODE
        c.state = RequestState.DECODE

        sched.schedule()  # all decoding

        # Finish A, add D.
        a.state = RequestState.FINISHED
        d = _make_request("d")
        sched.add_request(d)

        out = sched.schedule()
        assert [r.request_id for r in out.retired] == ["a"]
        assert [r.request_id for r in out.prefill] == ["d"]
        assert [r.request_id for r in out.decode] == ["b", "c"]

    def test_failed_request_retired(self) -> None:
        sched = ContinuousScheduler(_make_config())
        r = _make_request("r1")
        sched.add_request(r)
        sched.schedule()
        r.state = RequestState.FAILED

        out = sched.schedule()
        assert [r.request_id for r in out.retired] == ["r1"]


class TestCapacityLimit:
    def test_admits_up_to_max_batch_size(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=2))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)

        out = sched.schedule()
        assert [r.request_id for r in out.prefill] == ["a", "b"]
        # c stays in waiting.
        assert len(sched.waiting) == 1

    def test_admits_from_waiting_after_retire(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=2))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)

        sched.schedule()
        a.state = RequestState.DECODE
        b.state = RequestState.DECODE

        # Finish A → frees 1 slot → C admitted.
        a.state = RequestState.FINISHED

        out = sched.schedule()
        assert [r.request_id for r in out.retired] == ["a"]
        assert [r.request_id for r in out.prefill] == ["c"]
        assert [r.request_id for r in out.decode] == ["b"]

    def test_no_prefill_when_at_capacity(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=2))
        a, b = _make_request("a"), _make_request("b")
        sched.add_request(a)
        sched.add_request(b)
        sched.schedule()
        a.state = RequestState.DECODE
        b.state = RequestState.DECODE

        # Add c, but no capacity (both slots taken).
        c = _make_request("c")
        sched.add_request(c)

        out = sched.schedule()
        assert out.prefill == []
        assert len(out.decode) == 2
        assert len(sched.waiting) == 1


class TestQueueOverflow:
    def test_add_fails_when_queue_full(self) -> None:
        sched = ContinuousScheduler(_make_config(max_waiting_requests=2))
        assert sched.add_request(_make_request("r1")) is True
        assert sched.add_request(_make_request("r2")) is True
        assert sched.add_request(_make_request("r3")) is False

    def test_queue_frees_after_admit(self) -> None:
        sched = ContinuousScheduler(_make_config(max_waiting_requests=2, max_batch_size=8))
        sched.add_request(_make_request("r1"))
        sched.add_request(_make_request("r2"))

        # Admit moves requests from waiting to active.
        sched.schedule()
        assert len(sched.waiting) == 0

        # Now we can add more.
        assert sched.add_request(_make_request("r3")) is True


class TestFCFSOrdering:
    def test_insertion_order_preserved(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=5))
        ids = ["c", "a", "b"]
        for rid in ids:
            sched.add_request(_make_request(rid))
        out = sched.schedule()
        assert [r.request_id for r in out.prefill] == ids

    def test_waiting_queue_order_across_steps(self) -> None:
        """Requests admitted over multiple steps maintain FCFS order."""
        sched = ContinuousScheduler(_make_config(max_batch_size=1))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)

        # Step 1: only A admitted.
        out1 = sched.schedule()
        assert [r.request_id for r in out1.prefill] == ["a"]

        a.state = RequestState.DECODE
        a.state = RequestState.FINISHED

        # Step 2: B admitted.
        out2 = sched.schedule()
        assert [r.request_id for r in out2.prefill] == ["b"]

        b.state = RequestState.DECODE
        b.state = RequestState.FINISHED

        # Step 3: C admitted.
        out3 = sched.schedule()
        assert [r.request_id for r in out3.prefill] == ["c"]


class TestHasWork:
    def test_empty_no_work(self) -> None:
        sched = ContinuousScheduler(_make_config())
        assert sched.has_work() is False

    def test_waiting_has_work(self) -> None:
        sched = ContinuousScheduler(_make_config())
        sched.add_request(_make_request("r1"))
        assert sched.has_work() is True

    def test_active_has_work(self) -> None:
        sched = ContinuousScheduler(_make_config())
        sched.add_request(_make_request("r1"))
        sched.schedule()
        assert sched.has_work() is True

    def test_all_finished_no_work(self) -> None:
        sched = ContinuousScheduler(_make_config())
        req = _make_request("r1")
        sched.add_request(req)
        sched.schedule()
        req.state = RequestState.FINISHED
        sched.schedule()  # retires the request
        assert sched.has_work() is False


class TestNoWaitTimer:
    def test_admits_immediately_with_free_slots(self) -> None:
        """Unlike StaticScheduler, no timer — admits immediately."""
        sched = ContinuousScheduler(_make_config(max_batch_size=8))
        sched.add_request(_make_request("r1"))

        out = sched.schedule()
        assert len(out.prefill) == 1

    def test_admits_one_at_a_time_when_capacity_is_one(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=1))
        sched.add_request(_make_request("r1"))
        sched.add_request(_make_request("r2"))

        out = sched.schedule()
        assert len(out.prefill) == 1
        assert out.prefill[0].request_id == "r1"


class TestEmptyScheduleOutput:
    def test_empty_when_nothing_to_do(self) -> None:
        sched = ContinuousScheduler(_make_config())
        out = sched.schedule()
        assert out.prefill == []
        assert out.decode == []
        assert out.retired == []

    def test_only_retired_when_last_request_finishes(self) -> None:
        sched = ContinuousScheduler(_make_config())
        req = _make_request("r1")
        sched.add_request(req)
        sched.schedule()
        req.state = RequestState.DECODE
        sched.schedule()
        req.state = RequestState.FINISHED

        out = sched.schedule()
        assert [r.request_id for r in out.retired] == ["r1"]
        assert out.prefill == []
        assert out.decode == []


class TestMultipleRetireInOneStep:
    def test_multiple_retirements(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        reqs = [_make_request(f"r{i}") for i in range(4)]
        for r in reqs:
            sched.add_request(r)

        sched.schedule()
        for r in reqs:
            r.state = RequestState.DECODE
        sched.schedule()

        # Finish 3 out of 4.
        reqs[0].state = RequestState.FINISHED
        reqs[1].state = RequestState.FAILED
        reqs[2].state = RequestState.FINISHED

        out = sched.schedule()
        retired_ids = sorted(r.request_id for r in out.retired)
        assert retired_ids == ["r0", "r1", "r2"]
        assert [r.request_id for r in out.decode] == ["r3"]
