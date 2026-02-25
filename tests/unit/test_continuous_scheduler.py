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


class TestSplitInterface:
    """Tests for the retire() → admit() → decode_requests() split interface."""

    def test_retire_returns_finished(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        a, b = _make_request("a"), _make_request("b")
        for r in [a, b]:
            sched.add_request(r)
        sched.schedule()  # admit both
        a.state = RequestState.DECODE
        b.state = RequestState.DECODE

        a.state = RequestState.FINISHED
        retired = sched.retire()
        assert [r.request_id for r in retired] == ["a"]
        assert len(sched.active) == 1

    def test_admit_no_budget_contiguous(self) -> None:
        """admit(free_kv_tokens=None) checks only compute budget."""
        sched = ContinuousScheduler(_make_config(max_batch_size=2))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)

        new = sched.admit(free_kv_tokens=None)
        assert [r.request_id for r in new] == ["a", "b"]
        assert len(sched.waiting) == 1

    def test_admit_with_block_budget(self) -> None:
        """admit(free_kv_tokens=N) limits by 80% of available tokens."""
        sched = ContinuousScheduler(_make_config(max_batch_size=10))
        # Request with prompt_len=3 tokens each.
        a = _make_request("a", prompt_len=3)
        b = _make_request("b", prompt_len=3)
        c = _make_request("c", prompt_len=3)
        for r in [a, b, c]:
            sched.add_request(r)

        # 10 free tokens → 80% = 8 usable → admits a (3) and b (3), 2 remaining.
        new = sched.admit(free_kv_tokens=10)
        assert [r.request_id for r in new] == ["a", "b"]
        assert len(sched.waiting) == 1

    def test_admit_budget_too_small_for_first(self) -> None:
        """If the first request exceeds the budget, nothing is admitted."""
        sched = ContinuousScheduler(_make_config(max_batch_size=10))
        a = _make_request("a", prompt_len=10)
        sched.add_request(a)

        # 10 free tokens → 80% = 8, but prompt is 10.
        new = sched.admit(free_kv_tokens=10)
        assert new == []
        assert len(sched.waiting) == 1

    def test_decode_requests_returns_only_decode_state(self) -> None:
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        a, b = _make_request("a"), _make_request("b")
        for r in [a, b]:
            sched.add_request(r)
        sched.admit()  # admit both (WAITING state)

        # Only a is in DECODE state.
        a.state = RequestState.DECODE

        decode = sched.decode_requests()
        assert [r.request_id for r in decode] == ["a"]

    def test_split_matches_combined_for_contiguous(self) -> None:
        """Split interface produces same results as combined schedule()."""
        sched1 = ContinuousScheduler(_make_config(max_batch_size=4))
        sched2 = ContinuousScheduler(_make_config(max_batch_size=4))
        a1, b1 = _make_request("a"), _make_request("b")
        a2 = _make_request("a")
        b2 = _make_request("b")
        a2.prompt_token_ids = a1.prompt_token_ids
        b2.prompt_token_ids = b1.prompt_token_ids

        for r in [a1, b1]:
            sched1.add_request(r)
        for r in [a2, b2]:
            sched2.add_request(r)

        # Combined.
        out = sched1.schedule()

        # Split.
        retired = sched2.retire()
        prefill = sched2.admit(free_kv_tokens=None)
        decode = sched2.decode_requests()

        assert [r.request_id for r in out.retired] == [r.request_id for r in retired]
        assert [r.request_id for r in out.prefill] == [r.request_id for r in prefill]
        assert [r.request_id for r in out.decode] == [r.request_id for r in decode]

    def test_full_retire_free_admit_cycle(self) -> None:
        """Simulate the full retire → free → admit cycle the engine uses."""
        sched = ContinuousScheduler(_make_config(max_batch_size=2))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c", prompt_len=5)
        for r in [a, b, c]:
            sched.add_request(r)

        # Step 1: admit a, b.
        sched.retire()
        prefill = sched.admit(free_kv_tokens=None)
        assert [r.request_id for r in prefill] == ["a", "b"]
        a.state = RequestState.DECODE
        b.state = RequestState.DECODE

        # Step 2: a finishes, c admitted with budget.
        a.state = RequestState.FINISHED
        retired = sched.retire()
        assert [r.request_id for r in retired] == ["a"]

        # Simulate: engine frees a's blocks, queries budget.
        new = sched.admit(free_kv_tokens=100)
        assert [r.request_id for r in new] == ["c"]

        decode = sched.decode_requests()
        assert [r.request_id for r in decode] == ["b"]


class TestSplitInterfaceBoundary:
    """Boundary conditions for the split interface methods."""

    def test_admit_zero_budget_admits_nothing(self) -> None:
        """free_kv_tokens=0 → 80% of 0 = 0 → no request can fit."""
        sched = ContinuousScheduler(_make_config(max_batch_size=10))
        sched.add_request(_make_request("a", prompt_len=1))

        new = sched.admit(free_kv_tokens=0)
        assert new == []
        assert len(sched.waiting) == 1

    def test_admit_exact_budget_exhaustion(self) -> None:
        """Budget exactly consumed — next request blocked."""
        sched = ContinuousScheduler(_make_config(max_batch_size=10))
        # 10 free tokens → 80% = 8 usable. Two requests of 4 each = exactly 8.
        a = _make_request("a", prompt_len=4)
        b = _make_request("b", prompt_len=4)
        c = _make_request("c", prompt_len=1)
        for r in [a, b, c]:
            sched.add_request(r)

        new = sched.admit(free_kv_tokens=10)
        assert [r.request_id for r in new] == ["a", "b"]
        assert len(sched.waiting) == 1  # c still waiting

    def test_head_of_line_blocking(self) -> None:
        """A large first request blocks smaller ones behind it."""
        sched = ContinuousScheduler(_make_config(max_batch_size=10))
        big = _make_request("big", prompt_len=20)
        small = _make_request("small", prompt_len=1)
        for r in [big, small]:
            sched.add_request(r)

        # 10 tokens → 80% = 8, big needs 20 → blocks, small not admitted either.
        new = sched.admit(free_kv_tokens=10)
        assert new == []
        assert len(sched.waiting) == 2

    def test_retire_empty_active(self) -> None:
        """retire() on empty active list returns empty."""
        sched = ContinuousScheduler(_make_config())
        assert sched.retire() == []

    def test_retire_all_non_terminal(self) -> None:
        """retire() when all active are DECODE returns empty, active unchanged."""
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        a, b = _make_request("a"), _make_request("b")
        for r in [a, b]:
            sched.add_request(r)
        sched.admit()
        a.state = RequestState.DECODE
        b.state = RequestState.DECODE

        retired = sched.retire()
        assert retired == []
        assert len(sched.active) == 2

    def test_decode_requests_empty_active(self) -> None:
        sched = ContinuousScheduler(_make_config())
        assert sched.decode_requests() == []

    def test_decode_requests_excludes_prefill_and_finished(self) -> None:
        """Only DECODE state passes through, not PREFILL/WAITING/FINISHED."""
        sched = ContinuousScheduler(_make_config(max_batch_size=4))
        a, b, c = _make_request("a"), _make_request("b"), _make_request("c")
        for r in [a, b, c]:
            sched.add_request(r)
        sched.admit()

        a.state = RequestState.DECODE
        b.state = RequestState.PREFILL
        c.state = RequestState.FINISHED

        decode = sched.decode_requests()
        assert [r.request_id for r in decode] == ["a"]

    def test_admit_allows_exactly_one_when_budget_tight(self) -> None:
        """Budget allows precisely one request, blocks the next."""
        sched = ContinuousScheduler(_make_config(max_batch_size=10))
        a = _make_request("a", prompt_len=3)
        b = _make_request("b", prompt_len=3)
        for r in [a, b]:
            sched.add_request(r)

        # 5 tokens → 80% = 4 usable. a needs 3 → fits, b needs 3 → 1 remaining < 3.
        new = sched.admit(free_kv_tokens=5)
        assert [r.request_id for r in new] == ["a"]
        assert len(sched.waiting) == 1


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
