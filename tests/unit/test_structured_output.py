"""Unit tests for structured output integration (API, engine, sampling)."""

from __future__ import annotations

import json
import typing

import pytest
import torch
from outlines_core import Vocabulary

from infer.engine.request import Request
from infer.engine.sampler import SamplingParams, sample_token
from infer.structured.guide import compile_guide
from infer.structured.logit_mask import StructuredOutputState, apply_structured_output_mask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bool_vocab() -> Vocabulary:
    """Vocabulary with all characters needed for (true|false)."""
    return Vocabulary(
        99,
        {
            "t": [0],
            "r": [1],
            "u": [2],
            "e": [3],
            "f": [4],
            "a": [5],
            "l": [6],
            "s": [7],
            "true": [8],
            "false": [9],
            "tr": [10],
            "fal": [11],
            "se": [12],
        },
    )


BOOL_VOCAB_SIZE = 13


# ---------------------------------------------------------------------------
# ResponseFormat validation tests
# ---------------------------------------------------------------------------


class TestResponseFormatJsonSchema:
    def test_accepts_json_schema(self) -> None:
        """API should accept json_schema format."""
        from infer.server.api import ResponseFormat

        rf = ResponseFormat(type="json_schema", schema_={"type": "boolean"})
        assert rf.type == "json_schema"
        assert rf.schema_ == {"type": "boolean"}

    def test_accepts_regex(self) -> None:
        """API should accept regex format."""
        from infer.server.api import ResponseFormat

        rf = ResponseFormat(type="regex", pattern="(true|false)")
        assert rf.type == "regex"
        assert rf.pattern == "(true|false)"


class TestResponseFormatInvalidType:
    def test_rejects_invalid(self) -> None:
        """API should reject invalid format types."""
        from pydantic import ValidationError

        from infer.server.api import ResponseFormat

        with pytest.raises(ValidationError, match="type"):
            ResponseFormat(type="invalid")


# ---------------------------------------------------------------------------
# Zero-overhead tests
# ---------------------------------------------------------------------------


class TestNoResponseFormatZeroOverhead:
    def test_no_fsm_state_when_not_set(self) -> None:
        """When response_format is not set, Request should have no FSM state."""
        req = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
        )
        assert req.structured_output_state is None


class TestStructuredStateOnRequest:
    def test_request_carries_state(self) -> None:
        """Request should carry FSM state when structured output is configured."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        state = StructuredOutputState(guide=guide, current_state=guide.initial_state)

        req = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
            structured_output_state=state,
        )
        assert req.structured_output_state is not None
        assert req.structured_output_state.guide is guide


# ---------------------------------------------------------------------------
# End-to-end regex tests
# ---------------------------------------------------------------------------


class TestEndToEndRegexSimple:
    def test_regex_produces_matching_output(self) -> None:
        """Full pipeline with regex pattern produces matching output."""
        vocab = _make_bool_vocab()
        pattern = "(true|false)"
        guide = compile_guide(pattern, vocab)
        state = StructuredOutputState(guide=guide, current_state=guide.initial_state)

        id_to_str = {
            0: "t",
            1: "r",
            2: "u",
            3: "e",
            4: "f",
            5: "a",
            6: "l",
            7: "s",
            8: "true",
            9: "false",
            10: "tr",
            11: "fal",
            12: "se",
        }

        generated: list[int] = []
        for _ in range(20):
            if state.is_terminal():
                break
            logits = torch.randn(BOOL_VOCAB_SIZE)
            masked = apply_structured_output_mask(logits, state)
            params = SamplingParams(temperature=0.0)
            token = sample_token(masked, [], params)
            generated.append(token)
            state.advance(token)

        result = "".join(id_to_str[t] for t in generated)
        assert result in ("true", "false"), f"Got {result!r}"


class TestEndToEndJsonSchemaSimple:
    def test_json_schema_produces_valid_json(self) -> None:
        """Full pipeline with JSON schema produces valid JSON."""
        vocab = _make_bool_vocab()
        schema_str = json.dumps({"type": "boolean"})
        guide = compile_guide(schema_str, vocab, mode="json_schema")
        state = StructuredOutputState(guide=guide, current_state=guide.initial_state)

        id_to_str = {
            0: "t",
            1: "r",
            2: "u",
            3: "e",
            4: "f",
            5: "a",
            6: "l",
            7: "s",
            8: "true",
            9: "false",
            10: "tr",
            11: "fal",
            12: "se",
        }

        generated: list[int] = []
        for _ in range(20):
            if state.is_terminal():
                break
            logits = torch.randn(BOOL_VOCAB_SIZE)
            masked = apply_structured_output_mask(logits, state)
            params = SamplingParams(temperature=0.0)
            token = sample_token(masked, [], params)
            generated.append(token)
            state.advance(token)

        result = "".join(id_to_str[t] for t in generated)
        assert result in ("true", "false"), f"Got {result!r}"


# ---------------------------------------------------------------------------
# Sampling integration
# ---------------------------------------------------------------------------


class TestSampleTokenWithStructuredState:
    def test_structured_state_passed_to_sample(self) -> None:
        """sample_token should use structured_state when provided."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        state = StructuredOutputState(guide=guide, current_state=guide.initial_state)

        logits = torch.randn(BOOL_VOCAB_SIZE)
        params = SamplingParams(temperature=0.0)
        token = sample_token(logits, [], params, structured_state=state)
        assert token in state.allowed_tokens()

    def test_without_structured_state(self) -> None:
        """sample_token without structured_state should work normally."""
        logits = torch.tensor([1.0, 5.0, 3.0])
        params = SamplingParams(temperature=0.0)
        token = sample_token(logits, [], params, structured_state=None)
        assert token == 1  # argmax


# ---------------------------------------------------------------------------
# Runner helpers integration
# ---------------------------------------------------------------------------


class TestCheckStopWithStructuredOutput:
    def _make_vocab_for_ab(self) -> Vocabulary:
        return Vocabulary(99, {"a": [11], "b": [12]})

    def test_eos_at_terminal(self) -> None:
        """EOS should stop generation when FSM is at terminal state."""
        from infer.engine.runner_helpers import check_stop

        vocab = self._make_vocab_for_ab()
        guide = compile_guide("a", vocab)
        state = StructuredOutputState(guide=guide, current_state=guide.initial_state)
        state.advance(11)  # 'a' -> terminal

        class MockTokenizer:
            eos_token_ids: typing.ClassVar[set[int]] = {99}

        req = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
            structured_output_state=state,
        )
        req.generated_token_ids = [11]
        finished, reason = check_stop(req, 99, MockTokenizer())  # type: ignore[arg-type]
        assert finished
        assert reason == "eos"

    def test_eos_not_at_terminal(self) -> None:
        """EOS should NOT stop generation when FSM is NOT at terminal state."""
        from infer.engine.runner_helpers import check_stop

        vocab = self._make_vocab_for_ab()
        guide = compile_guide("ab", vocab)
        state = StructuredOutputState(guide=guide, current_state=guide.initial_state)
        state.advance(11)  # 'a' -> not terminal yet

        class MockTokenizer:
            eos_token_ids: typing.ClassVar[set[int]] = {99}

        req = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_new_tokens=100),
            arrival_time_s=0.0,
            structured_output_state=state,
        )
        req.generated_token_ids = [11]
        finished, _reason = check_stop(req, 99, MockTokenizer())  # type: ignore[arg-type]
        assert not finished

    def test_fsm_exhausted_stops(self) -> None:
        """When FSM is at terminal with only EOS in allowed tokens, generation should stop."""
        from infer.engine.runner_helpers import check_stop

        vocab = self._make_vocab_for_ab()
        guide = compile_guide("a", vocab)
        state = StructuredOutputState(guide=guide, current_state=guide.initial_state)
        state.advance(11)  # 'a' -> terminal

        # Mock tokenizer knows about EOS token 99 (same as vocab's eos_token_id).
        class MockTokenizer:
            eos_token_ids: typing.ClassVar[set[int]] = {99}

        req = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_new_tokens=100),
            arrival_time_s=0.0,
            structured_output_state=state,
        )
        req.generated_token_ids = [11]
        finished, reason = check_stop(req, 11, MockTokenizer())  # type: ignore[arg-type]
        assert finished
        assert reason == "eos"
