"""Unit tests for structured output integration (API, engine, sampling)."""

from __future__ import annotations

import json
import typing

import pytest
import torch

from infer.engine.request import Request
from infer.engine.sampler import SamplingParams, sample_token
from infer.structured.logit_mask import StructuredOutputState, apply_structured_output_mask
from infer.structured.token_fsm import TokenVocabularyIndex, compile_token_fsm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vocab() -> TokenVocabularyIndex:
    """A vocabulary with JSON-relevant tokens."""
    tokens: dict[str, int] = {
        "{": 0,
        "}": 1,
        '"': 2,
        ":": 3,
        ",": 4,
        " ": 5,
        "t": 6,
        "r": 7,
        "u": 8,
        "e": 9,
        "f": 10,
        "a": 11,
        "l": 12,
        "s": 13,
        "n": 14,
        "true": 15,
        "false": 16,
        "null": 17,
        "0": 18,
        "1": 19,
        "2": 20,
        "3": 21,
        "4": 22,
        "5": 23,
        "6": 24,
        "7": 25,
        "8": 26,
        "9": 27,
        "[": 28,
        "]": 29,
        ".": 30,
        "-": 31,
        "\\": 32,
    }
    return TokenVocabularyIndex(tokens)


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
        vocab = _make_vocab()
        fsm = compile_token_fsm("(true|false)", vocab)
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)

        req = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
            structured_output_state=state,
        )
        assert req.structured_output_state is not None
        assert req.structured_output_state.fsm is fsm


# ---------------------------------------------------------------------------
# End-to-end regex tests
# ---------------------------------------------------------------------------


class TestEndToEndRegexSimple:
    def test_regex_produces_matching_output(self) -> None:
        """Full pipeline with regex pattern produces matching output."""
        # Use simple vocabulary.
        vocab = _make_vocab()
        pattern = "(true|false)"
        fsm = compile_token_fsm(pattern, vocab)
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)

        # Simulate generation.
        vocab_size = 33
        generated: list[int] = []
        for _ in range(20):  # safety limit
            if state.is_terminal() and not state.allowed_tokens():
                break
            logits = torch.randn(vocab_size)
            masked = apply_structured_output_mask(logits, state)
            params = SamplingParams(temperature=0.0)
            token = sample_token(masked, [], params)
            generated.append(token)
            state.advance(token)

        # Build result string.
        token_strings = vocab.token_strings
        result = "".join(token_strings[t] for t in generated)
        assert result in ("true", "false"), f"Got {result!r}"


class TestEndToEndJsonSchemaSimple:
    def test_json_schema_produces_valid_json(self) -> None:
        """Full pipeline with JSON schema produces valid JSON."""
        vocab = _make_vocab()
        schema_str = json.dumps({"type": "boolean"})
        fsm = compile_token_fsm(schema_str, vocab, mode="json_schema")
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)

        # Simulate generation.
        vocab_size = 33
        generated: list[int] = []
        for _ in range(20):
            if state.is_terminal() and not state.allowed_tokens():
                break
            logits = torch.randn(vocab_size)
            masked = apply_structured_output_mask(logits, state)
            params = SamplingParams(temperature=0.0)
            token = sample_token(masked, [], params)
            generated.append(token)
            state.advance(token)

        token_strings = vocab.token_strings
        result = "".join(token_strings[t] for t in generated)
        assert result in ("true", "false"), f"Got {result!r}"


# ---------------------------------------------------------------------------
# Sampling integration
# ---------------------------------------------------------------------------


class TestSampleTokenWithStructuredState:
    def test_structured_state_passed_to_sample(self) -> None:
        """sample_token should use structured_state when provided."""
        vocab = _make_vocab()
        fsm = compile_token_fsm("(true|false)", vocab)
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)

        logits = torch.randn(33)
        params = SamplingParams(temperature=0.0)
        # The structured state should constrain sampling.
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
    def test_eos_at_terminal(self) -> None:
        """EOS should stop generation when FSM is at terminal state."""
        from infer.engine.runner_helpers import check_stop

        vocab = _make_vocab()
        fsm = compile_token_fsm("a", vocab)
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)
        state.advance(11)  # 'a' -> terminal

        # Create a mock tokenizer that reports token 99 as EOS.
        class MockTokenizer:
            eos_token_ids: typing.ClassVar[set[int]] = {99}

        req = Request(
            request_id="test",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(),
            arrival_time_s=0.0,
            structured_output_state=state,
        )
        req.generated_token_ids = [11]  # 'a'
        finished, reason = check_stop(req, 99, MockTokenizer())  # type: ignore[arg-type]
        assert finished
        assert reason == "eos"

    def test_eos_not_at_terminal(self) -> None:
        """EOS should NOT stop generation when FSM is NOT at terminal state."""
        from infer.engine.runner_helpers import check_stop

        vocab = _make_vocab()
        fsm = compile_token_fsm("ab", vocab)
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)
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
        req.generated_token_ids = [11]  # 'a'
        finished, _reason = check_stop(req, 99, MockTokenizer())  # type: ignore[arg-type]
        assert not finished

    def test_fsm_exhausted_stops(self) -> None:
        """When FSM is at terminal with no more allowed tokens, generation should stop."""
        from infer.engine.runner_helpers import check_stop

        vocab = _make_vocab()
        fsm = compile_token_fsm("a", vocab)
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)
        state.advance(11)  # 'a' -> terminal, no more tokens allowed

        class MockTokenizer:
            eos_token_ids: typing.ClassVar[set[int]] = set()  # no EOS tokens

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
