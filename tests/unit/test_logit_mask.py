"""Unit tests for logit masking and structured output state."""

from __future__ import annotations

import torch
from outlines_core import Vocabulary

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


def _make_abc_vocab() -> Vocabulary:
    """Vocabulary with characters for simple patterns."""
    return Vocabulary(
        99,
        {
            "a": [0],
            "b": [1],
            "c": [2],
            "ab": [3],
            "bc": [4],
            "abc": [5],
            "d": [6],
        },
    )


BOOL_VOCAB_SIZE = 13  # tokens 0-12
ABC_VOCAB_SIZE = 7  # tokens 0-6


def _make_state(
    pattern: str, vocab: Vocabulary | None = None, vocab_size: int | None = None
) -> StructuredOutputState:
    """Create a StructuredOutputState from a regex pattern."""
    if vocab is None:
        vocab = _make_abc_vocab()
    guide = compile_guide(pattern, vocab)
    return StructuredOutputState(guide=guide, current_state=guide.initial_state)


# ---------------------------------------------------------------------------
# apply_structured_output_mask tests
# ---------------------------------------------------------------------------


class TestMaskDisallowsTokens:
    def test_disallowed_tokens_get_neg_inf(self) -> None:
        """Tokens not in the allowed set should have -inf logits."""
        state = _make_state("abc")
        logits = torch.ones(ABC_VOCAB_SIZE)
        masked = apply_structured_output_mask(logits, state)
        allowed = state.allowed_tokens()
        for i in range(ABC_VOCAB_SIZE):
            if i in allowed:
                assert masked[i].item() == 1.0, f"Token {i} should be allowed"
            else:
                assert masked[i].item() == float("-inf"), f"Token {i} should be masked"


class TestMaskPreservesAllowed:
    def test_allowed_tokens_keep_original(self) -> None:
        """Allowed tokens should keep their original logit values."""
        vocab = _make_bool_vocab()
        state = _make_state("(true|false)", vocab)
        logits = torch.tensor([5.0, 3.0, 1.0, 2.0, 4.0, 6.0, 0.5, 0.1, 0.2, 3.5, 1.5, 2.5, 0.8])
        masked = apply_structured_output_mask(logits, state)
        allowed = state.allowed_tokens()
        for i in allowed:
            assert masked[i].item() == logits[i].item()


class TestMaskEmptyAllowedSet:
    def test_all_tokens_masked_when_no_valid_in_range(self) -> None:
        """When no allowed tokens are within vocab range, all should be -inf.

        outlines-core includes the EOS token at terminal states, but if the
        EOS token ID is outside the logits tensor range, all logits get masked.
        """
        state = _make_state("abc")
        # Move to terminal state by consuming 'abc'
        state.advance(5)  # 'abc' -> terminal
        assert state.is_terminal()
        # The only allowed token is EOS (id=99), which is outside vocab range.
        logits = torch.ones(ABC_VOCAB_SIZE)
        masked = apply_structured_output_mask(logits, state)
        assert (masked == float("-inf")).all()


# ---------------------------------------------------------------------------
# StructuredOutputState tests
# ---------------------------------------------------------------------------


class TestAdvanceState:
    def test_state_advances(self) -> None:
        """State should advance after consuming a token."""
        state = _make_state("abc")
        initial = state.current_state
        state.advance(0)  # 'a'
        assert state.current_state != initial

    def test_multi_step_advance(self) -> None:
        """State should advance through multiple tokens."""
        state = _make_state("abc")
        state.advance(0)  # 'a'
        state.advance(1)  # 'b'
        state.advance(2)  # 'c'
        assert state.is_terminal()


class TestIsTerminalCheck:
    def test_not_terminal_at_start(self) -> None:
        """Non-optional pattern should not be terminal at start."""
        state = _make_state("abc")
        assert not state.is_terminal()

    def test_terminal_after_match(self) -> None:
        """Should be terminal after consuming the full pattern."""
        state = _make_state("abc")
        state.advance(5)  # 'abc'
        assert state.is_terminal()

    def test_optional_terminal_at_start(self) -> None:
        """Optional pattern should be terminal at start."""
        state = _make_state("a?")
        assert state.is_terminal()


# ---------------------------------------------------------------------------
# Integration with sample_token
# ---------------------------------------------------------------------------


class TestIntegrationWithSampleToken:
    def test_masked_logits_produce_valid_token(self) -> None:
        """Sampling from masked logits should produce a valid token."""
        state = _make_state("abc")
        logits = torch.randn(ABC_VOCAB_SIZE)
        masked = apply_structured_output_mask(logits, state)

        params = SamplingParams(temperature=0.0)
        token = sample_token(masked, [], params)
        assert token in state.allowed_tokens()

    def test_full_generation_loop(self) -> None:
        """Simulate a full generation loop using logit masking."""
        state = _make_state("abc")
        generated: list[int] = []
        id_to_str = {0: "a", 1: "b", 2: "c", 3: "ab", 4: "bc", 5: "abc", 6: "d"}

        for _ in range(10):  # safety limit
            if state.is_terminal():
                break
            logits = torch.randn(ABC_VOCAB_SIZE)
            masked = apply_structured_output_mask(logits, state)
            params = SamplingParams(temperature=0.0)
            token = sample_token(masked, [], params)
            generated.append(token)
            state.advance(token)

        # Verify we reached terminal state.
        assert state.is_terminal()
        # Verify generated string matches pattern.
        text = "".join(id_to_str[t] for t in generated)
        assert text == "abc"


class TestZeroOverheadWhenNone:
    def test_no_mask_when_none(self) -> None:
        """When structured_state is None, sample_token should work normally."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        params = SamplingParams(temperature=0.0)
        # sample_token with no structured state should still work.
        token = sample_token(logits, [], params)
        assert token == 1  # argmax of [1, 5, 3, 2]
