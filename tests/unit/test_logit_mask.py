"""Unit tests for logit masking and structured output state."""

from __future__ import annotations

import torch

from infer.engine.sampler import SamplingParams, sample_token
from infer.structured.logit_mask import StructuredOutputState, apply_structured_output_mask
from infer.structured.token_fsm import TokenVocabularyIndex, compile_token_fsm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_vocab() -> TokenVocabularyIndex:
    """A small vocabulary for testing."""
    return TokenVocabularyIndex(
        {
            "a": 0,
            "b": 1,
            "c": 2,
            "ab": 3,
            "bc": 4,
            "abc": 5,
            "d": 6,
            "1": 7,
            "2": 8,
        }
    )


def _make_state(pattern: str, vocab: TokenVocabularyIndex | None = None) -> StructuredOutputState:
    """Create a StructuredOutputState from a regex pattern."""
    if vocab is None:
        vocab = _make_simple_vocab()
    fsm = compile_token_fsm(pattern, vocab)
    return StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)


# ---------------------------------------------------------------------------
# apply_structured_output_mask tests
# ---------------------------------------------------------------------------


class TestMaskDisallowsTokens:
    def test_disallowed_tokens_get_neg_inf(self) -> None:
        """Tokens not in the allowed set should have -inf logits."""
        state = _make_state("abc")
        logits = torch.ones(9)  # vocab size 9
        masked = apply_structured_output_mask(logits, state)
        # At initial state of 'abc', only tokens starting with 'a' are valid:
        # 'a' (0), 'ab' (3), 'abc' (5)
        allowed = state.allowed_tokens()
        for i in range(9):
            if i in allowed:
                assert masked[i].item() == 1.0, f"Token {i} should be allowed"
            else:
                assert masked[i].item() == float("-inf"), f"Token {i} should be masked"


class TestMaskPreservesAllowed:
    def test_allowed_tokens_keep_original(self) -> None:
        """Allowed tokens should keep their original logit values."""
        state = _make_state("a|b")
        logits = torch.tensor([5.0, 3.0, 1.0, 2.0, 4.0, 6.0, 0.5, 0.1, 0.2])
        masked = apply_structured_output_mask(logits, state)
        allowed = state.allowed_tokens()
        for i in allowed:
            assert masked[i].item() == logits[i].item()


class TestMaskEmptyAllowedSet:
    def test_all_tokens_masked(self) -> None:
        """When no tokens are allowed, all should be -inf."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        # Move to terminal state (after 'abc').
        state = StructuredOutputState(fsm=fsm, current_state=fsm.initial_state)
        state.advance(5)  # 'abc' -> terminal
        assert state.is_terminal()
        # No more tokens should be allowed.
        logits = torch.ones(9)
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
        logits = torch.randn(9)
        masked = apply_structured_output_mask(logits, state)

        params = SamplingParams(temperature=0.0)
        token = sample_token(masked, [], params)
        assert token in state.allowed_tokens()

    def test_full_generation_loop(self) -> None:
        """Simulate a full generation loop using logit masking."""
        state = _make_state("abc")
        generated: list[int] = []
        vocab_strings = _make_simple_vocab().token_strings

        for _ in range(10):  # safety limit
            if state.is_terminal() and not state.allowed_tokens():
                break
            logits = torch.randn(9)
            masked = apply_structured_output_mask(logits, state)
            params = SamplingParams(temperature=0.0)
            token = sample_token(masked, [], params)
            generated.append(token)
            state.advance(token)

        # Verify we reached terminal state.
        assert state.is_terminal()
        # Verify generated string matches pattern.
        text = "".join(vocab_strings[t] for t in generated)
        assert text == "abc"


class TestZeroOverheadWhenNone:
    def test_no_mask_when_none(self) -> None:
        """When structured_state is None, sample_token should work normally."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        params = SamplingParams(temperature=0.0)
        # sample_token with no structured state should still work.
        token = sample_token(logits, [], params)
        assert token == 1  # argmax of [1, 5, 3, 2]
