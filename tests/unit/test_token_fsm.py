"""Unit tests for TokenVocabularyIndex and TokenFSM."""

from __future__ import annotations

import pytest

from infer.structured.token_fsm import (
    TokenVocabularyIndex,
    clear_fsm_cache,
    compile_token_fsm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vocab(tokens: dict[str, int]) -> TokenVocabularyIndex:
    """Create a TokenVocabularyIndex from a simple vocabulary."""
    return TokenVocabularyIndex(tokens)


def _make_simple_vocab() -> TokenVocabularyIndex:
    """A small vocabulary for testing."""
    return _make_vocab(
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
            "12": 9,
            " ": 10,
        }
    )


# ---------------------------------------------------------------------------
# TokenVocabularyIndex tests
# ---------------------------------------------------------------------------


class TestVocabIndexBasic:
    def test_builds_from_vocab(self) -> None:
        vocab = _make_vocab({"hello": 0, "world": 1})
        assert vocab.token_strings[0] == "hello"
        assert vocab.token_strings[1] == "world"

    def test_sentencepiece_space(self) -> None:
        """SentencePiece \u2581 prefix should be decoded as space."""
        vocab = _make_vocab({"\u2581hello": 0})
        assert vocab.token_strings[0] == " hello"

    def test_empty_vocab(self) -> None:
        vocab = _make_vocab({})
        assert len(vocab.token_strings) == 0


class TestVocabIndexSpecialTokens:
    def test_skips_special_tokens(self) -> None:
        vocab = _make_vocab(
            {
                "<s>": 0,
                "</s>": 1,
                "<pad>": 2,
                "hello": 3,
            }
        )
        assert 0 not in vocab.token_strings
        assert 1 not in vocab.token_strings
        assert 2 not in vocab.token_strings
        assert vocab.token_strings[3] == "hello"

    def test_byte_fallback_printable(self) -> None:
        """Byte fallback tokens for printable ASCII should be decoded."""
        vocab = _make_vocab({"<0x41>": 0})  # 'A'
        assert vocab.token_strings[0] == "A"

    def test_byte_fallback_nonprintable(self) -> None:
        """Byte fallback tokens for non-printable chars should be skipped."""
        vocab = _make_vocab({"<0x01>": 0})
        assert 0 not in vocab.token_strings


# ---------------------------------------------------------------------------
# TokenFSM tests
# ---------------------------------------------------------------------------


class TestAllowedTokensInitialState:
    def test_simple_pattern(self) -> None:
        """Pattern 'abc' should allow tokens 'a' and 'ab' and 'abc' at initial state."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        allowed = fsm.allowed_tokens(fsm.initial_state)
        # Token 'a' (id=0) leads to state after 'a' — valid.
        assert 0 in allowed
        # Token 'ab' (id=3) leads to state after 'ab' — valid.
        assert 3 in allowed
        # Token 'abc' (id=5) leads to accept state — valid.
        assert 5 in allowed
        # Token 'b' (id=1) — 'b' is not a valid start for 'abc'.
        assert 1 not in allowed
        # Token 'd' (id=6) — not in pattern.
        assert 6 not in allowed

    def test_alternation_pattern(self) -> None:
        """Pattern 'a|b' should allow both 'a' and 'b' at initial state."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("a|b", vocab)
        allowed = fsm.allowed_tokens(fsm.initial_state)
        assert 0 in allowed  # 'a'
        assert 1 in allowed  # 'b'
        assert 2 not in allowed  # 'c'


class TestAllowedTokensMidState:
    def test_after_first_char(self) -> None:
        """After consuming 'a' in pattern 'abc', only 'b' and 'bc' should be allowed."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        # Advance past 'a'.
        state = fsm.next_state(fsm.initial_state, 0)  # token 'a'
        allowed = fsm.allowed_tokens(state)
        assert 1 in allowed  # 'b'
        assert 4 in allowed  # 'bc'
        assert 0 not in allowed  # 'a'
        assert 5 not in allowed  # 'abc'


class TestNextStateSingleChar:
    def test_single_char_token(self) -> None:
        """Advancing with single-char token 'a' in pattern 'abc'."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        state1 = fsm.next_state(fsm.initial_state, 0)  # 'a'
        assert state1 != fsm.initial_state
        assert not fsm.is_terminal(state1)


class TestNextStateMultiChar:
    def test_multi_char_token(self) -> None:
        """Advancing with multi-char token 'ab' in pattern 'abc' should skip to state after 'ab'."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        state_ab = fsm.next_state(fsm.initial_state, 3)  # 'ab'
        # Now only 'c' should be allowed.
        allowed = fsm.allowed_tokens(state_ab)
        assert 2 in allowed  # 'c'
        assert 0 not in allowed  # 'a'

    def test_full_token_reaches_accept(self) -> None:
        """Token 'abc' at initial state of pattern 'abc' should reach terminal."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        state = fsm.next_state(fsm.initial_state, 5)  # 'abc'
        assert fsm.is_terminal(state)


class TestIsTerminal:
    def test_initial_state_for_star(self) -> None:
        """Pattern 'a*' — initial state is terminal (zero matches)."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("a*", vocab)
        assert fsm.is_terminal(fsm.initial_state)

    def test_initial_state_for_required(self) -> None:
        """Pattern 'a' — initial state is NOT terminal."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("a", vocab)
        assert not fsm.is_terminal(fsm.initial_state)

    def test_after_match(self) -> None:
        """Pattern 'a' — state after 'a' is terminal."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("a", vocab)
        state = fsm.next_state(fsm.initial_state, 0)
        assert fsm.is_terminal(state)


class TestCacheHit:
    def test_same_pattern_cached(self) -> None:
        """Same pattern with same vocab should return cached FSM."""
        clear_fsm_cache()
        vocab = _make_simple_vocab()
        fsm1 = compile_token_fsm("abc", vocab)
        fsm2 = compile_token_fsm("abc", vocab)
        assert fsm1 is fsm2

    def test_different_pattern_not_cached(self) -> None:
        """Different patterns should produce different FSMs."""
        clear_fsm_cache()
        vocab = _make_simple_vocab()
        fsm1 = compile_token_fsm("abc", vocab)
        fsm2 = compile_token_fsm("ab", vocab)
        assert fsm1 is not fsm2


class TestMultiCharSpanningBoundary:
    def test_token_spanning_boundary(self) -> None:
        """Token 'ab' in pattern 'a(bc|bd)' — 'ab' spans from 'a' into the group."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc|abd", vocab)
        # 'ab' should be allowed at initial state: it matches 'ab' prefix of 'abc'.
        allowed = fsm.allowed_tokens(fsm.initial_state)
        assert 3 in allowed  # 'ab'


class TestNoAllowedTokensDeadState:
    def test_dead_state(self) -> None:
        """A state with no valid transitions should return empty allowed set."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        # After completing 'abc', we're in accept state.
        state = fsm.next_state(fsm.initial_state, 5)  # 'abc'
        # No more tokens should be allowed (pattern is complete).
        allowed = fsm.allowed_tokens(state)
        assert len(allowed) == 0


class TestFullGenerationSimulation:
    def test_simulate_generation(self) -> None:
        """Simulate token-by-token generation and verify the result matches."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)

        # Generate by picking tokens.
        state = fsm.initial_state
        generated: list[str] = []

        # Pick 'a' (id=0).
        assert 0 in fsm.allowed_tokens(state)
        state = fsm.next_state(state, 0)
        generated.append("a")

        # Pick 'bc' (id=4).
        assert 4 in fsm.allowed_tokens(state)
        state = fsm.next_state(state, 4)
        generated.append("bc")

        assert fsm.is_terminal(state)
        assert "".join(generated) == "abc"

    def test_simulate_with_multi_char(self) -> None:
        """Simulate with multi-char token 'abc' (id=5)."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)

        state = fsm.initial_state
        assert 5 in fsm.allowed_tokens(state)
        state = fsm.next_state(state, 5)
        assert fsm.is_terminal(state)


class TestJsonSchemaMode:
    def test_json_schema_compilation(self) -> None:
        """Compile a JSON schema to a TokenFSM."""
        clear_fsm_cache()
        vocab = _make_vocab(
            {
                "{": 10,
                "}": 11,
                '"': 12,
                ":": 13,
                "true": 14,
                "false": 15,
                ",": 16,
                "null": 17,
                "0": 18,
                "1": 19,
                " ": 20,
            }
        )
        import json

        schema = json.dumps({"type": "boolean"})
        fsm = compile_token_fsm(schema, vocab, mode="json_schema")
        # 'true' and 'false' should be allowed at initial state.
        allowed = fsm.allowed_tokens(fsm.initial_state)
        assert 14 in allowed  # 'true'
        assert 15 in allowed  # 'false'
        assert 10 not in allowed  # '{'


class TestInvalidTokenAtState:
    def test_raises_key_error(self) -> None:
        """next_state should raise KeyError for invalid token."""
        vocab = _make_simple_vocab()
        fsm = compile_token_fsm("abc", vocab)
        with pytest.raises(KeyError):
            fsm.next_state(fsm.initial_state, 6)  # 'd' not valid for 'abc'
