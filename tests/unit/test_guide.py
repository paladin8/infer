"""Unit tests for the outlines-core backed TokenGuide."""

from __future__ import annotations

import json

import pytest
from outlines_core import Vocabulary

from infer.structured.guide import build_vocabulary, clear_guide_cache, compile_guide

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
        },
    )


# ---------------------------------------------------------------------------
# TokenGuide basic tests
# ---------------------------------------------------------------------------


class TestBasicRegex:
    def test_allowed_tokens_at_start(self) -> None:
        """Correct tokens allowed at initial state."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        allowed = guide.allowed_tokens(guide.initial_state)
        # Should include tokens starting with 't' and 'f'
        assert 0 in allowed  # 't'
        assert 4 in allowed  # 'f'
        assert 8 in allowed  # 'true'
        assert 9 in allowed  # 'false'
        # Should not include tokens that don't start valid
        assert 1 not in allowed  # 'r'
        assert 2 not in allowed  # 'u'

    def test_next_state_advances(self) -> None:
        """next_state should advance to a new state."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        initial = guide.initial_state
        next_s = guide.next_state(initial, 0)  # 't'
        assert next_s != initial

    def test_terminal_after_full_match(self) -> None:
        """State should be terminal after consuming full pattern."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        state = guide.initial_state
        state = guide.next_state(state, 8)  # 'true'
        assert guide.is_terminal(state)

    def test_not_terminal_mid_match(self) -> None:
        """State should not be terminal mid-pattern."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        state = guide.initial_state
        state = guide.next_state(state, 0)  # 't'
        assert not guide.is_terminal(state)


class TestMultiCharTokens:
    def test_multi_char_token_advance(self) -> None:
        """Multi-character tokens should advance through multiple DFA states."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        state = guide.initial_state
        # 'tr' should advance two characters at once
        state = guide.next_state(state, 10)  # 'tr'
        assert not guide.is_terminal(state)
        # After 'tr', 'u' should be allowed
        allowed = guide.allowed_tokens(state)
        assert 2 in allowed  # 'u'

    def test_full_match_via_multi_char(self) -> None:
        """Reach terminal via multi-char tokens."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        state = guide.initial_state
        state = guide.next_state(state, 11)  # 'fal'
        state = guide.next_state(state, 12)  # 'se'
        assert guide.is_terminal(state)


class TestInvalidToken:
    def test_raises_key_error(self) -> None:
        """next_state should raise KeyError for disallowed tokens."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        with pytest.raises(KeyError):
            guide.next_state(guide.initial_state, 1)  # 'r' not valid at start

    def test_dead_state_returns_empty(self) -> None:
        """allowed_tokens should return empty set for invalid states."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        # Use a state that doesn't exist
        assert guide.allowed_tokens(999999) == set()


class TestCacheHit:
    def test_same_pattern_cached(self) -> None:
        """Same pattern + vocab should return cached guide."""
        clear_guide_cache()
        vocab = _make_bool_vocab()
        g1 = compile_guide("(true|false)", vocab)
        g2 = compile_guide("(true|false)", vocab)
        assert g1 is g2

    def test_different_pattern_not_cached(self) -> None:
        """Different patterns should produce different guides."""
        clear_guide_cache()
        vocab = _make_abc_vocab()
        g1 = compile_guide("abc", vocab)
        g2 = compile_guide("ab", vocab)
        assert g1 is not g2


# ---------------------------------------------------------------------------
# JSON schema mode
# ---------------------------------------------------------------------------


class TestJsonSchemaMode:
    def test_json_schema_compilation(self) -> None:
        """JSON schema should compile via build_regex_from_schema."""
        vocab = _make_bool_vocab()
        schema_str = json.dumps({"type": "boolean"})
        guide = compile_guide(schema_str, vocab, mode="json_schema")
        # Should allow tokens for 'true' and 'false'
        allowed = guide.allowed_tokens(guide.initial_state)
        assert 8 in allowed or 0 in allowed  # 'true' or 't'
        assert 9 in allowed or 4 in allowed  # 'false' or 'f'


class TestInvalidMode:
    def test_rejects_bad_mode(self) -> None:
        """Should raise ValueError for unsupported mode."""
        vocab = _make_bool_vocab()
        with pytest.raises(ValueError, match="Unsupported mode"):
            compile_guide("abc", vocab, mode="invalid")


# ---------------------------------------------------------------------------
# Full generation simulation
# ---------------------------------------------------------------------------


class TestFullGenerationSimulation:
    def test_simulate_generation(self) -> None:
        """Simulate token-by-token generation, verify result matches pattern."""
        vocab = _make_bool_vocab()
        guide = compile_guide("(true|false)", vocab)
        state = guide.initial_state
        generated: list[int] = []

        # Map token IDs to strings for verification
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

        for _ in range(10):
            if guide.is_terminal(state):
                break
            allowed = guide.allowed_tokens(state)
            if not allowed:
                break
            # Pick the first allowed token (deterministic)
            token = min(allowed)
            generated.append(token)
            state = guide.next_state(state, token)

        assert guide.is_terminal(state)
        text = "".join(id_to_str[t] for t in generated)
        assert text in ("true", "false"), f"Got {text!r}"


# ---------------------------------------------------------------------------
# build_vocabulary helper
# ---------------------------------------------------------------------------


class TestBuildVocabulary:
    def test_builds_from_tokenizer_vocab(self) -> None:
        """build_vocabulary should convert {str: int} to outlines-core Vocabulary."""
        raw_vocab = {"hello": 0, "world": 1, "<eos>": 2}
        v = build_vocabulary(raw_vocab, eos_token_id=2)
        assert v.get_eos_token_id() == 2
