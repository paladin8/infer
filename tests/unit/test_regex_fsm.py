"""Unit tests for the regex-to-FSM compiler."""

from __future__ import annotations

import pytest

from infer.structured.regex_fsm import DFA, compile_regex, minimize_dfa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accepts(pattern: str, text: str) -> bool:
    """Compile pattern and check if it accepts text."""
    dfa = compile_regex(pattern)
    return dfa.accepts(text)


# ---------------------------------------------------------------------------
# Literal matching
# ---------------------------------------------------------------------------


class TestLiteralMatch:
    def test_single_char(self) -> None:
        assert _accepts("a", "a")
        assert not _accepts("a", "b")
        assert not _accepts("a", "")
        assert not _accepts("a", "aa")

    def test_multi_char(self) -> None:
        assert _accepts("abc", "abc")
        assert not _accepts("abc", "ab")
        assert not _accepts("abc", "abcd")
        assert not _accepts("abc", "xyz")

    def test_empty_pattern(self) -> None:
        assert _accepts("", "")
        assert not _accepts("", "a")


# ---------------------------------------------------------------------------
# Alternation
# ---------------------------------------------------------------------------


class TestAlternation:
    def test_simple(self) -> None:
        assert _accepts("a|b", "a")
        assert _accepts("a|b", "b")
        assert not _accepts("a|b", "c")
        assert not _accepts("a|b", "")
        assert not _accepts("a|b", "ab")

    def test_multi_char_alternatives(self) -> None:
        assert _accepts("cat|dog", "cat")
        assert _accepts("cat|dog", "dog")
        assert not _accepts("cat|dog", "ca")
        assert not _accepts("cat|dog", "do")

    def test_three_alternatives(self) -> None:
        assert _accepts("a|b|c", "a")
        assert _accepts("a|b|c", "b")
        assert _accepts("a|b|c", "c")
        assert not _accepts("a|b|c", "d")

    def test_alternation_with_concat(self) -> None:
        assert _accepts("ab|cd", "ab")
        assert _accepts("ab|cd", "cd")
        assert not _accepts("ab|cd", "ac")


# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------


class TestCharacterClass:
    def test_simple_class(self) -> None:
        assert _accepts("[abc]", "a")
        assert _accepts("[abc]", "b")
        assert _accepts("[abc]", "c")
        assert not _accepts("[abc]", "d")

    def test_range(self) -> None:
        assert _accepts("[a-z]", "a")
        assert _accepts("[a-z]", "m")
        assert _accepts("[a-z]", "z")
        assert not _accepts("[a-z]", "A")
        assert not _accepts("[a-z]", "0")

    def test_negated_class(self) -> None:
        assert not _accepts("[^0-9]", "0")
        assert not _accepts("[^0-9]", "5")
        assert _accepts("[^0-9]", "a")
        assert _accepts("[^0-9]", "Z")

    def test_class_with_escape(self) -> None:
        assert _accepts("[\\d]", "5")
        assert not _accepts("[\\d]", "a")

    def test_mixed_range_and_literal(self) -> None:
        assert _accepts("[a-z0]", "a")
        assert _accepts("[a-z0]", "z")
        assert _accepts("[a-z0]", "0")
        assert not _accepts("[a-z0]", "1")


# ---------------------------------------------------------------------------
# Quantifiers
# ---------------------------------------------------------------------------


class TestQuantifierStar:
    def test_zero_matches(self) -> None:
        assert _accepts("a*", "")

    def test_one_match(self) -> None:
        assert _accepts("a*", "a")

    def test_many_matches(self) -> None:
        assert _accepts("a*", "aaa")

    def test_rejects_wrong_char(self) -> None:
        assert not _accepts("a*", "b")

    def test_with_suffix(self) -> None:
        assert _accepts("a*b", "b")
        assert _accepts("a*b", "ab")
        assert _accepts("a*b", "aab")
        assert not _accepts("a*b", "a")


class TestQuantifierPlus:
    def test_rejects_empty(self) -> None:
        assert not _accepts("a+", "")

    def test_one_match(self) -> None:
        assert _accepts("a+", "a")

    def test_many_matches(self) -> None:
        assert _accepts("a+", "aaa")

    def test_rejects_wrong_char(self) -> None:
        assert not _accepts("a+", "b")


class TestQuantifierOptional:
    def test_zero_matches(self) -> None:
        assert _accepts("a?", "")

    def test_one_match(self) -> None:
        assert _accepts("a?", "a")

    def test_rejects_two(self) -> None:
        assert not _accepts("a?", "aa")


class TestQuantifierRange:
    def test_exact(self) -> None:
        assert _accepts("a{3}", "aaa")
        assert not _accepts("a{3}", "aa")
        assert not _accepts("a{3}", "aaaa")

    def test_range(self) -> None:
        assert not _accepts("a{2,4}", "a")
        assert _accepts("a{2,4}", "aa")
        assert _accepts("a{2,4}", "aaa")
        assert _accepts("a{2,4}", "aaaa")
        assert not _accepts("a{2,4}", "aaaaa")

    def test_min_only(self) -> None:
        assert not _accepts("a{2,}", "a")
        assert _accepts("a{2,}", "aa")
        assert _accepts("a{2,}", "aaaaaaaaa")

    def test_zero_min(self) -> None:
        assert _accepts("a{0,2}", "")
        assert _accepts("a{0,2}", "a")
        assert _accepts("a{0,2}", "aa")
        assert not _accepts("a{0,2}", "aaa")


# ---------------------------------------------------------------------------
# Dot
# ---------------------------------------------------------------------------


class TestDot:
    def test_matches_any(self) -> None:
        assert _accepts(".", "a")
        assert _accepts(".", "Z")
        assert _accepts(".", "0")
        assert _accepts(".", " ")

    def test_rejects_empty(self) -> None:
        assert not _accepts(".", "")

    def test_rejects_newline(self) -> None:
        assert not _accepts(".", "\n")

    def test_dot_star(self) -> None:
        assert _accepts(".*", "")
        assert _accepts(".*", "hello world")


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_simple_group(self) -> None:
        assert _accepts("(ab)+", "ab")
        assert _accepts("(ab)+", "abab")
        assert not _accepts("(ab)+", "a")
        assert not _accepts("(ab)+", "")

    def test_nested_groups(self) -> None:
        assert _accepts("(a(bc))+", "abc")
        assert _accepts("(a(bc))+", "abcabc")
        assert not _accepts("(a(bc))+", "ab")

    def test_group_alternation(self) -> None:
        assert _accepts("(cat|dog)s", "cats")
        assert _accepts("(cat|dog)s", "dogs")
        assert not _accepts("(cat|dog)s", "cats1")


# ---------------------------------------------------------------------------
# Escape sequences
# ---------------------------------------------------------------------------


class TestEscapeSequences:
    def test_escape_dot(self) -> None:
        assert _accepts("\\.", ".")
        assert not _accepts("\\.", "a")

    def test_escape_backslash(self) -> None:
        assert _accepts("\\\\", "\\")

    def test_escape_star(self) -> None:
        assert _accepts("\\*", "*")
        assert not _accepts("\\*", "a")


# ---------------------------------------------------------------------------
# Shorthand classes
# ---------------------------------------------------------------------------


class TestShorthandClasses:
    def test_digit(self) -> None:
        assert _accepts("\\d", "0")
        assert _accepts("\\d", "9")
        assert not _accepts("\\d", "a")

    def test_word(self) -> None:
        assert _accepts("\\w", "a")
        assert _accepts("\\w", "Z")
        assert _accepts("\\w", "0")
        assert _accepts("\\w", "_")
        assert not _accepts("\\w", " ")

    def test_space(self) -> None:
        assert _accepts("\\s", " ")
        assert _accepts("\\s", "\t")
        assert _accepts("\\s", "\n")
        assert not _accepts("\\s", "a")

    def test_combined(self) -> None:
        assert _accepts("\\d+\\.\\d+", "3.14")
        assert _accepts("\\d+\\.\\d+", "0.0")
        assert not _accepts("\\d+\\.\\d+", "3.")
        assert not _accepts("\\d+\\.\\d+", ".14")


# ---------------------------------------------------------------------------
# Complex patterns
# ---------------------------------------------------------------------------


class TestComplexPatterns:
    def test_json_number(self) -> None:
        pattern = "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?"
        assert _accepts(pattern, "0")
        assert _accepts(pattern, "42")
        assert _accepts(pattern, "-1")
        assert _accepts(pattern, "3.14")
        assert _accepts(pattern, "-0.5")
        assert _accepts(pattern, "1e10")
        assert _accepts(pattern, "1E-5")
        assert _accepts(pattern, "1.5e+3")
        assert not _accepts(pattern, "01")  # leading zero
        assert not _accepts(pattern, ".")
        assert not _accepts(pattern, "e5")

    def test_json_string(self) -> None:
        # Simplified JSON string (no unicode escapes).
        pattern = '"([^"\\\\]|\\\\.)*"'
        assert _accepts(pattern, '""')
        assert _accepts(pattern, '"hello"')
        assert _accepts(pattern, '"hello world"')
        assert _accepts(pattern, '"a\\"b"')  # escaped quote
        assert not _accepts(pattern, '"')
        assert not _accepts(pattern, "hello")

    def test_json_boolean(self) -> None:
        pattern = "(true|false)"
        assert _accepts(pattern, "true")
        assert _accepts(pattern, "false")
        assert not _accepts(pattern, "True")
        assert not _accepts(pattern, "yes")

    def test_enum_values(self) -> None:
        pattern = '("red"|"green"|"blue")'
        assert _accepts(pattern, '"red"')
        assert _accepts(pattern, '"green"')
        assert _accepts(pattern, '"blue"')
        assert not _accepts(pattern, '"yellow"')
        assert not _accepts(pattern, "red")


# ---------------------------------------------------------------------------
# DFA minimization
# ---------------------------------------------------------------------------


class TestDFAMinimization:
    def test_removes_unreachable_states(self) -> None:
        """DFA with unreachable state should have it removed."""
        dfa = DFA(
            transitions={
                0: {"a": 1},
                1: {},
                2: {"b": 1},  # unreachable from 0
            },
            initial_state=0,
            accept_states={1},
            alphabet={"a", "b"},
        )
        minimized = minimize_dfa(dfa)
        # State 2 should be removed. Only states 0 and 1 remain.
        assert len(minimized.transitions) == 2

    def test_removes_dead_states(self) -> None:
        """DFA with dead state (no path to accept) should have it removed."""
        dfa = DFA(
            transitions={
                0: {"a": 1, "b": 2},
                1: {},  # accept
                2: {"c": 2},  # dead: loops forever, never reaches accept
            },
            initial_state=0,
            accept_states={1},
            alphabet={"a", "b", "c"},
        )
        minimized = minimize_dfa(dfa)
        # State 2 is dead — removed.
        assert minimized.accepts("a")
        assert not minimized.accepts("b")

    def test_preserves_valid_dfa(self) -> None:
        """A DFA with no unreachable/dead states should be unchanged."""
        dfa = compile_regex("ab")
        assert dfa.accepts("ab")
        minimized = minimize_dfa(dfa)
        assert minimized.accepts("ab")
        assert not minimized.accepts("a")


# ---------------------------------------------------------------------------
# DFA walk
# ---------------------------------------------------------------------------


class TestDFAWalk:
    def test_walk_returns_final_state(self) -> None:
        dfa = compile_regex("abc")
        state = dfa.walk("ab")
        assert state is not None
        assert state not in dfa.accept_states
        state2 = dfa.walk("c", start_state=state)
        assert state2 is not None
        assert state2 in dfa.accept_states

    def test_walk_returns_none_on_dead(self) -> None:
        dfa = compile_regex("abc")
        assert dfa.walk("x") is None

    def test_walk_empty_string(self) -> None:
        dfa = compile_regex("a*")
        state = dfa.walk("")
        assert state is not None
        assert state == dfa.initial_state


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_unmatched_paren(self) -> None:
        with pytest.raises(ValueError):
            compile_regex("(abc")

    def test_unmatched_bracket(self) -> None:
        with pytest.raises(ValueError):
            compile_regex("[abc")

    def test_empty_quantifier_braces(self) -> None:
        with pytest.raises(ValueError):
            compile_regex("a{}")
