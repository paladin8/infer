"""Token-level FSM guide backed by outlines-core.

Wraps ``outlines_core.Guide`` / ``Index`` / ``Vocabulary`` to provide
token-level guidance for structured output generation. Replaces the
from-scratch regex FSM, JSON schema compiler, and token FSM modules.
"""

from __future__ import annotations

from outlines_core import Index, Vocabulary
from outlines_core.json_schema import build_regex_from_schema


class TokenGuide:
    """Token-level FSM guide backed by outlines-core.

    Wraps an ``outlines_core.Index`` to provide:
    - ``initial_state``: starting state ID.
    - ``allowed_tokens(state)``: set of valid token IDs at a state.
    - ``next_state(state, token_id)``: advance after accepting a token.
    - ``is_terminal(state)``: whether generation can stop at a state.

    Args:
        index: A compiled ``outlines_core.Index``.
    """

    def __init__(self, index: Index) -> None:
        self._index = index

    @property
    def initial_state(self) -> int:
        """The FSM's initial state."""
        return self._index.get_initial_state()

    def allowed_tokens(self, state: int) -> set[int]:
        """Return the set of token IDs allowed at the given state.

        Args:
            state: Current FSM state.

        Returns:
            Set of valid token IDs. Empty set if the state is dead.
        """
        try:
            result = self._index.get_allowed_tokens(state)
            if result is None:
                return set()
            return set(result)
        except ValueError, TypeError:
            return set()

    def next_state(self, state: int, token_id: int) -> int:
        """Advance the FSM state after accepting a token.

        Args:
            state: Current FSM state.
            token_id: The accepted token ID.

        Returns:
            The new FSM state.

        Raises:
            KeyError: If the token is not allowed at this state.
        """
        try:
            result = self._index.get_next_state(state, token_id)
        except (ValueError, TypeError) as exc:
            raise KeyError(f"Token {token_id} is not allowed at state {state}") from exc
        if result is None:
            raise KeyError(f"Token {token_id} is not allowed at state {state}")
        return result

    def is_terminal(self, state: int) -> bool:
        """Check whether the given state is a terminal (accepting) state.

        Args:
            state: FSM state to check.

        Returns:
            True if generation can validly stop at this state.
        """
        try:
            return self._index.is_final_state(state)
        except ValueError, TypeError:
            return False


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------

_GUIDE_CACHE: dict[tuple[str, str, int], TokenGuide] = {}


def compile_guide(
    pattern: str,
    vocabulary: Vocabulary,
    *,
    mode: str = "regex",
) -> TokenGuide:
    """Compile a pattern into a TokenGuide, using cache if available.

    Args:
        pattern: Regex pattern or JSON schema string.
        vocabulary: An ``outlines_core.Vocabulary`` instance.
        mode: ``"regex"`` for regex patterns, ``"json_schema"`` for JSON schemas.

    Returns:
        A compiled TokenGuide ready for token-level guidance.

    Raises:
        ValueError: If mode is invalid or the pattern cannot be compiled.
    """
    cache_key = (mode, pattern, id(vocabulary))
    if cache_key in _GUIDE_CACHE:
        return _GUIDE_CACHE[cache_key]

    if mode == "json_schema":
        regex_pattern = build_regex_from_schema(pattern)
    elif mode == "regex":
        regex_pattern = pattern
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'regex' or 'json_schema'.")

    index = Index(regex_pattern, vocabulary)
    guide = TokenGuide(index)
    _GUIDE_CACHE[cache_key] = guide
    return guide


def clear_guide_cache() -> None:
    """Clear the guide compilation cache. Useful for testing."""
    _GUIDE_CACHE.clear()


def build_vocabulary(vocab: dict[str, int], eos_token_id: int) -> Vocabulary:
    """Build an ``outlines_core.Vocabulary`` from a tokenizer vocabulary.

    Converts the ``{token_string: token_id}`` mapping from HuggingFace tokenizers
    into the ``{token_string: [token_id]}`` format expected by outlines-core.

    Args:
        vocab: Mapping from token string to token ID.
        eos_token_id: The EOS token ID for this tokenizer.

    Returns:
        An ``outlines_core.Vocabulary`` instance.
    """
    # outlines-core expects dict[str, list[int]].
    # The EOS token must NOT be in the map (outlines-core manages it separately).
    oc_map: dict[str, list[int]] = {}
    for token_str, token_id in vocab.items():
        if token_id == eos_token_id:
            continue
        if token_str in oc_map:
            oc_map[token_str].append(token_id)
        else:
            oc_map[token_str] = [token_id]
    return Vocabulary(eos_token_id, oc_map)
