"""Token-level FSM for structured output generation.

Wraps a DFA and vocabulary index to provide token-level guidance:
- allowed_tokens(state): which token IDs are valid at a given DFA state
- next_state(state, token_id): advance the DFA after accepting a multi-char token
- is_terminal(state): whether generation can stop at this state

The key insight is that LLM tokens are multi-character strings. To check
whether a token is valid at a DFA state, we simulate feeding its decoded
string character-by-character through the DFA. If all characters lead to
valid transitions, the token is allowed and its destination state is the
final DFA state after consuming all characters.
"""

from __future__ import annotations

from infer.structured.json_schema import json_schema_to_regex
from infer.structured.regex_fsm import DFA, compile_regex

# ---------------------------------------------------------------------------
# TokenVocabularyIndex
# ---------------------------------------------------------------------------


class TokenVocabularyIndex:
    """Maps token IDs to their decoded string representations.

    Built once per tokenizer vocabulary. Filters out special tokens and
    tokens with empty string representations.

    Args:
        vocab: Mapping from token string to token ID (as returned by
            ``tokenizer.get_vocab()``).
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        self._token_strings: dict[int, str] = {}
        for token_str, token_id in vocab.items():
            # Skip special tokens (typically wrapped in angle brackets or start with special prefix).
            # We keep all tokens that have printable characters.
            # The actual decoded string is what matters for FSM matching.
            decoded = self._decode_token_string(token_str)
            if decoded:
                self._token_strings[token_id] = decoded

    @staticmethod
    def _decode_token_string(token_str: str) -> str:
        """Decode a raw vocabulary token string to its actual text.

        Handles common tokenizer conventions:
        - SentencePiece: leading \u2581 (lower one eighth block) = space
        - Byte-fallback tokens: <0xAB> patterns
        - Regular tokens: used as-is

        Args:
            token_str: Raw token string from vocabulary.

        Returns:
            Decoded string, or empty string if the token should be skipped.
        """
        # Skip obvious special tokens.
        if token_str.startswith("<") and token_str.endswith(">") and len(token_str) > 2:
            # Check for byte-fallback tokens like <0xAB>.
            inner = token_str[1:-1]
            if inner.startswith("0x") and len(inner) == 4:
                try:
                    byte_val = int(inner, 16)
                    if 32 <= byte_val < 127:
                        return chr(byte_val)
                    # Non-printable byte: skip.
                    return ""
                except ValueError:
                    pass
            # Other special tokens: skip.
            return ""

        # SentencePiece: replace leading \u2581 with space.
        result = token_str.replace("\u2581", " ")
        return result

    @property
    def token_strings(self) -> dict[int, str]:
        """Map from token ID to decoded string representation."""
        return self._token_strings


# ---------------------------------------------------------------------------
# TokenFSM
# ---------------------------------------------------------------------------


class TokenFSM:
    """FSM compiled against a specific vocabulary for token-level guidance.

    Pre-computes the mapping from each DFA state to the set of allowed
    token IDs by simulating each token's decoded string through the DFA.

    Args:
        dfa: The compiled DFA from a regex pattern.
        vocab_index: Token vocabulary index mapping IDs to strings.
    """

    def __init__(self, dfa: DFA, vocab_index: TokenVocabularyIndex) -> None:
        self._dfa = dfa
        self._vocab_index = vocab_index

        # Pre-compute: state -> set of allowed token IDs.
        self._allowed: dict[int, set[int]] = {}
        # Pre-compute: (state, token_id) -> next_state.
        self._next_state: dict[tuple[int, int], int] = {}

        self._precompute()

    def _precompute(self) -> None:
        """Pre-compute allowed tokens and next states for every DFA state."""
        all_states = set(self._dfa.transitions.keys())
        # Also include states that are targets of transitions.
        for trans in self._dfa.transitions.values():
            all_states.update(trans.values())

        token_strings = self._vocab_index.token_strings

        for state in all_states:
            allowed: set[int] = set()
            for token_id, token_str in token_strings.items():
                dest = self._dfa.walk(token_str, start_state=state)
                if dest is not None:
                    allowed.add(token_id)
                    self._next_state[(state, token_id)] = dest
            self._allowed[state] = allowed

    @property
    def initial_state(self) -> int:
        """The DFA's initial state."""
        return self._dfa.initial_state

    def allowed_tokens(self, state: int) -> set[int]:
        """Return the set of token IDs allowed at the given DFA state.

        Args:
            state: Current DFA state.

        Returns:
            Set of valid token IDs. Empty set if the state is dead.
        """
        return self._allowed.get(state, set())

    def next_state(self, state: int, token_id: int) -> int:
        """Advance the DFA state after accepting a token.

        Args:
            state: Current DFA state.
            token_id: The accepted token ID.

        Returns:
            The new DFA state.

        Raises:
            KeyError: If the token is not allowed at this state.
        """
        key = (state, token_id)
        if key not in self._next_state:
            raise KeyError(f"Token {token_id} is not allowed at state {state}")
        return self._next_state[key]

    def is_terminal(self, state: int) -> bool:
        """Check whether the given state is an accepting (terminal) state.

        Args:
            state: DFA state to check.

        Returns:
            True if generation can validly stop at this state.
        """
        return state in self._dfa.accept_states


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------

# Cache compiled TokenFSMs keyed by (mode, pattern, vocab_id).
_FSM_CACHE: dict[tuple[str, str, int], TokenFSM] = {}


def compile_token_fsm(
    pattern: str,
    vocab_index: TokenVocabularyIndex,
    *,
    mode: str = "regex",
) -> TokenFSM:
    """Compile a pattern into a TokenFSM, using cache if available.

    Args:
        pattern: Regex pattern or JSON schema string.
        vocab_index: Token vocabulary index.
        mode: ``"regex"`` for regex patterns, ``"json_schema"`` for JSON schemas.

    Returns:
        A compiled TokenFSM ready for token-level guidance.

    Raises:
        ValueError: If mode is invalid or the pattern cannot be compiled.
    """
    cache_key = (mode, pattern, id(vocab_index))
    if cache_key in _FSM_CACHE:
        return _FSM_CACHE[cache_key]

    if mode == "json_schema":
        import json

        schema = json.loads(pattern)
        regex_pattern = json_schema_to_regex(schema)
    elif mode == "regex":
        regex_pattern = pattern
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'regex' or 'json_schema'.")

    dfa = compile_regex(regex_pattern)
    fsm = TokenFSM(dfa, vocab_index)
    _FSM_CACHE[cache_key] = fsm
    return fsm


def clear_fsm_cache() -> None:
    """Clear the FSM compilation cache. Useful for testing."""
    _FSM_CACHE.clear()
