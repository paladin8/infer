# Phase 12: Structured Output

## Goal

- Constrain token generation to follow a schema (JSON schema or regex), enabling reliable structured output without post-hoc parsing or retries.

## Background

LLMs generate free-form text, but many applications need structured output (JSON objects, enum values, function call arguments). Structured output works by masking logits at each decode step: before sampling, set the logits of all tokens that would violate the grammar to `-inf`, so only valid continuations can be chosen. This requires tracking the current state in the grammar and computing which tokens are valid next — essentially intersecting the LLM's vocabulary with the set of strings accepted by the grammar from the current position.

## Design Decisions

1. **Build from scratch.** The FSM compiler is built from scratch rather than using `outlines-core`. This is an educational project; understanding the FSM construction, regex-to-NFA-to-DFS pipeline, and vocabulary indexing is the whole point. The implementation covers common JSON schema features but does not aim for full JSON Schema spec compliance.

2. **Regex as intermediate representation.** JSON schemas are compiled to regex patterns first, then regex patterns are compiled to FSMs. This avoids building two separate grammar backends and keeps the architecture simple: `JSON Schema -> Regex -> NFA -> DFA -> TokenFSM`.

3. **NFA-to-DFA via subset construction.** Standard Thompson construction for NFA, then powerset/subset construction for DFA. DFA states map directly to sets of allowed tokens. The DFA is minimized by removing unreachable states.

4. **Multi-character token handling.** When checking whether a token is allowed at a DFA state, we simulate feeding the token's decoded string character-by-character through the DFA. A token is allowed if the full string leads to a valid (non-dead) state. The destination state after consuming the full token string becomes the `next_state`.

5. **Pre-computed state-to-mask cache.** For each `(FSM, vocabulary)` pair, we pre-compute a mapping from each DFA state to the set of allowed token IDs. This is stored as a `dict[int, set[int]]`. The cache is keyed by the original pattern/schema string so repeated requests with the same schema skip recompilation.

6. **Integration via logit masking before sampling.** The `LogitMaskProcessor` applies the mask to raw logits before the existing sampling pipeline (repetition penalty -> temperature -> top-k -> top-p -> sample). This is a pure logit transformation, not a modification of the sampler itself.

7. **Per-request FSM state on Request dataclass.** Each `Request` carries an optional `StructuredOutputState` that tracks the current DFA state and a reference to the compiled `TokenFSM`. The state is advanced after each token is sampled.

8. **Zero overhead when disabled.** When `response_format` is not set, no FSM is compiled, no mask is applied, and no state is tracked. The only cost is a `None` check on the request's structured output state.

## Deliverables List

Ordered by dependency:

- **D1: `RegexFSM`** — Regex pattern to FSM compiler (NFA construction, DFA conversion, state transitions).
- **D2: `JsonSchemaCompiler`** — JSON schema to regex pattern compiler (object, array, string, number, boolean, null, enum, required, nested).
- **D3: `TokenVocabularyIndex` and `TokenFSM`** — Vocabulary preprocessing and state-to-allowed-tokens cache. `TokenFSM` wraps a DFA + vocabulary index to provide `allowed_tokens(state)`, `next_state(state, token_id)`, `is_terminal(state)`.
- **D4: `LogitMaskProcessor` and `StructuredOutputState`** — Logit masking processor and per-request FSM state tracking on `Request`.
- **D5: API extension and engine integration** — `response_format` field on `CompletionRequest`, FSM compilation in `Engine.add_request`, logit masking in sampling paths.

## Implementation Details

### D1: `RegexFSM` — `src/infer/structured/regex_fsm.py`

**Classes:**

```python
@dataclass
class NFAState:
    """A state in a Thompson NFA."""
    transitions: dict[str, list[int]]  # char -> list of target state IDs
    epsilon: list[int]  # epsilon transitions
    is_accept: bool = False

class NFA:
    """Thompson NFA built from a regex pattern."""
    states: list[NFAState]
    start: int
    accept: int

class DFA:
    """Deterministic finite automaton converted from an NFA."""
    transitions: dict[int, dict[str, int]]  # state -> char -> next_state
    initial_state: int
    accept_states: set[int]
    alphabet: set[str]
```

**Functions:**

```python
def parse_regex(pattern: str) -> NFA:
    """Parse a regex pattern into a Thompson NFA.

    Supported regex features:
    - Literal characters (with backslash escaping)
    - Character classes: [abc], [a-z], [^abc], \\d, \\w, \\s and negations
    - Quantifiers: *, +, ?, {n}, {n,m}, {n,}
    - Alternation: a|b
    - Grouping: (...)
    - Dot: . (matches any char except newline)
    - Anchors: not supported (patterns are always full-match)
    """

def nfa_to_dfa(nfa: NFA) -> DFA:
    """Convert NFA to DFA via subset construction."""

def minimize_dfa(dfa: DFA) -> DFA:
    """Remove unreachable and dead states from a DFA."""
```

### D2: `JsonSchemaCompiler` — `src/infer/structured/json_schema.py`

**Functions:**

```python
def json_schema_to_regex(schema: dict[str, Any]) -> str:
    """Compile a JSON schema to a regex pattern.

    Supported schema features:
    - type: "object", "array", "string", "number", "integer", "boolean", "null"
    - properties + required (for objects)
    - items (for arrays, single schema)
    - enum (string and number enums)
    - Nested objects and arrays
    - minItems, maxItems for arrays
    - minLength, maxLength for strings
    - No: $ref, allOf, anyOf, oneOf, additionalProperties, pattern
    """
```

**Key regex building blocks:**

- `JSON_STRING = r'"([^"\\]|\\.)*"'`
- `JSON_NUMBER = r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?'`
- `JSON_INTEGER = r'-?(0|[1-9][0-9]*)'`
- `JSON_BOOLEAN = r'(true|false)'`
- `JSON_NULL = r'null'`
- `JSON_WS = r'[ \t\n\r]*'` (optional whitespace between tokens)

### D3: `TokenVocabularyIndex` and `TokenFSM` — `src/infer/structured/token_fsm.py`

**Classes:**

```python
class TokenVocabularyIndex:
    """Maps token IDs to their decoded string representations.

    Built once per tokenizer vocabulary. Handles special tokens,
    byte-fallback tokens, and multi-byte UTF-8 sequences.
    """
    def __init__(self, vocab: dict[str, int]) -> None: ...

    @property
    def token_strings(self) -> dict[int, str]:
        """Map from token ID to decoded string."""

class TokenFSM:
    """FSM compiled against a specific vocabulary for token-level guidance.

    Wraps a DFA and TokenVocabularyIndex to provide:
    - allowed_tokens(state) -> set[int]: O(1) lookup of valid tokens
    - next_state(state, token_id) -> int: advance state after accepting a token
    - is_terminal(state) -> bool: whether generation can stop here
    """
    def __init__(self, dfa: DFA, vocab_index: TokenVocabularyIndex) -> None: ...

    @property
    def initial_state(self) -> int: ...

    def allowed_tokens(self, state: int) -> set[int]: ...
    def next_state(self, state: int, token_id: int) -> int: ...
    def is_terminal(self, state: int) -> bool: ...
```

**Multi-character token simulation:**

For each `(state, token_string)` pair, walk the DFA character-by-character. If all characters lead to valid transitions, the token is allowed and `next_state` is the final DFA state after consuming all characters. If any character leads to a dead state (no transition), the token is disallowed from that state.

**Caching:**

```python
_FSM_CACHE: dict[str, TokenFSM] = {}  # pattern_key -> compiled TokenFSM

def compile_token_fsm(
    pattern: str,
    vocab_index: TokenVocabularyIndex,
    *,
    mode: str = "regex",  # "regex" or "json_schema"
) -> TokenFSM:
    """Compile a pattern into a TokenFSM, using cache if available."""
```

### D4: `LogitMaskProcessor` and `StructuredOutputState` — `src/infer/structured/logit_mask.py`

**Classes:**

```python
@dataclass
class StructuredOutputState:
    """Per-request state for structured output generation."""
    fsm: TokenFSM
    current_state: int  # current DFA state

    def advance(self, token_id: int) -> None:
        """Advance FSM state after a token is accepted."""
        self.current_state = self.fsm.next_state(self.current_state, token_id)

    def is_terminal(self) -> bool:
        """Whether the current state is a valid completion point."""
        return self.fsm.is_terminal(self.current_state)

    def allowed_tokens(self) -> set[int]:
        """Valid token IDs from the current state."""
        return self.fsm.allowed_tokens(self.current_state)
```

**Function:**

```python
def apply_structured_output_mask(
    logits: Tensor,
    state: StructuredOutputState,
) -> Tensor:
    """Apply -inf mask to logits for tokens not allowed by the FSM.

    Args:
        logits: Raw logits, shape [vocab_size].
        state: Current structured output state with FSM reference.

    Returns:
        Masked logits (same shape and dtype). Disallowed tokens set to -inf.
    """
```

### D5: API and Engine Integration

**`src/infer/server/api.py` changes:**

Add `response_format` field to `CompletionRequest`:

```python
class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str  # "json_schema" or "regex"
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    pattern: str | None = None

class CompletionRequest(BaseModel):
    # ... existing fields ...
    response_format: ResponseFormat | None = None
```

**`src/infer/engine/request.py` changes:**

Add optional structured output state to `Request`:

```python
@dataclass
class Request:
    # ... existing fields ...
    structured_output_state: StructuredOutputState | None = field(default=None, repr=False)
```

**`src/infer/engine/engine.py` changes:**

In `add_request()`, if `response_format` is provided:
1. Build `TokenVocabularyIndex` (cached per tokenizer).
2. Compile the pattern/schema to a `TokenFSM` (cached per pattern).
3. Create `StructuredOutputState` and attach to `Request`.

**`src/infer/engine/sampler.py` changes:**

Add structured output mask application to `sample_token()`:

```python
def sample_token(
    logits: Tensor,
    context_token_ids: list[int],
    params: SamplingParams,
    generator: torch.Generator | None = None,
    structured_state: StructuredOutputState | None = None,
) -> int:
    """Sample a single token with optional structured output masking.

    When structured_state is provided, applies FSM mask before other transforms.
    """
```

**`src/infer/engine/continuous_runner.py` changes:**

After sampling each token, advance the FSM state:

```python
# After sample_token call:
if req.structured_output_state is not None:
    req.structured_output_state.advance(token)
```

Also check FSM terminal state as an additional stop condition in runner_helpers.

## Test Coverage Requirements

### D1 Tests — `tests/unit/test_regex_fsm.py`

- `test_literal_match`: Simple literal pattern matches/rejects strings.
- `test_alternation`: `a|b` accepts "a" and "b", rejects "c".
- `test_character_class`: `[a-z]` accepts lowercase, rejects digits.
- `test_negated_class`: `[^0-9]` accepts letters, rejects digits.
- `test_quantifier_star`: `a*` accepts "", "a", "aaa".
- `test_quantifier_plus`: `a+` accepts "a", "aaa", rejects "".
- `test_quantifier_optional`: `a?` accepts "" and "a", rejects "aa".
- `test_quantifier_range`: `a{2,4}` accepts "aa", "aaa", "aaaa", rejects "a", "aaaaa".
- `test_dot_matches_any`: `.` matches any single char except newline.
- `test_grouping`: `(ab)+` accepts "ab", "abab", rejects "a".
- `test_nested_groups`: `(a(bc))+` accepts "abc", "abcabc".
- `test_escape_special`: `\\.` matches literal dot.
- `test_shorthand_classes`: `\\d`, `\\w`, `\\s` match expected characters.
- `test_complex_pattern`: Full JSON-like pattern matching.
- `test_dfa_minimization`: Unreachable states removed.
- `test_empty_pattern`: Empty pattern accepts empty string.
- `test_single_char`: Single character pattern.

### D2 Tests — `tests/unit/test_json_schema.py`

- `test_string_type`: `{"type": "string"}` produces regex matching any JSON string.
- `test_number_type`: `{"type": "number"}` matches valid JSON numbers.
- `test_integer_type`: `{"type": "integer"}` matches integers, rejects floats.
- `test_boolean_type`: `{"type": "boolean"}` matches "true"/"false".
- `test_null_type`: `{"type": "null"}` matches "null".
- `test_simple_object`: Object with required string field.
- `test_object_multiple_fields`: Object with multiple fields, some optional.
- `test_nested_object`: Object containing object fields.
- `test_array_of_strings`: `{"type": "array", "items": {"type": "string"}}`.
- `test_array_of_objects`: Array containing objects.
- `test_enum_strings`: `{"enum": ["red", "green", "blue"]}`.
- `test_enum_mixed`: Enum with strings and numbers.
- `test_empty_object`: `{"type": "object", "properties": {}}`.
- `test_array_min_max_items`: Array with length constraints.
- `test_string_min_max_length`: String with length constraints.
- `test_nested_array_in_object`: Object containing array field.
- `test_boolean_in_object`: Object with boolean field.
- `test_nullable_field`: Field that can be string or null (via enum).
- `test_integer_field_in_object`: Object with integer field.
- `test_deeply_nested`: Three levels of object nesting.
- `test_all_types_object`: Object with one field per JSON type.
- `test_required_vs_optional`: Required fields must appear, optional may not.

### D3 Tests — `tests/unit/test_token_fsm.py`

- `test_vocab_index_basic`: Builds index from simple vocabulary.
- `test_vocab_index_special_tokens`: Special tokens handled correctly.
- `test_allowed_tokens_initial_state`: Correct tokens allowed at start.
- `test_allowed_tokens_mid_state`: After consuming partial input.
- `test_next_state_single_char_token`: Advances state for 1-char tokens.
- `test_next_state_multi_char_token`: Advances state for multi-char tokens.
- `test_is_terminal`: Terminal states identified correctly.
- `test_cache_hit`: Same pattern returns cached FSM.
- `test_cache_miss_different_pattern`: Different patterns produce different FSMs.
- `test_multi_char_token_spanning_boundary`: Token like `"}` that spans grammar boundaries.
- `test_no_allowed_tokens_dead_state`: Dead state returns empty allowed set.
- `test_full_generation_simulation`: Simulate token-by-token generation, verify result matches regex.

### D4 Tests — `tests/unit/test_logit_mask.py`

- `test_mask_disallows_tokens`: Disallowed tokens get -inf logits.
- `test_mask_preserves_allowed`: Allowed tokens keep original logits.
- `test_mask_empty_allowed_set`: All tokens masked (edge case handling).
- `test_advance_state`: State advances correctly after token.
- `test_is_terminal_check`: Terminal state detection works.
- `test_integration_with_sample_token`: Masked logits produce valid tokens.
- `test_zero_overhead_when_none`: No mask applied when state is None.

### D5 Tests — `tests/unit/test_structured_output.py`

- `test_response_format_json_schema`: API accepts json_schema format.
- `test_response_format_regex`: API accepts regex format.
- `test_response_format_invalid_type`: Rejects invalid format types.
- `test_no_response_format_zero_overhead`: No FSM state when format not set.
- `test_structured_state_on_request`: Request carries FSM state.
- `test_end_to_end_regex_simple`: Full pipeline with regex pattern produces matching output.
- `test_end_to_end_json_schema_simple`: Full pipeline with JSON schema produces valid JSON.

## Acceptance Criteria

1. `uv run pytest tests/unit/test_regex_fsm.py -v` — all pass
2. `uv run pytest tests/unit/test_json_schema.py -v` — all pass
3. `uv run pytest tests/unit/test_token_fsm.py -v` — all pass
4. `uv run pytest tests/unit/test_logit_mask.py -v` — all pass
5. `uv run pytest tests/unit/test_structured_output.py -v` — all pass
6. `uv run pytest` — all existing tests still pass (no regressions)
7. `uv run ruff check .` — no lint errors
8. `uv run mypy .` — no type errors
9. Test count increased from baseline (1096)
