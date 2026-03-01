# Phase 12: Structured Output

## Goal

- Constrain token generation to follow a schema (JSON schema or regex), enabling reliable structured output without post-hoc parsing or retries.

## Background

LLMs generate free-form text, but many applications need structured output (JSON objects, enum values, function call arguments). Structured output works by masking logits at each decode step: before sampling, set the logits of all tokens that would violate the grammar to `-inf`, so only valid continuations can be chosen. This requires tracking the current state in the grammar and computing which tokens are valid next — essentially intersecting the LLM's vocabulary with the set of strings accepted by the grammar from the current position.

## Design Decisions

1. **Use `outlines-core` for FSM construction.** The regex-to-DFA pipeline and vocabulary indexing are delegated to [`outlines-core`](https://github.com/dottxt-ai/outlines-core), a Rust-backed library used by vLLM and Outlines. It provides `build_regex_from_schema` (JSON schema to regex), `Vocabulary` (tokenizer vocabulary index), `Index` (regex + vocabulary to token-level FSM), and `Guide` (stateful token-level guidance). This replaces the from-scratch `regex_fsm.py`, `json_schema.py`, and `token_fsm.py` modules (originally D1-D3).

2. **Thin wrapper in `src/infer/structured/guide.py`.** A `TokenGuide` class wraps `outlines_core.Guide` to present the same interface used by `StructuredOutputState`: `allowed_tokens(state)`, `next_state(state, token_id)`, `is_terminal(state)`, `initial_state`. This isolates the rest of the codebase from the `outlines-core` API.

3. **Integration via logit masking before sampling.** The `LogitMaskProcessor` applies the mask to raw logits before the existing sampling pipeline (repetition penalty -> temperature -> top-k -> top-p -> sample). This is a pure logit transformation, not a modification of the sampler itself.

4. **Per-request FSM state on Request dataclass.** Each `Request` carries an optional `StructuredOutputState` that tracks the current state and a reference to the compiled `TokenGuide`. The state is advanced after each token is sampled.

5. **Zero overhead when disabled.** When `response_format` is not set, no FSM is compiled, no mask is applied, and no state is tracked. The only cost is a `None` check on the request's structured output state.

## Deliverables List

Ordered by dependency:

- **D1: `TokenGuide`** — Thin wrapper around `outlines_core.Guide` / `Index` / `Vocabulary`. Compiles regex or JSON schema into a token-level guide. Located at `src/infer/structured/guide.py`.
- **D2: `LogitMaskProcessor` and `StructuredOutputState`** — Logit masking processor and per-request FSM state tracking on `Request`. Located at `src/infer/structured/logit_mask.py`.
- **D3: API extension and engine integration** — `response_format` field on `CompletionRequest`, FSM compilation in `Engine.add_request`, logit masking in sampling paths.

## Implementation Details

### D1: `TokenGuide` — `src/infer/structured/guide.py`

```python
class TokenGuide:
    """Token-level FSM guide backed by outlines-core.

    Wraps outlines_core.Guide to provide:
    - initial_state: int
    - allowed_tokens(state) -> set[int]
    - next_state(state, token_id) -> int
    - is_terminal(state) -> bool
    """
    def __init__(self, regex_pattern: str, vocabulary: outlines_core.Vocabulary) -> None: ...

_GUIDE_CACHE: dict[tuple[str, int], TokenGuide] = {}

def compile_guide(
    pattern: str,
    vocabulary: outlines_core.Vocabulary,
    *,
    mode: str = "regex",  # "regex" or "json_schema"
) -> TokenGuide:
    """Compile a pattern into a TokenGuide, using cache if available."""
```

### D2: `LogitMaskProcessor` and `StructuredOutputState` — `src/infer/structured/logit_mask.py`

Unchanged from original design. `StructuredOutputState` now wraps `TokenGuide` instead of `TokenFSM`.

### D3: API and Engine Integration

**`src/infer/engine/engine.py` changes:**

In `_compile_structured_output()`, replace `TokenVocabularyIndex` + `compile_token_fsm` with `outlines_core.Vocabulary` + `compile_guide`.

All other integration points (sampler, runner_helpers, continuous_runner) remain unchanged — they interact only with `StructuredOutputState`.

## Test Coverage

### Guide tests — `tests/unit/test_guide.py`

- Basic regex compilation and token guidance
- JSON schema compilation via `build_regex_from_schema`
- Multi-character token handling
- Cache hit/miss behavior
- Terminal state detection
- Full generation simulation

### Logit mask tests — `tests/unit/test_logit_mask.py`

Unchanged — tests `StructuredOutputState` and `apply_structured_output_mask`.

### Integration tests — `tests/unit/test_structured_output.py`

Unchanged — tests API format validation, engine integration, sampling, and stop conditions.

## Acceptance Criteria

1. `uv run pytest tests/unit/test_guide.py -v` — all pass
2. `uv run pytest tests/unit/test_logit_mask.py -v` — all pass
3. `uv run pytest tests/unit/test_structured_output.py -v` — all pass
4. `uv run pytest` — all existing tests still pass (no regressions)
5. `uv run ruff check .` — no lint errors
6. `uv run mypy .` — no type errors
