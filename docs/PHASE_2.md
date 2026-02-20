# Phase 2: Autoregressive Generation

## Goal

Single-request token-by-token generation with a configurable sampling pipeline. Establish baseline throughput and latency measurements before any optimizations (no KV cache, no batching).

Dev models: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.

---

## Deliverables

### 1. Sampling parameters (`src/infer/engine/sampler.py`)

The `SamplingParams` dataclass matches the reference interface from the overall design:

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    max_new_tokens: int = 128
    stop: list[str] | None = None
    seed: int | None = None
```

Field semantics:
- **`temperature`**: Scales logits before softmax. `0.0` is treated as greedy (argmax). Values in `(0.0, 1.0)` sharpen the distribution, values above `1.0` flatten it.
- **`top_k`**: Keep only the top-k highest-probability tokens. `None` disables top-k filtering.
- **`top_p`**: Nucleus sampling. Keep the smallest set of tokens whose cumulative probability exceeds `top_p`. `1.0` disables top-p filtering.
- **`repetition_penalty`**: Penalize tokens that have already appeared in the context (prompt + generated tokens). `1.0` means no penalty. Uses the CTRL-paper formulation: positive logits are divided by the penalty, negative logits are multiplied.
- **`max_new_tokens`**: Maximum number of tokens to generate (excluding the prompt).
- **`stop`**: List of text strings that trigger early stopping. Generation halts when any stop string appears in the decoded output. The stop string itself is excluded from the returned text.
- **`seed`**: Random seed for reproducible sampling. `None` means non-deterministic.

Validation (performed at construction or by a `validate()` method):
- `temperature >= 0.0`
- `top_p` in `(0.0, 1.0]`
- `top_k is None or top_k >= 1`
- `repetition_penalty > 0.0`
- `max_new_tokens >= 1`

### 2. Sampler pipeline (`src/infer/engine/sampler.py`)

The sampler transforms raw logits into a sampled token ID. Transforms are applied in a fixed order:

1. **Repetition penalty** — reduce logits for tokens already in the context.
2. **Temperature scaling** — divide logits by temperature.
3. **Top-k filtering** — zero out all but the top-k logits.
4. **Top-p filtering** — zero out tokens below the nucleus threshold.
5. **Sample** — draw from the resulting distribution (or argmax for greedy).

```python
def sample_token(
    logits: Tensor,
    context_token_ids: list[int],
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    """Sample a single token from logits.

    Args:
        logits: Raw logits for a single position, shape ``[vocab_size]``.
        context_token_ids: All token IDs seen so far (prompt + generated).
            Used for repetition penalty.
        params: Sampling parameters.
        generator: Optional RNG for reproducible sampling.

    Returns:
        The sampled token ID.
    """
```

**Repetition penalty** (CTRL formulation):

```python
def apply_repetition_penalty(
    logits: Tensor,
    token_ids: list[int],
    penalty: float,
) -> Tensor:
    """Penalize tokens that appear in the context.

    For each unique token in ``token_ids``:
    - If the logit is positive, divide by ``penalty``.
    - If the logit is negative, multiply by ``penalty``.

    This makes repeated tokens less likely regardless of their
    original logit sign.
    """
```

Uses `torch.gather` / `torch.scatter_` for efficient indexing. Operates on unique token IDs only (via a set or `torch.unique`) to avoid double-penalizing. No-op when `penalty == 1.0`.

**Temperature scaling**:

```python
def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits by temperature. Returns logits unchanged if temperature == 1.0."""
```

`temperature == 0.0` is a special case handled at the sampling step (argmax), not here. If temperature is 0, this function is not called.

**Top-k filtering**:

```python
def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep only the top-k logits, set the rest to -inf."""
```

Uses `torch.topk` to find the k-th largest value, then masks everything below that threshold with `-inf`.

**Top-p (nucleus) filtering**:

```python
def apply_top_p(logits: Tensor, p: float) -> Tensor:
    """Keep the smallest set of tokens with cumulative probability >= p."""
```

Steps:
1. Sort logits descending.
2. Compute cumulative softmax probabilities over the sorted logits.
3. Find the cutoff index where cumulative probability first exceeds `p`.
4. Mask all tokens below that cutoff with `-inf`.
5. Unsort back to original token order.

No-op when `p == 1.0`.

**Sampling step**:

For `temperature == 0.0` (greedy): `torch.argmax(logits)`.

Otherwise: `torch.multinomial(softmax(logits), num_samples=1, generator=generator)`.

The `generator` is created from `SamplingParams.seed` when provided, ensuring deterministic output for a given seed.

### 3. Generation loop (`src/infer/engine/generate.py`)

The generation function ties together the model, tokenizer, and sampler into a complete generation pipeline.

```python
@dataclass
class GenerationResult:
    """Result of a single generation request."""

    token_ids: list[int]
    """Generated token IDs (excluding the prompt)."""

    text: str
    """Decoded generated text (truncated at stop string if applicable)."""

    finish_reason: str
    """Why generation stopped: ``"eos"``, ``"stop"``, or ``"length"``."""

    prompt_tokens: int
    """Number of tokens in the input prompt."""

    generated_tokens: int
    """Number of tokens actually generated (may be less than max_new_tokens)."""

    timing: GenerationTiming
    """Per-step and aggregate timing measurements."""


@dataclass
class GenerationTiming:
    """Timing breakdown for a single generation."""

    prefill_time_s: float
    """Time for the first forward pass (full prompt)."""

    decode_times_s: list[float]
    """Time for each individual decode step."""

    @property
    def decode_time_s(self) -> float:
        """Total decode time (sum of all decode steps)."""
        return sum(self.decode_times_s)

    @property
    def total_time_s(self) -> float:
        """Wall clock time (prefill + decode)."""
        return self.prefill_time_s + self.decode_time_s
```

```python
@torch.inference_mode()
def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt_token_ids: list[int],
    params: SamplingParams,
    *,
    device: str | torch.device = "cuda",
) -> GenerationResult:
    """Generate tokens autoregressively from a prompt.

    This is the naive (no KV cache) implementation: each decode step
    runs a full forward pass over the entire sequence (prompt + all
    generated tokens so far).  This gives O(n^2) compute for n
    generated tokens.  Phase 3 adds a KV cache to make decode O(n).

    Args:
        model: A loaded model (LlamaModel, Qwen3Model, or Gemma3Model).
        tokenizer: Tokenizer for the model.
        prompt_token_ids: Pre-tokenized prompt (list of token IDs).
        params: Sampling parameters.
        device: Device to run on.

    Returns:
        A GenerationResult with the generated text, token IDs,
        finish reason, and timing breakdown.
    """
```

**Generation loop pseudocode**:

```
validate params
create RNG generator from params.seed (if set)
resolve EOS token IDs from tokenizer (handle int | list[int])
tokens = list(prompt_token_ids)
generated_ids = []
finish_reason = "length"

# Prefill: first forward pass over the prompt
t0 = sync_and_time()
input_ids = tensor(tokens, device=device)  # [1, prompt_len]
logits = model(input_ids)                  # [1, prompt_len, vocab]
next_logits = logits[0, -1, :]             # last position
prefill_time = sync_and_time() - t0

# Sample first token
token = sample_token(next_logits, tokens, params, generator)
tokens.append(token)
generated_ids.append(token)

# Check stop conditions after first token
...

# Decode loop
for step in range(1, params.max_new_tokens):
    t0 = sync_and_time()
    input_ids = tensor(tokens, device=device)  # [1, prompt_len + step]
    logits = model(input_ids)
    next_logits = logits[0, -1, :]
    decode_step_time = sync_and_time() - t0

    token = sample_token(next_logits, tokens, params, generator)
    tokens.append(token)
    generated_ids.append(token)

    record decode_step_time

    # Check stop conditions
    if token in eos_token_ids:
        finish_reason = "eos"
        break
    if check_stop_strings(generated_ids, tokenizer, params.stop):
        finish_reason = "stop"
        break

# Decode final text, truncate at stop string if needed
text = decode_and_truncate(generated_ids, tokenizer, params.stop)
return GenerationResult(...)
```

**Timing**:

Use `torch.cuda.synchronize()` before each timing measurement to ensure GPU operations have completed. Fall back to no-op on CPU. Timing uses `time.perf_counter()` for high-resolution measurement.

```python
def _sync_device(device: torch.device) -> None:
    """Synchronize CUDA device if applicable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
```

**Prefill measurement (TTFT)**: start timer before the first forward pass, end after the first token is sampled. This captures the full time-to-first-token: forward pass + sampling. Sampling is negligible relative to the forward pass but is included because TTFT measures time until the first token is available to the caller.

**Decode step measurement**: start timer before each forward pass, end after the forward pass completes (before sampling). Per-step decode times are recorded in `GenerationTiming.decode_times_s`. Sampling cost is excluded to isolate the model execution time that dominates each step.

**CUDA synchronization**: `torch.cuda.synchronize()` is called before starting and after ending each timed region. This ensures GPU kernels have fully completed before reading the clock. Without sync, CUDA kernel launches return immediately and timing would measure only launch overhead.

### 4. Stop condition handling

Three stop conditions, checked after each generated token in this priority order:

1. **EOS token**: the generated token matches one of the model's EOS token IDs. Normalized to a `set[int]` at the start of generation.

    The `Tokenizer.eos_token_id` property returns the primary EOS token (`int | list[int]`), but some models use additional stop tokens in chat mode. For example, Llama 3.2 Instruct uses `128001` as the primary EOS but also treats `128008` (`<|end_of_turn|>`) and `128009` (`<|eot_id|>`) as generation-ending tokens. These additional tokens are not always returned by `eos_token_id`.

    To get the full set of EOS-like tokens, add an `eos_token_ids` property to `Tokenizer` that returns a `set[int]`:

    ```python
    @property
    def eos_token_ids(self) -> set[int]:
        """All token IDs that should terminate generation.

        Includes the primary eos_token_id plus any additional EOS tokens
        defined in the tokenizer config (e.g. Llama 3's eot_id and
        end_of_turn tokens).
        """
    ```

    Resolution strategy:
    - Start with the primary `eos_token_id` (always present).
    - Check `tokenizer.added_tokens_encoder` or `tokenizer_config.json` for tokens with names matching common EOS patterns (`<|eot_id|>`, `<|end_of_turn|>`, `<|im_end|>`, `<end_of_turn>`).
    - For the three dev models specifically: Llama 3.2 needs `{128001, 128008, 128009}`, Qwen 3 needs `{151645}` (`<|im_end|>`) plus `{151643}` (`<|endoftext|>`), Gemma 3 needs `{1}` (`<eos>`) plus `{106}` (`<end_of_turn>`).
    - The generation loop uses this `eos_token_ids` set for the EOS check.

2. **Stop strings**: a user-specified stop string appears in the decoded generated text. Checked by decoding the full generated token sequence after each token and searching for any stop string as a substring.

    ```python
    def _check_stop_strings(
        generated_ids: list[int],
        tokenizer: Tokenizer,
        stop_strings: list[str],
    ) -> str | None:
        """Check if any stop string appears in the decoded text.

        Returns the matched stop string, or None.
        """
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        for s in stop_strings:
            if s in text:
                return s
        return None
    ```

    When a stop string is found, the final `text` in `GenerationResult` is truncated at the first occurrence of the stop string (the stop string itself is excluded). The `token_ids` list includes all generated tokens up to and including the token that triggered the match (no attempt to trim tokens — text truncation is sufficient for Phase 2).

    **Performance note**: decoding the full generated sequence after each token is O(n) per step, making the stop string check O(n^2) total. This is acceptable in Phase 2 since the generation loop itself is already O(n^2) without KV cache.

    **Why not incremental decoding?** A seemingly simpler O(n) approach would decode only each new token and append to a running text buffer. However, tokenizer decode is **not compositional**: decoding tokens individually and concatenating does not always match decoding them together. Byte-level BPE tokenizers (Llama 3, Qwen 3) can split multi-byte UTF-8 sequences across token boundaries, and SentencePiece tokenizers (Gemma 3) have similar boundary effects. The full-decode approach is correct by construction. Phase 3+ (where KV cache makes the forward pass O(n) per step) is the right time to revisit this — a practical optimization there would be to search only the tail of the decoded text rather than the full string.

3. **Max tokens**: `len(generated_ids) >= params.max_new_tokens`. This is the fallback — generation always terminates.

`GenerationResult.finish_reason` is set to `"eos"`, `"stop"`, or `"length"` accordingly. Only the first matching condition triggers (e.g., if EOS is also a stop string match, `"eos"` takes priority).

### 5. Benchmark script (`benchmarks/bench_generation.py`)

A standalone script that loads a model, runs generation with controlled inputs, and reports throughput and latency metrics.

```
Ad-hoc single run:
    uv run python benchmarks/bench_generation.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --prompt-tokens 256 --max-new-tokens 256

Canonical suite:
    uv run python benchmarks/bench_generation.py --suite quick
    uv run python benchmarks/bench_generation.py --suite standard
    uv run python benchmarks/bench_generation.py --suite full
```

**Workload construction**:

The benchmark needs a prompt of a specific token length. Rather than using a real chat prompt (whose length varies by model template), it constructs a synthetic prompt by repeating a common token to reach the target length. This ensures consistent prompt length across models and runs.

```python
def make_synthetic_prompt(tokenizer: Tokenizer, target_tokens: int) -> list[int]:
    """Create a prompt of exactly target_tokens length.

    Encodes a repeated phrase, truncates or pads to exact length.
    """
```

Alternatively, for more realistic prompts, use a fixed passage of text and truncate the token sequence to the target length. The benchmark accepts both modes:
- `--prompt-tokens N`: synthetic prompt of exactly N tokens (default).
- `--prompt "text"`: user-provided prompt text (reports actual token count).

The benchmark includes 7 canonical configs (short-greedy through medium-full-pipeline) with 3 suite tiers (quick, standard, full). See `benchmarks/log/GENERATION_LOG.md` for config details and results.

**Benchmark flow**:

1. Load model with `load_model(model_id, dtype=dtype, device="cuda")`.
2. Load tokenizer.
3. Construct prompt.
4. Run `warmup_runs` generations (discard results — fills GPU caches, JIT compilation).
5. Run `trials` generations, collecting `GenerationResult` from each.
6. Compute and print summary statistics.

**Metrics reported**:

```
=== Phase 2 Generation Benchmark ===
Model:            meta-llama/Llama-3.2-1B-Instruct
Dtype:            bfloat16
Device:           cuda (NVIDIA GeForce RTX 5080, 16 GB)
CUDA version:     12.x
PyTorch version:  2.x.x
Prompt tokens:    256
Generated tokens: 256 (all trials)
Seed:             42

--- Prefill ---
TTFT (median):    32.1 ms
Prompt throughput: 7975 tok/s

--- Decode ---
Total decode time (median):  85.2 s
Decode throughput (median):  3.0 tok/s
Per-step latency:
  Mean:           332.8 ms
  P50:            330.1 ms
  P95:            510.2 ms
  P99:            520.5 ms
  Min:            170.3 ms  (step 1, seq_len=257)
  Max:            530.1 ms  (step 256, seq_len=512)

--- Total ---
Wall time (median):  85.3 s
End-to-end throughput: 3.0 tok/s

--- Memory ---
Post-load GPU:    ~2.0 GB
Peak GPU memory:  ~3.1 GB
```

**Metric definitions**:
- **TTFT (time to first token)**: `prefill_time_s` — time from start of first forward pass to first token available.
- **Prompt throughput**: `prompt_tokens / prefill_time_s` — how fast the model processes the prompt.
- **Decode throughput**: `generated_tokens / decode_time_s` — tokens generated per second during the decode phase.
- **Per-step latency**: distribution of `decode_times_s` entries. Since there's no KV cache, per-step latency grows linearly with sequence length (each step recomputes the full sequence).
- **Peak GPU memory**: `torch.cuda.max_memory_allocated()` after generation.
- **Post-load GPU**: `torch.cuda.memory_allocated()` after model load, before generation. This is primarily model weights plus small buffers (RoPE tables, etc.).

When `trials > 1`, report the **median** of each aggregate metric across trials. Per-step latency percentiles are computed over all steps across all trials.

**Output format**: prints to stdout as a formatted table. Optionally writes a JSON report to `benchmarks/reports/` for automated comparison across phases:

```python
{
    "phase": 2,
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "dtype": "bfloat16",
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "prompt_tokens": 256,
    "generated_tokens": 256,
    "seed": 42,
    "trials": 3,
    "ttft_median_ms": 32.1,
    "decode_throughput_median_tps": 3.0,
    "per_step_latency_p50_ms": 330.1,
    "per_step_latency_p95_ms": 510.2,
    "peak_gpu_memory_gb": 3.1,
    "timestamp": "2026-02-19T..."
}
```

**GPU information**: use `torch.cuda.get_device_name()` and `torch.cuda.get_device_properties()` for the report. Log `torch.__version__` and `torch.version.cuda`.

---

## File Layout

New files:

```
src/infer/
├── loader/
│   └── tokenizer.py            # MODIFIED: add eos_token_ids property
└── engine/
    ├── __init__.py             # NEW: re-exports SamplingParams, generate, GenerationResult
    ├── sampler.py              # NEW: SamplingParams + sampling transforms
    └── generate.py             # NEW: generate() + GenerationResult

benchmarks/
├── bench_generation.py         # NEW: Phase 2 benchmark script
├── reports/                    # NEW: directory for JSON benchmark reports
│   └── .gitkeep
└── log/
    └── GENERATION_LOG.md       # NEW: cross-phase generation benchmark log

tests/
├── unit/
│   ├── test_sampler.py         # NEW: sampler unit tests
│   └── test_generate.py        # NEW: generation loop unit tests
└── integration/
    └── test_generation.py      # NEW: real-model generation tests
```

---

## Testing Plan

### Sampler unit tests (`tests/unit/test_sampler.py`)

All tests use small synthetic logit tensors (no model loading required). Tests run on CPU.

**SamplingParams validation**:
- Valid defaults construct without error.
- `temperature=-1.0` raises `ValueError`.
- `top_p=0.0` raises `ValueError`.
- `top_p=1.5` raises `ValueError`.
- `top_k=0` raises `ValueError`.
- `repetition_penalty=0.0` raises `ValueError`.
- `max_new_tokens=0` raises `ValueError`.

**Temperature scaling**:
- `temperature=1.0` returns logits unchanged (within float tolerance).
- `temperature=0.5` doubles the logit magnitudes (logits / 0.5).
- `temperature=2.0` halves the logit magnitudes.
- Output shape matches input shape.
- Output dtype matches input dtype.

**Top-k filtering**:
- With `k=3` on a 10-element logit tensor: exactly 3 values are finite, the rest are `-inf`.
- The 3 finite values are the 3 largest from the original.
- `k >= vocab_size` returns logits unchanged.
- `k=1` leaves only the maximum value.

**Top-p filtering**:
- With `p=0.9` on a known distribution: the minimum set of tokens covering 90% cumulative probability is kept.
- `p=1.0` returns logits unchanged.
- Very small `p` (e.g. `0.01`) keeps only the top token (approximately — depends on distribution).
- Monotonic: smaller `p` keeps fewer or equal tokens.
- Already-filtered logits (`-inf` entries) are handled correctly (contribute 0 probability).

**Repetition penalty**:
- `penalty=1.0` returns logits unchanged.
- `penalty=2.0` with a positive logit token in context: that logit is halved.
- `penalty=2.0` with a negative logit token in context: that logit's magnitude is doubled (more negative).
- Tokens not in context are unchanged.
- Duplicate token IDs in context don't cause double-penalization.

**`sample_token` integration**:
- Greedy (`temperature=0.0`): always returns the argmax token.
- Deterministic with seed: same seed produces same token. Different seeds produce different tokens (with high probability on a non-degenerate distribution).
- Full pipeline: `temperature=0.8`, `top_k=50`, `top_p=0.9`, `repetition_penalty=1.1` — verify it returns a valid token ID.
- With uniform logits and `temperature=1.0`: sampling is approximately uniform (statistical test over many samples, chi-squared or similar).

### Generation loop unit tests (`tests/unit/test_generate.py`)

Tests use a **tiny mock model** that returns deterministic logits, avoiding real model loading. The mock model is a simple `nn.Module` that always returns the same logit pattern (e.g., logit `i` = vocab_size - i, making token 0 the greedy choice).

```python
class MockModel(nn.Module):
    """A tiny model that returns fixed logits for testing."""
    def __init__(self, vocab_size: int, fixed_next_token: int) -> None: ...
    def forward(self, input_ids: Tensor) -> Tensor: ...
```

And a `MockTokenizer` that does simple character-level or identity encoding/decoding.

**Basic generation**:
- Greedy decode with `max_new_tokens=5`: generates exactly 5 tokens, finish reason is `"length"`.
- Returns a `GenerationResult` with correct `prompt_tokens` and `generated_tokens` counts.

**EOS stopping**:
- Mock model that emits EOS after 3 tokens: generates 3 tokens, finish reason is `"eos"`.
- With multiple EOS token IDs: any matching EOS triggers stop.

**Stop string stopping**:
- Mock model + tokenizer where generating 4 tokens produces text containing a stop string: generates <= 4 tokens, finish reason is `"stop"`, text is truncated before the stop string.

**Determinism**:
- Same seed produces identical `token_ids` across two calls.
- Different seeds produce different `token_ids`.

**Timing**:
- `GenerationResult.timing.prefill_time_s > 0`.
- `len(timing.decode_times_s) == generated_tokens - 1` (first token is part of prefill).
- `timing.total_time_s >= timing.prefill_time_s + timing.decode_time_s`.

**Edge cases**:
- `max_new_tokens=1`: generates exactly 1 token.
- Empty stop list (`stop=[]`): no stop string checking.
- Prompt of length 1: works correctly.

### Integration tests (`tests/integration/test_generation.py`)

Marked `@pytest.mark.slow`. Tests skip gracefully if the model is not accessible. Uses a `device` fixture that auto-detects CUDA.

**Functional generation** (parametrized across all three dev models):

For each model, generate a response to a simple chat prompt and verify:
- Output is non-empty text.
- Output is coherent (not garbage — a rough heuristic: check that the output contains recognizable words or that the generated token IDs are in a valid range).
- `finish_reason` is one of `"eos"`, `"stop"`, `"length"`.
- `generated_tokens >= 1`.
- `timing.total_time_s > 0`.

```python
_DEV_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b-it",
]

_PROMPTS = [
    [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    [{"role": "user", "content": "Write a haiku about programming."}],
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "Name the four seasons."}],
]

@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_functional_generation(model_id: str, device: str) -> None:
    model, config = load_model(model_id, dtype=torch.bfloat16, device=device)
    tokenizer = Tokenizer(model_id)
    for messages in _PROMPTS:
        prompt = render_chat_template(messages, model_type=config.model_type)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        result = generate(
            model, tokenizer, prompt_ids,
            SamplingParams(temperature=0.0, max_new_tokens=64),
            device=device,
        )
        assert result.generated_tokens >= 1
        assert len(result.text) > 0
        assert result.finish_reason in ("eos", "stop", "length")
```

**Deterministic generation**:

For each model, verify that greedy decoding (`temperature=0.0`) produces identical output across two runs. Also verify that seeded sampling (`temperature=0.7, seed=42`) produces identical output across two runs.

```python
@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_deterministic_generation(model_id: str, device: str) -> None:
    model, config = load_model(model_id, dtype=torch.bfloat16, device=device)
    tokenizer = Tokenizer(model_id)
    prompt = render_chat_template(
        [{"role": "user", "content": "Count to five."}],
        model_type=config.model_type,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Greedy
    r1 = generate(model, tokenizer, prompt_ids,
                  SamplingParams(temperature=0.0, max_new_tokens=32), device=device)
    r2 = generate(model, tokenizer, prompt_ids,
                  SamplingParams(temperature=0.0, max_new_tokens=32), device=device)
    assert r1.token_ids == r2.token_ids

    # Seeded sampling
    r3 = generate(model, tokenizer, prompt_ids,
                  SamplingParams(temperature=0.7, seed=42, max_new_tokens=32), device=device)
    r4 = generate(model, tokenizer, prompt_ids,
                  SamplingParams(temperature=0.7, seed=42, max_new_tokens=32), device=device)
    assert r3.token_ids == r4.token_ids
```

**Stop string handling**:

Generate with a stop string and verify the output is truncated correctly.

```python
@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_stop_string(model_id: str, device: str) -> None:
    model, config = load_model(model_id, dtype=torch.bfloat16, device=device)
    tokenizer = Tokenizer(model_id)
    prompt = render_chat_template(
        [{"role": "user", "content": "List three colors, one per line."}],
        model_type=config.model_type,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # First generate WITHOUT stop string to confirm the model would
    # produce output containing the stop pattern.
    baseline = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=128),
        device=device,
    )
    # If the baseline doesn't contain "\n\n", this prompt can't test
    # stop strings — skip rather than pass vacuously.
    if "\n\n" not in baseline.text:
        pytest.skip("Model output does not contain stop pattern")

    # Now generate WITH stop string and verify truncation.
    result = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=128, stop=["\n\n"]),
        device=device,
    )
    assert result.finish_reason == "stop"
    assert "\n\n" not in result.text
    assert result.generated_tokens < baseline.generated_tokens
```

---

## Design Decisions

**No KV cache**: Phase 2 deliberately recomputes the full sequence at each decode step. This gives O(n^2) compute for n generated tokens and serves as the baseline for Phase 3's KV cache optimization. The generation loop is structured so Phase 3 can add prefill/decode code paths without rewriting the outer logic — the key change will be in how `model.forward()` is called (full sequence vs. single token with cached KV).

**Forward signature unchanged**: the model `forward(input_ids)` signature from Phase 1b is used as-is. Phase 3 will extend it with KV cache arguments. The generation loop in Phase 2 calls `model(input_ids)` where `input_ids` grows by one token each step.

**Timing inside `generate()`**: timing is collected inside the generation function rather than externally. This keeps the timing precise (CUDA sync boundaries are correct) and avoids duplicating the generation loop in the benchmark. The `GenerationTiming` dataclass provides all raw data needed for benchmark statistics.

**Stop string check via full decode**: each stop string check decodes the entire generated sequence. This is O(n) per step but correct even when stop strings span token boundaries. The O(n^2) total cost is dominated by the O(n^2) forward pass cost in Phase 2.

**Sampling on GPU**: the sampler transforms and `torch.multinomial` operate on the same device as the model. This avoids a GPU-to-CPU transfer per step (~512 KB for a 262K-vocab bf16 logit tensor). The transfer cost is negligible (~20 us at PCIe 4.0 speeds) relative to the forward pass (~20-500 ms), but keeping everything on-device is simpler and avoids an unnecessary copy. `torch.Generator(device=device)` is used for seeded sampling on CUDA. Note: CUDA `torch.multinomial` is deterministic for a given seed on the same GPU architecture, but results may differ across GPU architectures. This is acceptable for an educational runtime.

**Benchmark uses synthetic prompts**: to control prompt length precisely, the benchmark constructs prompts by encoding a repeated text passage and truncating to the target token count. This avoids variability from chat template differences across models.

---

## Exit Criteria

1. `generate()` produces coherent text for at least 3 prompts on each of the three dev models.
2. Greedy decoding (`temperature=0.0`) is deterministic: same prompt produces identical output across runs.
3. Seeded sampling (`temperature > 0, seed != None`) is deterministic: same seed produces identical output across runs.
4. Stop conditions work: EOS token stops generation, stop strings truncate output, `max_new_tokens` is respected.
5. All unit tests pass (`uv run pytest tests/unit/`).
6. Integration tests pass for all three dev models (`uv run pytest tests/integration/test_generation.py -m slow`).
7. Benchmark script runs and produces a report for at least one dev model.
8. `uv run ruff check .` and `uv run mypy .` pass cleanly.
