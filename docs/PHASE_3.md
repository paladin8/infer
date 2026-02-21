# Phase 3: KV Cache

## Goal

Prefill once, decode incrementally. Replace the O(n^2) no-cache generation with O(n) decode steps using a contiguous pre-allocated KV cache. Establish that the cached path produces identical output to Phase 2 under greedy and seeded decode.

Dev models: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.

---

## Deliverables

### 1. KV cache data structure (`src/infer/cache/simple.py`)

A contiguous pre-allocated cache that stores post-RoPE K and V tensors for all layers. Two monolithic tensors hold the key and value projections; a position counter tracks how many positions have been filled.

The file lives in `src/infer/cache/` following the overall design's cache subpackage structure (`simple.py` for contiguous, `paged.py` in Phase 6, etc.). Add `src/infer/cache/__init__.py` to re-export `KVCache`.

`KVCache` is a plain class (not a `@dataclass`) because it holds mutable GPU tensors and a mutable position counter. The `allocate` and `from_model_config` static methods are the intended construction API.

```python
class KVCache:
    """Pre-allocated contiguous KV cache.

    Stores key and value tensors for every layer in two contiguous
    allocations.  A position counter (``seq_len``) tracks how many
    positions have been filled.

    The cache stores K/V *after* QK-norm and RoPE, so cached entries
    are position-encoded and ready for attention.

    Attributes:
        k: Key cache, shape ``[num_layers, batch, num_kv_heads, max_seq_len, head_dim]``.
        v: Value cache, same shape as ``k``.
        seq_len: Number of positions currently filled (same for all layers).
    """

    def __init__(self, k: Tensor, v: Tensor, seq_len: int = 0) -> None:
        self.k = k
        self.v = v
        self.seq_len = seq_len
```

**Factory methods**:

```python
@staticmethod
def allocate(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> KVCache:
    """Pre-allocate cache tensors filled with zeros."""

@staticmethod
def from_model_config(
    config: ModelConfig,
    max_seq_len: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> KVCache:
    """Allocate a cache sized for the given model config.

    Extracts ``num_hidden_layers``, ``num_key_value_heads``, and
    ``computed_head_dim`` from ``config``.
    """
```

**Update method**:

```python
def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Store new K/V entries and return the full valid K/V for this layer.

    Writes ``k`` and ``v`` at positions ``[seq_len : seq_len + new_len]``
    in the cache for the given layer.  Returns views of the cache covering
    positions ``[0 : seq_len + new_len]``.

    Args:
        layer_idx: Which layer's cache to update.
        k: New key tensor ``[batch, num_kv_heads, new_len, head_dim]``.
        v: New value tensor ``[batch, num_kv_heads, new_len, head_dim]``.

    Returns:
        ``(cached_k, cached_v)`` covering all valid positions for this layer.
            Both have shape ``[batch, num_kv_heads, seq_len + new_len, head_dim]``.
    """
    new_len = k.shape[2]
    start = self.seq_len
    end = start + new_len
    assert end <= self.k.shape[3], (
        f"KV cache overflow: writing to position {end} "
        f"but max_seq_len is {self.k.shape[3]}"
    )
    self.k[layer_idx, :, :, start:end, :] = k
    self.v[layer_idx, :, :, start:end, :] = v
    return self.k[layer_idx, :, :, :end, :], self.v[layer_idx, :, :, :end, :]
```

**Advance method**:

```python
def advance(self, n: int) -> None:
    """Advance the position counter by ``n`` tokens.

    Called once per forward pass, after all layers have written their entries.
    """
    self.seq_len += n
```

**Memory reporting**:

```python
@property
def memory_bytes(self) -> int:
    """Total GPU memory used by cache tensors in bytes."""
    return self.k.nbytes + self.v.nbytes
```

**Memory budget**:

Using the formula from the overall design: `kv_bytes ≈ 2 × num_layers × num_kv_heads × head_dim × max_seq_len × dtype_bytes`

Example allocations at `max_seq_len=1024` in bfloat16 (2 bytes per element):

| Model        | num_layers | num_kv_heads | head_dim | Cache size |
|--------------|------------|--------------|----------|------------|
| Llama 3.2 1B |         16 |            8 |      128 |      64 MB |
| Qwen3 1.7B   |         28 |            2 |      128 |      28 MB |
| Gemma3 1B    |         26 |            1 |      256 |      26 MB |

At `max_seq_len=8192`, multiply by 8: Llama ~512 MB, Qwen3 ~224 MB, Gemma3 ~208 MB. All fit comfortably in 16 GB alongside model weights (~2-3.5 GB).

### 2. Modified attention (`src/infer/models/common.py`)

Update `Attention.forward` to optionally use a KV cache:

```python
def forward(
    self,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    mask: Tensor | None = None,
    kv_cache: KVCache | None = None,
    layer_idx: int = 0,
) -> Tensor:
    """Forward pass.

    Args:
        x: Input ``[batch, seq_len, hidden_size]``.
        cos: RoPE cosine table ``[seq_len, head_dim]``.
        sin: RoPE sine table ``[seq_len, head_dim]``.
        mask: Attention mask (additive, float).
            Without cache: ``[1, 1, seq_len, seq_len]``.
            With cache during prefill: ``[1, 1, seq_len, seq_len]``.
            With cache during decode: ``None`` for full-attention layers,
            ``[1, 1, 1, cached_len]`` for sliding-window layers (Gemma 3).
        kv_cache: Optional KV cache.  When provided, new K/V entries are
            stored and the full cached K/V is used for attention.
        layer_idx: Layer index for cache indexing.  Only used when
            ``kv_cache`` is not ``None``.

    Returns:
        Output tensor ``[batch, seq_len, hidden_size]``.
    """
```

The modified forward flow:

```
q, k, v = project(x)                          # [batch, heads, seq_len, head_dim]
q, k = qk_norm(q, k)                          # if enabled (Qwen 3, Gemma 3)
q, k = apply_rope(q, k, cos, sin)

if kv_cache is not None:
    k, v = kv_cache.update(layer_idx, k, v)    # write new + read full cached

k, v = gqa_expand(k, v)                       # expand KV heads to match Q heads
out = sdpa(q, k, v, mask)                     # scaled dot-product attention
return o_proj(reshape(out))
```

**Key change**: when `kv_cache` is provided, `k` and `v` after the `update` call have shape `[batch, num_kv_heads, cached_len, head_dim]` where `cached_len = kv_cache.seq_len + new_len`. The `q` tensor retains its original `seq_len` dimension (1 during decode, prompt_len during prefill). PyTorch's `scaled_dot_product_attention` handles asymmetric Q/KV sequence lengths natively.

**When `kv_cache` is None**: behavior is identical to Phase 2. The `layer_idx` parameter is ignored. Existing tests pass without modification.

### 3. Modified transformer blocks

Update `LlamaTransformerBlock`, `Qwen3TransformerBlock`, and `Gemma3TransformerBlock` to pass through KV cache arguments:

```python
def forward(
    self,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    mask: Tensor | None = None,
    kv_cache: KVCache | None = None,
    layer_idx: int = 0,
) -> Tensor:
```

No logic changes in any block — just forwarding `kv_cache` and `layer_idx` to `self.self_attn(...)`. The MLP sub-layer is unaffected by the KV cache.

### 4. Modified model forward methods

Store the model config for cache allocation and update `forward` to accept an optional `KVCache`.

**Common changes to all three models** (`LlamaModel`, `Qwen3Model`, `Gemma3Model`):

1. Store config: `self.config = config` in `__init__`.
2. New forward signature:

```python
def forward(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor:
    """Forward pass.

    Args:
        input_ids: Token IDs ``[batch, seq_len]``.
            During prefill: full prompt.
            During decode (with kv_cache): single new token ``[batch, 1]``.
        kv_cache: Optional KV cache.  When provided:
            - RoPE positions are offset by ``kv_cache.seq_len``.
            - Mask is adjusted for cached context.
            - After all layers, ``kv_cache.advance(seq_len)`` is called.
            - Logits are computed only for the last position.

    Returns:
        Logits.  Shape ``[batch, seq_len, vocab_size]`` without cache,
        or ``[batch, 1, vocab_size]`` with cache (last position only).
    """
```

**LlamaModel / Qwen3Model forward** (same structure, different block type):

```python
def forward(self, input_ids, kv_cache=None):
    x = self.embed_tokens(input_ids)
    seq_len = x.shape[1]

    if kv_cache is not None:
        pos = kv_cache.seq_len
        if seq_len > 1:
            assert pos == 0, "Chunked prefill not supported in Phase 3"
        cos = self.cos[pos : pos + seq_len]
        sin = self.sin[pos : pos + seq_len]
        if seq_len == 1:
            mask = None                   # decode: attend to all cached positions
        else:
            mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)  # prefill
    else:
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)

    for i, layer in enumerate(self.layers):
        x = layer(x, cos, sin, mask, kv_cache=kv_cache, layer_idx=i)

    if kv_cache is not None:
        kv_cache.advance(seq_len)
        x = x[:, -1:, :]                 # last-position optimization

    x = self.norm(x)
    return self.lm_head(x)
```

**RoPE position offset**: during decode, the new token is at position `kv_cache.seq_len` (not position 0). Slicing `self.cos[pos : pos + 1]` provides the correct positional encoding for that position. During prefill with an empty cache, `pos == 0` and the slice is `self.cos[:seq_len]` — identical to the no-cache path.

**Decode mask**: when `seq_len == 1` (single-token decode), no mask is needed. The single query token attends to all K/V entries returned by `kv_cache.update()`, which is exactly the correct behavior. SDPA with `attn_mask=None` applies no masking.

**Prefill mask**: when `seq_len > 1` with a cache, the standard causal mask applies to the new tokens. In Phase 3, prefill always starts with an empty cache (`pos == 0`), so the mask is identical to the no-cache path. (A future phase with chunked prefill would need a `[seq_len, pos + seq_len]` mask for `pos > 0`, but that is not needed here.)

**Last-position logit optimization**: when using the KV cache, only the last position's logits are needed for next-token prediction. Computing `lm_head` over the full sequence during prefill would waste significant compute (e.g. Llama 1B with prompt_len=1024 and vocab_size=128256: ~524B multiply-add operations just for the LM head). Slicing `x[:, -1:, :]` before `norm` and `lm_head` avoids this. This is safe because `RMSNorm` (and `Gemma3RMSNorm`) operate on the last dimension independently per position, so `norm(x[:, -1:, :]) == norm(x)[:, -1:, :]`.

**Gemma3Model forward** (additional complexity):

```python
def forward(self, input_ids, kv_cache=None):
    x = self.embed_tokens(input_ids)
    x = x * self.embedding_normalizer
    seq_len = x.shape[1]

    if kv_cache is not None:
        pos = kv_cache.seq_len
        if seq_len > 1:
            assert pos == 0, "Chunked prefill not supported in Phase 3"

        # RoPE tables: offset by current cache position.
        local_cos = self.local_cos[pos : pos + seq_len]
        local_sin = self.local_sin[pos : pos + seq_len]
        global_cos = self.global_cos[pos : pos + seq_len]
        global_sin = self.global_sin[pos : pos + seq_len]

        if seq_len == 1:
            # Single-token decode.
            cached_len = pos + 1       # total KV length after this token

            # Global attention layers: no mask needed.
            global_mask = None

            # Sliding-window layers: mask out positions beyond the window.
            cutoff = max(0, cached_len - self.sliding_window)
            if cutoff > 0:
                local_mask = torch.zeros(
                    1, 1, 1, cached_len, dtype=x.dtype, device=x.device
                )
                local_mask[:, :, :, :cutoff] = float("-inf")
            else:
                local_mask = None      # everything fits in the window
        else:
            # Prefill: standard masks for the new tokens.
            local_mask = sliding_window_causal_mask(
                seq_len, self.sliding_window, dtype=x.dtype, device=x.device
            )
            global_mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)
    else:
        local_cos = self.local_cos[:seq_len]
        local_sin = self.local_sin[:seq_len]
        global_cos = self.global_cos[:seq_len]
        global_sin = self.global_sin[:seq_len]
        local_mask = sliding_window_causal_mask(
            seq_len, self.sliding_window, dtype=x.dtype, device=x.device
        )
        global_mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)

    for i, layer in enumerate(self.layers):
        if self.layer_types[i] == "sliding_attention":
            x = layer(x, local_cos, local_sin, local_mask,
                      kv_cache=kv_cache, layer_idx=i)
        else:
            x = layer(x, global_cos, global_sin, global_mask,
                      kv_cache=kv_cache, layer_idx=i)

    if kv_cache is not None:
        kv_cache.advance(seq_len)
        x = x[:, -1:, :]

    x = self.norm(x)
    return self.lm_head(x)
```

**Sliding-window decode mask**: during single-token decode, global attention layers need no mask (one query attending to all cached entries). Sliding-window layers need a 1D mask of shape `[1, 1, 1, cached_len]` that sets positions older than `sliding_window` to `-inf`. Without this mask, sliding-window layers would attend to the full cache, violating the architecture's design and producing incorrect output. This is the key Gemma 3-specific complexity in Phase 3.

**Dual RoPE with cache**: both local and global RoPE tables are sliced with the same position offset (`pos : pos + seq_len`). Each layer type gets its own RoPE table (local for sliding-window layers, global for full-attention layers), but the position indexing is identical. The cache stores post-RoPE K/V regardless of which RoPE variant was used — that information is baked into the cached values.

### 5. Modified generation loop (`src/infer/engine/generate.py`)

Add a `use_kv_cache` parameter to `generate()`.

**Typing for `model.config`**: the function accesses `model.config` for cache allocation, but the parameter type is `nn.Module` which has no `config` attribute. To satisfy mypy, define a `Protocol` that the model must satisfy when using KV cache:

```python
class InferModel(Protocol):
    """Protocol for models that support KV-cached generation."""
    config: ModelConfig
    def __call__(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor: ...
    def parameters(self) -> Iterator[nn.Parameter]: ...
```

The `generate()` function uses `model: nn.Module` for backward compatibility (the `use_kv_cache=False` path doesn't need `config`). When `use_kv_cache=True`, the function accesses `model.config` via `getattr` with a clear error message:

```python
config = getattr(model, "config", None)
if config is None:
    raise TypeError(
        "model must have a .config attribute for KV cache allocation. "
        "Set use_kv_cache=False to use Phase 2 behavior."
    )
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
    use_kv_cache: bool = True,
) -> GenerationResult:
    """Generate tokens autoregressively from a prompt.

    When ``use_kv_cache=True`` (default), the KV cache eliminates
    redundant computation: the prompt is processed once (prefill),
    and each decode step processes only the new token.  This gives
    O(n) compute for *n* generated tokens.

    When ``use_kv_cache=False``, reverts to Phase 2 behavior
    (full-sequence recomputation at each step, O(n^2) compute).

    Args:
        model: A loaded model (LlamaModel, Qwen3Model, or Gemma3Model).
            Must have a ``.config`` attribute when ``use_kv_cache=True``.
        tokenizer: Tokenizer for the model.
        prompt_token_ids: Pre-tokenized prompt (list of token IDs).
        params: Sampling parameters.
        device: Device to run on.
        use_kv_cache: Whether to use the KV cache.

    Returns:
        A ``GenerationResult`` with the generated text, token IDs,
        finish reason, and timing breakdown.
    """
```

**Cache lifecycle**:

```python
if use_kv_cache:
    max_seq_len = len(prompt_token_ids) + params.max_new_tokens
    kv_cache = KVCache.from_model_config(
        model.config,
        max_seq_len=max_seq_len,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
else:
    kv_cache = None
```

The cache is allocated once before generation and garbage-collected when the function returns.

**Prefill** (unchanged structure, now with cache):

```python
input_ids = tensor([tokens], device=device)        # [1, prompt_len]
logits = model(input_ids, kv_cache=kv_cache)        # cache now has prompt_len entries
next_logits = logits[0, -1, :]
token = sample_token(next_logits, tokens, params, generator)
tokens.append(token)
generated_ids.append(token)
```

After prefill, when using the cache: `kv_cache.seq_len == len(prompt_token_ids)`.

**Decode loop** (key change):

```python
for step in range(1, params.max_new_tokens):
    _sync_device(device)
    t0 = time.perf_counter()

    if kv_cache is not None:
        input_ids = tensor([[tokens[-1]]], device=device)   # [1, 1] — single token
    else:
        input_ids = tensor([tokens], device=device)         # [1, n] — full sequence

    logits = model(input_ids, kv_cache=kv_cache)
    next_logits = logits[0, -1, :]

    _sync_device(device)
    step_time = time.perf_counter() - t0

    token = sample_token(next_logits, tokens, params, generator)
    tokens.append(token)
    generated_ids.append(token)
    ...
```

With cache: each decode step feeds only the most recently generated token (shape `[1, 1]`). The model processes this single token, writes its K/V to the cache, and produces logits using the full cached context. Each decode step still computes attention over the full cached sequence (O(T) where T is total sequence length), but it eliminates the O(T) cost of recomputing all K/V projections from scratch — only a single token's K/V projection is needed. This is the key improvement over Phase 2, where every step reprocesses the entire sequence through all layers.

Without cache: the full growing sequence is fed each step, identical to Phase 2.

**Repetition penalty context**: `sample_token` still receives the full `tokens` list for repetition penalty, which is correct. The penalty needs all tokens seen so far, not just the cached portion.

**Edge case — `max_new_tokens=1`**: the prefill forward pass processes the full prompt and samples the first (and only) token. The decode loop `range(1, 1)` is empty, so no decode steps execute. Result: 1 generated token, `decode_times_s = []`, `finish_reason = "length"` (unless the first token is EOS, in which case `"eos"`). This matches Phase 2 behavior.

### 6. Benchmark updates (`benchmarks/bench_generation.py`)

Update the benchmark script to support the `use_kv_cache` toggle and report comparisons.

**New CLI flags**:
- `--kv-cache` / `--no-kv-cache`: enable or disable KV cache (default: `--kv-cache`).
- `--compare`: run each benchmark config twice (with and without cache) and report the speedup ratio.

**New metrics**:
- `cache_memory_mb`: `kv_cache.memory_bytes / 1e6` (when using cache).
- `use_kv_cache`: boolean in JSON report metadata.
- `phase`: 3 in JSON report.

**Expected output with cache**:

```
=== Phase 3 Generation Benchmark ===
Model:            meta-llama/Llama-3.2-1B-Instruct
Dtype:            bfloat16
KV Cache:         enabled (64.0 MB)
...

--- Prefill ---
TTFT (median):    32.1 ms              (approximately unchanged from Phase 2)
Prompt throughput: 7975 tok/s

--- Decode ---
Total decode time (median):  1.2 s     (was 85.2 s)
Decode throughput (median):  213 tok/s  (was 3.0 tok/s)
Per-step latency:
  Mean:           4.7 ms               (was 332.8 ms)
  P50:            4.6 ms
  P95:            5.1 ms               (near-constant, no longer grows with sequence length)
  P99:            5.3 ms
  Min:            4.5 ms
  Max:            5.5 ms

--- Memory ---
Post-load GPU:    ~2.0 GB
Cache memory:     64.0 MB
Peak GPU memory:  ~2.1 GB
```

Per-step decode latency is now approximately constant (~5ms) instead of growing linearly with sequence length. The P95/P99 are close to the mean, confirming uniform step times.

---

## File Layout

New and modified files:

```
src/infer/
├── cache/
│   ├── __init__.py             # NEW: re-export KVCache
│   └── simple.py              # NEW: KVCache data structure
├── engine/
│   └── generate.py             # MODIFIED: add use_kv_cache parameter, cache lifecycle
└── models/
    ├── common.py               # MODIFIED: Attention.forward gains kv_cache, layer_idx params
    ├── llama.py                # MODIFIED: store config, forward gains kv_cache support
    ├── qwen3.py                # MODIFIED: store config, forward gains kv_cache support
    └── gemma3.py               # MODIFIED: store config, forward gains kv_cache support

benchmarks/
├── bench_generation.py         # MODIFIED: add kv-cache toggle, cache memory metrics
└── log/
    └── GENERATION_LOG.md       # MODIFIED: add Phase 3 results

tests/
├── unit/
│   ├── test_kv_cache.py        # NEW: KVCache unit tests
│   ├── test_components.py      # MODIFIED: add Attention + KV cache tests
│   └── test_generate.py        # MODIFIED: add KV cache generation tests
└── integration/
    └── test_generation.py      # MODIFIED: add output equivalence tests
```

---

## Testing Plan

### KV cache unit tests (`tests/unit/test_kv_cache.py`)

All tests use small synthetic tensors on CPU. No model loading required.

**Allocation**:
- `KVCache.allocate(num_layers=2, num_kv_heads=4, head_dim=8, max_seq_len=16)` creates tensors of shape `[2, 1, 4, 16, 8]` with the correct dtype.
- Tensors are initialized to zeros.
- `seq_len` starts at 0.

**Update and advance**:
- Write 4 tokens to layer 0: call `update(0, k, v)` with `k` shape `[1, 4, 4, 8]`. Returned K/V has shape `[1, 4, 4, 8]` (4 positions).
- After `advance(4)`, `seq_len == 4`.
- Write 1 more token to layer 0. Returned K/V has shape `[1, 4, 5, 8]` (5 positions).
- The first 4 positions in the returned K match the previously written values.
- After `advance(1)`, `seq_len == 5`.

**Multi-layer isolation**:
- Write different values to layer 0 and layer 1 at the same position.
- Verify each layer's cached K/V values are independent.

**Memory reporting**:
- `memory_bytes` matches expected calculation: `2 × num_layers × batch × kv_heads × max_seq × head_dim × dtype_bytes`.

**from_model_config**:
- Create a `ModelConfig` with known values, allocate cache, verify shapes match config dimensions.

**Boundary conditions**:
- `max_seq_len=1`: allocate, write 1 token, advance. Cache is full.
- Write with `new_len > 1` (multi-token update for prefill). Verify all positions are stored.

### Attention with KV cache (`tests/unit/test_components.py` — extend existing)

Extend the existing Attention unit tests with KV cache scenarios. Uses small synthetic Attention modules on CPU.

**Prefill then decode**:
- Create a small Attention module (hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8).
- Allocate a KVCache for 1 layer with max_seq_len=8.
- Forward pass with 4 tokens (prefill). Verify output shape `[1, 4, 32]`.
- Advance cache.
- Forward pass with 1 token (decode). Verify output shape `[1, 1, 32]`.
- Verify `kv_cache.seq_len == 5` after both advances.

**Output equivalence (no-cache vs. cache)**:
- Run a 5-token forward pass without cache (single Attention layer, standard causal mask).
- Run the same 5 tokens as prefill(4) + decode(1) with cache.
- Verify the output at the last position matches (within float tolerance).

This test validates that the cached path produces the same attention output as the full-sequence path for a single attention layer.

**Multiple decode steps**:
- Prefill with 2 tokens, then decode 3 tokens one at a time (with `advance` after each decode).
- Verify the output at each decode step matches a full-sequence forward pass up to that position.

### Generation with KV cache (`tests/unit/test_generate.py` — extend existing)

Extend the existing mock-model generation tests.

**MockModel updates**: modify `MockModel`, `SequenceMockModel`, and `UniformMockModel` to accept the `kv_cache` keyword argument in `forward`. The mocks ignore the cache (they don't implement caching logic) but must accept the parameter to match the calling convention. Add a `config` attribute to each mock with the minimal `ModelConfig` fields needed for cache allocation (`num_hidden_layers`, `num_key_value_heads`, `computed_head_dim`).

**KV-cached greedy generation**:
- `MockModel(vocab_size=10, fixed_next_token=3)`.
- `generate(..., use_kv_cache=True)` with `max_new_tokens=5`.
- Verify 5 tokens generated, all are token 3, `finish_reason == "length"`.

**Output equivalence (cache vs. no-cache)**:
- Run `generate(use_kv_cache=False)` and `generate(use_kv_cache=True)` with the same `SequenceMockModel` and seed.
- Verify `token_ids` are identical.

**EOS stopping with cache**:
- `SequenceMockModel` that emits EOS after 3 tokens.
- `generate(use_kv_cache=True)` stops at 3 tokens with `finish_reason == "eos"`.

**Stop string stopping with cache**:
- Mock model + tokenizer that produce a stop string after 4 tokens.
- `generate(use_kv_cache=True)` stops with `finish_reason == "stop"`, text is truncated.

**Timing with cache**:
- Verify `prefill_time_s > 0`.
- Verify `len(decode_times_s) == generated_tokens - 1`.

### Integration tests (`tests/integration/test_generation.py` — extend existing)

Marked `@pytest.mark.slow`. Uses real models. Tests skip gracefully if models are not accessible.

**Output equivalence** (the critical Phase 3 test):

For each dev model, verify that greedy decode with KV cache produces the exact same token sequence as greedy decode without cache:

```python
@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_kv_cache_output_equivalence(model_id: str, device: str) -> None:
    model, config = load_model(model_id, dtype=torch.bfloat16, device=device)
    tokenizer = Tokenizer(model_id)
    prompt = render_chat_template(
        [{"role": "user", "content": "Count to ten."}],
        model_type=config.model_type,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Phase 2 path: no cache
    r_nocache = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=64),
        device=device, use_kv_cache=False,
    )
    # Phase 3 path: with cache
    r_cached = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=64),
        device=device, use_kv_cache=True,
    )

    assert r_cached.token_ids == r_nocache.token_ids
    assert r_cached.text == r_nocache.text
    assert r_cached.finish_reason == r_nocache.finish_reason
```

This is the strongest correctness guarantee: bit-exact token-level equivalence under greedy decode.

**Seeded sampling equivalence**:

Same structure with `temperature=0.7, seed=42`. KV-cached and non-cached paths must produce identical token sequences for the same seed.

```python
@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_kv_cache_seeded_equivalence(model_id: str, device: str) -> None:
    model, config = load_model(model_id, dtype=torch.bfloat16, device=device)
    tokenizer = Tokenizer(model_id)
    prompt = render_chat_template(
        [{"role": "user", "content": "Write a short poem."}],
        model_type=config.model_type,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    r_nocache = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.7, seed=42, max_new_tokens=32),
        device=device, use_kv_cache=False,
    )
    r_cached = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.7, seed=42, max_new_tokens=32),
        device=device, use_kv_cache=True,
    )

    assert r_cached.token_ids == r_nocache.token_ids
```

**Throughput improvement** (informational, with a loose lower-bound assertion):

```python
@pytest.mark.slow
def test_kv_cache_throughput_improvement(device: str) -> None:
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model, config = load_model(model_id, dtype=torch.bfloat16, device=device)
    tokenizer = Tokenizer(model_id)
    prompt = render_chat_template(
        [{"role": "user", "content": "Explain the theory of relativity in detail."}],
        model_type=config.model_type,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    r_nocache = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=128),
        device=device, use_kv_cache=False,
    )
    r_cached = generate(
        model, tokenizer, prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=128),
        device=device, use_kv_cache=True,
    )

    nocache_tps = r_nocache.generated_tokens / r_nocache.timing.decode_time_s
    cached_tps = r_cached.generated_tokens / r_cached.timing.decode_time_s
    speedup = cached_tps / nocache_tps

    print(f"Decode throughput: {nocache_tps:.1f} -> {cached_tps:.1f} tok/s ({speedup:.1f}x)")
    # Sanity check: KV cache should provide meaningful speedup for 128 tokens.
    assert speedup > 2.0, f"KV cache speedup unexpectedly low: {speedup:.1f}x"
```

This is not a precise benchmark (no warmup, single trial) but validates that the cache provides a meaningful performance improvement with real models.

---

## Design Decisions

**Contiguous pre-allocated cache**: the KV cache uses two monolithic tensors (`k` and `v`) with shape `[num_layers, batch, kv_heads, max_seq_len, head_dim]`. A single contiguous allocation per tensor is simple, avoids per-layer allocation overhead, and directly demonstrates the "contiguous pre-allocated KV cache" concept from the overall design. Phase 6 replaces this with paged blocks — the contiguous design here provides a clear baseline to compare against.

**Post-RoPE storage**: K and V are stored *after* QK-norm and RoPE application. Each cached entry is already position-encoded and ready for attention. During decode, RoPE is applied only to the new token's Q and K (using the correct position index via `cos[pos : pos + 1]`), and cached K entries retain their original position encodings. The alternative — storing raw K/V and reapplying RoPE on every step — would add unnecessary compute and complexity.

**QK-norm correctness with cache**: for Qwen 3 and Gemma 3, QK-norm is applied per-head *before* RoPE. The attention flow is: project → QK-norm → RoPE → store in cache. During decode, the new token goes through the same pipeline (project → QK-norm → RoPE → store), so the dot product `Q_new · K_cached` compares identically-transformed representations: both sides have QK-norm and RoPE baked in. This is mathematically identical to the non-cached path where all tokens are processed together. No special handling is needed for QK-norm.

**In-place mutation with split update/advance**: the `update()` method writes new K/V entries and returns the full valid range, while `advance()` increments the position counter. These are separate calls so that all layers can write at the same starting position (each layer independently stores its K/V at positions `[seq_len : seq_len + new_len]`) before the counter advances. The model's `forward` calls `advance` once after the layer loop.

**Decode mask for sliding-window layers (Gemma 3)**: during single-token decode, full-attention layers need no mask (one query token attending to all cached entries is correct by construction). Sliding-window layers require a mask of shape `[1, 1, 1, cached_len]` to block positions beyond the window. Without this mask, sliding-window layers would attend to the entire cache, violating the architecture's attention pattern and breaking output equivalence. This is the only architecture-specific mask logic in Phase 3.

**Last-position logit optimization**: when using the KV cache, `forward` slices `x` to `x[:, -1:, :]` before the final norm and LM head. During prefill with a 1024-token prompt, this avoids computing ~131M unnecessary logit values (for vocab_size=128256). The optimization is safe because RMSNorm operates independently per position. The output shape changes from `[batch, seq_len, vocab]` to `[batch, 1, vocab]` when using the cache, but `generate()` always indexes with `logits[0, -1, :]`, which works for both shapes.

**Single `forward` method, not separate prefill/decode methods**: a single `forward(input_ids, kv_cache=None)` handles both prefill and decode. The code path is determined by the presence and state of `kv_cache` (empty cache → prefill, non-empty cache → decode) and by `seq_len` (> 1 → prefill, == 1 → decode). This is simpler than maintaining separate methods and aligns with Phase 2's note that "the key change will be in how `model.forward()` is called." The `kv_cache=None` default preserves Phase 2 behavior with zero code changes on the caller side. Note: the overall design lists "separate prefill/decode code paths" as a deliverable — this is satisfied by the distinct branches within `forward` (different mask, RoPE offset, and input construction for prefill vs. decode), rather than by separate method names.

**`use_kv_cache` toggle in `generate()`**: defaults to `True` because KV caching is the normal operating mode going forward. Setting it to `False` reproduces exact Phase 2 behavior for correctness testing (output equivalence) and benchmarking (direct comparison). This follows the project's principle that every optimization is independently toggleable.

**Model config stored on model**: each model class stores `self.config = config` in `__init__`. This allows `generate()` to access model dimensions for cache allocation (`model.config.num_hidden_layers`, etc.) without passing the config as a separate argument. The config is read-only after model construction.

---

## Exit Criteria

1. Greedy output equivalence: `generate(use_kv_cache=True)` produces identical `token_ids` as `generate(use_kv_cache=False)` for all three dev models.
2. Seeded sampling equivalence: same seed produces identical output with and without cache.
3. Measured decode throughput improvement: KV-cached decode is at least 2x faster than non-cached for 128+ generated tokens on at least one dev model.
4. All existing unit tests pass (Phase 1/2 tests are not broken by the signature changes).
5. New KV cache unit tests pass: allocation, update, advance, memory reporting.
6. New attention + cache unit tests pass: prefill/decode output matches full-sequence forward.
7. Integration tests pass for all three dev models (`uv run pytest tests/integration/test_generation.py -m slow`).
8. Benchmark script runs with `--kv-cache` and produces a JSON report.
9. `uv run ruff check .` and `uv run mypy .` pass cleanly.
