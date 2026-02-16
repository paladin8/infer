# Phase 1a: Model Loading and Single-Layer Forward

## Goal

Load HuggingFace weights and verify that a single transformer layer produces activations matching `transformers` for three architectures: Llama 3, Qwen 3, and Gemma 3.

Dev models: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.

---

## Deliverables

### 1. Config reader (`src/infer/loader/config.py`)

Read a HuggingFace `config.json` and parse it into a typed dataclass.

```python
@dataclass
class ModelConfig:
    # Identity
    model_type: str              # "llama", "qwen3", "gemma3_text"

    # Dimensions
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    head_dim: int | None = None  # None → hidden_size // num_attention_heads
                                 # Qwen 3 and Gemma 3 set this explicitly (can differ from computed)

    # Normalization
    rms_norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None  # Llama 3: {"rope_type": "llama3", ...}
    rope_local_base_freq: float | None = None   # Gemma 3: 10000.0 for sliding window layers

    # Projection biases (all False for Llama 3 / Qwen 3 / Gemma 3)
    attention_bias: bool = False
    mlp_bias: bool = False

    # Activation
    hidden_act: str = "silu"     # "silu" for SwiGLU (Llama/Qwen), "gelu_pytorch_tanh" for GeGLU (Gemma)

    # Attention scaling
    query_pre_attn_scalar: float | None = None  # None → use head_dim (standard)
                                                # Gemma 3 sets this to 256

    # Sliding window attention
    sliding_window: int | None = None           # Gemma 3 1B: 512, 4B/12B: 1024
    sliding_window_pattern: int | None = None   # Gemma 3: 6 (5 local + 1 global every 6 layers)

    # Layer types (resolved from sliding_window_pattern or loaded directly from config)
    layer_types: list[str] | None = None        # e.g. ["sliding_attention", ..., "full_attention", ...]
                                                # Newer transformers versions provide this directly;
                                                # older ones use sliding_window_pattern. Reader handles both.

    # Embeddings
    tie_word_embeddings: bool = False
```

Defaults are conservative baselines, not specific to any single dev model. All three dev models override `rope_theta` (Llama 3: 500000, Qwen 3: 1000000, Gemma 3: 1000000) and two override `rms_norm_eps` (Qwen 3 and Gemma 3 use 1e-6). The config reader always loads actual values from `config.json`.

A `computed_head_dim` property computes the fallback when `head_dim` is `None`:

```python
@property
def computed_head_dim(self) -> int:
    return self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
```

The config reader normalizes HF config quirks:
- **`hidden_activation` → `hidden_act`**: Gemma 3 uses a different field name — reader maps it.
- **Nested `text_config` extraction**: Gemma 3 multimodal configs (4B+) nest text config under `text_config` with its own `model_type` — reader extracts it. The 1B model (`gemma3_text`) has a flat config.
- **JSON `null` handling**: real HF configs sometimes have `null` for fields like `mlp_bias`. The reader drops `None` values so non-nullable fields (e.g. `bool`, `float`) fall back to their dataclass defaults rather than storing `None`.
- **`layer_types` resolution**: if the config has `sliding_window_pattern` but no `layer_types`, the reader generates the list: every `pattern`-th layer (1-indexed) is `"full_attention"`, the rest are `"sliding_attention"`. For Gemma 3 with pattern=6 and 26 layers, this gives global layers at indices 5, 11, 17, 23. If `layer_types` is already present (newer transformers configs), it is used as-is.

Reads from a local path (`str` or `Path`). Fails fast if `model_type` is not in the supported set (`"llama"`, `"qwen3"`, `"gemma3_text"`). Unknown fields are silently ignored.

### 2. Safetensors weight loader (`src/infer/loader/weights.py`)

Load model weights from safetensors files into a flat `dict[str, torch.Tensor]`.

Two code paths:
- **Single-file**: a lone `model.safetensors` file. No external metadata to validate against — the loader returns whatever tensors are in the file.
- **Sharded**: an `model.safetensors.index.json` pointing to multiple shard files. Validate that the set of loaded tensor names matches the index's `weight_map` (no missing, no unexpected).

The loader returns the raw HF-namespaced tensor dict. It does not do any renaming — that's the model's responsibility when it calls `load_state_dict`.

Key details:
- Use the `safetensors.torch.load_file` API for each shard.
- Accept `device` argument (passed to `load_file`).
- Accept `dtype` argument — applied as a post-load `.to(dtype)` conversion on each tensor, since `safetensors.torch.load_file` does not support dtype natively.

### 3. Weight name mapping (`src/infer/loader/weight_map.py`)

Per-architecture functions that map HF checkpoint tensor names to our internal module names.

```python
def llama_weight_map(num_layers: int) -> dict[str, str]: ...
def qwen3_weight_map(num_layers: int) -> dict[str, str]: ...
def gemma3_weight_map(num_layers: int) -> dict[str, str]: ...
```

Key differences between architectures:
- Qwen 3 and Gemma 3 have additional `q_norm`/`k_norm` weight tensors per layer.
- Gemma 3 has 4 norm weight tensors per layer (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`) vs 2 for Llama/Qwen.
- `lm_head.weight` is always included in the map. Real HF checkpoints are inconsistent about including it when `tie_word_embeddings=True` (e.g. Qwen3-1.7B includes it even when tied). Whether to reuse `embed_tokens.weight` as the LM head is a model construction concern.

A dispatcher selects the right map based on `model_type`:

```python
def get_weight_map(config: ModelConfig) -> dict[str, str]: ...
```

### 4. Tokenizer wrapper (`src/infer/loader/tokenizer.py`)

Thin wrapper around HuggingFace `AutoTokenizer`.

```python
class Tokenizer:
    def __init__(self, model_path: str) -> None: ...
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...
    @property
    def eos_token_id(self) -> int | list[int]: ...
    @property
    def bos_token_id(self) -> int | None: ...
    @property
    def vocab_size(self) -> int: ...
```

`eos_token_id` returns `int | list[int]` because Llama 3.2 defines multiple EOS tokens (`[128001, 128008, 128009]`).

No custom tokenization logic — this just provides a stable interface so the rest of the codebase doesn't import `transformers` directly.

### 5. Shared components (`src/infer/models/common.py`)

Components shared across all three architectures:

**RMSNorm**
- `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight`
- Parameters: `weight` (dim,)
- Used for layer norms (dim=hidden_size) and QK-norms (dim=head_dim).

**RoPE (Rotary Position Embeddings)**

Two concerns, cleanly separated:

1. **Precompute cos/sin tables** — a factory function that returns the right tables based on config. Three variants:
   - **Vanilla**: standard `theta^(-2i/d)` frequencies. Used by Qwen 3 (theta=1M), and Gemma 3 local layers (theta=10K).
   - **Llama3-scaled**: frequency-dependent scaling per `rope_scaling` config. Applies different scale factors to different frequency bands based on `factor`, `high_freq_factor`, `low_freq_factor`, and `original_max_position_embeddings`. Used by Llama 3.
   - **Linear-scaled**: uniform frequency scaling by `factor`. Used by Gemma 3 global layers (4B+).

2. **Apply rotation** — a single shared function that rotates Q and K given precomputed cos/sin tables. Same for all architectures.

For Gemma 3, two sets of cos/sin tables are needed: one with `rope_theta` (for global attention layers) and one with `rope_local_base_freq` (for sliding window layers). The `TransformerBlock` selects the right tables based on its layer type. Note: the 1B model has `rope_scaling: null`, so both sets use vanilla RoPE (just different theta values). Linear scaling only applies to 4B+ models.

**Attention module**
- Q/K/V linear projections with configurable bias. Projection dimensions are `num_attention_heads * head_dim` for Q and `num_key_value_heads * head_dim` for K/V — not `hidden_size`, since `head_dim` can be decoupled (e.g. Gemma 3 1B: `hidden_size=1152` but `num_heads * head_dim = 4 * 256 = 1024`).
- Optional QK-norm (RMSNorm on Q and K per-head, applied after projection, before RoPE). Used by Qwen 3 and Gemma 3.
- K/V head expansion for GQA (`repeat_interleave` when `num_kv_heads < num_attention_heads`).
- Configurable attention scaling: pass `scale=head_dim**-0.5` (default) or `scale=query_pre_attn_scalar**-0.5` to `F.scaled_dot_product_attention`'s `scale` parameter.
- `F.scaled_dot_product_attention` with caller-provided `attn_mask`. The mask construction is the caller's responsibility — standard causal mask for most layers, sliding window causal mask for Gemma 3 local attention layers.
- Output projection (`head_dim * num_attention_heads` → `hidden_size`).

**Gated MLP**
- gate_proj, up_proj, down_proj linear layers with configurable bias.
- Configurable activation function (SiLU for SwiGLU, GELU-tanh for GeGLU).
- `down_proj(act_fn(gate_proj(x)) * up_proj(x))`

**Mask helpers**

Masks use the float additive convention (`0.0` for attend, `-inf` for mask) with shape `[1, 1, seq_len, seq_len]`, matching `transformers` and `F.scaled_dot_product_attention`'s `attn_mask` parameter.

- `causal_mask(seq_len)` — standard lower-triangular causal mask.
- `sliding_window_causal_mask(seq_len, window_size)` — causal mask that also masks positions beyond the sliding window.

### 6. Per-architecture transformer blocks

Llama 3 and Qwen 3 have identical block-level structure (pre-norm with 2 block norms). They differ only in whether the attention module uses QK-norm. Gemma 3 has a structurally different block (sandwich norm with 4 block norms).

Per-architecture files exist for forward-looking reasons — Phase 1b will add full model classes (embedding, layer stacking, final norm, lm_head) which diverge more.

**Llama 3** (`src/infer/models/llama.py`) — pre-norm, 2 block norms, no QK-norm:
```
residual = x
x = input_layernorm(x)
x = attention(x, positions, mask)     # no QK-norm, scale=1/sqrt(head_dim)
x = residual + x
residual = x
x = post_attention_layernorm(x)
x = mlp(x)                            # SwiGLU
x = residual + x
```

**Qwen 3** (`src/infer/models/qwen3.py`) — pre-norm, 2 block norms, with QK-norm:

Same block structure as Llama 3. The only difference is that the attention module has QK-norm enabled (RMSNorm on Q and K per-head, after projection, before RoPE).

```
residual = x
x = input_layernorm(x)
x = attention(x, positions, mask)     # QK-norm on Q,K before RoPE, scale=1/sqrt(head_dim)
x = residual + x
residual = x
x = post_attention_layernorm(x)
x = mlp(x)                            # SwiGLU
x = residual + x
```

**Gemma 3** (`src/infer/models/gemma3.py`) — sandwich norm, 4 block norms, with QK-norm:

Structurally different: post-sub-layer norms applied *before* the residual add (sandwich pattern). Takes a `layer_type` parameter (`"sliding_attention"` or `"full_attention"`, from `config.layer_types[i]`) to select the right attention mask and RoPE frequencies.

```
residual = x
x = input_layernorm(x)
x = attention(x, positions, mask)     # QK-norm, scale=1/sqrt(query_pre_attn_scalar)
x = post_attention_layernorm(x)       # post-norm BEFORE residual add
x = residual + x
residual = x
x = pre_feedforward_layernorm(x)
x = mlp(x)                            # GeGLU (gelu_pytorch_tanh)
x = post_feedforward_layernorm(x)     # post-norm BEFORE residual add
x = residual + x
```

Note: we only build a single `TransformerBlock` per architecture in this phase. The full model stack (embed + N layers + final norm + lm_head) is Phase 1b. Gemma 3 also scales embeddings by `sqrt(hidden_size)` before the first layer — this is a Phase 1b concern since single-layer tests use synthetic inputs.

---

## File Layout

```
src/infer/
├── __init__.py
├── loader/
│   ├── __init__.py
│   ├── config.py        # ModelConfig dataclass + reader
│   ├── weights.py       # safetensors loader
│   ├── weight_map.py    # HF->internal name mapping (per architecture)
│   └── tokenizer.py     # AutoTokenizer wrapper
└── models/
    ├── __init__.py
    ├── common.py         # RMSNorm, RoPE factory, Attention, GatedMLP, mask helpers
    ├── llama.py          # LlamaTransformerBlock
    ├── qwen3.py          # Qwen3TransformerBlock
    └── gemma3.py         # Gemma3TransformerBlock
```

---

## Testing Plan

### Config tests (`tests/unit/test_config.py`) — DONE

Sample configs in tests use values from real HF config.json files (Llama-3.2-1B-Instruct, Qwen3-1.7B, gemma-3-1b-it).

- **Config reader**: parse each architecture, verify all fields populate correctly. Test `str` and `Path` inputs.
- **Error handling**: unsupported `model_type`, missing `model_type`, missing required fields.
- **Config normalization**: `hidden_activation` → `hidden_act`, `text_config` extraction (with and without `model_type`), extra fields ignored, JSON `null` → defaults, no input mutation.
- **`computed_head_dim`**: explicit, inferred (hidden_size // num_heads), and decoupled (Gemma 3: head_dim=256 ≠ hidden_size/num_heads=288).
- **`layer_types`**: resolution from `sliding_window_pattern`, no resolution without pattern, explicit `layer_types` preserved.

### Weight loader tests (`tests/unit/test_weights.py`) — DONE

- **Single-file loading**: create a small safetensors file with known tensors, load it, verify tensor names/shapes/values. Verify dtype conversion preserves values. Verify `str` path and `torch.device` object inputs work.
- **Sharded loading**: create two small shards + index JSON, load, verify all tensors present and correct. Verify dtype conversion preserves values. Verify index takes precedence when both layouts exist.
- **Validation**: missing tensors (in index but not in shards) and unexpected tensors (in shards but not in index) both raise `ValueError`. Missing shard file raises `FileNotFoundError` with a helpful message naming the missing shard.
- **Mixed dtypes**: file with mixed fp32/fp16 tensors preserves original dtypes without `dtype` arg, converts all with `dtype` arg.
- **Device propagation**: verify `device` argument is passed through to loaded tensors.

### Weight map tests (`tests/unit/test_weight_map.py`) — DONE

- **Per-architecture structure**: verify each architecture's mapping produces the expected per-layer weight count (Llama: 9, Qwen3: 11, Gemma3: 13) and total weight count. Verify Llama has no QK-norm, Qwen3/Gemma3 have q_norm/k_norm, Gemma3 has 4 sandwich norms.
- **Global weights**: embed_tokens, norm, and lm_head always present. lm_head.weight always included regardless of tie_word_embeddings (real HF checkpoints are inconsistent about omitting it).
- **Dispatcher**: get_weight_map routes correctly by model_type, uses num_hidden_layers from config.
- **Dev model layer counts**: parametrized test at real layer counts (Llama 16, Qwen3 28, Gemma3 26).
- **Consistency**: internal names equal HF names with `model.` prefix stripped (all architectures). Mappings are bijective (no collisions). Qwen3 is a strict superset of Llama (extra keys are only q_norm/k_norm).

### Tokenizer tests (`tests/unit/test_tokenizer.py`)

- Encode a known string, verify token IDs match `AutoTokenizer` directly.
- Decode token IDs back, verify text.
- Check `eos_token_id` (including list case), `bos_token_id`, `vocab_size` properties.

These tests require a model to be available. Use `meta-llama/Llama-3.2-1B-Instruct`. Marked `@pytest.mark.slow`. Tests skip gracefully if the model is not accessible.

### Component tests (`tests/unit/test_components.py`)

- **RMSNorm**: create with random weight, feed random input, verify output matches the formula. Check output shape and dtype.
- **RoPE factory**: verify cos/sin table shapes for vanilla, llama3-scaled, and linear-scaled variants. Verify that different `rope_theta` values produce different tables.
- **RoPE apply**: apply rotation to a known Q tensor, verify against a hand-computed rotation for a small example.
- **Attention**: construct a small module (e.g. 4 heads, 2 KV heads, head_dim=8), feed random input, verify output shape. Test with and without QK-norm. Test with custom `query_pre_attn_scalar`. Test with both causal and sliding window masks.
- **GatedMLP**: verify output shape and that the computation matches `down(act(gate(x)) * up(x))` for both SiLU and GELU-tanh activations.
- **Mask helpers**: verify causal mask shape and values. Verify sliding window mask blocks positions outside the window.

### Single-layer parity tests (`tests/integration/test_single_layer_parity.py`)

Three parity tests, one per architecture. Each marked `@pytest.mark.slow`. Tests skip gracefully if the model is not accessible.

For each model:
1. Load model with `transformers`.
2. Load the same layer 0 weights into our `TransformerBlock` using our loader + weight map.
3. Generate random hidden states (shape `[1, seq_len, hidden_size]`, bf16) as input — avoids dependency on embedding implementation.
4. Precompute `position_embeddings` (cos/sin tuple) from the reference model's `rotary_emb` module — modern `transformers` decoder layers expect pre-computed RoPE tables, not raw `position_ids`. For Gemma 3, the model precomputes two sets (global and local); extract the correct one based on layer type.
5. Feed the same input, position embeddings, and attention mask through both implementations.
6. Compare output activations.

| Test    | Model                   | Key differences exercised                                                              |
|---------|-------------------------|----------------------------------------------------------------------------------------|
| Llama 3 | `Llama-3.2-1B-Instruct` | Pre-norm, no QK-norm, head_dim=64, llama3 RoPE scaling                                 |
| Qwen 3  | `Qwen3-1.7B`            | QK-norm, head_dim=128, vanilla RoPE with theta=1M                                      |
| Gemma 3 | `gemma-3-1b-it`         | Sandwich norm, QK-norm, GeGLU, query_pre_attn_scalar=256, head_dim=256, sliding window |

For the Gemma 3 test, layer 0 is a sliding window layer (`sliding_window_pattern=6`, so layers 0-4 are local, layer 5 is global). The test uses a sliding window causal mask with `window_size=512`.

Tolerance thresholds (bf16 only — fp32 tests are not required for exit criteria):
- **bf16**: max absolute error < 1e-2, mean absolute error < 1e-3

Document actual measured errors in test output for future tightening.

---

## Exit Criteria

1. All unit tests pass (`uv run pytest tests/unit/`).
2. Single-layer parity tests pass for all three models at bf16 (`uv run pytest tests/integration/ -m slow`).
3. `uv run ruff check .` and `uv run mypy .` pass cleanly.
4. Loader handles both single-file and sharded safetensors checkpoints correctly.
