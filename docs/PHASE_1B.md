# Phase 1b: Full Model and Logits Parity

## Goal

Assemble the full models for all three architectures and verify end-to-end logits match `transformers`. Build the chat template renderer and layer-by-layer debugging tools.

Dev models: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.

---

## Deliverables

### 1. Full model classes

Add full model `nn.Module` subclasses to the existing architecture files. Each model stacks embedding, all transformer blocks, a final norm, and an LM head.

**LlamaModel** (`src/infer/models/llama.py`):

```python
class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None: ...
    def forward(self, input_ids: Tensor) -> Tensor: ...
```

Structure:
```
x = embed_tokens(input_ids)                     # [batch, seq_len, hidden_size]
mask = causal_mask(seq_len, dtype, device)
cos, sin = rope_cos_sin[:seq_len]               # llama3-scaled RoPE
for layer in layers:
    x = layer(x, cos, sin, mask)
x = norm(x)                                     # RMSNorm
logits = lm_head(x)                             # [batch, seq_len, vocab_size]
```

Components:
- `embed_tokens`: `nn.Embedding(vocab_size, hidden_size)`.
- `layers`: `nn.ModuleList` of `LlamaTransformerBlock` (existing class, one per layer).
- `norm`: `RMSNorm(hidden_size, eps=rms_norm_eps)`.
- `lm_head`: `nn.Linear(hidden_size, vocab_size, bias=False)`.
- RoPE tables: precomputed via `build_rope_cos_sin(computed_head_dim, max_position_embeddings, rope_theta, rope_scaling)`, registered as non-persistent buffers.

**Qwen3Model** (`src/infer/models/qwen3.py`):

Same structure as Llama. Differences:
- Uses `Qwen3TransformerBlock` (has QK-norm).
- Vanilla RoPE with `theta=1000000`.
- `tie_word_embeddings=True` for Qwen3-1.7B: `lm_head.weight` shares `embed_tokens.weight` (see weight loading below).

**Gemma3Model** (`src/infer/models/gemma3.py`):

```python
class Gemma3Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None: ...
    def forward(self, input_ids: Tensor) -> Tensor: ...
```

Structure:
```
x = embed_tokens(input_ids)                     # [batch, seq_len, hidden_size]
x = x * sqrt(hidden_size)                       # Gemma 3 embedding normalizer
for i, layer in enumerate(layers):
    layer_type = layer_types[i]
    if layer_type == "sliding_attention":
        cos, sin = local_cos_sin[:seq_len]
        mask = sliding_window_causal_mask(seq_len, sliding_window, dtype, device)
    else:
        cos, sin = global_cos_sin[:seq_len]
        mask = causal_mask(seq_len, dtype, device)
    x = layer(x, cos, sin, mask)
x = norm(x)                                     # Gemma3RMSNorm
logits = lm_head(x)                             # [batch, seq_len, vocab_size]
```

Components:
- `embed_tokens`: `nn.Embedding(vocab_size, hidden_size)`.
- `layers`: `nn.ModuleList` of `Gemma3TransformerBlock` (existing class, one per layer).
- `norm`: `Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)` — **not** standard `RMSNorm`. Uses the `(1 + weight)` convention to match HF checkpoints.
- `lm_head`: `nn.Linear(hidden_size, vocab_size, bias=False)`.
- `layer_types`: stored as a plain `list[str]` instance attribute from `config.layer_types` (not a parameter or buffer). Used in `forward` to select per-layer mask and RoPE tables.
- `embedding_normalizer`: `float` stored as `config.hidden_size ** 0.5`. Applied as a scalar multiply after embedding lookup.
- Local RoPE tables: precomputed via `build_rope_cos_sin(computed_head_dim, max_position_embeddings, rope_local_base_freq)`, registered as non-persistent buffers (`local_cos`, `local_sin`).
- Global RoPE tables: precomputed via `build_rope_cos_sin(computed_head_dim, max_position_embeddings, rope_theta, rope_scaling)`, registered as non-persistent buffers (`global_cos`, `global_sin`). For the 1B model (`rope_scaling: null`), this is vanilla RoPE with `theta=1000000`. For 4B+ models, this uses linear-scaled RoPE.

Differences from Llama/Qwen:
- **Embedding normalizer**: multiply embedding output by `sqrt(hidden_size)` before the first layer.
- **Dual RoPE tables**: two sets of precomputed cos/sin — one with `rope_local_base_freq` (for sliding window layers) and one with `rope_theta` (for global layers).
- **Per-layer mask selection**: sliding window layers use `sliding_window_causal_mask`, global layers use `causal_mask`.
- **Gemma3RMSNorm** for the final norm (existing `(1 + weight)` variant).
- `tie_word_embeddings=True` for gemma-3-1b-it.

**Common forward signature** — all three models:

```python
def forward(self, input_ids: Tensor) -> Tensor:
    """
    Args:
        input_ids: Token IDs, shape [batch, seq_len].

    Returns:
        Logits, shape [batch, seq_len, vocab_size].
    """
```

This is the minimal useful signature. KV cache arguments are added in Phase 3.

**Mask dtype and device**: the masks produced by `causal_mask` and `sliding_window_causal_mask` (from `common.py`) currently return float32 tensors on CPU. The full model `forward` must cast and move masks to match the model's dtype and device. Update the mask helpers to accept `dtype` and `device` arguments for this purpose, with defaults of `torch.float32` and `"cpu"` to preserve backward compatibility.

**RoPE buffer registration**: use `register_buffer("cos", cos, persistent=False)` and `register_buffer("sin", sin, persistent=False)`. Non-persistent buffers move with `.to(device)` but are excluded from `state_dict`, avoiding conflicts during weight loading.

**RoPE table memory**: tables are precomputed for the full `max_position_embeddings` length (Llama: 131072, Qwen3: 40960, Gemma3: 32768). Each cos/sin pair is `[max_seq_len, head_dim]` float32. Llama: ~64 MB total (2 tables). Gemma 3: ~128 MB total (4 tables — two local, two global, with `head_dim=256`). This is modest relative to model weights and matches HF's precomputation approach. For Phase 1b tests with short sequences, only a small prefix of each table is used.

**RoPE dtype handling**: tables are precomputed in float32 for precision during the trigonometric computation. They are cast to the model dtype (e.g. bf16) once at load time via `model.to(device=device, dtype=dtype)` in `load_model`, matching HF's behavior. There is no per-forward-call dtype casting.

### 2. Model loading function (`src/infer/loader/model_loader.py`)

A top-level function that combines config loading, model construction, weight loading, and state dict application.

```python
def load_model(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[nn.Module, ModelConfig]:
    """Load a model from a local directory or HuggingFace Hub model ID.

    Args:
        model_path: Path to a local directory containing config.json and
            safetensors weights, or a HuggingFace Hub model ID
            (e.g. "meta-llama/Llama-3.2-1B-Instruct").
        dtype: Target dtype for model parameters.
        device: Target device.

    Returns:
        A tuple of (model, config). The model has weights loaded and is in eval mode.
    """
```

Steps:

1. `local_path = _resolve_model_path(model_path)` — resolve Hub ID to local cache path (or pass through if already a local directory).
2. `config = load_config(local_path)` — parse config.json.
3. `model = _build_model(config)` — dispatch by `model_type` to construct the right model class.
4. `raw_weights = load_weights(local_path, device="cpu", dtype=dtype)` — load safetensors as flat dict with HF names. Load to CPU first to avoid doubling GPU memory (random init weights + loaded weights).
5. `weight_map = get_weight_map(config)` — get HF→internal name mapping.
6. Rename: `renamed = {internal: raw_weights[hf] for hf, internal in weight_map.items() if hf in raw_weights}`.
7. Handle tied embeddings: if `"lm_head.weight"` not in `renamed` and `"embed_tokens.weight"` in `renamed`, copy it. Some HF configs omit the `tie_word_embeddings` flag entirely (e.g. Gemma 3), so we check by absence of the weight rather than relying on the flag.
8. `model.load_state_dict(renamed, strict=True)` — strict mode to catch missing/unexpected keys. RoPE buffers are non-persistent so they are excluded from strict checking.
9. `model.to(device=device, dtype=dtype)` — move to target device and cast to target dtype. This also converts the float32 RoPE buffers to the model dtype (e.g. bf16), matching HF's behavior.
10. `model.eval()` — set to eval mode.
11. Return model and config as a tuple: `(model, config)`.

**Dispatcher** `_build_model(config: ModelConfig) -> nn.Module`:

```python
_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "llama": LlamaModel,
    "qwen3": Qwen3Model,
    "gemma3_text": Gemma3Model,
}
```

Fails fast with `ValueError` if `model_type` is not in the registry.

**Tied embeddings handling**: Qwen3-1.7B and gemma-3-1b-it have `tie_word_embeddings=True`. Their HF checkpoints may or may not include `lm_head.weight`:
- **Present in checkpoint**: our weight map includes it, it gets loaded, and `load_state_dict` receives both `embed_tokens.weight` and `lm_head.weight` as separate tensors (they happen to be equal). No special handling needed.
- **Missing from checkpoint**: our weight map expects it but it's absent. Step 7 copies `embed_tokens.weight` to fill the gap. This is detected by checking for absence of `lm_head.weight` in the renamed dict (not by the `tie_word_embeddings` config flag, since some models like Gemma 3 omit the flag entirely).

After loading, the model's `embed_tokens.weight` and `lm_head.weight` will be separate tensors with identical values. They are not sharing memory. This is correct for inference — shared memory is a training optimization that doesn't affect correctness.

**HuggingFace Hub support**: `model_path` can be either a local directory containing `config.json` and safetensors weights, or a HuggingFace Hub model ID (e.g. `"meta-llama/Llama-3.2-1B-Instruct"`). Hub IDs are resolved to a local cache path via `huggingface_hub.snapshot_download`.

### 3. Chat template renderer (`src/infer/loader/chat_template.py`)

Jinja2-based renderer that formats a list of chat messages into a prompt string. Each supported model family has its own template.

```python
def render_chat_template(
    messages: list[dict[str, str]],
    model_type: str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat messages into a formatted prompt string.

    Args:
        messages: List of message dicts, each with "role" and "content" keys.
            Roles: "system", "user", "assistant".
        model_type: One of "llama", "qwen3", "gemma3_text".
        add_generation_prompt: Whether to append the assistant turn header
            at the end (for prompting the model to generate).

    Returns:
        The formatted prompt string, including special token text.

    Raises:
        ValueError: If model_type is not supported.
    """
```

The returned string includes special token text (e.g. `<|begin_of_text|>`). The caller tokenizes it with `tokenizer.encode(prompt, add_special_tokens=False)` — the tokenizer recognizes these as added tokens and produces the correct token IDs.

**Llama 3 template**:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Jinja2 source:

```jinja2
{{- "<|begin_of_text|>" -}}
{% for message in messages -%}
<|start_header_id|>{{ message["role"] }}<|end_header_id|>

{{ message["content"] }}<|eot_id|>
{% endfor -%}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>

{% endif -%}
```

Supported roles: `system`, `user`, `assistant`, `ipython`.

**Qwen 3 template** (ChatML-based):

```
<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
```

Jinja2 source:

```jinja2
{% for message in messages -%}
<|im_start|>{{ message["role"] }}
{{ message["content"] }}<|im_end|>
{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
```

Supported roles: `system`, `user`, `assistant`.

**Gemma 3 template**:

Gemma 3 does **not** support a `system` turn token. System messages are folded into the first user turn, separated by a blank line. The BOS token (`<bos>`) is prepended.

Without system message:
```
<bos><start_of_turn>user
{user_content}<end_of_turn>
<start_of_turn>model
```

With system message:
```
<bos><start_of_turn>user
{system_content}

{user_content}<end_of_turn>
<start_of_turn>model
```

Jinja2 source:

```jinja2
{{- "<bos>" -}}
{% if messages[0]["role"] == "system" -%}
    {%- set first_user_prefix = messages[0]["content"] + "\n\n" -%}
    {%- set loop_messages = messages[1:] -%}
{% else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{% endif -%}
{% for message in loop_messages -%}
{% if message["role"] == "assistant" -%}
<start_of_turn>model
{{ message["content"] }}<end_of_turn>
{% else -%}
<start_of_turn>{{ message["role"] }}
{{ first_user_prefix if loop.first else "" }}{{ message["content"] }}<end_of_turn>
{% endif -%}
{% endfor -%}
{% if add_generation_prompt -%}
<start_of_turn>model
{% endif -%}
```

Gemma uses `model` as the assistant role name. The template maps `"assistant"` → `"model"` so callers can use the standard role name. System messages are extracted from position 0 and prepended to the first user turn's content.

Supported roles: `user`, `assistant` (rendered as `model`). `system` is accepted only as the first message and is folded into the first user turn.

**Implementation notes**:
- Templates are compiled once at import time using `jinja2.Environment(keep_trailing_newline=True)`. The `keep_trailing_newline` setting ensures that trailing newlines in the template source are preserved, which matters for correct formatting.
- Templates are stored as module-level constants (plain strings) in `chat_template.py`.
- The renderer does not validate message content or role sequences — it just formats what it receives. Role validation is a higher-level concern.

**Parity validation**: for each model, verify that our rendered output matches `transformers`'s `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` for a set of test conversations. This validates both the template text and the Jinja2 rendering behavior.

### 4. Layer-by-layer activation diff tooling (`src/infer/debug.py`)

A utility for debugging numerical mismatches between our model and HF `transformers`. Hooks into both models, runs the same input, and reports per-layer activation differences.

```python
@dataclass
class LayerDiff:
    """Activation diff for a single layer."""
    layer_index: int
    max_abs_error: float
    mean_abs_error: float
    our_norm: float        # L2 norm of our output
    ref_norm: float        # L2 norm of reference output

@dataclass
class ModelDiff:
    """Full model activation diff report."""
    embedding_diff: LayerDiff    # after embedding (before first layer)
    layer_diffs: list[LayerDiff] # per-layer after each transformer block
    final_norm_diff: LayerDiff   # after final norm
    logits_diff: LayerDiff       # final logits

def compare_models(
    our_model: nn.Module,
    ref_model: nn.Module,
    input_ids: Tensor,
) -> ModelDiff:
    """Compare activations between our model and a reference model.

    Registers forward hooks on both models to capture intermediate
    activations at each layer boundary. Runs the same input through
    both and computes per-layer error metrics.

    Args:
        our_model: Our model instance (LlamaModel, Qwen3Model, or Gemma3Model).
        ref_model: HuggingFace transformers model (AutoModelForCausalLM).
        input_ids: Token IDs, shape [batch, seq_len].

    Returns:
        A ModelDiff with per-layer error metrics.
    """
```

**Hook strategy**:

For **our model**: register a forward hook on each transformer block in `model.layers` to capture its output tensor. Also hook `model.norm` (post-final-norm) and capture the final logits from the forward return value.

For the **HF model**: register hooks on the corresponding submodules. The exact module paths depend on the architecture:
- Llama: `model.model.layers[i]`, `model.model.norm`
- Qwen3: `model.model.layers[i]`, `model.model.norm`
- Gemma3: `model.model.layers[i]`, `model.model.norm`

**Embedding comparison**: use a forward **pre-hook** on `model.layers[0]` (ours) and `model.model.layers[0]` (HF) to capture the input to the first transformer block. This captures the embedding **after** any model-specific transforms (e.g. Gemma 3's `sqrt(hidden_size)` scaling), avoiding the need to replicate those transforms in the diff tooling. A post-hook on `embed_tokens` would miss the scaling since it happens in the model's `forward`, not inside `embed_tokens`.

Since the HF model structure places the causal LM head outside the backbone, capture logits from the HF model's forward return value (`output.logits`).

**Output format**: `compare_models` returns a `ModelDiff` which can be printed as a summary table. Provide a `format_diff(diff: ModelDiff) -> str` function that produces a readable table:

```
Layer       Max Abs Err   Mean Abs Err   Our Norm     Ref Norm
embed       1.23e-04      4.56e-05       125.3        125.3
layer_0     2.34e-04      5.67e-05       130.1        130.1
layer_1     3.45e-04      6.78e-05       128.9        128.9
...
final_norm  4.56e-04      7.89e-05       127.5        127.5
logits      5.67e-03      1.23e-03       450.2        450.2
```

This tooling is used interactively during development and by integration tests. It is not part of the production inference path.

---

## File Layout

New and modified files:

```
src/infer/
├── debug.py                    # NEW: layer-by-layer diff tooling
├── loader/
│   ├── chat_template.py        # NEW: Jinja2 chat template renderer
│   └── model_loader.py         # NEW: load_model() entry point
└── models/
    ├── common.py               # MODIFIED: add dtype/device args to mask helpers
    ├── llama.py                # MODIFIED: add LlamaModel class
    ├── qwen3.py                # MODIFIED: add Qwen3Model class
    └── gemma3.py               # MODIFIED: add Gemma3Model class
```

---

## Testing Plan

### Chat template tests (`tests/unit/test_chat_template.py`)

Test the Jinja2 templates against known expected outputs.

- **Per-architecture rendering**: for each model type, render a standard conversation (system + user + assistant + user) and verify the output string matches expected format character-for-character.
- **Generation prompt**: verify `add_generation_prompt=True` appends the assistant header, `False` does not.
- **Role mapping**: verify Gemma 3 maps `"assistant"` → `"model"` in the output.
- **System message**: verify system messages render correctly for Llama (header-based) and Qwen (ChatML `<|im_start|>system`). Verify Gemma folds system messages into the first user turn (no `<start_of_turn>system` — system content is prepended to the first user message, separated by a blank line).
- **Empty conversation**: single user message with no system, verify minimal output.
- **Unsupported model_type**: verify `ValueError` for unknown model types.
- **Parity against transformers** (`@pytest.mark.slow`): for each dev model, compare our rendered output against `transformers.AutoTokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. Parametrized across models. Tests skip gracefully if the model tokenizer is not accessible.

### Model loader tests (`tests/unit/test_model_loader.py`)

- **Dispatcher**: verify `_build_model` returns the right class for each model_type. Verify `ValueError` for unsupported types.
- **Tied embeddings**: construct a model with `tie_word_embeddings=True`, load weights where `lm_head.weight` is missing from the checkpoint, verify `load_model` fills it from `embed_tokens.weight` and `load_state_dict` succeeds.

### Mask helper update tests

- **dtype/device args**: verify `causal_mask` and `sliding_window_causal_mask` produce tensors with the requested dtype and device. Verify backward compatibility (defaults match current behavior).

### Full-model logits parity tests (`tests/integration/test_logits_parity.py`)

Six parity tests: three architectures × two dtypes (float32, bfloat16). Each marked `@pytest.mark.slow`. Tests skip gracefully if the model is not accessible. Tests auto-detect CUDA via a `device` fixture.

For each model × dtype combination:
1. Load the HF model with `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)`.
2. Load our model with `load_model(model_id, dtype=dtype, device=device)`.
3. Create a test input: a short token sequence (e.g. `tokenizer.encode("The capital of France is")`, length ~10 tokens).
4. Run forward on both models with the same `input_ids`.
5. Compare the output logits tensors against dtype-specific thresholds.

| Test    | Model                    | Key differences exercised                                                                     |
|---------|--------------------------|-----------------------------------------------------------------------------------------------|
| Llama 3 | `Llama-3.2-1B-Instruct`  | 16 layers, llama3-scaled RoPE, no tied embeddings, head_dim=64                                |
| Qwen 3  | `Qwen3-1.7B`             | 28 layers, vanilla RoPE theta=1M, tied embeddings, QK-norm, head_dim=128                     |
| Gemma 3 | `gemma-3-1b-it`          | 26 layers, dual RoPE, tied embeddings, embedding scaling, sliding window, sandwich norm       |

Tolerance thresholds:
- **float32**: max < 1e-4, mean < 1e-5. CPU gives exact (0.0) parity. CUDA SDPA kernels introduce tiny non-deterministic rounding (~1e-5).
- **bfloat16**: max < 2.5, mean < 0.2. Rounding accumulation across many transformer layers.

Both dtypes are tested for each architecture. bf16 errors come from rounding accumulation across layers, not from any single component — verified by confirming float32 gives near-exact parity.

Observed errors (dev models, CUDA, short inputs):

| Model      | Dtype | Max Abs Error | Mean Abs Error |
|------------|-------|--------------|----------------|
| Llama 1B   | fp32  | ~0.0         | ~0.0           |
| Llama 1B   | bf16  | ~0.24        | ~0.03          |
| Qwen3 1.7B | fp32  | ~1.2e-5      | ~2.5e-6        |
| Qwen3 1.7B | bf16  | ~1.65        | ~0.12          |
| Gemma3 1B  | fp32  | ~0.0         | ~0.0           |
| Gemma3 1B  | bf16  | ~0.33        | ~0.04          |

Document actual measured errors in test output for future tightening. If errors exceed thresholds, use the diff tooling to identify the first divergent layer.

### End-to-end test (`tests/integration/test_end_to_end.py`)

One parametrized test across all three architectures that exercises the full pipeline: load model, tokenize a real prompt via chat template, forward pass, verify logits shape and parity. Marked `@pytest.mark.slow`. Uses a `device` fixture that auto-detects CUDA. Runs in bfloat16.

```python
@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_end_to_end(model_id: str, device: str) -> None:
    model, config = load_model(model_id, dtype=torch.bfloat16, device=device)
    tokenizer = Tokenizer(model_id)
    prompt = render_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        model_type=config.model_type,
    )
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    logits = model(input_ids)

    # Verify shape
    assert logits.shape == (1, input_ids.shape[1], config.vocab_size)

    # Verify logits match HF (load HF model, compare)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    ref_logits = ref_model.to(device)(input_ids).logits
    # bf16 thresholds
    assert (logits.float() - ref_logits.float()).abs().max().item() < 2.5
    assert (logits.float() - ref_logits.float()).abs().mean().item() < 0.2
```

This validates the full pipeline: config loading → model construction → weight loading → tokenization → chat template → forward pass → logits parity.

### Diff tooling tests (`tests/unit/test_debug.py`)

- **Smoke test**: run `compare_models` with a small synthetic model (2 layers) and verify it returns a `ModelDiff` with the right number of layer diffs.
- **Identical models**: create two identical small models, verify all errors are zero.
- **Format test**: verify `format_diff` produces a string with the expected table format.

---

## Exit Criteria

1. Full-model logits parity tests pass for all three dev models at both float32 and bfloat16 (`uv run pytest tests/integration/test_logits_parity.py -m slow`).
2. End-to-end tests pass for all three dev models (`uv run pytest tests/integration/test_end_to_end.py -m slow`).
3. All unit tests pass (`uv run pytest tests/unit/`).
4. Chat template output matches `transformers.apply_chat_template` for Qwen 3 and Gemma 3. (Llama excluded from parity check — HF's template injects a default "Cutting Knowledge Date" system prompt that our simplified template omits.)
5. `uv run ruff check .` and `uv run mypy .` pass cleanly.
6. Max absolute error thresholds documented by dtype in test output.
