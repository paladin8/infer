# Phase 10: Weight Quantization (FP8 + INT8)

## Goal

Load and serve quantized 8B-class models on 16 GB VRAM with reasonable output quality. Two quantization formats are supported:

1. **FP8 (float8_e4m3fn)**: Block-wise quantization with per-block scale factors. Target: `Qwen/Qwen3-8B-FP8`.
2. **INT8 (int8)**: Per-channel symmetric quantization with per-row scale factors. Target: `nytopop/Qwen3-8B.w8a8`.

Both formats halve weight memory from ~16 GB (bf16) to ~8 GB, making 8B-class models practical on a single 16 GB GPU. Decode throughput is memory-bandwidth-bound (reading weights dominates each step), so halving weight size translates to meaningful speedup. Both implementations use eager dequant-then-matmul (dequantize to bf16, then call `F.linear`), which works on any GPU with bf16 support (compute capability >= 8.0, i.e. Ampere+).

Benchmark models: `Qwen/Qwen3-8B-FP8`, `nytopop/Qwen3-8B.w8a8`.

---

## Background: FP8 Checkpoint Format

The `Qwen/Qwen3-8B-FP8` checkpoint uses block-wise FP8 quantization following the format popularized by DeepSeek-V3. Each quantized linear layer has two tensors:

1. **Weight** (`{name}.weight`): stored as `float8_e4m3fn`, shape `[out_features, in_features]`.
2. **Scale** (`{name}.weight_scale_inv`): stored as `float32`, shape `[ceil(out/128), ceil(in/128)]`.

The name `weight_scale_inv` means "inverse of the quantization scale" — the value you *multiply* by to undo quantization. During quantization, blocks are divided by this value; during dequantization, blocks are multiplied. Each scalar in the scale tensor corresponds to one 128x128 block:

```
dequantized[i*128:(i+1)*128, j*128:(j+1)*128] =
    weight_fp8[i*128:(i+1)*128, j*128:(j+1)*128].to(bf16) * weight_scale_inv[i, j]
```

The checkpoint's `config.json` contains a `quantization_config` field:

```json
{
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128]
  }
}
```

Non-quantized layers (embeddings, layer norms, QK norms, LM head) remain in bf16. For Qwen3-8B-FP8, `tie_word_embeddings` may be true, so `lm_head.weight` may be absent from the checkpoint — the existing tied-embeddings logic in `model_loader.py` handles this.

---

## Background: INT8 Checkpoint Format (compressed-tensors)

The `nytopop/Qwen3-8B.w8a8` checkpoint uses per-channel symmetric INT8 quantization in the `compressed-tensors` format. This is produced by LLM Compressor using SmoothQuant + GPTQ with the W8A8 scheme. Each quantized linear layer has two tensors:

1. **Weight** (`{name}.weight`): stored as `int8`, shape `[out_features, in_features]`. Raw signed 8-bit integers, no bit-packing.
2. **Scale** (`{name}.weight_scale`): stored as `bfloat16`, shape `[out_features, 1]`. One scale factor per output channel (per-row). Stored internally as `float32` for precision consistency with the FP8 path (bf16 from checkpoint is coerced to float32 by `load_state_dict`).

The quantization is **symmetric** with zero point implicitly 0 (no zero-point tensor stored). Dequantization is a simple multiply:

```
dequantized[row, :] = weight_int8[row, :].to(bf16) * weight_scale[row, 0]
```

The checkpoint's `config.json` contains a `quantization_config` field:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 8,
          "type": "int",
          "symmetric": true,
          "strategy": "channel"
        },
        "targets": ["Linear"]
      }
    },
    "ignore": ["lm_head"]
  }
}
```

Key properties:
- **Per-channel** (per-row) quantization — one scale per output feature.
- **Symmetric** — no zero point needed (`zero_point = 0` implicitly).
- **8-bit signed integer** — values in `[-128, 127]`.
- **`lm_head` excluded** — kept in bf16 for quality.
- **Dynamic input activation quantization** — `input_activations.dynamic = true` means activation scales are computed at runtime. We only implement weight dequantization (W8A16 inference), not quantized matmul (W8A8).

Non-quantized layers remain in bf16: embeddings, layer norms, QK norms, LM head.

---

## Architecture

```text
load_model(model_path, dtype, device, quantization=None)
│
├── load_config(model_path)
│   └── Parse config.json including quantization_config
│       → ModelConfig with quantization_config field
│
├── Auto-detect quantization from config if not explicitly set
│   └── quantization_config.quant_method == "fp8" → quantization = "fp8"
│
├── _build_model(config)
│   └── Build model skeleton (same as before, with nn.Linear)
│
├── If quantization == "fp8":
│   └── replace_linear_with_fp8(model)
│       → Walk module tree, replace nn.Linear with FP8Linear
│       → Skip: embed_tokens, lm_head, *_norm layers
│       → Assert: all replaced layers have bias=False
│
├── load_weights(model_path, device="cpu", dtype=None)
│   └── Load safetensors as-is (pass dtype=None to skip _apply_dtype)
│       → FP8 tensors stay float8_e4m3fn, scales stay float32,
│         non-quantized tensors stay bf16
│
├── get_weight_map(config, quantization)
│   └── Base weight map + scale tensor entries when quantization="fp8"
│       e.g. "model.layers.0.self_attn.q_proj.weight_scale_inv"
│            → "layers.0.self_attn.q_proj.weight_scale_inv"
│
├── model.load_state_dict(renamed, strict=True)
│   → FP8Linear receives both weight (fp8) and weight_scale_inv (f32)
│
└── Selective dtype/device move:
    → Iterate parameters and buffers
    → Move all to target device
    → Convert to target dtype ONLY if not float8_e4m3fn
    → model.eval()
```

### FP8Linear Module

```text
FP8Linear(in_features, out_features)
│
├── Attributes:
│   ├── weight: Parameter[float8_e4m3fn, (out, in)]
│   │   └── Initialized as empty fp8 tensor; populated by load_state_dict
│   ├── weight_scale_inv: Buffer[float32, (ceil(out/128), ceil(in/128))]
│   │   └── Persistent buffer (included in state_dict for load_state_dict)
│   └── block_size: int = 128
│
├── _apply() override:
│   └── Skip dtype conversion for float8_e4m3fn parameters
│       (prevents model.to(dtype=bf16) from destroying FP8 weights)
│
└── forward(x: Tensor[bf16, (batch, seq, in)]) -> Tensor[bf16, (batch, seq, out)]
    │
    ├── Dequantize weight to bf16:
    │   w_bf16 = fp8_block_dequant(self.weight, self.weight_scale_inv, block_size=128)
    │   → Cast fp8→float32 first (F.pad doesn't support fp8), multiply by scale
    │     in float32 for precision, then cast to bf16
    │   → Shape: (out, in) in bf16
    │
    └── F.linear(x, w_bf16)
        → Standard matmul with dequantized weights
```

**Design decision: eager dequant-then-matmul.** The simplest correct approach is to dequantize the full weight matrix to bf16, then call `F.linear`. This avoids writing a fused Triton FP8 GEMM kernel, which is significantly more complex. The dequantization cost is small relative to the matmul for the Qwen3-8B layer sizes (the largest projection `gate_proj` at [12288, 4096] has a dequant that is ~100x cheaper than the matmul). A fused kernel (`src/infer/kernels/fp8_dequant.py` in the project structure) is deferred to a future optimization pass.

**Inference mode requirement.** The forward pass must run under `torch.no_grad()` or `torch.inference_mode()` to ensure temporary dequantized weight tensors are freed promptly. The model is in `.eval()` mode and the generate/runner code already uses `torch.no_grad()`.

### INT8Linear Module

```text
INT8Linear(in_features, out_features)
│
├── Attributes:
│   ├── weight: Parameter[int8, (out, in)]
│   │   └── Initialized as empty int8 tensor; populated by load_state_dict
│   ├── weight_scale: Buffer[float32, (out, 1)]
│   │   └── Persistent buffer (float32 for precision; bf16 from checkpoint
│   │       coerced to float32 by load_state_dict)
│   └── (no block_size — per-channel quantization)
│
├── _apply() override:
│   └── Skip dtype conversion for int8 parameters
│       (prevents model.to(dtype=bf16) from destroying INT8 weights)
│
└── forward(x: Tensor[bf16, (batch, seq, in)]) -> Tensor[bf16, (batch, seq, out)]
    │
    ├── Dequantize weight to bf16:
    │   w_bf16 = (self.weight.to(float32) * self.weight_scale).to(bf16)
    │   → Multiply in float32 for precision, then cast to bf16
    │   → Shape: (out, in) in bf16
    │
    └── F.linear(x, w_bf16)
        → Standard matmul with dequantized weights
```

**Design decision: per-channel dequant.** INT8 per-channel dequantization is simpler than FP8 block-wise dequant — just cast int8 to bf16 and multiply by a `[out, 1]` scale that broadcasts along the in_features dimension. No padding or reshaping needed.

### Auto-detection Logic

The `_detect_quantization` function in `model_loader.py` is extended to handle both formats:

- `quantization_config.quant_method == "fp8"` → `"fp8"`
- `quantization_config.quant_method == "compressed-tensors"` AND `quantization_config.format == "int-quantized"` → `"int8"`
- Everything else → `None`

The `format` field is the most reliable discriminator for compressed-tensors checkpoints. No need to navigate nested `config_groups` for detection.

---

## Deliverables

### D1: ModelConfig quantization_config field

Add `quantization_config: dict[str, Any] | None = None` to `ModelConfig`. The existing `load_config` field-filtering logic will automatically pick this up from the JSON since it uses `fields(ModelConfig)` to determine known fields.

**Files:** `src/infer/loader/config.py`

### D2: FP8Linear module

Create `src/infer/quant/fp8_linear.py` with the `FP8Linear` module:

- `weight` is an `nn.Parameter` with `dtype=float8_e4m3fn`. Override `_apply` to skip dtype conversion for FP8 parameters (prevents `model.to(dtype=...)` from destroying them).
- `weight_scale_inv` is a persistent registered buffer (`float32`). Must be persistent so `load_state_dict` can populate it.
- `forward()` dequantizes weight to bf16 using block-wise scaling, then calls `F.linear`.
- `fp8_block_dequant(weight, scale, block_size)` helper: vectorized via tensor reshaping (reshape to `[out/B, B, in/B, B]`, broadcast-multiply by `scale[:, None, :, None]`, reshape back). Cast fp8→float32 before multiply for precision, then cast result to bf16.
- Does not support bias. Surgery asserts `bias is False` for replaced layers.

**Files:** `src/infer/quant/__init__.py`, `src/infer/quant/fp8_linear.py`

### D3: Model surgery — replace nn.Linear with FP8Linear

Add `replace_linear_with_fp8(model)` function that walks the module tree and replaces `nn.Linear` instances with `FP8Linear`. Skip modules that should remain in full precision:
- `embed_tokens` (embedding layer, not a linear)
- `lm_head` (final projection to vocab)
- Any module whose name contains `norm` (RMSNorm, QK norms)

Assert that replaced `nn.Linear` modules have `bias is None` (all Qwen3 linear layers are bias-free). Raise `ValueError` if a biased linear is encountered.

**Files:** `src/infer/quant/fp8_linear.py`

### D4: Weight map extension for scale tensors

Add `quantization: str | None = None` parameter to `get_weight_map()` (breaking change — update the single caller in `model_loader.py`). When `quantization == "fp8"`, add `weight_scale_inv` entries alongside each linear layer's weight entry. The per-architecture functions (`llama_weight_map`, `qwen3_weight_map`, `gemma3_weight_map`) gain an optional `quantization` parameter.

**Files:** `src/infer/loader/weight_map.py`

### D5: Model loader FP8 path

Modify `load_model` to:
1. Accept `quantization: str | None = None` parameter.
2. Auto-detect FP8: if `quantization is None` and `config.quantization_config` has `quant_method == "fp8"`, set `quantization = "fp8"`.
3. When FP8: pass `dtype=None` to `load_weights` (skip `_apply_dtype`), apply `replace_linear_with_fp8(model)` after `_build_model`, pass `quantization` to `get_weight_map`.
4. After `load_state_dict`: selective dtype/device move — iterate all parameters and buffers, move to device, convert to target dtype only if not `float8_e4m3fn`.
5. When not FP8: existing behavior unchanged.

**Files:** `src/infer/loader/model_loader.py`

### D6: EngineConfig quantization field

Add `quantization: str | None = None` to `EngineConfig` with validation. Allowed values: `None`, `"fp8"`. Pass through to `load_model` in the engine initialization path.

**Files:** `src/infer/engine/config.py`, `src/infer/engine/engine.py`

### D7: Tests

- Unit test for `fp8_block_dequant`: create a known bf16 weight, quantize to FP8 + scales, dequantize, verify close to original. Include a test case with non-multiple-of-128 dimensions to exercise the padding path.
- Unit test for `FP8Linear`: verify forward pass produces same output as `nn.Linear` with dequantized weights.
- Unit test for `replace_linear_with_fp8`: verify module surgery replaces the right layers, skips norms/embeddings/lm_head, raises on biased layers.
- Unit test for weight map with quantization: verify scale tensor entries are generated.
- Integration test (GPU, requires model): load `Qwen/Qwen3-8B-FP8`, generate a short response, verify it's coherent.

**Files:** `tests/unit/test_fp8_linear.py`, `tests/unit/test_weight_map_fp8.py`

### D8: INT8Linear module

Create `src/infer/quant/int8_linear.py` with the `INT8Linear` module:

- `weight` is an `nn.Parameter` with `dtype=torch.int8`. Override `_apply` to skip dtype conversion for int8 parameters (prevents `model.to(dtype=...)` from destroying them).
- `weight_scale` is a persistent registered buffer (`float32`). Must be persistent so `load_state_dict` can populate it. Checkpoint stores bf16 scales which are coerced to float32 by `load_state_dict`.
- `forward()` dequantizes weight to bf16 via `(weight.to(float32) * weight_scale).to(bf16)`, then calls `F.linear`.
- `int8_channel_dequant(weight, scale)` helper: cast int8 → float32, broadcast-multiply by scale `[out, 1]`, cast to bf16.
- Does not support bias (all target architectures use bias-free linears).

**Files:** `src/infer/quant/int8_linear.py`

### D9: Model surgery — replace nn.Linear with INT8Linear

Add `replace_linear_with_int8(model)` function that walks the module tree and replaces `nn.Linear` instances with `INT8Linear`. Same skip logic as FP8: skip `embed_tokens`, `lm_head`, and `*norm*` layers. Assert `bias is None`.

**Files:** `src/infer/quant/int8_linear.py`

### D10: Weight map and loader extension for INT8

Extend `get_weight_map()` to handle `quantization="int8"`. Add `weight_scale` entries (not `weight_scale_inv`) alongside each linear layer's weight entry. The INT8 scale tensor name in the checkpoint is `{name}.weight_scale`, mapped to `{internal_name}.weight_scale`.

Update `_detect_quantization` to recognize `compressed-tensors` format with int weights as `"int8"`.

Update `_selective_to` to preserve `int8` parameter dtypes (same pattern as FP8).

**Files:** `src/infer/loader/weight_map.py`, `src/infer/loader/model_loader.py`

### D11: EngineConfig and CLI update for INT8

Add `"int8"` to `_VALID_QUANTIZATIONS`. Update `--quantization` CLI choices to include `"int8"`.

**Files:** `src/infer/engine/config.py`, `src/infer/server/__main__.py`

### D12: INT8 tests

- Unit test for `int8_channel_dequant`: create known bf16 weight, quantize to INT8 + scales, dequantize, verify close to original.
- Unit test for `INT8Linear`: verify forward pass produces same output as `nn.Linear` with dequantized weights.
- Unit test for `replace_linear_with_int8`: verify module surgery.
- Unit test for weight map with INT8 quantization.
- Unit test for `_detect_quantization` with compressed-tensors config.

**Files:** `tests/unit/test_int8_linear.py`, `tests/unit/test_weight_map_fp8.py` (extend existing)

---

## Implementation Details

### Block-wise dequantization (vectorized)

```python
def fp8_block_dequant(
    weight: Tensor,      # [out, in], float8_e4m3fn
    scale: Tensor,       # [ceil(out/B), ceil(in/B)], float32
    block_size: int = 128,
) -> Tensor:
    """Dequantize block-wise FP8 weight to bf16.

    Casts fp8 → float32 for the multiply (precision), then → bf16.
    """
    out_features, in_features = weight.shape
    # Pad to multiple of block_size if needed
    pad_out = (block_size - out_features % block_size) % block_size
    pad_in = (block_size - in_features % block_size) % block_size
    if pad_out > 0 or pad_in > 0:
        weight = F.pad(weight.to(torch.float32), (0, pad_in, 0, pad_out))
    else:
        weight = weight.to(torch.float32)

    # Reshape to [out/B, B, in/B, B], multiply by scale [out/B, 1, in/B, 1]
    out_blocks = weight.shape[0] // block_size
    in_blocks = weight.shape[1] // block_size
    weight = weight.reshape(out_blocks, block_size, in_blocks, block_size)
    weight = weight * scale[:, None, :, None]

    # Reshape back and trim padding
    weight = weight.reshape(out_blocks * block_size, in_blocks * block_size)
    weight = weight[:out_features, :in_features]
    return weight.to(torch.bfloat16)
```

### Per-channel INT8 dequantization

```python
def int8_channel_dequant(
    weight: Tensor,      # [out, in], int8
    scale: Tensor,       # [out, 1], float32
) -> Tensor:
    """Dequantize per-channel symmetric INT8 weight to bf16."""
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)
```

This is simpler than FP8 block-wise dequant because:
- No padding or reshaping needed (per-channel, not per-block).
- Scale broadcasts naturally along the input dimension.
- Multiply in float32 for precision (consistent with FP8 path), then cast to bf16.

Note: The checkpoint name `w8a8` is the calibration scheme (SmoothQuant + GPTQ with 8-bit weights and 8-bit activations). We serve this as W8A16 — weights are int8, activations are bf16. Dynamic activation quantization is not implemented; we dequantize weights to bf16 and compute in bf16.

### Weight loading without dtype conversion

When loading FP8 checkpoints, pass `dtype=None` to `load_weights`. This skips the `_apply_dtype` step entirely. Tensors arrive in their checkpoint dtype: FP8 weights as `float8_e4m3fn`, scales as `float32`, non-quantized tensors as `bf16`. After `load_state_dict`, the selective dtype/device move handles placing everything correctly.

### Selective dtype/device move

After `load_state_dict`, instead of `model.to(device=device, dtype=dtype)`:

```python
for param in model.parameters():
    if param.dtype != torch.float8_e4m3fn:
        param.data = param.data.to(device=device, dtype=dtype)
    else:
        param.data = param.data.to(device=device)
for buf in model.buffers():
    buf.data = buf.data.to(device=device)
    # Scale buffers stay float32; RoPE/norm buffers get target dtype
```

### Layers NOT quantized

These layers remain in bf16 (full precision):
- `embed_tokens` (embedding lookup, not a matmul)
- `lm_head` (final projection to vocab — quantizing this degrades quality significantly; may be tied to `embed_tokens`)
- `input_layernorm`, `post_attention_layernorm` (RMSNorm, tiny parameters)
- `q_norm`, `k_norm` (QK-norm, tiny parameters)

### Model dimensions (Qwen3-8B)

| Layer | Weight shape | Scale shape | FP8 bytes | bf16 bytes | Savings |
|---|---|---|---|---|---|
| `q_proj` | [4096, 4096] | [32, 32] | 16 MB | 32 MB | 2x |
| `k_proj` | [1024, 4096] | [8, 32] | 4 MB | 8 MB | 2x |
| `v_proj` | [1024, 4096] | [8, 32] | 4 MB | 8 MB | 2x |
| `o_proj` | [4096, 4096] | [32, 32] | 16 MB | 32 MB | 2x |
| `gate_proj` | [12288, 4096] | [96, 32] | 48 MB | 96 MB | 2x |
| `up_proj` | [12288, 4096] | [96, 32] | 48 MB | 96 MB | 2x |
| `down_proj` | [4096, 12288] | [32, 96] | 48 MB | 96 MB | 2x |

Per layer: 184 MB (FP8) vs 368 MB (bf16). With 36 layers: ~6.5 GB (FP8) vs ~13 GB (bf16).
Total with embeddings, norms, LM head: ~8 GB (FP8) vs ~16 GB (bf16).

### Model dimensions — INT8 (Qwen3-8B, nytopop/Qwen3-8B.w8a8)

| Layer | Weight shape | Scale shape | INT8 bytes | bf16 bytes | Savings |
|---|---|---|---|---|---|
| `q_proj` | [4096, 4096] | [4096, 1] | 16 MB + 8 KB | 32 MB | ~2x |
| `k_proj` | [1024, 4096] | [1024, 1] | 4 MB + 2 KB | 8 MB | ~2x |
| `v_proj` | [1024, 4096] | [1024, 1] | 4 MB + 2 KB | 8 MB | ~2x |
| `o_proj` | [4096, 4096] | [4096, 1] | 16 MB + 8 KB | 32 MB | ~2x |
| `gate_proj` | [12288, 4096] | [12288, 1] | 48 MB + 24 KB | 96 MB | ~2x |
| `up_proj` | [12288, 4096] | [12288, 1] | 48 MB + 24 KB | 96 MB | ~2x |
| `down_proj` | [4096, 12288] | [4096, 1] | 48 MB + 8 KB | 96 MB | ~2x |

Per layer: 184 MB (INT8) vs 368 MB (bf16) — same savings ratio as FP8. Scale tensor overhead is negligible (~76 KB per layer vs 184 MB weights).

---

## Risks and Mitigations

1. **Dequantization overhead.** Eager dequant materializes a full bf16 weight before matmul, temporarily using extra memory. Mitigation: only one layer's worth of dequantized weight exists at a time during the sequential forward pass (~96 MB for the largest projection). Under `torch.no_grad()`, the temporary is freed as soon as `F.linear` returns. Total peak overhead is well within the ~8 GB freed by quantization.

2. **Numerical precision.** FP8 E4M3 has limited dynamic range (max ~448, min ~1.95e-3). Block-wise quantization with 128x128 granularity mitigates this by adapting scale per block, but some quality degradation is expected vs bf16. The checkpoint publisher (Qwen) has already validated quality.

3. **`nn.Module.to()` destroying FP8 weights.** Any call to `model.to(dtype=...)` would convert FP8 parameters to bf16 (garbage values). Mitigation: `FP8Linear` overrides `_apply` to skip dtype conversion for `float8_e4m3fn` parameters. The model loader uses selective dtype/device move instead of `model.to(dtype=...)`.

4. **safetensors FP8 support.** The `safetensors` library must support loading `float8_e4m3fn` tensors. Versions 0.4+ support this. The project's `pyproject.toml` pins `safetensors>=0.7`, which is sufficient.

5. **Dequantization on every forward pass.** Every token generation step dequantizes all 252 linear projections (36 layers x 7 projections). For the target model, each dequantization involves reshaping and broadcasting a multi-MB tensor. This is compute-cheap relative to the matmul but adds up. If benchmarks show significant overhead, the fallback is to cache dequantized weights in bf16 (trading ~6.5 GB memory for zero dequant cost). This is a future optimization, not needed for the initial implementation.

---

## Exit Criteria

- `Qwen/Qwen3-8B-FP8` loads successfully and fits in 16 GB VRAM.
- `nytopop/Qwen3-8B.w8a8` loads successfully and fits in 16 GB VRAM.
- Generated text is coherent and reasonable on standard prompts for both models.
- All existing tests pass with `quantization=None` (no regression).
- Unit tests for FP8 dequantization, INT8 dequantization, and model surgery pass.
- VRAM usage logged: ~8 GB for weights vs ~16 GB for bf16.
