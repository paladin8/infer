# Phase 3.1: Triton Kernel Optimization

## Goal

Close the gap between current decode throughput and the hardware memory-bandwidth limit via fused Triton kernels. Current performance on RTX 5080:

| Model | Decode tok/s | Step latency | Theoretical min (bandwidth-limited) |
|-------|-------------|-------------|--------------------------------------|
| Llama 3.2 1B | ~148 | ~6.8ms | ~2.6ms (2.5 GB weights / 960 GB/s) |
| Qwen3 1.7B | ~74 | ~13.4ms | ~4.5ms (4.3 GB weights / 960 GB/s) |
| Gemma3 1B | ~60 | ~16.5ms | ~2.7ms (2.6 GB weights / 960 GB/s) |

The gap comes from kernel launch overhead (many small PyTorch ops per step), redundant HBM traffic (intermediate tensors), and CUDA dispatch costs. Fused Triton kernels target each of these.

Dev models: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.

---

## Profiling Methodology

### Script: `benchmarks/profile_generation.py`

Uses `torch.profiler.profile` with:
- `activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`
- `record_shapes=True`, `with_stack=True`
- Warmup phase (excluded from trace)

Outputs:
- Chrome trace JSON (`benchmarks/profiles/`) — viewable in `chrome://tracing` or Perfetto.
- Text summary table: top-N CUDA kernels by total GPU time, launch counts, memory ops.
- Separate prefill and decode phase breakdown.

Usage:
```bash
uv run python benchmarks/profile_generation.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --prompt-tokens 256 --decode-tokens 64
```

---

## Kernel Candidates

Ordered by expected impact. Each kernel replaces multiple PyTorch ops with a single Triton kernel launch.

### 1. Fused RMSNorm

**Current ops**: cast→pow→mean→rsqrt→mul→cast→mul (7+ CUDA kernel launches per call).

**Frequency**: 33 calls per Llama decode step (2 per layer × 16 layers + 1 final norm). Similar counts for Qwen3 (57) and Gemma3 (105 — 4 per-block norms × 26 layers + 1 final).

**Kernel**: Single Triton kernel that:
1. Loads the input row (last dimension) into SRAM.
2. Computes mean of squares in a single reduction.
3. Multiplies by rsqrt and weight in one pass.
4. Writes output.

Handles both standard RMSNorm (`weight * normed`) and Gemma3 variant (`(1 + weight) * normed`). Input/output dtype is preserved (bf16/fp16), computation is done in fp32 internally.

**File**: `src/infer/kernels/rms_norm.py`

### 2. Fused RoPE

**Current ops**: slice→cat→negate→mul→add (5+ kernel launches for `_rotate_half` + the elementwise ops).

**Frequency**: Applied to both Q and K at each layer. 16 calls per step (Llama), 28 (Qwen3), 26 (Gemma3).

**Kernel**: Single Triton kernel that:
1. Loads Q (or K) row and cos/sin entries.
2. Computes the rotation in-place: `out[i] = x[i]*cos[i] + x[paired]*sin[i]` where paired index handles the half-swap.
3. Writes output directly.

Processes Q and K in one fused call (two pointer sets).

**File**: `src/infer/kernels/rope.py`

### 3. Fused Residual + RMSNorm

**Pattern**: `x = residual + x; x = rmsnorm(x)` at every sub-layer boundary.

**Current cost**: Residual add writes result to HBM, then RMSNorm reads it back. Fusing saves one full HBM round-trip per sub-layer.

**Kernel**: Single Triton kernel that:
1. Loads `residual` and `x` rows.
2. Computes `combined = residual + x`.
3. Applies RMSNorm to `combined` (same as fused RMSNorm kernel logic).
4. Writes both `combined` (needed as the new residual for the next sub-layer) and `normed` output.

Supports both standard and Gemma3 `(1 + weight)` variants.

**File**: `src/infer/kernels/fused_norm_residual.py`

### 4. Fused SwiGLU/GeGLU Activation

**Pattern**: `act_fn(gate) * up` in the gated MLP. The linear projections (`gate_proj`, `up_proj`) are cuBLAS GEMMs and stay as-is.

**Current cost**: `silu(gate)` writes an intermediate tensor, then `* up` reads it back and writes the final result. Two HBM round-trips for the intermediate-size tensor.

**Kernel**: Single Triton kernel that:
1. Loads `gate` and `up` rows (both of intermediate_size).
2. Computes `act_fn(gate) * up` element-wise.
3. Writes output directly.

Supports SiLU (Llama, Qwen3) and GELU-tanh (Gemma3) activations selected via a compile-time constant.

**File**: `src/infer/kernels/activation.py`

### What NOT to Replace

**`F.scaled_dot_product_attention`** — PyTorch already dispatches to FlashAttention or memory-efficient backends. Only revisit if profiling shows it's a bottleneck. The attention kernels are already well-optimized and would require complex Triton implementations to match.

---

## Integration Pattern

### Module structure

```
src/infer/kernels/
├── __init__.py              # Package marker
├── rms_norm.py              # triton_rms_norm (standard + Gemma3 variant)
├── rope.py                  # triton_apply_rope
├── fused_norm_residual.py   # triton_fused_residual_rms_norm
└── activation.py            # triton_fused_gated_activation
```

Triton is a required dependency — there are no fallback paths or feature flags. This keeps the codebase simple and avoids maintaining two code paths for every operation.

### Integration points in `src/infer/models/common.py`

1. **`RMSNorm.forward`**: calls `triton_rms_norm` directly.
2. **`apply_rope`**: calls `triton_apply_rope` directly.
3. **`GatedMLP.forward`**: calls `triton_fused_gated_activation` directly.

### Integration in model forward (residual + norm fusion)

The `LlamaTransformerBlock.forward`, `Qwen3TransformerBlock.forward`, and `Gemma3TransformerBlock.forward` call `triton_fused_residual_rms_norm` directly. The pattern `residual + x; rmsnorm(...)` becomes a single kernel launch.

---

## Precision Notes

### FMA vs separate multiply-add in RoPE

The RoPE rotation `x * cos + rotate_half(x) * sin` compiles to FMA (fused multiply-add) instructions inside the Triton kernel. FMA uses a single rounding step for `a*b + c`, whereas HuggingFace's PyTorch implementation stores each multiply result to bf16 memory before adding (two rounding steps via separate CUDA kernel launches).

This produces a consistent per-element difference of ~0.016 max in bf16, independent of head dimension. Verified against fp64 ground truth, the Triton FMA result is **more precise**:

- ~9% of elements are closer to fp64 truth (vs ~5% for PyTorch)
- ~7% lower total absolute error
- ~85% of elements are identical

The per-element error is amplified by downstream RMSNorm layers. Architectures with more norms per block show larger block-level divergence despite identical per-element RoPE error:

| Architecture | Norms/block | Block max error | Block mean error |
|-------------|------------|----------------|-----------------|
| Llama 3 | 2 | ~0.016 | ~8e-5 |
| Qwen 3 | 2 + QK-norm | ~0.031 | ~9e-4 |
| Gemma 3 | 4 sandwich + QK-norm | ~2.0 | ~1.1e-2 |

### RMSNorm and fused residual+norm

These kernels upcast to f32 internally and match PyTorch's precision conventions exactly (standard: `weight.to(input_dtype) * normed.to(input_dtype)`; Gemma3: `(1 + weight_f32) * normed_f32`). No measurable precision difference vs HF.

---

## Testing Strategy

### Unit tests (`tests/unit/test_kernels.py`)

For each kernel:
1. Generate random inputs of varying shapes (small, medium, large).
2. Run both the PyTorch reference and the Triton kernel.
3. Assert `torch.allclose` within dtype tolerance.

**Shapes tested**:
- Small: `(1, 128)`, `(1, 1, 64)`
- Medium: `(4, 1024)`, `(2, 8, 256)`
- Large: `(1, 4096)`, `(1, 16, 2048)`

**Dtypes tested**: `bfloat16`, `float16`, `float32`.

**Tolerance**: `atol=1e-2, rtol=1e-2` for bf16/fp16 (limited precision), `atol=1e-5, rtol=1e-5` for fp32.

Tests are skipped when CUDA is not available (module-level `pytest.skip`).

### Model-level regression

All existing tests (`tests/unit/`, `tests/integration/`) must pass. The integration parity tests compare single-layer output against HuggingFace with model-specific tolerances that account for FMA precision differences (see Precision Notes above).

### Sanity check script

`scripts/sanity_check.py` runs each model on a simple prompt and prints the output, timing, and throughput. Use for quick manual verification after changes:

```bash
uv run python scripts/sanity_check.py
uv run python scripts/sanity_check.py --models llama qwen3
uv run python scripts/sanity_check.py --prompt "Explain gravity in one sentence."
```

---

## File Layout

```
docs/
├── OVERALL_DESIGN.md           # MODIFIED: Phase 3.1 section, design decisions
└── PHASE_3_1.md                # This design doc

benchmarks/
├── profile_generation.py       # Profiling script
└── profiles/                   # Chrome trace output

src/infer/kernels/
├── __init__.py                 # Package marker
├── rms_norm.py                 # Fused RMSNorm kernel
├── rope.py                     # Fused RoPE kernel
├── fused_norm_residual.py      # Fused residual + RMSNorm kernel
└── activation.py               # Fused SwiGLU/GeGLU activation kernel

src/infer/models/
├── common.py                   # Direct Triton kernel calls in RMSNorm, apply_rope, GatedMLP
├── llama.py                    # Fused residual+norm in block forward
├── qwen3.py                    # Fused residual+norm in block forward
└── gemma3.py                   # Fused residual+norm in block forward

scripts/
└── sanity_check.py             # Quick generation sanity check for all models

tests/unit/
└── test_kernels.py             # Kernel correctness tests (Triton vs PyTorch reference)

benchmarks/log/
└── GENERATION_LOG.md           # Benchmark results
```

---

## Exit Criteria

1. All 4 Triton kernels implemented and unit-tested.
2. Measurable decode throughput improvement on all 3 dev models.
3. All existing tests pass (unit, integration, parity).
4. Parity test tolerances documented with root-cause analysis (FMA precision).
5. Profiling script produces valid Chrome traces and summary tables.
6. Sanity check script (`scripts/sanity_check.py`) runs all 3 models successfully.
7. Benchmark log updated with Triton-enabled results.
8. `uv run ruff check .` and `uv run mypy .` pass cleanly.
