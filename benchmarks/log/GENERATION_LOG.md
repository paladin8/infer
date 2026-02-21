# Generation Benchmark Log

E2E throughput (tok/s, median of 3 trials). Higher is better.

**Hardware**: NVIDIA GeForce RTX 5080 (16 GB VRAM, 84 SMs, compute 12.0, CUDA 12.8)

**Configs** (all bfloat16, seed=42, 2 warmup runs):

| Key       | Prompt |  Gen | Sampling                            | Purpose                         |
|-----------|-------:|-----:|-------------------------------------|---------------------------------|
| short     |     64 |   64 | greedy                              | Quick Q&A, baseline latency     |
| medium    |    256 |  256 | greedy                              | Standard chat (headline number) |
| prefill   |   1024 |   64 | greedy                              | Isolates prefill performance    |
| decode    |     64 | 1024 | greedy                              | Isolates decode performance     |
| long      |   1024 | 1024 | greedy                              | Stress test, longest sequence   |
| sampled   |    256 |  256 | temp=0.8 top_k=50 top_p=0.9         | Sampling transform overhead     |
| full-pipe |    256 |  256 | temp=0.8 top_k=50 top_p=0.9 rep=1.1 | All transforms active           |

**Suite tiers** (`--suite <tier>`):
- **quick**: short, medium (fast iteration)
- **standard**: quick + prefill, decode, sampled
- **full**: all 7 configs (milestone comparisons)

Model loads once per suite. `torch.cuda.empty_cache()` runs between configs to prevent memory accumulation.

**Dev models**:
- `meta-llama/Llama-3.2-1B-Instruct` (Llama)
- `Qwen/Qwen3-1.7B` (Qwen3)
- `google/gemma-3-1b-it` (Gemma3)

**How to run**:

```bash
# Run the full suite for all three dev models:
uv run python benchmarks/bench_generation.py --suite full --model meta-llama/Llama-3.2-1B-Instruct
uv run python benchmarks/bench_generation.py --suite full --model Qwen/Qwen3-1.7B
uv run python benchmarks/bench_generation.py --suite full --model google/gemma-3-1b-it

# Quick suite (short + medium only) for fast iteration:
uv run python benchmarks/bench_generation.py --suite quick --model meta-llama/Llama-3.2-1B-Instruct

# Ad-hoc single run with custom parameters:
uv run python benchmarks/bench_generation.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --prompt-tokens 256 --max-new-tokens 256 --temperature 0.8 --seed 42
```

Defaults: bfloat16, 2 warmup runs, 3 timed trials, seed=42. JSON reports saved to `benchmarks/reports/` (suppress with `--no-save`).

**Recording results**: After running the full suite, copy the median E2E throughput (tok/s) from the suite summary table into the appropriate row below.

---

## 2026-02-20 — KV cache implementation (with `--kv-cache`, the default)

Per-step decode latency is now approximately constant across sequence lengths, confirming KV cache eliminates the O(n) per-step recomputation.

### E2E throughput (tok/s)

| Config    | Llama | Qwen3 | Gemma3 |
|-----------|------:|------:|-------:|
| short     | 148.4 |  75.0 |   57.5 |
| medium    | 144.7 |  71.3 |   61.8 |
| prefill   | 138.4 |  70.6 |   54.9 |
| decode    | 143.6 |  73.6 |   62.0 |
| long      | 144.4 |  73.7 |   54.8 |
| sampled   | 151.1 |  75.9 |   60.5 |
| full-pipe | 147.8 |  75.9 |   60.3 |

### Decode throughput (tok/s)

| Config    | Llama | Qwen3 | Gemma3 |
|-----------|------:|------:|-------:|
| short     | 151.4 |  76.4 |   58.6 |
| medium    | 145.6 |  71.8 |   62.1 |
| prefill   | 148.5 |  75.1 |   56.8 |
| decode    | 143.7 |  73.7 |   63.2 |
| long      | 145.0 |  73.9 |   54.9 |
| sampled   | 152.1 |  76.4 |   60.8 |
| full-pipe | 149.1 |  76.3 |   61.0 |

### Per-step decode latency (ms)

| Config    | Llama P50 | Llama P95 | Qwen3 P50 | Qwen3 P95 | Gemma3 P50 | Gemma3 P95 |
|-----------|----------:|----------:|----------:|----------:|-----------:|-----------:|
| short     |       6.6 |       7.6 |      13.2 |      15.0 |       16.9 |       20.1 |
| medium    |       6.8 |       8.2 |      13.7 |      16.6 |       16.1 |       17.5 |
| prefill   |       6.8 |       7.3 |      13.5 |      14.4 |       17.6 |       20.3 |
| decode    |       6.8 |       7.9 |      13.4 |      14.7 |       16.1 |       18.6 |
| long      |       6.9 |       7.4 |      13.6 |      15.4 |       18.1 |       20.3 |
| sampled   |       6.6 |       7.0 |      13.0 |      14.4 |       16.2 |       18.4 |
| full-pipe |       6.7 |       7.3 |      13.0 |      14.2 |       16.2 |       17.8 |

### TTFT / prefill (ms)

| Config    | Llama | Qwen3 | Gemma3 |
|-----------|------:|------:|-------:|
| short     |   8.0 |  15.3 |   18.2 |
| medium    |  11.8 |  21.1 |   20.1 |
| prefill   |  31.5 |  54.2 |   34.7 |
| long      |  31.1 |  53.3 |   35.3 |

### KV cache memory (MB) and peak GPU allocation (GB)

| Config    | Llama cache | Llama peak | Qwen3 cache | Qwen3 peak | Gemma3 cache | Gemma3 peak |
|-----------|------------:|-----------:|------------:|-----------:|-------------:|------------:|
| short     |         4.2 |        2.8 |        14.7 |        3.8 |          3.4 |         2.6 |
| medium    |        16.8 |        2.9 |        58.7 |        3.9 |         13.6 |         2.6 |
| prefill   |        35.7 |        2.9 |       124.8 |        4.0 |         29.0 |         2.6 |
| decode    |        35.7 |        2.9 |       124.8 |        3.9 |         29.0 |         2.6 |
| long      |        67.1 |        3.0 |       234.9 |        4.1 |         54.5 |         2.6 |

### Notes

- **Llama 3.2 1B**: fastest decode (~148 tok/s, ~6.8ms/step). Largest KV cache per token (8 KV heads, 128 head_dim) but fewest layers (16).
- **Qwen3 1.7B**: ~74 tok/s decode, ~13.4ms/step. Largest model (28 layers) but fewest KV heads (2), so cache is moderate.
- **Gemma3 1B**: slowest decode (~60 tok/s, ~16.5ms/step) despite being the smallest model. 26 layers of sliding-window + global attention alternation adds overhead. Hits EOS early on short/decode configs (55 tokens generated instead of 64/1024).
- Step latency is near-constant across all configs (short vs long), confirming KV cache works correctly — no O(n) per-step growth.
- Sampling transforms (top-k, top-p, repetition penalty) add negligible overhead.
- Long-context (1024+1024) cache sizes: Llama 67 MB, Qwen3 235 MB, Gemma3 55 MB — all fit comfortably alongside model weights.

---

## 2026-02-21 — Triton fused kernels (Phase 3.1)

Fused Triton kernels for RMSNorm, RoPE, residual+RMSNorm, and SwiGLU/GeGLU activation. Also replaced `repeat_interleave` with zero-copy `expand` for GQA head expansion.

Kernels: `src/infer/kernels/{rms_norm,rope,fused_norm_residual,activation}.py`.
Triton kernels are always used (required dependency, no fallback paths).

### E2E throughput (tok/s)

| Config    | Llama | Qwen3 | Gemma3 |
|-----------|------:|------:|-------:|
| short     | 215.1 | 133.2 |  120.0 |
| medium    | 213.1 | 133.8 |  121.4 |
| prefill   | 194.2 | 120.5 |  118.9 |
| decode    | 214.9 | 134.1 |  128.5 |
| long      | 208.1 | 131.7 |  116.1 |
| sampled   | 214.4 | 133.9 |  127.0 |
| full-pipe | 214.2 | 135.3 |  126.0 |

### Decode throughput (tok/s)

| Config    | Llama | Qwen3 | Gemma3 |
|-----------|------:|------:|-------:|
| short     | 219.0 | 135.5 |  122.2 |
| medium    | 214.8 | 134.9 |  122.3 |
| prefill   | 213.4 | 132.3 |  124.1 |
| decode    | 215.1 | 134.2 |  130.9 |
| long      | 209.3 | 132.5 |  116.4 |
| sampled   | 216.0 | 135.0 |  127.7 |
| full-pipe | 216.3 | 136.4 |  127.6 |

### Per-step decode latency (ms)

| Config    | Llama P50 | Llama P95 | Qwen3 P50 | Qwen3 P95 | Gemma3 P50 | Gemma3 P95 |
|-----------|----------:|----------:|----------:|----------:|-----------:|-----------:|
| short     |       4.6 |       5.6 |       7.4 |       8.7 |        8.2 |        9.8 |
| medium    |       4.6 |       5.2 |       7.3 |       8.4 |        8.0 |        9.3 |
| prefill   |       4.7 |       5.2 |       7.5 |       8.6 |        8.0 |        9.3 |
| decode    |       4.6 |       5.1 |       7.3 |       8.3 |        7.6 |        8.6 |
| long      |       4.7 |       5.2 |       7.5 |       8.3 |        8.6 |        9.5 |
| sampled   |       4.6 |       5.0 |       7.3 |       8.1 |        7.8 |        8.9 |
| full-pipe |       4.6 |       5.1 |       7.3 |       8.0 |        7.8 |        8.9 |

### TTFT / prefill (ms)

| Config    | Llama | Qwen3 | Gemma3 |
|-----------|------:|------:|-------:|
| short     |   5.4 |   8.6 |    8.8 |
| medium    |   8.5 |  14.8 |    9.4 |
| prefill   |  29.6 |  47.1 |   22.8 |
| long      |  28.8 |  45.8 |   22.8 |

### KV cache memory (MB) and peak GPU allocation (GB)

| Config    | Llama cache | Llama peak | Qwen3 cache | Qwen3 peak | Gemma3 cache | Gemma3 peak |
|-----------|------------:|-----------:|------------:|-----------:|-------------:|------------:|
| short     |         4.2 |        2.8 |        14.7 |        3.8 |          3.4 |         2.6 |
| medium    |        16.8 |        2.9 |        58.7 |        3.9 |         13.6 |         2.6 |
| prefill   |        35.7 |        2.9 |       124.8 |        4.0 |         29.0 |         2.6 |
| decode    |        35.7 |        2.9 |       124.8 |        3.9 |         29.0 |         2.6 |
| long      |        67.1 |        3.0 |       234.9 |        4.1 |         54.5 |         2.6 |

### Speedup vs Phase 3 (KV cache baseline)

| Model   | Phase 3 decode | Phase 3.1 decode | Speedup | Step latency reduction |
|---------|---------------:|-----------------:|--------:|-----------------------:|
| Llama   |     148 tok/s  |       215 tok/s  |   1.45x |   6.8ms → 4.6ms (32%)  |
| Qwen3   |      74 tok/s  |       135 tok/s  |   1.82x |  13.4ms → 7.3ms (46%)  |
| Gemma3  |      60 tok/s  |       122 tok/s  |   2.03x |  16.5ms → 8.0ms (52%)  |

### Notes

- **Llama 3.2 1B**: 45% faster (148→215 tok/s). Already memory-bandwidth-bound at 6.8ms/step, so gains come primarily from reducing kernel launch overhead and intermediate memory traffic.
- **Qwen3 1.7B**: 82% faster (74→135 tok/s). Biggest absolute gain from fused kernels — 28 layers means 28× the savings from fused residual+norm and fused activation per step.
- **Gemma3 1B**: 103% faster (60→122 tok/s). Largest relative gain. The 26-layer sliding-window architecture had the most kernel launch overhead to eliminate. GQA expand (replacing `repeat_interleave` with zero-copy `expand`) also helps since Gemma3 has only 1 KV head (4× expansion).
- **TTFT improved** across all models (Gemma3 prefill: 35ms→23ms, Llama: 31ms→29ms) from fused kernels reducing overhead during the prefill forward pass.
- **Memory unchanged** — fused kernels reduce intermediate tensor allocations but cache sizes and peak allocation stay the same since model weights dominate.
- Kernels: fused RMSNorm (standard + Gemma style), fused RoPE (single kernel for Q+K with separate input/output strides), fused residual+RMSNorm, fused SwiGLU/GeGLU activation.
