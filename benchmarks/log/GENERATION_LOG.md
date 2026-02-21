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
