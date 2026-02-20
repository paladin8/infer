# Generation Benchmark Log

E2E throughput (tok/s, median of 3 trials). Higher is better.

**Hardware**: NVIDIA GeForce RTX 5080 (16 GB VRAM, 84 SMs, compute 12.0, CUDA 12.8)

**Configs** (all bfloat16, seed=42, 2 warmup runs):

| Key       | Prompt |  Gen | Sampling                            | Purpose                         |
|-----------|-------:|-----:|-------------------------------------|---------------------------------|
| short     |     64 |   64 | greedy                              | Quick Q&A, baseline latency     |
| medium    |    256 |  256 | greedy                              | Standard chat (headline number) |
| prefill   |   1024 |   64 | greedy                              | Isolates prefill performance    |
| decode    |     64 |  512 | greedy                              | Isolates decode performance     |
| long      |   1024 |  512 | greedy                              | Stress test, longest sequence   |
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
