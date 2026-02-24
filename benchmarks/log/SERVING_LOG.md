# Serving Benchmark Log

Serving throughput (output tok/s), TTFT (ms), and ITL (ms) against a running `infer` server. Measures end-to-end serving performance including HTTP overhead, batching, and scheduling.

**Hardware**: NVIDIA GeForce RTX 5080 (16 GB VRAM, 84 SMs, compute 12.0, CUDA 12.8)

## Workloads

| Workload            | Requests | Prompt lengths                                  | Max tokens | Arrival pattern                  | Measures                                              |
|---------------------|----------|-------------------------------------------------|------------|----------------------------------|-------------------------------------------------------|
| baseline            |       10 | ~256 tok                                        | 256        | Sequential (wait for completion) | Single-request overhead floor                         |
| continuous_batching |       32 | ~[64, 512] tok                                  | [64, 256]  | Uniform 4 RPS                    | TTFT P95/P99 (benefit of per-step admit/retire)       |
| paged_attention     |       48 | Bimodal: ~50% [32,128], ~50% [512,1024] tok     | [128, 512] | Bursts of 8, 500ms apart         | Throughput + errors (memory waste from contiguous KV) |
| chunked_prefill     |       48 | ~75% [1024,2048], ~25% [64,128] tok             | [64, 256]  | Poisson 6 RPS                    | ITL P95/P99 (long prefills block decode)              |
| prefix_caching      |       48 | ~1024 tok shared prefix + ~[32,128] tok suffix  | 256        | Uniform 8 RPS                    | TTFT P50/P95 (repeated prefill cost)                  |

Prompts are realistic English text (assembled from a fixed corpus). Token counts are approximate (~4 chars/token); actual counts come from the server.

## Benchmark models

- `meta-llama/Llama-3.2-3B-Instruct` (Llama)
- `Qwen/Qwen3-4B` (Qwen3)
- `google/gemma-3-1b-it` (Gemma3) — text-only variant; 4B is multimodal and unsupported

## How to run

Start the server for each model, then run the benchmark against it:

```bash
# Start the server (in a separate terminal):
uv run python -m infer.server --model meta-llama/Llama-3.2-3B-Instruct --max-batch-size 8

# Run all workloads:
uv run python benchmarks/bench_serving.py \
    --server http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --workload all --seed 42

# Run a single workload:
uv run python benchmarks/bench_serving.py \
    --server http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --workload baseline --seed 42
```

Repeat for each model:

```bash
# Qwen3-4B
uv run python -m infer.server --model Qwen/Qwen3-4B --max-batch-size 8
uv run python benchmarks/bench_serving.py --server http://localhost:8000 --model Qwen/Qwen3-4B --workload all --seed 42

# Gemma3-1B (text-only)
uv run python -m infer.server --model google/gemma-3-1b-it --max-batch-size 8
uv run python benchmarks/bench_serving.py --server http://localhost:8000 --model google/gemma-3-1b-it --workload all --seed 42
```

Defaults: seed=42, 2 warmup requests per workload, JSON reports saved to `benchmarks/reports/` (suppress with `--no-save`).

**Recording results**: After running all workloads for a model, copy the values from the cross-workload comparison table and per-workload detail tables into the appropriate cells below. Round throughput to 1 decimal, TTFT/ITL to integer ms.

---

## Phase 4 — Static Batching

### baseline — Sequential Single Requests

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |   88.5 |   67.4 |   91.9 |
| TTFT P50 (ms)      |     86 |     93 |     75 |
| TTFT P99 (ms)      |     88 |     96 |     77 |
| ITL P50 (ms)       |     11 |     14 |     11 |
| ITL P99 (ms)       |     13 |     18 |     14 |
| Latency P50 (s)    |  2.890 |  3.767 |  2.748 |

### continuous_batching — Staggered Arrivals

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  254.0 |  192.4 |  231.8 |
| TTFT P50 (ms)      |   4037 |   6123 |   4242 |
| TTFT P99 (ms)      |   9678 |  14226 |  10960 |
| ITL P50 (ms)       |     17 |     23 |     23 |
| ITL P99 (ms)       |     19 |     27 |     29 |
| Latency P50 (s)    |  6.491 |  9.422 |  7.141 |
| Errors             |      0 |      0 |      0 |

### paged_attention — Bursty Arrivals, Bimodal Lengths

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  217.3 |  154.5 |  254.1 |
| TTFT P50 (ms)      |  28121 |  43266 |  24010 |
| TTFT P99 (ms)      |  59125 |  94320 |  52792 |
| ITL P50 (ms)       |     21 |     30 |     24 |
| ITL P99 (ms)       |     24 |     69 |     28 |
| Latency P50 (s)    | 33.051 | 53.743 | 30.207 |
| Errors             |      0 |      0 |      0 |

### chunked_prefill — Long Prompts, Poisson Arrivals

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  150.5 |   96.5 |  209.2 |
| TTFT P50 (ms)      |  16810 |  27334 |  12262 |
| TTFT P99 (ms)      |  35625 |  58949 |  23125 |
| ITL P50 (ms)       |     29 |     43 |     24 |
| ITL P99 (ms)       |     36 |     78 |     29 |
| Latency P50 (s)    | 23.564 | 37.883 | 16.769 |
| Errors             |      0 |      0 |      0 |

### prefix_caching — Shared System Prompt

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  300.0 |  212.9 |  283.7 |
| TTFT P50 (ms)      |  14419 |  20665 |  14418 |
| TTFT P99 (ms)      |  30405 |  44815 |  31665 |
| ITL P50 (ms)       |     23 |     32 |     25 |
| ITL P99 (ms)       |     26 |     40 |     33 |
| Latency P50 (s)    | 20.704 | 29.543 | 21.178 |
| Errors             |      0 |      0 |      0 |
