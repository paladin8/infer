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

---

## Phase 5 — Continuous Batching

Server launched with `--batching-mode continuous`. Same hardware, models, seeds, and workload definitions as Phase 4. Multiple new requests arriving in the same step are batched into a single prefill forward pass (right-padded to the longest prompt) to amortize weight loading.

### baseline — Sequential Single Requests

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |   87.1 |   65.8 |   86.3 |
| TTFT P50 (ms)      |     35 |     43 |     25 |
| TTFT P99 (ms)      |     37 |     44 |     27 |
| ITL P50 (ms)       |     11 |     15 |     11 |
| ITL P99 (ms)       |     13 |     17 |     16 |
| Latency P50 (s)    |  2.939 |  3.875 |  2.966 |

### continuous_batching — Staggered Arrivals

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  316.9 |  241.9 |  249.8 |
| TTFT P50 (ms)      |   1825 |   3378 |   2965 |
| TTFT P99 (ms)      |   4352 |   7918 |   7334 |
| ITL P50 (ms)       |     21 |     28 |     28 |
| ITL P99 (ms)       |     58 |     71 |     40 |
| Latency P50 (s)    |  5.149 |  7.659 |  7.180 |
| Errors             |      0 |      0 |      0 |

### paged_attention — Bursty Arrivals, Bimodal Lengths

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  281.0 |  197.5 |  258.4 |
| TTFT P50 (ms)      |  17341 |  28951 |  19037 |
| TTFT P99 (ms)      |  40552 |  67679 |  50193 |
| ITL P50 (ms)       |     26 |     38 |     29 |
| ITL P99 (ms)       |     64 |     68 |     44 |
| Latency P50 (s)    | 25.657 | 41.632 | 31.483 |
| Errors             |      0 |      0 |      0 |

### chunked_prefill — Long Prompts, Poisson Arrivals

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  197.4 |  137.3 |  240.4 |
| TTFT P50 (ms)      |  13156 |  20259 |   9931 |
| TTFT P99 (ms)      |  25153 |  39454 |  18861 |
| ITL P50 (ms)       |     36 |     51 |     29 |
| ITL P99 (ms)       |    140 |    191 |     64 |
| Latency P50 (s)    | 17.881 | 27.012 | 14.063 |
| Errors             |      0 |      0 |      0 |

### prefix_caching — Shared System Prompt

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  256.4 |  104.2 |  258.0 |
| TTFT P50 (ms)      |  17139 |  45287 |  16896 |
| TTFT P99 (ms)      |  34569 |  90710 |  34148 |
| ITL P50 (ms)       |     29 |     67 |     29 |
| ITL P99 (ms)       |     95 |    191 |     51 |
| Latency P50 (s)    | 23.779 | 64.245 | 24.676 |
| Errors             |      0 |      0 |      0 |

### Phase 5 vs Phase 4 — Comparison

The continuous_batching workload is the headline comparison. Changes relative to Phase 4 static batching:

| Metric             |   Llama |   Qwen3 |  Gemma3 |
|--------------------|--------:|--------:|--------:|
| Throughput         |   +25%  |   +26%  |    +8%  |
| TTFT P50           |   -55%  |   -45%  |   -30%  |
| TTFT P99           |   -55%  |   -44%  |   -33%  |
| ITL P50            |   +24%  |   +22%  |   +22%  |
| ITL P99            |  +205%  |  +163%  |   +38%  |

**What improved**: TTFT dropped 30-55% across all models — requests no longer wait for an entire batch to drain before being admitted. Throughput improved 8-26% because the batch stays fuller (no wasted slots for already-finished requests). Batched prefill amortizes weight loading when multiple requests arrive in the same step.

**What got worse**: ITL P99 increased 38-205% due to prefill stalls — when new requests are admitted, their prefill blocks decode for all active requests. Larger models (Llama 3B, Qwen3 4B) suffer more because their prefills take longer. This is the problem Phase 7 (chunked prefill) is designed to solve.

**chunked_prefill workload**: Qwen3-4B improved from 96.5 (static) to 137.3 tok/s (+42% throughput), but ITL P99 increased from 78ms to 191ms. Long-prompt prefills still block decode even when batched — the stall is shorter (one batched pass vs N individual passes) but still present.

**prefix_caching regression on Qwen3**: throughput dropped from 212.9 to 104.2 tok/s (-51%). At 8 RPS with ~1024-token prompts, batched prefills of the arriving requests are large (multiple 1024-token sequences padded together), creating long decode stalls. The problem is amplified on Qwen3-4B due to the model's higher per-token compute cost. Llama and Gemma3 showed mild regression (-15% and -9% respectively).
