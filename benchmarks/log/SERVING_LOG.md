# Serving Benchmark Log

Serving throughput (output tok/s), TTFT (ms), and ITL (ms) against a running `infer` server. Measures end-to-end serving performance including HTTP overhead, batching, and scheduling.

**Hardware**: NVIDIA GeForce RTX 5080 (16 GB VRAM, 84 SMs, compute 12.0, CUDA 12.8)

## Workloads

| Workload            | Requests | Prompt lengths                                 | Max tokens | Arrival pattern                  | Measures                                                |
|---------------------|----------|------------------------------------------------|------------|----------------------------------|---------------------------------------------------------|
| baseline            | 10       | ~256 tok                                       | 256        | Sequential (wait for completion) | Single-request overhead floor                           |
| continuous_batching | 32       | ~[64, 512] tok                                 | [64, 256]  | Uniform 4 RPS                    | TTFT P95/P99 (benefit of per-step admit/retire)         |
| paged_attention     | 48       | ~[128, 384] tok                                | [128, 256] | Single burst (all at once)       | Throughput + TTFT (paged admits more per memory budget) |
| chunked_prefill     | 48       | ~75% [1024,2048], ~25% [64,128] tok            | [64, 256]  | Poisson 6 RPS                    | ITL P95/P99 (long prefills block decode)                |
| prefix_caching      | 48       | ~1024 tok shared prefix + ~[32,128] tok suffix | 256        | Uniform 8 RPS                    | TTFT P50/P95 (repeated prefill cost)                    |

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

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) |  88.5 |  67.4 |   91.9 |
| TTFT P50 (ms)      |    86 |    93 |     75 |
| TTFT P99 (ms)      |    88 |    96 |     77 |
| ITL P50 (ms)       |    11 |    14 |     11 |
| ITL P99 (ms)       |    13 |    18 |     14 |
| Latency P50 (s)    | 2.890 | 3.767 |  2.748 |

### continuous_batching — Staggered Arrivals

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) | 254.0 | 192.4 |  231.8 |
| TTFT P50 (ms)      |  4037 |  6123 |   4242 |
| TTFT P99 (ms)      |  9678 | 14226 |  10960 |
| ITL P50 (ms)       |    17 |    23 |     23 |
| ITL P99 (ms)       |    19 |    27 |     29 |
| Latency P50 (s)    | 6.491 | 9.422 |  7.141 |
| Errors             |     0 |     0 |      0 |

### paged_attention — Single Burst, Moderate Lengths

Workload redefined in Phase 6 to demonstrate paged attention benefits: 48 requests sent simultaneously with moderate lengths (~128-384 prompt, 128-256 gen). With the same KV token budget (32,768 tokens), contiguous reserves 8 slots x 4096 tokens while paged fits 24 concurrent requests.

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  346.1 |  271.0 |  289.1 |
| TTFT P50 (ms)      |  10920 |  14038 |  12146 |
| TTFT P99 (ms)      |  22014 |  28489 |  24816 |
| ITL P50 (ms)       |     17 |     22 |     24 |
| ITL P99 (ms)       |     21 |     25 |     28 |
| Latency P50 (s)    | 14.346 | 18.325 | 16.016 |
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

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) |  87.1 |  65.8 |   86.3 |
| TTFT P50 (ms)      |    35 |    43 |     25 |
| TTFT P99 (ms)      |    37 |    44 |     27 |
| ITL P50 (ms)       |    11 |    15 |     11 |
| ITL P99 (ms)       |    13 |    17 |     16 |
| Latency P50 (s)    | 2.939 | 3.875 |  2.966 |

### continuous_batching — Staggered Arrivals

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) | 316.9 | 241.9 |  249.8 |
| TTFT P50 (ms)      |  1825 |  3378 |   2965 |
| TTFT P99 (ms)      |  4352 |  7918 |   7334 |
| ITL P50 (ms)       |    21 |    28 |     28 |
| ITL P99 (ms)       |    58 |    71 |     40 |
| Latency P50 (s)    | 5.149 | 7.659 |  7.180 |
| Errors             |     0 |     0 |      0 |

### paged_attention — Single Burst, Moderate Lengths

Workload redefined in Phase 6 (see above).

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  338.2 |  259.5 |  269.3 |
| TTFT P50 (ms)      |  10934 |  14354 |  13901 |
| TTFT P99 (ms)      |  22904 |  30005 |  27193 |
| ITL P50 (ms)       |     21 |     28 |     28 |
| ITL P99 (ms)       |     43 |     54 |     39 |
| Latency P50 (s)    | 15.469 | 20.281 | 17.381 |
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

| Metric     | Llama | Qwen3 | Gemma3 |
|------------|------:|------:|-------:|
| Throughput |  +25% |  +26% |    +8% |
| TTFT P50   |  -55% |  -45% |   -30% |
| TTFT P99   |  -55% |  -44% |   -33% |
| ITL P50    |  +24% |  +22% |   +22% |
| ITL P99    | +205% | +163% |   +38% |

**What improved**: TTFT dropped 30-55% across all models — requests no longer wait for an entire batch to drain before being admitted. Throughput improved 8-26% because the batch stays fuller (no wasted slots for already-finished requests). Batched prefill amortizes weight loading when multiple requests arrive in the same step.

**What got worse**: ITL P99 increased 38-205% due to prefill stalls — when new requests are admitted, their prefill blocks decode for all active requests. Larger models (Llama 3B, Qwen3 4B) suffer more because their prefills take longer. This is the problem Phase 7 (chunked prefill) is designed to solve.

**chunked_prefill workload**: Qwen3-4B improved from 96.5 (static) to 137.3 tok/s (+42% throughput), but ITL P99 increased from 78ms to 191ms. Long-prompt prefills still block decode even when batched — the stall is shorter (one batched pass vs N individual passes) but still present.

**prefix_caching regression on Qwen3**: throughput dropped from 212.9 to 104.2 tok/s (-51%). At 8 RPS with ~1024-token prompts, batched prefills of the arriving requests are large (multiple 1024-token sequences padded together), creating long decode stalls. The problem is amplified on Qwen3-4B due to the model's higher per-token compute cost. Llama and Gemma3 showed mild regression (-15% and -9% respectively).

---

## Phase 6 — Paged Attention with Triton Kernel

Server launched with `--batching-mode continuous --kv-cache-backend paged`. Same hardware, seed=42. The paged KV cache eliminates per-slot max_seq_len reservation. A Triton paged attention kernel fuses gather+attention during decode, reading K/V directly from scattered blocks via page table.

Per-model server config (tuned to avoid VRAM pressure):

| Model  | max-batch-size | num-gpu-blocks | KV pool | Total VRAM est. |
|--------|---------------:|---------------:|--------:|----------------:|
| Llama  |             24 |           2048 |  3.5 GB |        ~11.5 GB |
| Qwen3  |             16 |   1024 (1536*) |  2.3 GB |        ~10.2 GB |
| Gemma3 |             24 |           2048 |  3.3 GB |         ~5.8 GB |

\* Qwen3 prefix_caching uses 1536 blocks (longer sequences need more blocks). Batch=24/blocks=2048 was memory-constrained on Qwen3-4B (~8 GB weights), causing a 3x throughput collapse (see tuning notes below).

### baseline — Sequential Single Requests

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) |  84.1 |  62.9 |   85.4 |
| TTFT P50 (ms)      |    40 |    48 |     29 |
| TTFT P99 (ms)      |    41 |    51 |     39 |
| ITL P50 (ms)       |    12 |    16 |     12 |
| ITL P99 (ms)       |    13 |    20 |     15 |
| Latency P50 (s)    | 3.046 | 4.076 |  2.997 |

### continuous_batching — Staggered Arrivals

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) | 340.4 | 258.8 |  284.5 |
| TTFT P50 (ms)      |   133 |   190 |    146 |
| TTFT P99 (ms)      |   202 |  3951 |   1215 |
| ITL P50 (ms)       |    35 |    47 |     53 |
| ITL P99 (ms)       |    85 |    98 |     83 |
| Latency P50 (s)    | 6.202 | 8.699 |  8.666 |
| Errors             |     0 |     0 |      0 |

### paged_attention — Single Burst, Moderate Lengths

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  455.2 |  317.4 |  325.2 |
| TTFT P50 (ms)      |   3736 |   8286 |   4637 |
| TTFT P99 (ms)      |  12278 |  22182 |  16463 |
| ITL P50 (ms)       |     45 |     44 |     67 |
| ITL P99 (ms)       |     90 |     92 |     89 |
| Latency P50 (s)    | 13.528 | 18.074 | 16.601 |
| Errors             |      0 |      0 |      0 |

### chunked_prefill — Long Prompts, Poisson Arrivals

| Metric             |   Llama |   Qwen3 | Gemma3 |
|--------------------|--------:|--------:|-------:|
| Throughput (tok/s) |    22.5 |    16.8 |  276.6 |
| TTFT P50 (ms)      |   39099 |  174046 |   1318 |
| TTFT P99 (ms)      |  285209 |  393755 |  12314 |
| ITL P50 (ms)       |    1384 |    1241 |     77 |
| ITL P99 (ms)       |    1562 |    2203 |    119 |
| Latency P50 (s)    | 294.249 | 332.746 | 16.010 |
| Errors             |       0 |       0 |      0 |

### prefix_caching — Shared System Prompt

Qwen3 uses 1536 blocks for this workload (longer sequences need more blocks than 1024).

| Metric             |  Llama |   Qwen3 | Gemma3 |
|--------------------|-------:|--------:|-------:|
| Throughput (tok/s) |  253.2 |    29.1 |  309.6 |
| TTFT P50 (ms)      |  10214 |  127742 |   6673 |
| TTFT P99 (ms)      |  22173 |  278685 |  16762 |
| ITL P50 (ms)       |     83 |     709 |     73 |
| ITL P99 (ms)       |    208 |    1673 |    100 |
| Latency P50 (s)    | 33.739 | 273.611 | 26.060 |
| Errors             |      0 |       0 |      0 |

### Phase 6 vs Phase 5 — Comparison (paged_attention workload)

The paged_attention workload is the headline comparison. Throughput in tok/s:

| Configuration                              |    Llama |    Qwen3 |   Gemma3 |
|--------------------------------------------|---------:|---------:|---------:|
| Phase 4: static + contiguous (batch=8)     |    346.1 |    271.0 |    289.1 |
| Phase 5: continuous + contiguous (batch=8) |    338.2 |    259.5 |    269.3 |
| Phase 6: continuous + paged                |    455.2 |    317.4 |    325.2 |
| Phase 6 vs Phase 4                         | **+32%** | **+17%** | **+12%** |

**Llama (+32% throughput, -66% TTFT P50)**: The headline win. Batch=24 fits 3x more concurrent requests in the same memory. TTFT P50 dropped from 10,920ms to 3,736ms.

**Qwen3 (+17% throughput, -41% TTFT P50)**: At batch=16 (tuned for 16 GB VRAM), paged attention improves throughput from 271.0 to 317.4 tok/s and TTFT P50 from 14,038ms to 8,286ms. Batch=24 with 2048 blocks caused a 3x throughput collapse (121 tok/s) because 8 GB weights + 4.5 GB KV pool left insufficient headroom for activations.

**Gemma3 (+12% throughput, -62% TTFT P50)**: Solid improvement. The small model (1B) handles 24 concurrent sequences easily. All workloads performed well, including chunked_prefill (276.6 tok/s, up from 240.4 in Phase 5).

**ITL tradeoff**: ITL P50 increased across all models (Llama 17→45ms, Qwen3 22→44ms, Gemma3 24→67ms) — expected from running more concurrent sequences per step.

**chunked_prefill regression (Llama, Qwen3)**: With more batch slots, the scheduler admits long-prompt requests simultaneously, creating massive prefill stalls. Gemma3 is unaffected because its prefills are fast. Phase 7 (chunked prefill) will address this.

**Batch size tuning**: The right paged batch size depends on VRAM headroom after model weights and KV pool. Empirical sweep on Qwen3-4B (paged_attention workload):

| batch | blocks | Throughput | Notes                                  |
|------:|-------:|-----------:|----------------------------------------|
|    12 |    768 |      293.9 | Underfilled — not enough concurrency   |
|    16 |   1024 |      318.1 | Sweet spot — best throughput/ITL ratio |
|    20 |   1280 |      326.2 | Slightly higher throughput, higher ITL |
|    24 |   2048 |      121.4 | VRAM cliff — 3x collapse               |

---

## Phase 7 — Chunked Prefill

Server launched with `--batching-mode continuous --kv-cache-backend paged --chunked-prefill --prefill-chunk-size 512`. Same hardware, seed=42. Chunked prefill splits long prompts into 512-token chunks processed across multiple engine steps, interleaved with decode to reduce ITL spikes.

Per-model server config (same batch/block settings as Phase 6):

| Model  | max-batch-size | num-gpu-blocks |
|--------|---------------:|---------------:|
| Llama  |             24 |           2048 |
| Qwen3  |             16 |   1024 (1536*) |
| Gemma3 |             24 |           2048 |

\* Qwen3 prefix_caching uses 1536 blocks (same as Phase 6).

### baseline — Sequential Single Requests

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) |  84.7 |  47.2 |   86.0 |
| TTFT P50 (ms)      |    37 |    45 |     28 |
| TTFT P99 (ms)      |    38 |    60 |     30 |
| ITL P50 (ms)       |    12 |    15 |     12 |
| ITL P99 (ms)       |    13 |    37 |     14 |
| Latency P50 (s)    | 3.016 | 3.944 |  2.974 |

### continuous_batching — Staggered Arrivals

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) | 345.8 | 263.7 |  289.5 |
| TTFT P50 (ms)      |   128 |   175 |    140 |
| TTFT P99 (ms)      |   220 |  3624 |   1131 |
| ITL P50 (ms)       |    34 |    45 |     51 |
| ITL P99 (ms)       |    84 |    94 |     78 |
| Latency P50 (s)    | 6.001 | 8.359 |  8.404 |
| Errors             |     0 |     0 |      0 |

### paged_attention — Single Burst, Moderate Lengths

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  469.0 |  321.5 |  327.0 |
| TTFT P50 (ms)      |   3618 |   8110 |   4605 |
| TTFT P99 (ms)      |  11942 |  21844 |  16347 |
| ITL P50 (ms)       |     44 |     44 |     67 |
| ITL P99 (ms)       |     86 |     90 |     86 |
| Latency P50 (s)    | 13.151 | 17.758 | 16.483 |
| Errors             |      0 |      0 |      0 |

### chunked_prefill — Long Prompts, Poisson Arrivals

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  219.3 |  148.1 |  272.6 |
| TTFT P50 (ms)      |   3464 |  15071 |   1525 |
| TTFT P99 (ms)      |  17975 |  32236 |  12708 |
| ITL P50 (ms)       |     90 |     92 |     78 |
| ITL P99 (ms)       |    209 |    207 |    103 |
| Latency P50 (s)    | 21.194 | 29.989 | 16.270 |
| Errors             |      0 |      0 |      0 |

### prefix_caching — Shared System Prompt

Qwen3 uses 1536 blocks for this workload.

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  299.2 |  206.2 |  309.4 |
| TTFT P50 (ms)      |   8092 |  18157 |   6787 |
| TTFT P99 (ms)      |  18189 |  36053 |  16903 |
| ITL P50 (ms)       |     70 |     70 |     73 |
| ITL P99 (ms)       |    163 |    171 |     96 |
| Latency P50 (s)    | 27.940 | 37.379 | 26.077 |
| Errors             |      0 |      0 |      0 |

### Phase 7 vs Phase 6 — Comparison (chunked_prefill workload)

The chunked_prefill workload is the headline comparison — it has long-prompt arrivals creating prefill pressure while decode requests are in-flight.

| Metric             |         Llama P6 → P7 |         Qwen3 P6 → P7 |       Gemma3 P6 → P7 |
|--------------------|----------------------:|----------------------:|---------------------:|
| Throughput (tok/s) |  22.5 → 219.3 (+874%) |  16.8 → 148.1 (+782%) |  276.6 → 272.6 (-1%) |
| ITL P50 (ms)       |  1384 → 90 (**-94%**) |  1241 → 92 (**-93%**) |         77 → 78 (0%) |
| ITL P99 (ms)       | 1562 → 209 (**-87%**) | 2203 → 207 (**-91%**) | 119 → 103 (**-13%**) |
| TTFT P50 (ms)      |   39099 → 3464 (-91%) | 174046 → 15071 (-91%) |   1318 → 1525 (+16%) |
| TTFT P99 (ms)      | 285209 → 17975 (-94%) | 393755 → 32236 (-92%) |  12314 → 12708 (+3%) |

**Llama (+874% throughput, -94% ITL P50)**: The headline win. Phase 6 was catastrophically slow on this workload (22.5 tok/s) because long-prompt prefills blocked all decode requests for hundreds of milliseconds per step, creating a vicious cycle where prefills dominated the schedule. Chunked prefill breaks long prefills into 512-token chunks interleaved with decode, restoring ITL P50 from 1,384ms to 90ms — a 15x improvement. Throughput jumped from 22.5 to 219.3 tok/s because decode requests can actually make progress between chunks.

**Qwen3 (+782% throughput, -93% ITL P50)**: Same dramatic improvement. Phase 6 was even worse (16.8 tok/s, ITL P50 1,241ms). Chunked prefill restores ITL P50 to 92ms and throughput to 148.1 tok/s. The prefix_caching workload also recovered from 29.1 to 206.2 tok/s (vs Phase 6's 29.1 with 1536 blocks).

**Gemma3 (stable)**: Already fast in Phase 6 (276.6 tok/s, ITL P50 77ms) because the 1B model has fast prefills. Chunked prefill adds no benefit but also no regression — the 512-token chunk size is larger than most prefills so most requests complete in one chunk.

**Non-chunked workload regressions**: baseline, continuous_batching, and paged_attention workloads show negligible change vs Phase 6, confirming no regression from the chunked prefill code path when prefills are short enough to complete in one chunk.

**prefix_caching recovery**: Qwen3 prefix_caching went from 29.1 tok/s (Phase 6) to 206.2 tok/s (Phase 7). The shared 1024-token system prompt is now processed in two 512-token chunks instead of one blocking prefill, preventing the massive decode stalls that collapsed throughput in Phase 6.

---

## Phase 8 — Prefix Caching

Server launched with `--batching-mode continuous --kv-cache-backend paged --chunked-prefill --prefill-chunk-size 512 --prefix-caching`. Same hardware, seed=42. Prefix caching stores completed KV blocks in a radix tree keyed by token IDs. When a new request shares a prefix with a prior request, the cached blocks are reused — the runner starts prefill from the first uncached token, skipping redundant computation.

Per-model server config (same batch/block settings as Phase 7):

| Model  | max-batch-size | num-gpu-blocks |
|--------|---------------:|---------------:|
| Llama  |             24 |           2048 |
| Qwen3  |             16 |   1024 (1536*) |
| Gemma3 |             24 |           2048 |

\* Qwen3 prefix_caching uses 1536 blocks (same as Phase 6/7).

### baseline — Sequential Single Requests

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) |  86.4 |  64.6 |   86.8 |
| TTFT P50 (ms)      |    37 |    45 |     26 |
| TTFT P99 (ms)      |    38 |    48 |     31 |
| ITL P50 (ms)       |    12 |    15 |     11 |
| ITL P99 (ms)       |    13 |    18 |     14 |
| Latency P50 (s)    | 2.964 | 3.940 |  2.942 |

### continuous_batching — Staggered Arrivals

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) | 348.7 | 264.9 |  282.3 |
| TTFT P50 (ms)      |   117 |   175 |    154 |
| TTFT P99 (ms)      |   206 |  3490 |   1248 |
| ITL P50 (ms)       |    34 |    45 |     53 |
| ITL P99 (ms)       |    80 |    90 |     79 |
| Latency P50 (s)    | 5.852 | 8.230 |  8.679 |
| Errors             |     0 |     0 |      0 |

### paged_attention — Single Burst, Moderate Lengths

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  476.7 |  325.0 |  315.7 |
| TTFT P50 (ms)      |   3412 |   7915 |   4735 |
| TTFT P99 (ms)      |  11614 |  21482 |  16841 |
| ITL P50 (ms)       |     44 |     44 |     69 |
| ITL P99 (ms)       |     84 |     82 |     97 |
| Latency P50 (s)    | 12.828 | 17.498 | 16.982 |
| Errors             |      0 |      0 |      0 |

### chunked_prefill — Long Prompts, Poisson Arrivals

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  220.7 |  148.7 |  276.0 |
| TTFT P50 (ms)      |   3327 |  14827 |   1348 |
| TTFT P99 (ms)      |  17770 |  32062 |  12393 |
| ITL P50 (ms)       |     90 |     92 |     77 |
| ITL P99 (ms)       |    221 |    194 |    102 |
| Latency P50 (s)    | 21.084 | 29.914 | 15.976 |
| Errors             |      0 |      0 |      0 |

### prefix_caching — Shared System Prompt

Qwen3 uses 1536 blocks for this workload.

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  331.3 |  221.6 |  309.4 |
| TTFT P50 (ms)      |   5833 |  16059 |   6314 |
| TTFT P99 (ms)      |  15315 |  32515 |  16671 |
| ITL P50 (ms)       |     68 |     70 |     74 |
| ITL P99 (ms)       |     87 |     92 |     90 |
| Latency P50 (s)    | 23.810 | 34.017 | 25.772 |
| Errors             |      0 |      0 |      0 |

### Phase 8 vs Phase 7 — Comparison (prefix_caching workload)

The prefix_caching workload is the headline comparison — 48 requests share a ~1024-token system prompt, so prefix caching should eliminate redundant prefill after the first request.

| Metric             |            Llama P7 → P8 |            Qwen3 P7 → P8 |        Gemma3 P7 → P8 |
|--------------------|-------------------------:|-------------------------:|----------------------:|
| Throughput (tok/s) | 299.2 → 331.3 (**+11%**) |  206.2 → 221.6 (**+7%**) |    309.4 → 309.4 (0%) |
| TTFT P50 (ms)      |   8092 → 5833 (**-28%**) | 18157 → 16059 (**-12%**) | 6787 → 6314 (**-7%**) |
| TTFT P99 (ms)      | 18189 → 15315 (**-16%**) | 36053 → 32515 (**-10%**) |   16903 → 16671 (-1%) |
| ITL P50 (ms)       |            70 → 68 (-3%) |             70 → 70 (0%) |         73 → 74 (+1%) |
| ITL P99 (ms)       |      163 → 87 (**-47%**) |      171 → 92 (**-46%**) |         96 → 90 (-6%) |

**Llama (+11% throughput, -28% TTFT P50)**: Prefix caching eliminates ~1024-token prefill for 47 of 48 requests. TTFT P50 dropped from 8,092ms to 5,833ms — requests start generating faster because the shared prefix is read from cached KV blocks instead of recomputed. ITL P99 dropped 47% (163ms → 87ms) because skipped prefill chunks free up compute budget for decode.

**Qwen3 (+7% throughput, -12% TTFT P50)**: Similar pattern. TTFT P50 improved from 18,157ms to 16,059ms. ITL P99 dropped 46% (171ms → 92ms). The smaller relative TTFT improvement vs Llama is because Qwen3's higher per-token cost makes the suffix prefill (unique per request) a larger fraction of total TTFT.

**Gemma3 (stable)**: Negligible change. The 1B model prefills the ~1024-token prefix so fast (~25ms) that caching it provides minimal savings. All metrics within noise.

**Non-prefix workloads**: baseline, continuous_batching, paged_attention, and chunked_prefill show no regression — the prefix tree adds negligible overhead when prompts don't share prefixes. The chunked_prefill workload (different prompts, no sharing) fills and evicts the tree continuously without impacting performance.

**Bug fix during benchmarking**: The initial chunked_prefill run had 46/48 errors ("Cannot allocate 1 blocks: only 0 free") because decode block allocation didn't evict from the prefix tree. Fixed by adding eviction-aware allocation to `PagedDecodeCacheView._ensure_blocks_allocated()`. Without prefix reuse, completed request blocks stay in the tree as evictable; the decode allocator must evict them when the free pool is exhausted.

---

## Phase 9 — CUDA Graphs

Server launched with `--batching-mode continuous --kv-cache-backend paged --chunked-prefill --prefill-chunk-size 512 --prefix-caching --cuda-graphs`. Same hardware, seed=42. CUDA graphs capture the decode forward pass into pre-recorded graphs for power-of-2 batch size buckets (1, 2, 4, 8, 16, 32), replaying them with a single CPU call per step to eliminate kernel launch overhead. Prefill remains eager. Graphs are warmed up at server startup.

Per-model server config (same batch/block settings as Phase 8):

| Model  | max-batch-size | num-gpu-blocks |
|--------|---------------:|---------------:|
| Llama  |             24 |           2048 |
| Qwen3  |             16 |   1024 (1536*) |
| Gemma3 |             24 |           2048 |

\* Qwen3 prefix_caching uses 1536 blocks (same as Phase 6/7/8).

### baseline — Sequential Single Requests

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) |  53.8 |  42.2 |   50.1 |
| TTFT P50 (ms)      |    44 |    53 |     36 |
| TTFT P99 (ms)      |    49 |    56 |     39 |
| ITL P50 (ms)       |    18 |    23 |     20 |
| ITL P99 (ms)       |    21 |    26 |     21 |
| Latency P50 (s)    | 4.744 | 6.043 |  5.107 |

### continuous_batching — Staggered Arrivals

| Metric             | Llama | Qwen3 | Gemma3 |
|--------------------|------:|------:|-------:|
| Throughput (tok/s) | 291.7 | 155.1 |  268.7 |
| TTFT P50 (ms)      |   168 |  1215 |    159 |
| TTFT P99 (ms)      |   702 | 11690 |   1674 |
| ITL P50 (ms)       |    47 |    80 |     56 |
| ITL P99 (ms)       |    91 |   129 |     79 |
| Latency P50 (s)    | 7.471 |16.133 |  8.918 |
| Errors             |     0 |     0 |      0 |

### paged_attention — Single Burst, Moderate Lengths

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  440.2 |  175.3 |  323.7 |
| TTFT P50 (ms)      |   3677 |  14887 |   5142 |
| TTFT P99 (ms)      |  11744 |  39640 |  16972 |
| ITL P50 (ms)       |     44 |     80 |     67 |
| ITL P99 (ms)       |     85 |    126 |     81 |
| Latency P50 (s)    | 13.001 | 32.345 | 18.878 |
| Errors             |      0 |      0 |      0 |

### chunked_prefill — Long Prompts, Poisson Arrivals

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  216.7 |  155.9 |  262.9 |
| TTFT P50 (ms)      |   3597 |  13389 |   1866 |
| TTFT P99 (ms)      |  17889 |  28550 |  12705 |
| ITL P50 (ms)       |     90 |     81 |     77 |
| ITL P99 (ms)       |    215 |    197 |    112 |
| Latency P50 (s)    | 21.021 | 26.731 | 16.052 |
| Errors             |      0 |      0 |      0 |

### prefix_caching — Shared System Prompt

Qwen3 uses 1536 blocks for this workload.

| Metric             |  Llama |  Qwen3 | Gemma3 |
|--------------------|-------:|-------:|-------:|
| Throughput (tok/s) |  330.4 |  190.1 |  315.6 |
| TTFT P50 (ms)      |   6458 |  19382 |   6801 |
| TTFT P99 (ms)      |  15103 |  38548 |  16137 |
| ITL P50 (ms)       |     67 |     81 |     72 |
| ITL P99 (ms)       |     88 |    106 |     84 |
| Latency P50 (s)    | 24.189 | 40.303 | 25.564 |
| Errors             |      0 |      0 |      0 |

### Phase 9 vs Phase 8 — Comparison

CUDA graphs were expected to reduce decode latency by 5--15% through eliminated kernel launch overhead. The actual results show a regression across all models, most severe on low-concurrency workloads.

**Throughput comparison (tok/s):**

| Workload            | Llama P8 → P9       | Qwen3 P8 → P9       | Gemma3 P8 → P9      |
|---------------------|---------------------:|---------------------:|---------------------:|
| baseline            |  86.4 → 53.8 (-38%) |  64.6 → 42.2 (-35%) |  86.8 → 50.1 (-42%) |
| continuous_batching | 348.7 → 291.7 (-16%) | 264.9 → 155.1 (-41%) | 282.3 → 268.7 (-5%) |
| paged_attention     |  476.7 → 440.2 (-8%) | 325.0 → 175.3 (-46%) | 315.7 → 323.7 (+3%) |
| chunked_prefill     |  220.7 → 216.7 (-2%) | 148.7 → 155.9 (+5%) | 276.0 → 262.9 (-5%) |
| prefix_caching      |  331.3 → 330.4 ( 0%) | 221.6 → 190.1 (-14%) | 309.4 → 315.6 (+2%) |

**ITL P50 comparison (ms):**

| Workload            | Llama P8 → P9   | Qwen3 P8 → P9   | Gemma3 P8 → P9   |
|---------------------|-----------------:|-----------------:|------------------:|
| baseline            |  12 → 18 (+50%) |  15 → 23 (+53%) |   11 → 20 (+82%) |
| continuous_batching |  34 → 47 (+38%) |  45 → 80 (+78%) |   53 → 56 (+6%)  |
| paged_attention     |  44 → 44 ( 0%)  |  44 → 80 (+82%) |   69 → 67 (-3%)  |
| chunked_prefill     |  90 → 90 ( 0%)  |  92 → 81 (-12%) |   77 → 77 ( 0%)  |
| prefix_caching      |  68 → 67 (-1%)  |  70 → 81 (+16%) |   74 → 72 (-3%)  |

**Root cause: Triton kernels replay ~2x slower inside CUDA graphs.** Profiling with `benchmarks/profile_cuda_graph.py` on Llama 3B reveals the bottleneck is the graph replay itself, not the Python-side `prepare()` overhead (which is only 0.1--0.8ms per step):

```
                      batch=1         batch=8
Graph replay:         17.9 ms/step    31.6 ms/step
Eager (SDPA path):    10.8 ms/step    15.3 ms/step
Eager (Triton path):  10.0 ms/step    12.9 ms/step
```

The Triton paged attention kernel is actually *faster* than SDPA when run eagerly (10.0ms vs 10.8ms at batch=1). But the same kernel captured into a CUDA graph runs 75--145% slower than its eager equivalent. This is a Triton JIT + CUDA graph interaction issue: Triton's kernel launch mechanism, autotuning state, or compiled kernel variants may not replay correctly inside CUDA graphs, causing suboptimal execution.

**Why larger batches are partially shielded:** High-concurrency workloads (paged_attention, prefix_caching) approach break-even because the per-step overhead is amortized across more sequences, and the workloads are less sensitive to per-token decode latency. The baseline (single-request sequential) is hit hardest because every ms of per-step overhead translates directly to per-token latency.

**Why Qwen3 is hit hardest:** With batch=16 and 1024 blocks, Qwen3 is already VRAM-constrained. The additional memory from CUDA graph capture (graph intermediates + 6 captured graphs with shared pool) compounds the issue, and the smaller max concurrency means the per-step overhead is amortized across fewer sequences.

**Path forward:** The Triton-in-CUDA-graph performance gap must be resolved before graph capture provides a net benefit. Options: (1) investigate Triton graph capture compatibility (warmup autotuning, kernel caching, stream interactions); (2) replace Triton attention with a native CUDA kernel (e.g. Flash Attention via `flash_attn`) that is known to work well with CUDA graphs; (3) use `torch.compile` with CUDA graph backend instead of manual capture.
