# infer

An educational LLM inference runtime built from scratch on PyTorch. The goal is to understand how systems like vLLM, TGI, and SGLang work by building one incrementally — from loading weights and producing a single token up through continuous batching, paged attention, speculative decoding, and structured output.

Supports three model architectures: **Llama 3**, **Qwen 3**, and **Gemma 3**.

## Features

- Custom model implementations (no `transformers` dependency except for tokenization)
- Fused Triton kernels (RMSNorm, RoPE, residual+norm, SwiGLU/GeGLU, paged attention)
- Contiguous and paged KV cache backends
- Static and continuous batching
- Chunked prefill for interleaving long prefills with decode
- Radix-tree prefix caching with LRU eviction
- CUDA graph capture for decode (experimental)
- FP8 and INT8 weight quantization (serve 8B models on 16 GB VRAM)
- Speculative decoding with a draft model (lossless 1.5-2.5x decode speedup)
- Structured output via JSON schema or regex constraints (backed by `outlines-core`)
- OpenAI-compatible SSE streaming API (`POST /v1/completions`)
- Every optimization independently toggleable via `EngineConfig`

## Project structure

```
src/infer/
├── models/          # Model architectures (Llama, Qwen3, Gemma3) and shared components
├── loader/          # Weight loading, config parsing, tokenizer, chat templates
├── engine/          # Generation loop, sampler, scheduler, runners, engine config
├── cache/           # KV cache backends (simple, slotted, paged) and prefix tree
├── kernels/         # Triton kernels (RMSNorm, RoPE, fused norm, activation, paged attn)
├── quant/           # Weight quantization (FP8Linear, INT8Linear, model surgery)
├── structured/      # Structured output (TokenGuide, logit masking via outlines-core)
└── server/          # FastAPI server with SSE streaming
benchmarks/          # Serving benchmarks, profiling, workload definitions, reports
scripts/             # Sanity check script for quick model verification
tests/               # Unit, integration, and stress tests (1136 tests)
docs/                # Per-phase design documents
```

## Setup

Requires Python 3.14+, CUDA 12.x, and a GPU with bf16 support.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies
uv sync --dev

# Verify CUDA visibility
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Running the server

```bash
# Basic — static batching, contiguous KV cache
uv run python -m infer.server --model meta-llama/Llama-3.2-3B-Instruct

# Continuous batching with paged attention
uv run python -m infer.server --model meta-llama/Llama-3.2-3B-Instruct \
    --batching-mode continuous --kv-cache-backend paged \
    --max-batch-size 24 --num-gpu-blocks 2048

# + Chunked prefill (reduces ITL spikes from long prompts)
uv run python -m infer.server --model meta-llama/Llama-3.2-3B-Instruct \
    --batching-mode continuous --kv-cache-backend paged \
    --max-batch-size 24 --num-gpu-blocks 2048 \
    --chunked-prefill --prefill-chunk-size 512

# + Prefix caching (reuses shared prefixes across requests)
uv run python -m infer.server --model meta-llama/Llama-3.2-3B-Instruct \
    --batching-mode continuous --kv-cache-backend paged \
    --max-batch-size 24 --num-gpu-blocks 2048 \
    --chunked-prefill --prefill-chunk-size 512 --prefix-caching

# Quantized 8B model (FP8 — auto-detected from checkpoint)
uv run python -m infer.server --model Qwen/Qwen3-8B-FP8 \
    --batching-mode continuous --kv-cache-backend paged \
    --max-batch-size 24 --num-gpu-blocks 2048

# Quantized 8B model (INT8)
uv run python -m infer.server --model nytopop/Qwen3-8B.w8a8 \
    --batching-mode continuous --kv-cache-backend paged \
    --max-batch-size 24 --num-gpu-blocks 2048

# Speculative decoding (1B draft model speeds up 3B target)
uv run python -m infer.server --model meta-llama/Llama-3.2-3B-Instruct \
    --batching-mode continuous --kv-cache-backend paged \
    --max-batch-size 8 --num-gpu-blocks 2048 \
    --speculative-decoding --draft-model meta-llama/Llama-3.2-1B-Instruct \
    --spec-length 5
```

### Server flags

| Flag                       | Default      | Description                                  |
|----------------------------|--------------|----------------------------------------------|
| `--model`                  | (required)   | HuggingFace model ID or local path           |
| `--host`                   | `0.0.0.0`    | Bind address                                 |
| `--port`                   | `8000`       | Bind port                                    |
| `--dtype`                  | `bfloat16`   | Weight dtype (`bfloat16` or `float16`)       |
| `--max-seq-len`            | `4096`       | Maximum total sequence length                |
| `--max-batch-size`         | `8`          | Maximum concurrent requests                  |
| `--batching-mode`          | `static`     | `static` or `continuous`                     |
| `--kv-cache-backend`       | `contiguous` | `contiguous` or `paged`                      |
| `--block-size`             | `16`         | Tokens per KV block (paged only)             |
| `--num-gpu-blocks`         | auto         | Total KV blocks (paged only)                 |
| `--chunked-prefill`        | off          | Enable chunked prefill                       |
| `--prefill-chunk-size`     | `512`        | Tokens per prefill chunk                     |
| `--prefix-caching`         | off          | Enable prefix caching                        |
| `--cuda-graphs`            | off          | CUDA graph capture for decode (experimental) |
| `--quantization`           | auto-detect  | Weight quantization (`fp8` or `int8`)        |
| `--speculative-decoding`   | off          | Enable speculative decoding                  |
| `--draft-model`            | none         | Draft model for speculative decoding         |
| `--spec-length`            | `5`          | Candidate tokens per speculation round       |
| `--seed`                   | none         | Global random seed                           |

### Sending requests

The server exposes `POST /v1/completions` with SSE streaming:

```bash
curl -N http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "prompt": "Explain the concept of attention in transformers.",
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": true
    }'
```

Structured output (JSON schema constraint):

```bash
curl -N http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "prompt": "Extract the name and age: John is 30 years old.",
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }'
```

## Running benchmarks

Start the server in one terminal, then run the benchmark client:

```bash
# Run all workloads
uv run python benchmarks/bench_serving.py \
    --server http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --workload all --seed 42

# Run a single workload
uv run python benchmarks/bench_serving.py \
    --server http://localhost:8000 \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --workload paged_attention --seed 42
```

### Workloads

| Workload              | Requests       | Pattern                 | Measures                              |
|-----------------------|----------------|-------------------------|---------------------------------------|
| `baseline`            | 10 sequential  | Wait for completion     | Single-request overhead floor         |
| `continuous_batching` | 32 at 4 RPS    | Staggered arrivals      | TTFT benefit from per-step scheduling |
| `paged_attention`     | 48 burst       | All at once             | Throughput with higher concurrency    |
| `chunked_prefill`     | 48 at 6 RPS    | 75% long prompts        | ITL stability under prefill pressure  |
| `prefix_caching`      | 48 at 8 RPS    | Shared ~1024-tok prefix | TTFT from prefix reuse                |

Reports are saved to `benchmarks/reports/` as JSON.

### Profiling

```bash
uv run python benchmarks/profile_generation.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --prompt-tokens 256 --decode-tokens 64
```

Outputs a Chrome trace JSON (viewable in `chrome://tracing` or Perfetto) and a CUDA kernel summary.

## Performance results

All results on an RTX 5080 (16 GB VRAM). Three benchmark models: Llama 3.2-3B, Qwen3-4B, Gemma3-1B.

### Throughput progression (tok/s)

Each row adds one optimization. The workload column shows which benchmark best demonstrates the improvement.

| Configuration         | Workload              | Llama 3B    | Qwen3 4B    | Gemma3 1B    |
|-----------------------|-----------------------|-------------|-------------|--------------|
| Static batching       | paged_attention       | 346         | 271         | 289          |
| + Continuous batching | continuous_batching   | 317 (+25%)  | 242 (+26%)  | 250 (+8%)    |
| + Paged attention     | paged_attention       | 455 (+32%)  | 317 (+17%)  | 325 (+12%)   |
| + Chunked prefill     | chunked_prefill       | 219 (vs 23) | 148 (vs 17) | 273 (stable) |
| + Prefix caching      | prefix_caching        | 331 (+11%)  | 222 (+7%)   | 309 (stable) |

### Key improvements by phase

**Continuous batching** (Phase 5): TTFT dropped 30-55% — requests no longer wait for entire batches to drain. Throughput improved 8-26%.

**Paged attention** (Phase 6): +12-32% throughput on burst workloads by fitting 3x more concurrent requests in the same memory. TTFT P50 dropped 41-66%.

**Chunked prefill** (Phase 7): Broke the prefill stall problem. On the long-prompt workload, Llama went from 23 to 219 tok/s (+874%) and ITL P50 dropped from 1384ms to 90ms (-94%).

**Prefix caching** (Phase 8): On the shared-prefix workload (48 requests, ~1024-token common prefix), Llama gained +11% throughput, -28% TTFT P50, -47% ITL P99. Gemma3-1B showed negligible change since the 1B model prefills too fast for caching to help.

**CUDA graphs** (Phase 9): Captures the decode forward pass into a CUDA graph to eliminate CPU-side kernel launch overhead. In practice, Triton kernels replay slower inside CUDA graphs than they execute eagerly, so this is kept as a reference implementation (`--cuda-graphs`) rather than a recommended optimization.

**Weight quantization** (Phase 10): FP8 (block-wise, `float8_e4m3fn`) and INT8 (per-channel symmetric) quantization halve weight memory, making 8B-class models (e.g. `Qwen/Qwen3-8B-FP8`, `nytopop/Qwen3-8B.w8a8`) servable on 16 GB VRAM. Uses eager dequant-then-matmul — quantization format is auto-detected from the checkpoint.

**Speculative decoding** (Phase 11): A small draft model (Llama 3.2-1B) proposes K candidate tokens per step, verified in a single target model (Llama 3.2-3B) forward pass. Output is mathematically identical to pure target-model sampling (lossless). Typical acceptance rates of 70-90% yield 1.5-2.5x effective decode throughput.

**Structured output** (Phase 12): Constrains generation to follow a JSON schema or regex pattern by masking invalid tokens at each decode step. Uses `outlines-core` for FSM construction and token-level guidance. Zero overhead when disabled.

## Development

```bash
uv run pytest              # Run tests
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run mypy .              # Type check
```

Quick sanity check across all supported models:

```bash
uv run python scripts/sanity_check.py
```

## Design

See `docs/OVERALL_DESIGN.md` for the full design document covering architecture, phased implementation plan, API contract, and benchmarking methodology. Per-phase design docs are in `docs/PHASE_*.md`.
