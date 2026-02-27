# infer: An Educational LLM Inference Runtime

## 1. Overview

`infer` is an LLM inference runtime built from scratch on top of PyTorch. The goal is to understand how systems like vLLM, TGI, and SGLang work by building one incrementally, starting from "load weights and produce one token" and working up to continuous batching, paged attention, and chunked prefill.

### Goals

- Educational depth over production polish.
- Incremental milestones with measurable improvement at each stage.
- Real HuggingFace models (Llama-family first), not toy architectures.
- Single-GPU optimization first.
- Quantifiable impact for each optimization via toggleable features.

### Non-Goals

- Competing with vLLM on absolute performance.
- Multi-GPU and tensor parallel execution.
- Supporting all HuggingFace architectures.
- Training and fine-tuning workflows.

### Definition of Done for v1

v1 is complete at the end of **Phase 5**:

- Continuous batching works with streaming output.
- OpenAI-compatible `POST /v1/completions` endpoint is available.
- Correctness tests pass (parity checks + scheduler/cache tests).
- A benchmark report exists comparing baseline vs. each enabled optimization through Phase 5.

Phases 6-8 are advanced extensions, not required for v1.

---

## 2. Scope and Milestone Boundaries

To keep implementation tractable, phases are grouped into milestones:

| Milestone | Phases | Outcome |
|----------|--------|---------|
| M1 Core Inference | 1a-3.1 | Single-request generation with KV cache + Triton kernels |
| M2 Serving MVP | 4-5 | Multi-request serving with continuous batching + SSE |
| M3 Advanced Memory/Scheduling | 6-8 | Paged attention, chunked prefill, prefix caching |

Guardrails:

- Do not start M2 until Phase 3 correctness and speedup checks are complete.
- Do not start M3 until M2 benchmarks and API stability checks are complete.
- M3 features remain optional feature flags and must not regress M2 behavior when disabled.

---

## 3. Environment and Assumptions

### Baseline environment

- OS: Linux (x86_64)
- Python: 3.14+
- PyTorch: 2.4+ with CUDA-enabled build
- CUDA runtime: 12.x
- GPU: NVIDIA with bf16 support preferred (fp16 fallback allowed)

### Hardware tiers

- Dev tier: 16 GB VRAM (primary development target, RTX 5080 class)
- Stretch tier: 24 GB+ VRAM (for larger experiments and more aggressive concurrency)

### Model assumptions

- Target architectures: Llama 3, Qwen 3, Gemma 3.
- Dev models (unit tests, quick iteration): `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.
- Benchmark models (serving benchmarks, Phase 4+): `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-4B`, `google/gemma-3-1b-it`.
- Practical unquantized target range on dev hardware: 1B-4B class models.
- Larger models (for example 7B-8B class) are advanced and typically require quantization and/or reduced concurrency.
- Use local model paths when possible to avoid benchmark variance from download latency.

### Reproducibility requirements

- Pin core dependencies in project config.
- Log model revision/commit, dtype, and all runtime flags in benchmark output.
- Set random seeds for deterministic sampling tests.

---

## 4. Architecture Overview

```text
┌─────────────────────────────────────────────────┐
│                   API Server                    │
│            (FastAPI + SSE streaming)            │
└────────────────────┬────────────────────────────┘
                     │ Completion requests
                     ▼
┌─────────────────────────────────────────────────┐
│                   Scheduler                     │
│  - Request queue                                │
│  - Batching policy (static -> continuous)       │
│  - Admission control                             │
└────────────────────┬────────────────────────────┘
                     │ Batch of sequences
                     ▼
┌─────────────────────────────────────────────────┐
│                 Model Runner                    │
│  - Executes forward passes                      │
│  - Manages KV cache reads/writes                │
│  - Sampling pipeline                            │
└─────────┬───────────────────┬───────────────────┘
          │                   │
          ▼                   ▼
┌──────────────────┐ ┌──────────────────────────┐
│   Model Loader   │ │   KV Cache Manager       │
│  - HF weights    │ │  - Contiguous/Paged mode │
│  - Arch registry │ │  - Allocator/page tables │
│  - Tokenization  │ │  - Eviction policy       │
└──────────────────┘ └──────────────────────────┘
```

Runtime boundary:

- The engine is compute-only (`step()` has no network I/O).
- The API layer owns request/response translation, backpressure, and client disconnect handling.

---

## 5. External API Contract

### v1 endpoint

- `POST /v1/completions`

Supported request fields for v1:

- `model: str`
- `prompt: str | list[int]`
- `max_tokens: int`
- `temperature: float`
- `top_p: float`
- `top_k: int | null`
- `repetition_penalty: float`
- `stream: bool` (always treated as `true`; accepted for OpenAI compatibility)
- `stop: str | list[str] | null`
- `seed: int | null`

Optional fields can be added later, but unsupported fields must return a clear validation error.

### Response format (v1, SSE only)

SSE streaming is the only response mode. Non-streaming clients can collect the full stream and assemble the final response.

Event types:

- `token` for newly generated token text and token id
- `done` for terminal marker (includes `usage` counters: `prompt_tokens`, `completion_tokens`, `total_tokens`)
- `error` for fatal per-request error

### Error responses

- `400` — malformed request (bad JSON, missing required fields).
- `422` — unsupported or invalid field values (unknown `model`, out-of-range `temperature`, etc.).
- `503` — server overloaded (request queue full).

Errors are returned as standard HTTP responses before the SSE stream starts. Per-request errors that occur *during* generation are delivered via the `error` SSE event.

### Ordering guarantees

- Tokens are emitted in generation order per request.
- Exactly one terminal event (`done` or `error`) is emitted.

---

## 6. Internal Interfaces and Data Model

Core request state machine:

`WAITING -> PREFILL -> DECODE -> FINISHED`

Error state:

`FAILED`

Reference interfaces:

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    max_new_tokens: int = 128
    stop: list[str] | None = None
    seed: int | None = None

@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time_s: float

    # Mutable state
    state: RequestState = RequestState.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    generator: torch.Generator | None = None
    slot_idx: int | None = None           # Cache pool slot (Phase 5+)
    prefill_progress: int = 0             # Chunked prefill token count (Phase 7)
    output_queue: asyncio.Queue[StepOutput] | None = None

@dataclass
class StepOutput:
    request_id: str
    token_id: int | None
    text_delta: str
    finished: bool
    finish_reason: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

Engine invariants:

- `step()` performs one full scheduler iteration.
- `step()` is deterministic for a fixed random seed and fixed request queue.
- Scheduler state updates are atomic per iteration (no partial updates).

KV cache budget formula (for planning):

`kv_bytes ~= 2 * num_layers * num_kv_heads * head_dim * active_tokens * dtype_bytes`

Use this estimate in admission control before allocating new sequences.

---

## 7. Phased Implementation Plan

Each phase has explicit deliverables and exit criteria to reduce ambiguity.

### Phase 1a: Model Loading and Single-Layer Forward

Goal:

- Load HF weights and verify a single transformer layer produces correct activations for Llama 3, Qwen 3, and Gemma 3.

Deliverables:

- Safetensors loader with shard/index handling.
- HF `config.json` reader and weight name mapping (per architecture).
- `AutoTokenizer` wrapper (use HF `transformers` for tokenization).
- Shared transformer components (RMSNorm, RoPE, GQA core, gated MLP).
- Per-architecture `TransformerBlock` implementations: Llama 3 (pre-norm), Qwen 3 (pre-norm + QK-norm), Gemma 3 (sandwich norm + QK-norm + GeGLU).

Exit criteria:

- Single-layer activation parity tests against `transformers` for all three dev models: `Llama-3.2-1B-Instruct`, `Qwen3-1.7B`, `gemma-3-1b-it`.
- Max absolute error thresholds documented by dtype.
- Unit tests for loader (single-file and sharded checkpoints) and tokenizer wrapper.

### Phase 1b: Full Model and Logits Parity

Goal:

- Assemble full models for all three architectures (Llama 3, Qwen 3, Gemma 3) and verify end-to-end logits match `transformers`.

Deliverables:

- Complete `nn.Module` for each architecture (stack all layers, final norm, LM head) and a `load_model` entry point.
- Jinja2-based chat template renderer with custom templates per model (no `transformers.apply_chat_template`).
- Layer-by-layer activation diff tooling for debugging mismatches.

Exit criteria:

- Full-model logits parity tests against `transformers` for all three dev models, with max absolute error thresholds documented by dtype.
- At least one end-to-end test per model: load model, tokenize prompt, forward pass, verify logits.

### Phase 2: Autoregressive Generation

Goal:

- Single-request token-by-token generation.

Deliverables:

- Generation loop.
- Sampler pipeline (temperature, top-k/top-p, repetition penalty).
- Stop condition handling.

Exit criteria:

- Functional generation for at least 3 prompts.
- Deterministic generation test with fixed seed.

### Phase 3: KV Cache

Goal:

- Prefill once, decode incrementally.

Deliverables:

- Contiguous pre-allocated KV cache.
- Separate prefill/decode code paths.
- Sequence length/position tracking.

Exit criteria:

- Output equivalence vs Phase 2 under greedy decode.
- Measured decode throughput improvement over Phase 2 on same hardware/model.

### Phase 3.1: Triton Kernel Optimization

Goal:

- Close the gap between current decode throughput and hardware memory-bandwidth limits via fused Triton kernels, reducing kernel launch overhead and redundant HBM traffic.

Deliverables:

- Profiling script (`benchmarks/profile_generation.py`) with Chrome trace export and CUDA kernel summary.
- Fused Triton kernels: RMSNorm, RoPE, residual+RMSNorm, SwiGLU/GeGLU activation.
- Triton kernels used unconditionally (no fallback paths — Triton is a required dependency).
- Unit tests for each kernel (Triton vs PyTorch correctness).
- Sanity check script (`scripts/sanity_check.py`) for quick manual verification of all models.

Exit criteria:

- Measurable decode throughput improvement on all 3 dev models.
- All existing tests pass (Phase 1–3 regressions checked).
- Benchmark log updated with Triton-enabled results.

### Phase 4: Static Batching and SSE API

Goal:

- Serve multiple concurrent requests in fixed batches with a streaming API from the start.

Deliverables:

- Request queue and batch builder.
- Padded/varlen attention support for mixed prompt lengths.
- FastAPI endpoint (`POST /v1/completions`) with SSE streaming (the only API surface — no separate non-streaming path).
- HTTP error responses for invalid requests (400 for bad params, 422 for unsupported fields, 503 when queue is full).
- Custom benchmark script (`bench_serving.py`) measuring TTFT, ITL, and throughput under concurrent load.

Exit criteria:

- Throughput under concurrent load exceeds sequential serving baseline.
- Regression tests cover mixed sequence lengths and EOS handling in batch.
- Benchmark script produces a valid report with TTFT, ITL, and throughput metrics.

### Phase 5: Continuous Batching (v1 completion target)

Goal:

- Iteration-level scheduling with per-step admit/retire.

Deliverables:

- Continuous scheduler (admit/retire each step, FCFS policy).
- Slotted KV cache pool with per-slot position tracking.
- Adaptive prefill: individual (N=1) or batched (N>1, right-padded with scatter-write via `BatchedPrefillCacheView`).
- Model `position_ids` support for batched decode with per-sequence RoPE.
- Triton RoPE kernel update for 3D cos/sin.
- Per-sequence lifecycle tracking.

Exit criteria:

- Tail latency improvement vs static batching on mixed-length workload.
- No starvation in stress test (bounded wait for queued requests).
- Benchmark results recorded comparing static vs continuous batching.
- v1 Definition of Done satisfied.

### Phase 6: Paged Attention

Goal:

- Replace contiguous per-sequence allocation with paged KV memory.

Deliverables:

- `CachePoolProtocol` abstracting pool interface (both `SlottedKVCache` and `PagedKVCachePool` satisfy it).
- `BlockAllocator` with O(1) alloc/free, double-free detection, owner tracking, and leak detection.
- `PagedKVCachePool` with block tensor storage, page tables, and audit/reclaim for leaked blocks.
- Three paged cache views (`PagedPrefillCacheView`, `PagedDecodeCacheView`, `PagedBatchedPrefillCacheView`).
- Split scheduler interface (`retire()` → `admit(free_kv_tokens)` → `decode_requests()`).
- Vectorized flat-gather decode path (indices computed once per step, reused across layers).
- Triton paged attention kernel (fused gather + attention for decode, auto-dispatched).

Exit criteria:

- Higher max concurrent sequence capacity than contiguous mode at same VRAM limit.
- Correctness tests pass with randomized page mappings.
- `CachePoolProtocol` satisfied by both backends; no `isinstance` checks in runner or engine.
- All Phase 1-5 tests pass with `kv_cache_backend="contiguous"` (no regression).
- `audit_blocks()` returns empty after normal request lifecycle.

**Status: COMPLETE.** All deliverables implemented, benchmarked on all three models (Llama 3.2 3B, Qwen3-4B, Gemma 3 1B). Triton paged attention kernel delivers +12-32% throughput over contiguous baseline depending on model. 817 tests pass.

See `docs/PHASE_6.md` for the full design.

### Phase 7: Chunked Prefill

Goal:

- Break long prefills into fixed-size chunks that interleave with decode steps, preventing decode stalls when new requests arrive.

Deliverables:

- Configurable chunk size (`prefill_chunk_size` in `EngineConfig`).
- Scheduler support for partial prefill: track per-request prefill progress and resume across iterations.
- Mixed prefill-chunk + decode batches within a single `step()` call.
- Attention mask handling for partial prefill sequences alongside decoding sequences.

Exit criteria:

- ITL stability improvement under concurrent prefill+decode workload vs unchunked baseline.
- No throughput regression on decode-only workload (chunking disabled or chunk size >= prompt length).
- Correctness: chunked prefill produces identical logits to full prefill under greedy decode.

**Status: COMPLETE.** All deliverables implemented, benchmarked on all three models. Chunked prefill (512-token chunks, paged backend) dramatically improves ITL under long-prompt prefill pressure: Llama ITL P50 1384ms to 90ms (-94%), Qwen3 1241ms to 92ms (-93%), with throughput gains of +874% and +782% respectively on the chunked_prefill workload. No regressions on non-chunked workloads. Both contiguous and paged backends supported. 902 tests pass.

See `docs/PHASE_7.md` for the full design.

### Phase 8: Prefix Caching

Goal:

- Reuse shared prefill prefixes.

Deliverables:

- Radix/prefix tree keyed by token ids.
- Block refcounting and LRU eviction policy.
- Prefix-aware allocation and free in `PagedKVCachePool`.
- Full-hit optimization (single last-token forward pass when entire prefix is cached).

Exit criteria:

- TTFT improvement on shared-prefix workload.
- Correctness: identical output with/without prefix caching (greedy decode).
- Cache correctness tests for refcount + eviction edge cases.
- No regression when disabled.

---

## 8. Engine Configuration and Feature Flags

Every optimization must be independently toggleable via `EngineConfig`.

```python
@dataclass
class EngineConfig:
    # Core
    model: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_seq_len: int = 4096
    seed: int | None = None

    # Capacity controls
    max_batch_size: int = 32
    max_num_batched_tokens: int = 8192

    # KV cache (always on; backend selects implementation)
    kv_cache_backend: str = "contiguous"  # "contiguous" | "paged"
    block_size: int = 16                  # paged backend only
    num_gpu_blocks: int | None = None     # paged backend only; None = auto

    # Batching
    batching_mode: str = "continuous"  # "static" | "continuous"
    scheduler_policy: str = "fcfs"     # "fcfs" | "prefill_first" | "decode_round_robin"

    # Chunked prefill
    use_chunked_prefill: bool = False
    prefill_chunk_size: int = 512
    max_prefill_chunks_per_step: int | None = None  # None = no cap

    # Prefix caching
    use_prefix_caching: bool = False

    # Attention backend (Triton paged kernel auto-dispatches for paged decode)
    attention_backend: str = "sdpa"    # "naive" | "sdpa" | "flash"
```

Compatibility rules to enforce in config validation:

- `use_chunked_prefill=True` requires `batching_mode="continuous"`.
- `use_prefix_caching=True` requires `kv_cache_backend="paged"` and `use_chunked_prefill=True`.

Benchmark matrix baseline:

| Config       | KV Backend | Batching   | Chunked Prefill | Prefix |
|--------------|------------|------------|-----------------|--------|
| Static Batch | contiguous | static     | off             | off    |
| +Continuous  | contiguous | continuous | off             | off    |
| +Paged       | paged      | continuous | off             | off    |
| +Chunked     | paged      | continuous | on              | off    |
| +Prefix      | paged      | continuous | on              | on     |

Scheduler policy sweep (Phase 5+):

- Run `W2 Mixed` and `W3 Shared Prefix` for each `scheduler_policy`.
- Report throughput, TTFT (P50/P99), ITL (P50/P99), and starvation incidents.

---

## 9. Component Deep Dives

### 9.1 Model loader and registry

- Read `config.json` and dispatch by `model_type`.
- Map HF tensor names to internal module names with explicit checks for missing/unexpected tensors.
- Support both single-file and sharded safetensors checkpoints.

### 9.2 Chat template handling

- Use the `jinja2` library for template rendering (no custom parser).
- Write and maintain our own Jinja2 template strings per supported model family (no `transformers.apply_chat_template` dependency).
- Templates live alongside model definitions and are selected via the model registry.
- If a new model needs a template we haven't written yet, fail fast with a clear error pointing to the template directory.

### 9.3 Scheduler behavior

- Keep policy modular.
- Keep `scheduler_policy` runtime-configurable and benchmark all supported policies.
- Minimum required fairness rule for v1:
  - Decode-active sequences must receive service at least once every `N` scheduler iterations.
- Admission control checks token budget and KV budget before enqueueing into active set.

### 9.4 Engine async-readiness

- Engine remains synchronous internally.
- API integration runs `step()` in a dedicated loop/task.
- All request outputs pass through per-request queues/channels to isolate API concerns from runtime logic.

---

## 10. Metrics and Benchmarking

Use a custom benchmark script (`bench_serving.py`) as the primary harness and add internal counters from the engine.

Core metrics:

- Throughput (output tokens/sec across all requests)
- TTFT (time to first token)
- ITL (inter-token latency)
- Request latency (P50/P95/P99)
- Queue wait time (scheduler-level)
- GPU memory usage split (weights vs KV vs other)

Standard workloads to run for every milestone:

- `W1 Single`: one request, prompt 256, generate 256.
- `W2 Mixed`: 16 concurrent requests on dev tier (optionally 64 on stretch tier), prompt lengths 32-1024, generate 64-256.
- `W3 Shared Prefix`: 16 concurrent chat requests on dev tier (optionally 64 on stretch tier) sharing same system prompt.

Benchmark report must include:

- Hardware and software versions.
- Full `EngineConfig`.
- Summary table vs baseline.

---

## 11. Testing Strategy and Quality Gates

Test layers:

- Unit tests:
  - tensor shape and dtype checks
  - sampler transform correctness
  - template parser behavior
- Integration tests:
  - logits parity against `transformers`
  - end-to-end generation for fixed seeds
  - API contract tests (streaming and non-streaming)
- Stress tests:
  - queue growth and backpressure
  - cancellation/disconnect handling
  - OOM handling (advanced phases)

Milestone gates:

- M1 gate: single-layer parity (1a) + full-model parity (1b) + deterministic generation + KV cache regression tests.
- M2 gate: API compatibility tests + load test with no deadlocks/starvation.
- M3 gate: new feature enabled/disabled equivalence tests and no M2 regressions.

---

## 12. Risks and Mitigations

- Chat template complexity can stall Phase 1b.
  - Mitigation: use `jinja2` library for rendering; write our own template strings per model family.
- Numerical mismatches vs `transformers` can be hard to debug.
  - Mitigation: layer-by-layer activation diff tooling and deterministic seeds.
- Memory fragmentation and OOM in advanced phases.
  - Mitigation: strict allocator invariants, fuzz tests for page table operations, hard caps in admission control.
- Benchmark noise can hide regressions.
  - Mitigation: fixed workloads, warmup runs, and repeated trials with median reporting.

---

## 13. Project Structure

```text
infer/
├── src/
│   └── infer/
│       ├── debug.py
│       ├── models/
│       │   ├── common.py
│       │   ├── llama.py
│       │   ├── qwen3.py
│       │   └── gemma3.py
│       ├── loader/
│       │   ├── config.py
│       │   ├── weights.py
│       │   ├── weight_map.py
│       │   ├── tokenizer.py
│       │   ├── chat_template.py
│       │   └── model_loader.py
│       ├── engine/
│       │   ├── config.py
│       │   ├── engine.py
│       │   ├── request.py
│       │   ├── runner.py
│       │   ├── runner_helpers.py
│       │   ├── continuous_runner.py
│       │   ├── scheduler.py
│       │   ├── sampler.py
│       │   └── generate.py
│       ├── cache/
│       │   ├── protocol.py
│       │   ├── simple.py
│       │   ├── slotted.py
│       │   ├── paged.py
│       │   └── prefix.py
│       ├── kernels/
│       │   ├── __init__.py
│       │   ├── rms_norm.py
│       │   ├── rope.py
│       │   ├── fused_norm_residual.py
│       │   ├── activation.py
│       │   └── paged_attention.py
│       └── server/
│           ├── __init__.py
│           ├── api.py
│           └── __main__.py
├── benchmarks/
│   ├── run_matrix.py
│   ├── workloads/
│   └── reports/
├── scripts/
│   └── sanity_check.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── stress/
└── docs/
    ├── OVERALL_DESIGN.md
    ├── PHASE_1A.md
    ├── PHASE_1B.md
    ├── PHASE_2.md
    ├── PHASE_3.md
    ├── PHASE_3_1.md
    ├── PHASE_4.md
    ├── PHASE_5.md
    ├── PHASE_6.md
    └── PHASE_7.md
```

---

## 14. Design Decisions Log

- Primary dev hardware target is 16 GB VRAM; unquantized model work should stay mostly in the 1B-3B range.
- Support three architectures from Phase 1a: Llama 3, Qwen 3, Gemma 3. Shared components (RMSNorm, RoPE, attention, gated MLP) with per-architecture transformer blocks and weight maps.
- Start with custom model code, use `transformers` only for `AutoTokenizer` and parity checks.
- Use `jinja2` for chat template rendering with our own template strings per model (no `transformers.apply_chat_template`, no custom parser).
- Use SDPA as default fast path before Triton specialization.
- Triton is a required dependency; kernels are always used (no PyTorch fallback paths). This simplifies the codebase and avoids maintaining two code paths. The fused RoPE kernel uses FMA instructions, producing results that are slightly different from but more precise than HF's PyTorch implementation (verified against fp64 ground truth).
- The Triton paged attention kernel auto-dispatches based on runtime conditions (`is_paged()`, `seq_len == 1`, `mask is None`, `hasattr(write_only)`) rather than an explicit config flag, keeping model code decoupled from cache backend details.
- Keep engine internals synchronous with a clean `step()` boundary for async API integration.
- Make continuous batching policy runtime-configurable and benchmark each policy.
- Use a custom `bench_serving.py` as the primary serving benchmark harness (our SSE format uses custom event types that don't match external tools' OpenAI assumptions).
- KV caching is always on — no toggle to disable it. The `kv_cache_backend` config selects the implementation (`"contiguous"` for Phase 3's pre-allocated cache, `"paged"` for Phase 6's block-allocated cache).
- Use `src/infer/` package layout to avoid import ambiguity with the repo name.

---

## 15. Open Questions

None currently. Re-open after Phase 5 policy benchmarking if additional scheduler policies are needed.

---

## 16. References

- [vLLM: Easy, Fast, and Cheap LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [vLLM source code](https://github.com/vllm-project/vllm)
- [SGLang source code](https://github.com/sgl-project/sglang)
