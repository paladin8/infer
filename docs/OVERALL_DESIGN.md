# infer: An Educational LLM Inference Runtime

## 1. Overview

`infer` is an LLM inference runtime built from scratch on top of PyTorch. The goal is to understand how systems like vLLM, TGI, and SGLang work by building one incrementally, starting from "load weights and produce one token" and working up to continuous batching, paged attention, and speculative decoding.

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

Phases 6-9 are advanced extensions, not required for v1.

---

## 2. Scope and Milestone Boundaries

To keep implementation tractable, phases are grouped into milestones:

| Milestone | Phases | Outcome |
|----------|--------|---------|
| M1 Core Inference | 1a-3 | Single-request generation with KV cache |
| M2 Serving MVP | 4-5 | Multi-request serving with continuous batching + SSE |
| M3 Advanced Memory/Scheduling | 6-9 | Paged attention, speculative decoding, prefix caching, CPU offload preemption |

Guardrails:

- Do not start M2 until Phase 3 correctness and speedup checks are complete.
- Do not start M3 until M2 benchmarks and API stability checks are complete.
- M3 features remain optional feature flags and must not regress M2 behavior when disabled.

---

## 3. Environment and Assumptions

### Baseline environment

- OS: Linux (x86_64)
- Python: 3.11+
- PyTorch: 2.4+ with CUDA-enabled build
- CUDA runtime: 12.x
- GPU: NVIDIA with bf16 support preferred (fp16 fallback allowed)

### Hardware tiers

- Dev tier: 16 GB VRAM (primary development target, RTX 5080 class)
- Stretch tier: 24 GB+ VRAM (for larger experiments and more aggressive concurrency)

### Model assumptions

- Default dev model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Practical unquantized target range on dev hardware: 1B-3B class models.
- Default speculative pair target: ~3B target with ~1B draft (or smaller), chosen based on fit and stability on 16 GB.
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
│  - Admission control / preemption               │
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
│  - Tokenization  │ │  - Eviction/preemption   │
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

Error/preemption states for advanced phases:

`FAILED`, `PREEMPTED`

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
class GenerationRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    arrival_time_s: float

@dataclass
class StepOutput:
    request_id: str
    token_id: int | None
    text_delta: str
    finished: bool
    error: str | None = None
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

- Load HF weights and verify a single transformer layer produces correct activations.

Deliverables:

- Safetensors loader with shard/index handling.
- HF `config.json` reader and weight name mapping.
- `AutoTokenizer` wrapper (use HF `transformers` for tokenization).
- Single transformer block implementation (RMSNorm, RoPE, GQA, SwiGLU).

Exit criteria:

- Single-layer activation parity test against `transformers` with max absolute error threshold documented by dtype.
- Unit tests for loader (single-file and sharded checkpoints) and tokenizer wrapper.

### Phase 1b: Full Model and Logits Parity

Goal:

- Assemble the full Llama model and verify end-to-end logits match `transformers`.

Deliverables:

- Complete Llama `nn.Module` (stack all layers, final norm, LM head).
- Jinja2-based chat template renderer with custom templates per model (no `transformers.apply_chat_template`).
- Layer-by-layer activation diff tooling for debugging mismatches.

Exit criteria:

- Full-model logits parity test against `transformers` with max absolute error threshold documented by dtype.
- At least one end-to-end test: load model, tokenize prompt, forward pass, verify logits.

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

### Phase 4: Static Batching and SSE API

Goal:

- Serve multiple concurrent requests in fixed batches with a streaming API from the start.

Deliverables:

- Request queue and batch builder.
- Padded/varlen attention support for mixed prompt lengths.
- FastAPI endpoint (`POST /v1/completions`) with SSE streaming (the only API surface — no separate non-streaming path).
- HTTP error responses for invalid requests (400 for bad params, 422 for unsupported fields, 503 when queue is full).
- Early `genai-perf` validation: confirm the SSE streaming format is compatible and run a baseline benchmark.

Exit criteria:

- Throughput under concurrent load exceeds sequential serving baseline.
- Regression tests cover mixed sequence lengths and EOS handling in batch.
- `genai-perf` can drive the API endpoint and produce a valid report.

Note: Confirm `genai-perf` SSE compatibility early in this phase. If incompatible, fall back to a simple custom benchmark script (concurrent requests measuring TTFT/ITL) for M2 and revisit `genai-perf` integration later.

### Phase 5: Continuous Batching (v1 completion target)

Goal:

- Iteration-level scheduling with per-step admit/retire.

Deliverables:

- Continuous scheduler (admit/retire each step).
- Per-sequence lifecycle tracking.
- Configurable scheduling policy selection (`fcfs`, `prefill_first`, `decode_round_robin`).

Exit criteria:

- Tail latency improvement vs static batching on mixed-length workload.
- No starvation in stress test (bounded wait for queued requests).
- Policy sweep benchmarks are recorded for each scheduling policy.
- v1 Definition of Done satisfied.

### Phase 6: Paged Attention

Goal:

- Replace contiguous per-sequence allocation with paged KV memory.

Deliverables:

- Block allocator + page tables.
- Gather-based paged attention path.
- Optional Triton gather kernel.

Exit criteria:

- Higher max concurrent sequence capacity than contiguous mode at same VRAM limit.
- Correctness tests pass with randomized page mappings.

### Phase 7: Speculative Decoding

Goal:

- Draft/verify loop for faster decode.

Deliverables:

- Draft model integration.
- Acceptance/rejection logic with distribution-preserving algorithm.
- Scheduler integration for variable verification step sizes.

Exit criteria:

- Throughput gain measured on compatible draft/target pair.
- Distribution correctness checks for sampled outputs (statistical tests).

### Phase 8: Prefix Caching

Goal:

- Reuse shared prefill prefixes.

Deliverables:

- Radix/prefix tree keyed by token ids.
- Block refcounting and eviction policy.
- Scheduler hook for prefix-hit short-circuit in prefill.

Exit criteria:

- TTFT improvement on shared-prefix workload.
- Cache correctness tests for refcount + eviction edge cases.

### Phase 9: CPU Offload Preemption (Advanced)

Goal:

- Keep high-priority requests making progress under KV pressure by offloading preempted sequence state to CPU memory and later restoring it.

Deliverables:

- Preemption mode supporting CPU offload + resume.
- Host-pinned memory buffers for offloaded KV/state payloads.
- Scheduler policy hooks to choose victims and re-admit offloaded sequences.

Exit criteria:

- Under memory pressure, service remains live without hard OOM for tested workloads.
- Resumed requests produce equivalent outputs to non-preempted execution under deterministic settings.
- Benchmark includes throughput/latency impact of offload preemption vs recompute preemption.

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

    # KV cache
    use_kv_cache: bool = True
    use_paged_attention: bool = False
    block_size: int = 16

    # Batching
    batching_mode: str = "continuous"  # "none" | "static" | "continuous"
    scheduler_policy: str = "fcfs"     # "fcfs" | "prefill_first" | "decode_round_robin"

    # Speculative decoding
    use_speculative: bool = False
    draft_model: str | None = None
    spec_tokens: int = 5

    # Prefix caching
    use_prefix_caching: bool = False

    # Preemption
    preemption_mode: str = "recompute"  # "none" | "recompute" | "cpu_offload"

    # Attention backend
    attention_backend: str = "sdpa"    # "naive" | "sdpa" | "flash" | "triton_paged"
```

Compatibility rules to enforce in config validation:

- `use_paged_attention=True` requires `use_kv_cache=True`.
- `use_speculative=True` requires `draft_model` and batching mode not equal to `"none"`.
- `use_prefix_caching=True` requires paged attention mode.
- `preemption_mode="cpu_offload"` requires `batching_mode="continuous"` and paged attention mode.

Benchmark matrix baseline:

| Config | KV Cache | Batching | Paged | Prefix | Speculative | Preemption |
|-------|----------|----------|-------|--------|-------------|------------|
| Baseline | off | none | off | off | off | none |
| +KV Cache | on | none | off | off | off | none |
| +Static Batch | on | static | off | off | off | none |
| +Continuous | on | continuous | off | off | off | recompute |
| +Paged | on | continuous | on | off | off | recompute |
| +Prefix | on | continuous | on | on | off | recompute |
| +Speculative | on | continuous | on | on | on | recompute |
| +CPU Offload | on | continuous | on | on | on | cpu_offload |

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
  - Decode-active sequences must receive service at least once every `N` scheduler iterations unless preempted.
- Admission control checks token budget and KV budget before enqueueing into active set.

### 9.4 Engine async-readiness

- Engine remains synchronous internally.
- API integration runs `step()` in a dedicated loop/task.
- All request outputs pass through per-request queues/channels to isolate API concerns from runtime logic.

---

## 10. Metrics and Benchmarking

Use NVIDIA `genai-perf` as the primary harness and add internal counters from the engine.

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
  - OOM/preemption paths (advanced phases), including CPU offload + resume behavior

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
│       ├── models/
│       │   ├── registry.py
│       │   ├── llama.py
│       │   └── common.py
│       ├── loader/
│       │   ├── weights.py
│       │   ├── tokenizer.py
│       │   └── chat_template.py
│       ├── engine/
│       │   ├── config.py
│       │   ├── engine.py
│       │   ├── runner.py
│       │   ├── scheduler.py
│       │   └── sampler.py
│       ├── cache/
│       │   ├── simple.py
│       │   ├── paged.py
│       │   ├── prefix.py
│       │   └── offload.py
│       ├── kernels/
│       │   └── paged_attention.py
│       ├── speculative/
│       │   └── spec_decode.py
│       └── server/
│           └── api.py
├── benchmarks/
│   ├── run_matrix.py
│   ├── workloads/
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── stress/
└── docs/
    └── OVERALL_DESIGN.md
```

---

## 14. Design Decisions Log

- Primary dev hardware target is 16 GB VRAM; unquantized model work should stay mostly in the 1B-3B range.
- Start with custom model code, use `transformers` only for `AutoTokenizer` and parity checks.
- Use `jinja2` for chat template rendering with our own template strings per model (no `transformers.apply_chat_template`, no custom parser).
- Use SDPA as default fast path before Triton specialization.
- Scope Triton work to paged gather kernel first.
- Keep engine internals synchronous with a clean `step()` boundary for async API integration.
- Make continuous batching policy runtime-configurable and benchmark each policy.
- Add CPU offload preemption as a late advanced feature (Phase 9), after paged attention/prefix work.
- Use `genai-perf` as primary benchmark harness; validate SSE compatibility in Phase 4 with a simple custom script as fallback.
- Use `src/infer/` package layout to avoid import ambiguity with the repo name.

---

## 15. Open Questions

None currently. Re-open after Phase 5 policy benchmarking if additional scheduler policies are needed.

---

## 16. References

- [vLLM: Easy, Fast, and Cheap LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [vLLM source code](https://github.com/vllm-project/vllm)
- [SGLang source code](https://github.com/sgl-project/sglang)
