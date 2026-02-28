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

| Milestone                      | Phases | Outcome                                                     |
|--------------------------------|--------|-------------------------------------------------------------|
| M1 Core Inference              | 1a-3.1 | Single-request generation with KV cache + Triton kernels    |
| M2 Serving MVP                 | 4-5    | Multi-request serving with continuous batching + SSE        |
| M3 Advanced Memory/Scheduling  | 6-8    | Paged attention, chunked prefill, prefix caching            |
| M4 Execution Efficiency        | 9-10   | CUDA graphs for decode, weight quantization for larger models |
| M5 Advanced Decoding           | 11-12  | Speculative decoding, structured output                     |

Guardrails:

- Do not start M2 until Phase 3 correctness and speedup checks are complete.
- Do not start M3 until M2 benchmarks and API stability checks are complete.
- M3 features remain optional feature flags and must not regress M2 behavior when disabled.
- Do not start M4 until M3 benchmarks are complete.
- M4 and M5 features remain optional feature flags and must not regress M3 behavior when disabled.
- M5 phases (11-12) are independent of each other and can be implemented in either order.

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
- Eviction-aware decode block allocation (decode path evicts from prefix tree when allocator is exhausted).
- Full-hit optimization (single last-token forward pass when entire prefix is cached).

Exit criteria:

- TTFT improvement on shared-prefix workload.
- Correctness: identical output with/without prefix caching (greedy decode).
- Cache correctness tests for refcount + eviction edge cases.
- No regression when disabled.

**Status: COMPLETE.** All deliverables implemented, benchmarked on all three models. Prefix caching on the shared-prefix workload (48 requests, ~1024-token common prefix): Llama +11% throughput, -28% TTFT P50, -47% ITL P99; Qwen3 +7% throughput, -12% TTFT P50, -46% ITL P99; Gemma3 stable (1B model prefills too fast for caching to help). No regression on non-prefix workloads. 976 tests pass.

See `docs/PHASE_8.md` for the full design.

### Phase 9: CUDA Graphs

Goal:

- Eliminate CPU-side kernel launch overhead during decode by capturing the decode forward pass into a CUDA graph and replaying it each step.

Background:

During decode, each step executes the same sequence of GPU kernels with the same shapes — only the tensor contents change. Normally the CPU submits each kernel individually, and for small/fast decode kernels the GPU can finish before the CPU has submitted the next one, leaving the GPU idle between launches. A Llama-3B decode step has ~100+ kernel launches at 5-15us each, adding 0.5-1.5ms of pure CPU overhead per step against ~10-15ms of GPU compute. CUDA graphs capture the full kernel sequence once, then replay it with a single CPU call, eliminating the gaps.

Deliverables:

- `CUDAGraphRunner` wrapper that captures and replays the decode forward pass.
- Pre-allocated placeholder tensors for all graph inputs/outputs (fixed memory addresses required by CUDA graphs).
- `GraphPagedDecodeCacheView` with static GPU tensors for page tables, sequence lengths, and KV write indices.
- Graph pool keyed by batch size — power-of-2 buckets (1, 2, 4, 8, 16, 32) pre-captured at startup, with padding to nearest bucket.
- Model forward change: skip mask creation for decode when `padding_mask=None`, enabling Triton paged attention dispatch and graph capture.
- Triton paged attention kernel extended with `window_size` parameter for Gemma 3 sliding-window layers.
- Integration with `ContinuousRunner`: intercept only the decode path (prefill stays eager since shapes vary per request).
- Eager warmup at server startup to pre-record graphs for all power-of-2 batch sizes up to `max_batch_size`. Batch sizes exceeding the largest bucket fall back to eager mode.
- `use_cuda_graphs: bool` flag in `EngineConfig` (default `False`).

Constraints:

- All tensors touched by the graph must remain at fixed addresses between replays. Intermediate activation buffers must be pre-allocated, not dynamically created.
- No Python-level control flow inside the captured region. The decode path must be branch-free (already the case — decode is always `seq_len == 1` per sequence).
- Paged attention compatibility: the page table tensor stays at a fixed address; its contents are updated before replay.
- KV cache writes must land in the same backing storage (already satisfied by paged/slotted pools).

Exit criteria:

- Measurable decode throughput improvement (expect 5-15%) on baseline and continuous_batching workloads.
- No correctness regression: greedy decode output identical with and without CUDA graphs.
- No regression when disabled.
- Graph memory overhead documented (VRAM for placeholder tensors + graph capture).

**Status: COMPLETE (not recommended for use).** All deliverables implemented and benchmarked on all three models. CUDA graph capture/replay is functionally correct, but profiling reveals that Triton-compiled kernels replay 75--145% slower inside CUDA graphs than they execute eagerly, resulting in a net throughput regression across all workloads (e.g. Llama baseline -38%, Gemma3 baseline -42%). The implementation and `--cuda-graphs` flag are kept for reference but should not be enabled for performance. See `docs/PHASE_9.md` for design and `benchmarks/log/SERVING_LOG.md` Phase 9 for profiling details. 1000 tests pass.

See `docs/PHASE_9.md` for the full design.

### Phase 10: Weight Quantization (FP8 + INT8)

Goal:

- Support weight-only quantization to reduce memory footprint, enabling 8B-class models on 16 GB VRAM and improving memory-bandwidth-bound decode throughput. Two formats are supported:
  - **FP8**: Block-wise float8_e4m3fn quantization. Target: `Qwen/Qwen3-8B-FP8`.
  - **INT8**: Per-channel symmetric int8 quantization. Target: `nytopop/Qwen3-8B.w8a8`.

Background:

Unquantized bf16 weights for an 8B model require ~16 GB, leaving no room for KV cache on a 16 GB card. Quantization halves weight memory to ~8 GB, making 8B models practical. Decode throughput is memory-bandwidth-bound (reading weights dominates), so smaller weights directly translate to faster decode. Both FP8 and INT8 achieve approximately 2x memory reduction (1 byte per element vs 2 bytes for bf16), with negligible scale tensor overhead.

FP8 uses block-wise quantization (128x128 blocks with per-block scale factors) in the DeepSeek-V3 format. INT8 uses per-channel symmetric quantization (one scale per output channel) in the compressed-tensors format, which is simpler but equally effective for weight-only dequantization.

Deliverables:

- Quantization-aware weight loader: detect quantization format from `quantization_config` in config.json. Support both FP8 (`quant_method: "fp8"`) and INT8 (`quant_method: "compressed-tensors"` with int weights).
- `FP8Linear` module for block-wise FP8: stores float8_e4m3fn weight and per-block float32 scale tensor (`weight_scale_inv`). Forward pass dequantizes to bf16 and computes matmul.
- `INT8Linear` module for per-channel INT8: stores int8 weight and per-channel float32 scale tensor (`weight_scale`). Forward pass dequantizes to bf16 and computes matmul.
- Model surgery functions: `replace_linear_with_fp8` and `replace_linear_with_int8` to swap `nn.Linear` modules with quantized versions.
- Per-architecture support: quantize all linear layers in attention (Q/K/V/O projections) and MLP (gate/up/down projections). Norms, embeddings, and LM head stay in bf16.
- `quantization: str | None` config field in `EngineConfig` (`None`, `"fp8"`, `"int8"`). Auto-detected from checkpoint when `None`.
- Weight map extension: map scale tensors alongside weight tensors for each quantized linear layer (`weight_scale_inv` for FP8, `weight_scale` for INT8).

Constraints:

- Generation quality must be verified: quantized models produce coherent, reasonable output on standard prompts.
- FP8: block-wise quantization (128x128) must be supported — this is the format used by Qwen/Qwen3-8B-FP8.
- INT8: per-channel symmetric quantization must be supported — this is the format used by nytopop/Qwen3-8B.w8a8.
- Non-quantized layers (embeddings, norms, LM head) remain in bf16.
- All existing tests must pass with `quantization=None` (no regression).

Exit criteria:

- Successfully load and serve `Qwen/Qwen3-8B-FP8` on 16 GB VRAM.
- Successfully load and serve `nytopop/Qwen3-8B.w8a8` on 16 GB VRAM.
- Generation quality sanity check: both quantized models produce coherent, reasonable output on standard prompts.
- All existing tests pass with `quantization=None` (no regression).
- Unit tests for both FP8 and INT8 dequantization, model surgery, and weight maps.

### Phase 11: Speculative Decoding

Goal:

- Accelerate decode by using a smaller draft model to propose multiple tokens per step, verified in a single forward pass of the target model.

Background:

Autoregressive decode is sequential — one forward pass per token. Speculative decoding breaks this by running a fast draft model (e.g. Llama-1B) to generate K candidate tokens, then verifying all K in one forward pass of the target model (e.g. Llama-3B). If the target model agrees with N of the K candidates, you get N+1 tokens for the cost of one target forward pass plus K cheap draft passes. At high acceptance rates (70-90% for well-matched draft/target pairs), this yields 2-3x effective decode speedup without any change to output distribution.

Deliverables:

- `DraftTargetRunner` that orchestrates the draft-then-verify loop.
- Draft model loading: load a second, smaller model alongside the target model. Both models share the same tokenizer.
- Draft generation loop: run the draft model autoregressively for K steps (configurable `spec_length`, default 5), producing K candidate token IDs and their log-probabilities.
- Target verification pass: run the target model on all K candidates in a single forward pass, compare target log-probabilities against draft log-probabilities to determine acceptance.
- Rejection sampling: use the standard speculative decoding acceptance criterion (accept token i if `target_prob[i] >= draft_prob[i]`, otherwise accept with probability `target_prob[i] / draft_prob[i]`) to guarantee the output distribution is identical to pure target-model sampling.
- KV cache management: draft model gets its own lightweight KV cache. On rejection, roll back both draft and target KV caches to the last accepted position.
- Integration with continuous batching: speculative decoding applies per-sequence within the existing scheduler. Sequences using speculation coexist with non-speculative sequences in the same batch.
- `use_speculative_decoding: bool` and `draft_model: str | None` fields in `EngineConfig`.

Constraints:

- Output distribution must be provably identical to non-speculative decoding (this is the key property of speculative decoding — it's lossless).
- Draft model must share the same tokenizer/vocabulary as the target model.
- VRAM budget must account for both models: draft model weights + draft KV cache + target model weights + target KV cache. This favors small draft models (1B class).
- Greedy decode shortcut: under greedy sampling (`temperature=0`), acceptance simplifies to exact token match (no rejection sampling needed).

Exit criteria:

- Measurable tokens-per-second improvement on single-request decode (expect 1.5-2.5x depending on draft/target pair and acceptance rate).
- Correctness: output is statistically identical to non-speculative sampling (verified via greedy decode exact match and sampling distribution tests with fixed seeds).
- Acceptance rate logging per request for tuning `spec_length`.
- No regression when disabled.
- Benchmark comparing speculative vs non-speculative decode on baseline workload.

### Phase 12: Structured Output

Goal:

- Constrain token generation to follow a schema (JSON schema or regex), enabling reliable structured output without post-hoc parsing or retries.

Background:

LLMs generate free-form text, but many applications need structured output (JSON objects, enum values, function call arguments). Structured output works by masking logits at each decode step: before sampling, set the logits of all tokens that would violate the grammar to `-inf`, so only valid continuations can be chosen. This requires tracking the current state in the grammar and computing which tokens are valid next — essentially intersecting the LLM's vocabulary with the set of strings accepted by the grammar from the current position.

Deliverables:

- Grammar representation: compile a JSON schema or regex pattern into a finite-state machine (FSM) where each state maps to a set of allowed token IDs.
- `FSMCompiler` that takes a JSON schema or regex and produces an `FSM` with:
  - `initial_state`: starting state.
  - `allowed_tokens(state) -> set[int]`: valid token IDs from the current state.
  - `next_state(state, token_id) -> state`: advance the FSM after a token is accepted.
  - `is_terminal(state) -> bool`: whether the current state is a valid completion point.
- Vocabulary pre-processing: for each FSM state, pre-compute the set of allowed token IDs by testing every vocabulary entry against the grammar transitions. Cache this mapping (state -> token mask) since it's expensive to compute but reusable across requests with the same schema.
- `LogitMaskProcessor` that plugs into the sampler pipeline: given the current FSM state, applies `-inf` mask to disallowed tokens before temperature/top-k/top-p.
- Per-request FSM state tracking: each `Request` carries its own FSM state, advanced after each token is sampled.
- API extension: add `response_format` field to the completions request (`{"type": "json_schema", "schema": {...}}` or `{"type": "regex", "pattern": "..."}`).
- Support for common JSON schema features: object, array, string, number, boolean, null, enum, required fields, nested objects. Does not need to cover the full JSON Schema spec.

Constraints:

- Token healing: some tokens span grammar boundaries (e.g. `"}` includes both a string close and object close). The FSM must handle multi-character tokens correctly by checking whether the token's decoded string is a valid prefix or completion of the expected grammar production.
- Performance: logit masking must not add significant per-token latency. Pre-computed state-to-mask lookup should be O(1) per token. FSM compilation can be slower (one-time cost per unique schema).
- Unconstrained mode must have zero overhead (no masking when `response_format` is not set).

Exit criteria:

- JSON schema mode produces valid JSON matching the schema for 100% of test cases (on a suite of at least 20 diverse schemas).
- Regex mode produces strings matching the pattern for 100% of test cases.
- No measurable throughput regression when structured output is not requested.
- Correctness tests for edge cases: empty objects, nested arrays, enum constraints, numeric ranges.
- Benchmark comparing unconstrained vs structured output generation speed.

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

    # CUDA graphs (Phase 9)
    use_cuda_graphs: bool = False

    # Quantization (Phase 10)
    quantization: str | None = None    # None | "fp8" | "int8"

    # Speculative decoding (Phase 11)
    use_speculative_decoding: bool = False
    draft_model: str | None = None
    spec_length: int = 5               # candidate tokens per speculation step

    # Attention backend (Triton paged kernel auto-dispatches for paged decode)
    attention_backend: str = "sdpa"    # "naive" | "sdpa" | "flash"
```

Compatibility rules to enforce in config validation:

- `use_chunked_prefill=True` requires `batching_mode="continuous"`.
- `use_prefix_caching=True` requires `kv_cache_backend="paged"` and `use_chunked_prefill=True`.
- `use_cuda_graphs=True` requires `batching_mode="continuous"` and `kv_cache_backend="paged"`.
- `use_speculative_decoding=True` requires `draft_model` to be set.
- `quantization` must be `None`, `"fp8"`, or `"int8"`.

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
- M4 gate: CUDA graph decode functional correctness (note: not a performance win due to Triton/graph incompatibility) + quantized model loads and generates (both FP8 and INT8) + no M3 regressions when disabled.
- M5 gate: speculative decode distribution correctness + structured output schema compliance + no M4 regressions when disabled.

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
│       │   ├── cuda_graph_runner.py
│       │   ├── speculative_runner.py
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
│       │   ├── paged_attention.py
│       │   └── fp8_dequant.py
│       ├── quant/
│       │   ├── __init__.py
│       │   ├── fp8_linear.py
│       │   └── int8_linear.py
│       ├── grammar/
│       │   ├── __init__.py
│       │   ├── fsm.py
│       │   └── compiler.py
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
    ├── PHASE_7.md
    ├── PHASE_8.md
    ├── PHASE_9.md
    ├── PHASE_10.md
    ├── PHASE_11.md
    └── PHASE_12.md
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

- ~~Phase 9: Should CUDA graphs be recorded lazily (on first encounter of each batch size) or eagerly (pre-record a fixed set at startup)?~~ **Resolved in Phase 9 design:** Eager warmup at startup for power-of-2 batch sizes (1, 2, 4, 8, 16, 32). This avoids first-encounter latency during serving while keeping the number of captured graphs bounded. Batch sizes exceeding the largest bucket fall back to eager mode. See `docs/PHASE_9.md` for details.
- ~~Phase 10: Prioritize pre-quantized checkpoint loading (AWQ/GPTQ) or on-the-fly quantization from bf16?~~ **Resolved:** Load pre-quantized checkpoints. FP8: block-wise FP8 with 128x128 blocks and per-block `weight_scale_inv` (Qwen/Qwen3-8B-FP8). INT8: per-channel symmetric INT8 with per-row `weight_scale` (nytopop/Qwen3-8B.w8a8, compressed-tensors format). No on-the-fly quantization needed.
- Phase 11: What draft/target model pairs to benchmark? Llama-1B/Llama-3B is the natural pair, but Qwen3-1.7B/Qwen3-4B is also viable. Need to verify tokenizer compatibility.
- Phase 12: Build the FSM compiler from scratch or use an existing library (e.g. `outlines-core`)? Building from scratch is more educational but significantly more work.

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
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [Efficient Guided Generation for Large Language Models (Outlines)](https://arxiv.org/abs/2307.09702)
- [CUDA Graphs (NVIDIA docs)](https://developer.nvidia.com/blog/cuda-graphs/)
