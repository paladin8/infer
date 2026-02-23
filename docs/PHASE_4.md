# Phase 4: Static Batching and SSE API

## Goal

Serve multiple concurrent requests in fixed batches with a streaming API. This is the first phase where `infer` becomes a server — clients send completion requests over HTTP and receive tokens as SSE events in real time.

Phase 4 introduces five areas of work: model changes for padded batching, a request data model and scheduler, an engine that orchestrates batched forward passes, and a FastAPI server exposing the `POST /v1/completions` endpoint. The engine remains synchronous internally; the API layer runs the engine step loop in a dedicated asyncio task.

Static batching means: a batch is formed from queued requests, all requests in the batch are right-padded to a uniform prompt length and run together from prefill through completion, and no new requests are admitted until the entire batch finishes. Phase 5 replaces this with continuous batching (per-step admit/retire).

Dev models: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.

---

## Architecture

```text
                  HTTP POST /v1/completions
                             │
                             ▼
                ┌─────────────────────────┐
                │      FastAPI Server     │
                │  (api.py)               │
                │  - Validates request    │
                │  - Enqueues request     │
                │  - SSE streams output   │
                └────────────┬────────────┘
                             │ asyncio.Queue per request
                             ▼
                ┌─────────────────────────┐
                │      Engine Loop        │
                │  (engine.py)            │
                │  - Runs in background   │
                │    asyncio task         │
                │  - Calls step()         │
                └────────────┬────────────┘
                             │
                             ▼
              ┌───────────────────────────────┐
              │         Engine.step()         │
              │  1. Scheduler builds batch    │
              │  2. Runner executes batched   │
              │     forward pass              │
              │  3. Sampler produces tokens   │
              │     per request               │
              │  4. Token deltas → queues     │
              └───────────────────────────────┘
```

The engine step loop is a `while True` loop in a background asyncio task. Each iteration calls `Engine.step()`, which is synchronous and compute-bound. Between steps, `await asyncio.sleep(0)` yields to the event loop so the API can process new connections and send pending SSE events.

All forward passes within a batch are batched into a single model call. Right-padding with attention masks handles mixed prompt lengths within a batch.

---

## Deliverables

### 1. Request data model (`src/infer/engine/request.py`)

A `Request` tracks the lifecycle of a single completion request from arrival through generation to completion.

```python
import asyncio
from dataclasses import dataclass, field
from enum import Enum

import torch

from infer.engine.sampler import SamplingParams


class RequestState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    FAILED = "failed"


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

    # Per-request RNG — created from sampling_params.seed at enqueue time.
    # None means non-deterministic sampling.
    generator: torch.Generator | None = None

    # Output channel — set by the server when the request is enqueued.
    output_queue: asyncio.Queue[StepOutput] | None = None
```

KV cache is not stored on the `Request` — it is allocated per batch (see section 5). The `generator` field is created from `sampling_params.seed` when the request is enqueued by the engine:

```python
if params.seed is not None:
    req.generator = torch.Generator(device=device)
    req.generator.manual_seed(params.seed)
```

State machine transitions:

```text
WAITING ──(added to batch)──> PREFILL ──(prefill done)──> DECODE ──(EOS/stop/length)──> FINISHED
   │                            │                           │
   └──────────────(error)──>  FAILED  <───(error)───────────┘
```

`StepOutput` carries per-step results from the engine to the API layer:

```python
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

The field is named `text_delta` to match the overall design's `StepOutput` definition. The `prompt_tokens` and `completion_tokens` fields extend the overall design's interface for usage reporting in the SSE `done` event.

### 2. Engine configuration (`src/infer/engine/config.py`)

```python
@dataclass
class EngineConfig:
    model: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_seq_len: int = 4096
    max_batch_size: int = 8
    max_waiting_requests: int = 64
    seed: int | None = None

    # Batching and scheduling — fixed for Phase 4, extended in Phase 5.
    batching_mode: str = "static"        # "static" | "continuous"
    scheduler_policy: str = "fcfs"       # "fcfs" (Phase 5 adds more)
    batch_wait_timeout_s: float = 0.05   # max seconds to wait for batch to fill

    # KV cache backend — Phase 6 adds "paged".
    kv_cache_backend: str = "contiguous"  # "contiguous" | "paged"
```

`max_batch_size` controls how many requests are grouped into a static batch. `max_waiting_requests` controls admission — when the waiting queue exceeds this limit, new requests get 503. `max_seq_len` is the maximum total sequence length (prompt + generation) for any single request; it sizes the KV cache allocation per batch. `batch_wait_timeout_s` controls how long the scheduler waits for the batch to fill before dispatching (see section 3).

KV caching is always on — there is no toggle to disable it. The `kv_cache_backend` field selects the cache implementation: `"contiguous"` (Phase 3's pre-allocated `KVCache`) is the only supported backend in Phase 4; Phase 6 adds `"paged"` for block-allocated paged attention. The engine raises `ValueError` on unsupported values.

The `batching_mode` and `scheduler_policy` fields exist for Phase 5 extensibility. In Phase 4, `batching_mode` must be `"static"` — the engine raises `ValueError` on unsupported values. The overall design defaults `max_batch_size` to 32; we use 8 for Phase 4 to fit within 16 GB dev-tier VRAM with per-batch contiguous caches (see memory budget below).

### 3. Scheduler (`src/infer/engine/scheduler.py`)

The static-batching scheduler forms batches from the waiting queue with a bounded wait: when no batch is active and at least one request is waiting, the scheduler waits up to `batch_wait_timeout_s` (default 50ms) for the batch to fill to `max_batch_size`, then dispatches whatever is queued. The batch runs until all requests finish. Then the next batch is formed.

The bounded wait balances latency and throughput. Without it, a burst of requests arriving milliseconds apart would be split across multiple serial batches instead of being coalesced into one. The 50ms default is short enough that single-request latency is barely affected (~1 decode step worth of delay) but long enough to capture concurrent arrivals from a typical load pattern.

```python
class StaticScheduler:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self._batch_wait_deadline: float | None = None  # set when first request arrives

    def add_request(self, request: Request) -> bool:
        """Add a request to the waiting queue.

        Returns False if the queue is full (caller should return 503).
        """

    def schedule(self) -> list[Request]:
        """Return the current active batch, starting a new batch if needed.

        Called at the start of each engine step. Behavior:

        1. If ``self.active`` has unfinished requests, return it unchanged.
        2. If all active requests are finished, clear the active list.
        3. If the waiting queue is non-empty and no batch is forming, start
           the wait timer (``batch_wait_deadline = now + batch_wait_timeout_s``).
        4. If the waiting queue has ``max_batch_size`` requests OR the wait
           timer has expired, pull requests into ``self.active`` and dispatch.
        5. Otherwise return empty (still waiting for the batch to fill).
        """

    def has_work(self) -> bool:
        """True if there are active requests or waiting requests."""
```

Requests in the waiting queue are ordered FCFS by insertion order (the `deque` preserves `add_request` call order). No priority, no preemption. Phase 5 adds per-step scheduling.

The engine loop integrates with the bounded wait naturally: when `schedule()` returns an empty list because the wait timer hasn't expired, the loop yields to the event loop (`await asyncio.sleep(0)`) and calls `schedule()` again on the next iteration. The timer is checked inside `schedule()` using `time.perf_counter()`.

### 4. Model changes for padded batching

To support batched forward passes with mixed prompt lengths, the model forward methods gain an optional `padding_mask` parameter. This is the only model-level change in Phase 4.

**New signature** (all three models — `LlamaModel`, `Qwen3Model`, `Gemma3Model`):

```python
def forward(
    self,
    input_ids: Tensor,
    kv_cache: KVCache | None = None,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Forward pass.

    Args:
        input_ids: Token IDs, shape ``[batch, seq_len]``.
        kv_cache: Optional KV cache for incremental decoding.
        padding_mask: Optional boolean mask, shape ``[batch, total_kv_len]``.
            ``True`` for real token positions, ``False`` for padding positions.
            When provided, the model combines this with its internal causal
            (and sliding-window) masks to exclude padding from attention.
            When ``None``, behavior is identical to Phase 3 (no padding).

    Returns:
        Logits.  Without padding_mask or without cache: ``[batch, seq_len, vocab_size]``.
        With cache and no padding_mask: ``[batch, 1, vocab_size]`` (last-position optimization).
        With cache and padding_mask: ``[batch, seq_len, vocab_size]`` during prefill
        (caller gathers per-sequence logits), ``[batch, 1, vocab_size]`` during decode.
    """
```

**Backward compatibility**: when `padding_mask is None`, the model constructs masks exactly as before (Phase 3 behavior). All existing tests pass without modification. The `padding_mask` parameter is only used by the engine's batched runner.

**Right-padding and positional correctness**: the runner right-pads prompts (appends PAD tokens after real tokens). This means real tokens naturally occupy positions `[0, prompt_len)` and get correct RoPE positional encodings. No custom `position_ids` are needed — the existing `cos[pos : pos + seq_len]` slicing works because all sequences share the same position range `[0, max_prompt_len)`.

**Mask construction for Llama/Qwen3** (inside `forward`):

```python
if padding_mask is not None and kv_cache is not None:
    kv_len = kv_cache.seq_len + seq_len  # total KV length after this step
    # Start from the internal mask (causal for prefill, None for decode).
    if seq_len > 1:
        # Prefill: causal mask [1, 1, seq_len, seq_len] → expand to [batch, ...]
        mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)
        mask = mask.expand(batch_size, -1, -1, -1).clone()
    else:
        # Decode: [batch, 1, 1, kv_len] initialized to zero (attend to all)
        mask = torch.zeros(batch_size, 1, 1, kv_len, dtype=x.dtype, device=x.device)
    # Apply padding: set -inf where padding_mask is False
    pad_mask = ~padding_mask[:, None, None, :kv_len]  # [batch, 1, 1, kv_len]
    mask.masked_fill_(pad_mask, float("-inf"))
```

**Mask construction for Gemma3** (same pattern, applied to both `local_mask` and `global_mask`):

```python
if padding_mask is not None and kv_cache is not None:
    kv_len = kv_cache.seq_len + seq_len
    if seq_len > 1:
        # Prefill: start from sliding/causal masks, expand + add padding
        local_mask = sliding_window_causal_mask(...).expand(batch_size, ...).clone()
        global_mask = causal_mask(...).expand(batch_size, ...).clone()
    else:
        # Decode: start from sliding-window mask, add padding
        # Global: zeros of [batch, 1, 1, kv_len], then mask padding
        # Local: zeros + sliding-window cutoff, then mask padding
        ...
    pad_mask = ~padding_mask[:, None, None, :kv_len]
    local_mask.masked_fill_(pad_mask, float("-inf"))
    global_mask.masked_fill_(pad_mask, float("-inf"))
```

The model's internal mask logic (causal, sliding window) is preserved and augmented with padding awareness. The `padding_mask` is combined after the internal masks are constructed, so the architecture-specific mask patterns (Gemma3's sliding window vs global per layer) remain untouched.

**Last-position optimization with padding**: during prefill with `padding_mask` (batched mode), the last-position optimization (`x = x[:, -1:, :]`) is skipped because the last position in the padded sequence may be a padding position for shorter sequences. The model returns full `[batch, seq_len, vocab_size]` logits, and the runner gathers the correct last-real-token logits per sequence. During decode (`seq_len == 1`), the optimization is a no-op. During non-batched prefill (`padding_mask is None`), the optimization is applied as before.

```python
if kv_cache is not None:
    kv_cache.advance(seq_len)
    if padding_mask is None:
        x = x[:, -1:, :]  # single-request optimization (Phase 3 behavior)
    # else: return full logits for batched prefill; caller gathers.
x = self.norm(x)
return self.lm_head(x)
```

**Transformer block and Attention are unchanged**: the `mask` parameter is already passed through to `F.scaled_dot_product_attention`, which handles `[batch, 1, q_len, kv_len]` masks natively. Only the model-level forward methods change.

### 5. Model runner (`src/infer/engine/runner.py`)

The runner executes batched forward passes for a batch of requests. It manages right-padding, mask construction, KV cache allocation, and per-request sampling.

```python
class ModelRunner:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        config: EngineConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]

        # Active batch state (set during prefill, cleared when batch completes).
        self._kv_cache: KVCache | None = None
        self._prompt_lens: list[int] = []     # real prompt length per request
        self._max_prompt_len: int = 0
        self._padding_mask: Tensor | None = None  # [batch, max_total_seq_len]
        self._prev_text_lens: list[int] = []  # for incremental text_delta tracking

    @torch.inference_mode()
    def prefill(self, requests: list[Request]) -> list[StepOutput]:
        """Run batched prefill for a batch of requests.

        Right-pads all prompts to the longest prompt length, allocates a
        single batched KV cache, runs one forward pass, and samples the
        first token per request.

        Returns one StepOutput per request.
        """

    @torch.inference_mode()
    def decode_step(self, requests: list[Request]) -> list[StepOutput]:
        """Run one batched decode step for active requests.

        Feeds each request's last generated token (or a dummy token for
        finished requests) as a [batch, 1] input, runs one forward pass,
        and samples the next token per active request.

        Returns one StepOutput per request.
        """

    def clear_batch(self) -> None:
        """Free batch-level state (KV cache, masks)."""
        self._kv_cache = None
        self._prompt_lens = []
        self._max_prompt_len = 0
        self._padding_mask = None
```

**Batching strategy: right-padded batched forward passes**

All requests in a batch are right-padded to the longest prompt length and processed in a single model forward call. This gives GPU parallelism across the batch — the GPU processes all requests simultaneously, not one at a time.

Right-padding (appending PAD tokens after real content) is chosen over left-padding because real tokens naturally start at position 0 and get correct RoPE positional encodings without custom `position_ids`. The tradeoff is that logit extraction during prefill requires per-sequence gathering (the last real token is at a different position per sequence), but this is a trivial index operation.

**Batched KV cache**: a single `KVCache` is allocated with `batch_size=len(requests)`. All sequences share the same `seq_len` counter (they're padded to the same length and advance together). Padding positions store garbage K/V, but those are masked out in attention by the `padding_mask`.

**Prefill flow**:

```python
@torch.inference_mode()
def prefill(self, requests):
    batch_size = len(requests)
    device = self.config.device

    # Record per-request prompt lengths.
    self._prompt_lens = [len(req.prompt_token_ids) for req in requests]
    self._max_prompt_len = max(self._prompt_lens)

    # Right-pad all prompts to max_prompt_len (pad with 0).
    padded = []
    for req in requests:
        tokens = req.prompt_token_ids
        padded.append(tokens + [0] * (self._max_prompt_len - len(tokens)))
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)  # [batch, max_prompt_len]

    # Build padding mask: True for real tokens, False for padding.
    # Shape: [batch, max_total_seq_len] where max_total_seq_len covers
    # prefill + max decode tokens. Extended during decode as positions grow.
    max_decode = max(req.sampling_params.max_new_tokens for req in requests)
    max_total = min(self._max_prompt_len + max_decode, self.config.max_seq_len)
    self._padding_mask = torch.zeros(batch_size, max_total, dtype=torch.bool, device=device)
    for i, plen in enumerate(self._prompt_lens):
        self._padding_mask[i, :plen] = True  # real prompt tokens

    # Allocate batched KV cache.
    self._kv_cache = KVCache.from_model_config(
        self.model.config, max_seq_len=max_total,
        batch_size=batch_size, dtype=self.dtype, device=device,
    )

    # Batched forward pass.
    for req in requests:
        req.state = RequestState.PREFILL
    logits = self.model(input_ids, kv_cache=self._kv_cache, padding_mask=self._padding_mask)
    # logits: [batch, max_prompt_len, vocab_size] (full, no last-position opt)

    # Gather logits at each request's last real token position.
    last_positions = torch.tensor(
        [plen - 1 for plen in self._prompt_lens], dtype=torch.long, device=device
    )
    # [batch, vocab_size]
    next_logits = logits[torch.arange(batch_size, device=device), last_positions, :]

    # Sample first token per request.
    outputs = []
    for i, req in enumerate(requests):
        context = req.prompt_token_ids
        token = sample_token(next_logits[i], context, req.sampling_params, req.generator)
        req.generated_token_ids.append(token)
        req.state = RequestState.DECODE

        # Mark the generated token's position as valid in the padding mask.
        self._padding_mask[i, self._max_prompt_len] = True

        finished, reason = self._check_stop(req, token)
        if finished:
            req.state = RequestState.FINISHED
            req.finish_reason = reason
        outputs.append(self._make_step_output(req, token, finished, reason))

    return outputs
```

**Decode flow**:

```python
@torch.inference_mode()
def decode_step(self, requests):
    device = self.config.device
    batch_size = len(requests)

    # Build input: last generated token for active requests, dummy (0) for finished.
    tokens = []
    for req in requests:
        if req.state == RequestState.DECODE:
            tokens.append(req.generated_token_ids[-1])
        else:
            tokens.append(0)  # dummy token for finished requests
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)  # [batch, 1]

    # Forward pass.
    logits = self.model(input_ids, kv_cache=self._kv_cache, padding_mask=self._padding_mask)
    # logits: [batch, 1, vocab_size]

    # Sample per request.
    outputs = []
    for i, req in enumerate(requests):
        if req.state != RequestState.DECODE:
            # Already finished — emit no-op output.
            outputs.append(self._make_finished_noop(req))
            continue

        context = req.prompt_token_ids + req.generated_token_ids
        token = sample_token(logits[i, -1, :], context, req.sampling_params, req.generator)
        req.generated_token_ids.append(token)

        # Mark this decode position as valid in padding mask.
        decode_pos = self._kv_cache.seq_len  # position AFTER advance (set by model forward)
        # Actually, the model already called advance(1), so the new token
        # was written at (seq_len_before_advance). We mark it valid:
        current_pos = self._max_prompt_len + len(req.generated_token_ids) - 1
        self._padding_mask[i, current_pos] = True

        finished, reason = self._check_stop(req, token)
        if finished:
            req.state = RequestState.FINISHED
            req.finish_reason = reason
        outputs.append(self._make_step_output(req, token, finished, reason))

    return outputs
```

**Stop condition checking**: `_check_stop` mirrors `generate.py`'s approach. It checks:
1. Is the token in the model's EOS token set → `finish_reason = "eos"`.
2. Does the decoded text (from all `generated_token_ids`) contain any stop string → `finish_reason = "stop"`.
3. Has `len(generated_token_ids) >= max_new_tokens` → `finish_reason = "length"`.

The stop string check decodes all generated tokens each step via `tokenizer.decode(generated_ids, skip_special_tokens=True)` and checks for substring matches. This is O(n) per step for n generated tokens — acceptable for Phase 4, matching the existing `generate.py` behavior.

**Finished requests in decode**: requests that finish before the batch completes continue to occupy their slot in the batch tensor. They receive a dummy token (0) as input, and their forward pass output is ignored (no sampling). This wastes some compute but keeps the batch size constant — the batch runs until all requests finish.

**Padding mask update during decode**: each generated token position is marked `True` in the `padding_mask` so that subsequent decode steps can attend to it. The gap between a short prompt's real tokens and the start of decoded tokens (the right-padding region) stays `False` and is always masked out in attention.

**Memory budget**: with `max_batch_size=8` and `max_seq_len=4096`, worst-case KV cache memory per model. Note: a single batched cache is allocated (not per-request).

| Model        | Per-sequence cache (4K) | 8-sequence batch cache | Weights | Total  |
|--------------|------------------------:|-----------------------:|--------:|-------:|
| Llama 3.2 1B |                  256 MB |                 2.0 GB |  2.0 GB | 4.0 GB |
| Qwen3 1.7B   |                   57 MB |                 0.5 GB |  3.5 GB | 4.0 GB |
| Gemma3 1B    |                   52 MB |                 0.4 GB |  2.6 GB | 3.0 GB |

All fit within the 16 GB dev-tier budget. Default `max_batch_size=8` is conservative; can be raised on stretch-tier hardware.

### 6. Engine (`src/infer/engine/engine.py`)

The engine ties together the scheduler and runner. It owns the model and exposes a synchronous `step()` method.

```python
class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.model = ...    # loaded in __init__ or via a factory
        self.tokenizer = ...
        self.scheduler = StaticScheduler(config)
        self.runner = ModelRunner(self.model, self.tokenizer, config)
        self.model_id: str = config.model  # full HF model ID for validation

    @classmethod
    def from_components(
        cls, config: EngineConfig, model: nn.Module, tokenizer: object,
    ) -> Self:
        """Create an engine from pre-built components (for testing).

        Bypasses model/tokenizer loading so tests can inject mock objects.
        """

    def add_request(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        output_queue: asyncio.Queue[StepOutput],
    ) -> bool:
        """Create a Request, tokenize if needed, and add to the scheduler.

        Validates that prompt + max_new_tokens <= max_seq_len.
        Creates a per-request torch.Generator from sampling_params.seed.
        Returns False if the queue is full (503).
        """

    def step(self) -> None:
        """Execute one scheduler iteration.

        1. Call scheduler.schedule() to get the active batch.
        2. If the batch is newly formed (requests in WAITING state),
           call runner.prefill(batch).
        3. If the batch is in decode phase, call runner.decode_step(batch).
        4. For each StepOutput, push to the request's output_queue.
        5. If all requests in the batch are FINISHED or FAILED,
           call runner.clear_batch().
        """

    def has_work(self) -> bool:
        """True if the engine has pending or active requests."""
```

**`step()` is deterministic**: for a fixed random seed and fixed request queue, `step()` produces the same outputs. No async I/O, no non-deterministic operations inside `step()`.

**Token delivery**: each `StepOutput` is pushed to the request's `asyncio.Queue`. The API layer reads from this queue and emits SSE events. This decouples the engine's synchronous compute from the API's async I/O.

**Error handling**: if a forward pass raises an exception (e.g. GPU OOM during KV cache allocation), the engine catches it, transitions all affected requests to `FAILED`, pushes `StepOutput(error=str(e))` to each request's queue, and calls `runner.clear_batch()`. The API layer emits an SSE `error` event and closes the stream.

**Prompt length validation**: `add_request()` checks that `len(prompt_token_ids) + sampling_params.max_new_tokens <= config.max_seq_len`. If not, it rejects the request immediately (the API returns 422 before starting the SSE stream).

**Thread safety**: both `add_request()` (called from API coroutines) and `step()` (called from the engine loop task) run on the same asyncio event loop thread. Python's single-threaded asyncio model guarantees that they do not execute concurrently — `add_request()` runs between `await` points, and `step()` is synchronous (no yield points). The `deque` operations in the scheduler are therefore safe without locks.

### 7. API server (`src/infer/server/api.py`)

FastAPI application with a single endpoint.

**Endpoint**: `POST /v1/completions`

**Request body**:

```python
from pydantic import BaseModel, ConfigDict

class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str | list[int]
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    stream: bool = True      # accepted but always treated as true
    stop: str | list[str] | None = None
    seed: int | None = None
```

The `extra="forbid"` config rejects unknown fields with a 422 error, matching the overall design's requirement that "unsupported fields must return a clear validation error." The `top_k` and `repetition_penalty` fields extend the overall design's v1 API contract (which lists `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `stream`, `stop`, `seed`).

**Validation**:

- `model` must match the engine's loaded model ID (the full HuggingFace model ID, e.g. `meta-llama/Llama-3.2-1B-Instruct`) → 422 if mismatch.
- `max_tokens` must be >= 1 → 422.
- `temperature` must be >= 0 → 422.
- `top_p` must be in (0, 1] → 422.
- `prompt` must be a non-empty string or non-empty list of ints → 400.
- `prompt` token count + `max_tokens` must not exceed `max_seq_len` → 422.

**Response: SSE stream**

Content-Type: `text/event-stream`.

Event types (matching `OVERALL_DESIGN.md` section 5):

```
event: token
data: {"token": "Hello", "token_id": 9906}

event: token
data: {"token": " world", "token_id": 1917}

event: done
data: {"finish_reason": "eos", "usage": {"prompt_tokens": 15, "completion_tokens": 42, "total_tokens": 57}}

event: error
data: {"error": "KV cache overflow: sequence length exceeds max_seq_len"}
```

**Error responses** (HTTP, before SSE stream starts):

- `400` — bad JSON, empty prompt, wrong types.
- `422` — invalid field values (wrong model, out-of-range temperature, unsupported field, prompt too long).
- `503` — request queue full.

**SSE implementation**: uses `sse-starlette`'s `EventSourceResponse` with an async generator that reads from the request's `asyncio.Queue[StepOutput]` and yields SSE events:

```python
@app.post("/v1/completions")
async def completions(req: CompletionRequest) -> EventSourceResponse:
    # Validate, create output queue, enqueue request
    output_queue: asyncio.Queue[StepOutput] = asyncio.Queue()
    request_id = str(uuid4())

    if not engine.add_request(request_id, req.prompt, sampling_params, output_queue):
        raise HTTPException(status_code=503, detail="Server overloaded")

    async def event_generator():
        while True:
            step_out = await output_queue.get()
            if step_out.error:
                yield ServerSentEvent(
                    event="error",
                    data=orjson.dumps({"error": step_out.error}).decode(),
                )
                return
            if not step_out.finished:
                yield ServerSentEvent(
                    event="token",
                    data=orjson.dumps({
                        "token": step_out.text_delta,
                        "token_id": step_out.token_id,
                    }).decode(),
                )
            else:
                # Emit the final token if present.
                if step_out.text_delta:
                    yield ServerSentEvent(
                        event="token",
                        data=orjson.dumps({
                            "token": step_out.text_delta,
                            "token_id": step_out.token_id,
                        }).decode(),
                    )
                yield ServerSentEvent(
                    event="done",
                    data=orjson.dumps({
                        "finish_reason": step_out.finish_reason,
                        "usage": {
                            "prompt_tokens": step_out.prompt_tokens,
                            "completion_tokens": step_out.completion_tokens,
                            "total_tokens": step_out.prompt_tokens + step_out.completion_tokens,
                        },
                    }).decode(),
                )
                return

    return EventSourceResponse(event_generator())
```

**Client disconnect handling**: when a client disconnects mid-stream, the `EventSourceResponse` generator exits. The request continues running in the engine (it cannot be removed mid-batch in Phase 4 static batching). The `output_queue` fills up and is garbage-collected when the batch finishes. Phase 5 can add request cancellation.

### 8. Engine loop and lifecycle (`src/infer/server/api.py`)

The engine step loop runs as a background asyncio task started on `app.lifespan`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model, create engine
    engine = Engine(engine_config)
    app.state.engine = engine
    # Start the engine loop
    loop_task = asyncio.create_task(engine_loop(engine))
    yield
    # Shutdown: cancel the loop
    loop_task.cancel()
    with suppress(asyncio.CancelledError):
        await loop_task

async def engine_loop(engine: Engine) -> None:
    while True:
        if engine.has_work():
            engine.step()
            await asyncio.sleep(0)  # yield to event loop
        else:
            await asyncio.sleep(0.001)  # 1ms idle poll
```

The `asyncio.sleep(0)` between steps is critical — without it, the engine would starve the event loop and SSE events would not be sent until the batch completes. With it, token events are flushed after each step.

The `asyncio.sleep(0.001)` when idle prevents busy-waiting. 1ms is short enough that new requests are picked up promptly.

**App factories**: two factory functions create the FastAPI app:

- `create_app(config: EngineConfig)` — production factory. Loads a real model inside the lifespan. Routes are registered at app creation using an `_EngineProxy` that forwards attribute access to the real engine once the lifespan starts. This avoids registering routes inside the lifespan (which would miss the route table).
- `create_app_with_engine(engine: Engine)` — testing factory. Accepts a pre-built engine (typically from `Engine.from_components()` with a mock model) so integration tests can run without loading real models.

### 9. Server entry point (`src/infer/server/__main__.py`)

```bash
uv run python -m infer.server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-batch-size 8
```

CLI arguments map to `EngineConfig` fields. Uses `uvicorn.run()` internally.

### 10. Benchmark script (`benchmarks/bench_serving.py`)

A concurrent load generator using `httpx.AsyncClient` to measure serving throughput and latency.

**Workloads**:

- `W1 Single`: 1 request, prompt 256 tokens, generate 256 tokens. Baseline for single-request serving overhead vs direct `generate()`.
- `W2 Mixed`: 16 concurrent requests, prompt lengths uniformly sampled from [32, 1024], generate [64, 256] tokens. Measures throughput under load.

**Metrics collected per request**:

- TTFT (time to first `token` event from request send).
- ITL (inter-token latency between consecutive `token` events).
- Request latency (time from send to `done` event).
- Completion tokens.

**Aggregate metrics**:

- Total throughput (output tok/s across all requests).
- TTFT P50/P95/P99.
- ITL P50/P95/P99.
- Request latency P50/P95/P99.

**Comparison baseline**: run the same prompts sequentially through `generate()` and measure wall-clock time. Static batching throughput under concurrent load should exceed this sequential baseline.

---

## File Layout

New and modified files:

```
src/infer/
├── engine/
│   ├── __init__.py             # MODIFIED: re-export new types
│   ├── config.py               # NEW: EngineConfig dataclass
│   ├── request.py              # NEW: Request, RequestState, StepOutput
│   ├── scheduler.py            # NEW: StaticScheduler
│   ├── runner.py               # NEW: ModelRunner (batched prefill + decode)
│   ├── engine.py               # NEW: Engine (step loop orchestration)
│   ├── sampler.py              # UNCHANGED
│   └── generate.py             # UNCHANGED (kept for standalone usage)
├── models/
│   ├── llama.py                # MODIFIED: add padding_mask parameter to forward
│   ├── qwen3.py                # MODIFIED: add padding_mask parameter to forward
│   └── gemma3.py               # MODIFIED: add padding_mask parameter to forward
├── cache/
│   └── simple.py               # MODIFIED: from_model_config gains batch_size parameter
└── server/
    ├── __init__.py             # NEW: package marker
    ├── api.py                  # NEW: FastAPI app, SSE endpoint, engine loop
    └── __main__.py             # NEW: CLI entry point

benchmarks/
├── bench_serving.py            # NEW: concurrent load generator
└── log/
    └── GENERATION_LOG.md       # MODIFIED: add Phase 4 results section

tests/
├── unit/
│   ├── test_request.py         # NEW: Request state machine tests
│   ├── test_scheduler.py       # NEW: StaticScheduler tests
│   ├── test_runner.py          # NEW: ModelRunner tests (mock model)
│   └── test_components.py      # MODIFIED: add padding_mask tests for Attention
├── integration/
│   ├── test_api.py             # NEW: API endpoint tests (TestClient)
│   └── test_logits_parity.py   # MODIFIED: verify padding_mask doesn't break parity
└── stress/
    └── test_backpressure.py    # NEW: queue growth and 503 behavior
```

**Preserved files**: `generate.py` and `sampler.py` are unchanged. The standalone `generate()` function remains the primary entry point for benchmarks, scripts, and non-server usage. The new engine reuses `SamplingParams` and `sample_token` from `sampler.py` but does not call `generate()` — the engine manages the generation loop itself (batched KV cache, per-step output delivery). `models/common.py` is unchanged — padding support is implemented in the model-level forward methods, not in the shared `Attention` class.

**`KVCache.from_model_config` change**: add `batch_size` parameter (default 1 for backward compatibility):

```python
@staticmethod
def from_model_config(
    config: ModelConfig,
    max_seq_len: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> KVCache:
```

---

## Testing Plan

### Request state machine tests (`tests/unit/test_request.py`)

- Create a `Request` with all required fields. Verify initial state is `WAITING`.
- Verify `StepOutput` construction with and without error fields.
- Test `Request` field defaults (empty generated_token_ids, generator is None, etc.).
- Verify `generator` creation from seed.

### Scheduler tests (`tests/unit/test_scheduler.py`)

All tests use `Request` objects with dummy prompt IDs (no model loading).

**Basic batch formation**:
- Add 3 requests to scheduler with `max_batch_size=4`.
- Call `schedule()`. Verify 3 requests returned, all transitioned from waiting to active.
- Call `schedule()` again. Verify same 3 requests returned (batch is still active).

**Batch completion**:
- Add 2 requests. `schedule()` returns them.
- Mark both as `FINISHED`. Call `schedule()` again.
- Verify active list is cleared and waiting queue was consulted (returns empty if no new requests).

**Partial completion**:
- Add 3 requests. `schedule()`. Mark 2 as `FINISHED`, leave 1 in `DECODE`.
- `schedule()` returns the same batch (still has active requests).
- Mark the last as `FINISHED`. `schedule()` now clears the batch.

**Queue overflow**:
- Set `max_waiting_requests=2`. Add 2 requests → both succeed.
- Add a 3rd → `add_request()` returns `False`.

**FCFS ordering**:
- Add 3 requests in order A, B, C.
- `schedule()` returns them in insertion order: [A, B, C].

**Batch size limit**:
- Set `max_batch_size=2`. Add 5 requests.
- First `schedule()` returns 2. After they finish, next `schedule()` returns 2. Then 1.

**`has_work`**:
- Empty scheduler → `has_work()` is `False`.
- Add a request → `True`.
- Schedule and finish all → `False`.

**Bounded wait — dispatches early when full**:
- Set `max_batch_size=4`, `batch_wait_timeout_s=1.0` (long timeout).
- Add 4 requests. Call `schedule()` immediately.
- Returns 4 requests (batch is full, no waiting needed).

**Bounded wait — dispatches on timeout**:
- Set `max_batch_size=4`, `batch_wait_timeout_s=0.01` (10ms).
- Add 2 requests. Call `schedule()` → returns empty (timer started, not expired).
- Sleep 15ms. Call `schedule()` → returns 2 requests (timer expired).

**Bounded wait — timer resets between batches**:
- Complete a batch. Add 1 new request.
- Verify the timer starts fresh (not carried over from the previous batch).

### Runner tests (`tests/unit/test_runner.py`)

Uses mock models (similar pattern to existing `test_generate.py` mocks). Mock models accept `padding_mask` kwarg. No real model loading.

**Single-request prefill**:
- Create a `MockModel` that returns fixed logits. Create a `ModelRunner`.
- Call `prefill([req])`. Verify:
  - `req.state` is `DECODE` (or `FINISHED` if first token is EOS).
  - Runner's `_kv_cache` is allocated with `batch_size=1`.
  - `req.generated_token_ids` has 1 entry.
  - Returns 1 `StepOutput` with the correct token.

**Multi-request batched prefill with mixed prompt lengths**:
- Create 3 requests with prompt lengths [3, 7, 5].
- Call `prefill(requests)`. Verify:
  - Runner's `_max_prompt_len` is 7.
  - KV cache is allocated with `batch_size=3`.
  - `_padding_mask` has correct True/False pattern per sequence.
  - All 3 requests have 1 generated token and are in DECODE state.

**Decode step**:
- After prefill, call `decode_step(requests)`.
- Verify each request's `generated_token_ids` grows by 1.
- Returned `StepOutput` has the new token per request.

**EOS handling**:
- Mock model that returns EOS token ID logits for one request in the batch.
- After prefill or decode, verify that request has `state == FINISHED`, `finish_reason == "eos"`.
- Other requests in the batch continue decoding.

**Max tokens**:
- `max_new_tokens=3`. Run prefill + 2 decode steps.
- After step 3, verify `req.state == FINISHED`, `finish_reason == "length"`.

**Batch runs until all finish**:
- 2 requests: one hits EOS after 2 tokens, other has `max_new_tokens=5`.
- Run prefill + 4 decode steps. Verify the first request finishes early and the second continues.
- After all finish, verify `clear_batch()` frees KV cache.

**Stop string handling**:
- Mock model + tokenizer that produce a stop string after 3 tokens.
- Verify `finish_reason == "stop"`.

### Model padding tests (`tests/unit/test_components.py` — extend existing)

**Padding mask backward compatibility**:
- Run a model forward without `padding_mask` → identical output to Phase 3.

**Padding equivalence**:
- Forward pass with batch=1, no padding, prompt [A, B, C].
- Forward pass with batch=2, right-padded: [[A, B, C, 0, 0], [D, E, F, G, H]].
- Verify logits at position 2 (last real token of sequence 0) match the single-request logits at position 2.

### API integration tests (`tests/integration/test_api.py`)

Uses FastAPI's `TestClient` (or `httpx.AsyncClient` with `ASGITransport`). Can use a mock engine or a real engine with a small mock model.

**Valid request → SSE stream**:
- POST a valid request. Read events.
- Verify at least one `token` event and exactly one `done` event.
- Verify `done` event has `usage` with correct `prompt_tokens` and `completion_tokens`.

**Empty prompt → 400**:
- POST with `prompt: ""` → 400 response.

**Wrong model name → 422**:
- POST with `model: "nonexistent"` → 422 response.

**Invalid temperature → 422**:
- POST with `temperature: -1.0` → 422 response.

**Unknown field → 422**:
- POST with an extra field `{"foo": "bar"}` → 422 response.

**Prompt too long → 422**:
- POST with a prompt whose token count + max_tokens exceeds max_seq_len → 422.

**Queue full → 503**:
- Set `max_waiting_requests=1`. Send 2+ concurrent requests.
- At least one should get 503.

**String prompt tokenization**:
- POST with `prompt: "Hello world"` (string). Verify tokens are generated.

**Token ID prompt**:
- POST with `prompt: [1, 2, 3]` (list of ints). Verify tokens are generated.

**Stop string**:
- POST with `stop: ["end"]`. Verify generation stops and `finish_reason` is `"stop"` in the `done` event.

### Parity regression tests (`tests/integration/test_logits_parity.py` — extend existing)

- Add a test that runs the existing logits parity check with `padding_mask=None`, verifying that Phase 3 behavior is not regressed by the new parameter.

### Stress tests (`tests/stress/test_backpressure.py`)

Marked `@pytest.mark.slow`. Uses a real or mock engine.

**Queue growth under load**:
- Send `max_waiting_requests + 10` requests concurrently.
- Verify 10 get 503 responses.
- Verify the remaining `max_waiting_requests` complete successfully.

**No deadlock under concurrent disconnect**:
- Send 8 requests, immediately cancel/disconnect 4 of them.
- Verify the remaining 4 complete and the server remains responsive.

---

## Design Decisions

**Right-padded batched forward passes**: all requests in a batch are right-padded to the longest prompt length and processed in a single model forward call. Right-padding is chosen over left-padding because real tokens naturally occupy positions `[0, prompt_len)` and receive correct RoPE positional encodings without custom `position_ids`. The tradeoff is that logit extraction during prefill requires per-sequence gathering (the last real token position varies per sequence), but this is trivial. During decode, all sequences have `seq_len=1`, so gathering is not needed.

**Single batched KV cache**: a single `KVCache` with `batch_size=len(batch)` is allocated per batch, rather than per-request caches. All sequences share the same `seq_len` counter (they advance together in static batching). Padding positions store garbage K/V which is masked out in attention. The single allocation is simpler and enables true batched forward passes through the model. Phase 6 replaces this with paged block allocation.

**`padding_mask` on model, not on Attention**: the `padding_mask` parameter is added to the model-level forward methods (LlamaModel, Qwen3Model, Gemma3Model), not to the Attention class. The model combines padding info with its internal causal/sliding-window masks before passing the result to Attention. This keeps the Attention class unchanged and confines the padding logic to the model layer, where architecture-specific mask construction (Gemma3's dual masks) already lives.

**Last-position optimization skipped during batched prefill**: when `padding_mask` is provided, the model returns full `[batch, seq_len, vocab_size]` logits during prefill instead of applying the last-position optimization. This is because the last position in the padded sequence may be padding for shorter sequences. The runner gathers logits at each sequence's actual last token. The cost of computing full logits during batched prefill is small (~1ms) relative to the attention compute. During decode, `seq_len=1` so the optimization is a no-op regardless.

**Engine remains synchronous**: `Engine.step()` does no async I/O. The API layer runs the step loop in a background task and uses `asyncio.Queue` to bridge sync compute with async SSE delivery. This keeps the engine testable without async infrastructure.

**`generate.py` preserved**: the standalone `generate()` function remains unchanged and is not used by the engine. The engine has its own generation logic split across `ModelRunner.prefill()` and `ModelRunner.decode_step()`. This avoids coupling the engine's per-step, per-request control flow with the standalone function's single-request loop. Both share `SamplingParams` and `sample_token` from `sampler.py`.

**Token delivery via asyncio.Queue**: each request gets a `Queue[StepOutput]`. The engine pushes after each step; the API pops and emits SSE events. This decouples the engine from the API and makes it easy to add multiple consumers later (metrics, logging).

**No request cancellation in Phase 4**: when a client disconnects, the engine continues generating for that request until the batch finishes naturally. The output queue is simply not consumed. This is fine for static batching where the batch runs to completion anyway. Phase 5 can add cancellation when per-step scheduling makes it possible to remove individual requests.

**Bounded wait for batch formation**: the scheduler waits up to `batch_wait_timeout_s` (default 50ms) for the batch to fill before dispatching. This coalesces concurrent arrivals into larger batches for better GPU utilization, while keeping single-request latency impact minimal (~1 decode step). The timer starts when the first request arrives after a batch completes, and resets between batches. If the batch fills to `max_batch_size` before the timeout, it dispatches immediately. The alternative (dispatch immediately) would split burst traffic across serial batches, wasting GPU parallelism.

**Conservative default batch size**: `max_batch_size=8` keeps worst-case KV cache memory under ~2 GB across all dev models at 4K sequence length, leaving ample headroom for model weights on 16 GB hardware. The overall design defaults to 32; 8 is used here because per-batch contiguous caches are larger than Phase 6's paged allocation would be.

**Model name matching via HF model ID**: the `model` field in `CompletionRequest` is validated against the full HuggingFace model ID (e.g. `meta-llama/Llama-3.2-1B-Instruct`) stored at engine initialization. This is the natural identifier used throughout the project.

**Reject unknown fields**: `CompletionRequest` uses Pydantic's `extra="forbid"` to reject requests containing unsupported fields with a 422 error, matching the overall design's requirement.

**Prompt length validation at admission**: `Engine.add_request()` rejects prompts where `len(prompt_token_ids) + max_new_tokens > max_seq_len` before enqueuing. This prevents KV cache overflow during generation and gives the client an immediate 422 error rather than a mid-stream failure.

**orjson for SSE serialization**: SSE data payloads are serialized with `orjson` (already a dependency) for speed. The payloads are small, so this is micro-optimization, but orjson is already in the dependency list and avoids importing `json`.

---

## Exit Criteria

1. `POST /v1/completions` returns a valid SSE stream with `token` and `done` events for all three dev models.
2. HTTP error responses: 400 for malformed requests, 422 for invalid field values and unknown fields, 503 when queue is full.
3. Throughput under concurrent load (W2 Mixed, 16 requests) exceeds sequential serving baseline (running the same prompts one at a time through `generate()`).
4. Static batch lifecycle works: batch forms, all requests complete, batch clears, next batch forms.
5. Mixed prompt lengths in a single batch produce correct output (right-padding masks work, per-sequence logit gathering works).
6. Regression tests cover: EOS handling in batch, stop string handling, max_tokens limit, partial batch completion.
7. Queue backpressure: excess requests get 503, no deadlocks, no unbounded memory growth.
8. All existing unit and integration tests pass (Phases 1-3.1 not broken by `padding_mask` addition).
9. New unit tests pass: request state machine, scheduler, runner (including batched prefill/decode).
10. API integration tests pass: valid requests, error responses, SSE event format, prompt length validation.
11. Benchmark script (`bench_serving.py`) runs and produces a report with TTFT, ITL, and throughput metrics.
12. `uv run ruff check .` and `uv run mypy .` pass cleanly.
