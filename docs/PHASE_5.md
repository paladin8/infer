# Phase 5: Continuous Batching

## Goal

Replace static batching with iteration-level scheduling: admit new requests and retire finished ones every engine step, rather than waiting for an entire batch to complete. This eliminates head-of-line blocking where short requests are held hostage by long ones, improving tail latency and keeping the GPU busy with a fuller batch.

Phase 5 is the v1 completion target. After this phase, `infer` has continuous batching with streaming output, an OpenAI-compatible endpoint, and benchmark reports comparing each optimization through Phase 5.

Benchmark models: `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-4B`, `google/gemma-3-1b-it` (same as Phase 4).

---

## Architecture

```text
               HTTP POST /v1/completions
                          │
                          ▼
             ┌─────────────────────────┐
             │      FastAPI Server     │  (unchanged from Phase 4)
             └────────────┬────────────┘
                          │ asyncio.Queue per request
                          ▼
             ┌──────────────────────────┐
             │      Engine.step()       │
             │                          │
             │  1. scheduler.schedule() │──► retire finished, admit new
             │  2. runner.step(...)     │──► prefill new + decode active
             │  3. push StepOutputs     │
             └─────────────┬────────────┘
                           │
           ┌───────────────┼─────────────────┐
           ▼               ▼                 ▼
   ┌───────────────┐ ┌────────────┐ ┌────────────────┐
   │ Continuous    │ │ Continuous │ │ Slotted KV     │
   │ Scheduler     │ │ Runner     │ │ Cache Pool     │
   │               │ │            │ │                │
   │ waiting queue │ │ prefill()  │ │ max_batch_size │
   │ active set    │ │ decode()   │ │ slots          │
   │ retire/admit  │ │            │ │ per-slot       │
   │ per step      │ │            │ │ seq_lens       │
   └───────────────┘ └────────────┘ └────────────────┘
```

**Key difference from Phase 4**: the scheduler no longer holds a fixed batch until all requests finish. Each `step()` call retires completed requests (freeing their cache slots), admits new requests from the waiting queue (up to free slots), prefills the new arrivals, and runs one batched decode step for all active requests.

**Step order within `Engine.step()`**:
1. `scheduler.schedule()` — retire finished, admit new, return schedule
2. Free cache slots for retired requests
3. Batched decode for all active decode requests (one forward pass)
4. Prefill new requests — single request uses individual prefill (no padding overhead), multiple requests use batched prefill (right-padded to longest prompt, one forward pass)
5. Push StepOutputs to per-request queues

Decode runs before prefill (step 3 before step 4) to prioritize inter-token latency for in-flight requests. A newly arrived request waits one step before prefill begins — this adds ~10ms to TTFT (one decode step) but keeps ITL stable for existing requests. When multiple requests arrive in the same step, batched prefill amortizes weight loading across all of them in a single forward pass (right-padded to the longest prompt). A long prefill blocks decode for its duration — Phase 7 (chunked prefill) addresses this by breaking prefills into chunks interleaved with decode steps.

**No API changes**: the `POST /v1/completions` endpoint, SSE event format, and error responses are identical to Phase 4. Continuous batching is an internal engine improvement selected via `batching_mode="continuous"` in `EngineConfig`.

---

## Deliverables

### 1. Cache protocol and slotted KV cache pool

#### Cache protocol (`src/infer/cache/protocol.py`)

Models currently type-hint `kv_cache: KVCache | None`, but Phase 5 introduces two new cache types (`PrefillCacheView`, `DecodeCacheView`) that implement the same interface without inheriting from `KVCache`. A `Protocol` makes this contract explicit and verifiable at type-check time:

```python
class KVCacheProtocol(Protocol):
    """Interface that all KV cache implementations must satisfy.

    Models call ``update()`` per layer to store and retrieve K/V,
    ``advance()`` once per forward pass, and read ``seq_len`` for
    mask width calculation.  Phase 5 views, Phase 6 paged views,
    and the original ``KVCache`` all implement this protocol.
    """

    seq_len: int

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]: ...
    def advance(self, n: int) -> None: ...
```

Model forward signatures change from `kv_cache: KVCache | None` to `kv_cache: KVCacheProtocol | None`. The `TYPE_CHECKING` import is updated accordingly. Existing `KVCache` satisfies the protocol without modification.

#### Slotted KV cache pool (`src/infer/cache/slotted.py`)

A pre-allocated cache pool with `max_batch_size` independent slots, each tracking its own position.

```python
class SlottedKVCache:
    """Pre-allocated KV cache pool for continuous batching.

    Allocates cache for ``max_batch_size`` sequences at engine startup.
    Each slot has an independent position counter, enabling sequences at
    different generation stages to coexist in the same cache tensor.

    Attributes:
        k: Key cache, shape ``[num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim]``.
        v: Value cache, same shape as ``k``.
        seq_lens: Per-slot position counters, shape ``[max_batch_size]``.
    """

    def __init__(
        self,
        k: Tensor,
        v: Tensor,
        max_batch_size: int,
    ) -> None:
        self.k = k
        self.v = v
        self.max_batch_size = max_batch_size
        self.seq_lens: list[int] = [0] * max_batch_size
        self._free_slots: set[int] = set(range(max_batch_size))

    @staticmethod
    def from_model_config(
        config: ModelConfig,
        max_seq_len: int,
        max_batch_size: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> SlottedKVCache:
        """Allocate a pool sized for the given model config."""

    def allocate_slot(self) -> int:
        """Claim a free slot. Raises RuntimeError if none available."""

    def free_slot(self, slot: int) -> None:
        """Release a slot and reset its position counter."""

    def free_slot_count(self) -> int:
        """Number of available slots."""

    def prefill_view(self, slot: int) -> PrefillCacheView:
        """Return a single-slot view for prefill (KVCache-compatible)."""

    def decode_view(self, active_slots: list[int]) -> DecodeCacheView:
        """Return a multi-slot view for batched decode."""

    def batched_prefill_view(
        self, slots: list[int], prompt_lens: list[int],
    ) -> BatchedPrefillCacheView:
        """Return a multi-slot view for batched prefill."""
```

**`PrefillCacheView`** wraps a single slot and provides the same interface as `KVCache` so the existing model forward code works unchanged:

```python
class PrefillCacheView:
    """KVCache-compatible view for single-slot prefill.

    Writes K/V directly into the pool at the assigned slot.
    The model forward code calls ``update()`` and ``advance()``
    exactly as it does with a regular KVCache.
    """

    def __init__(self, pool: SlottedKVCache, slot: int) -> None:
        self.pool = pool
        self.slot = slot
        self.seq_len: int = 0  # starts at 0 for new request

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store K/V at this slot's position and return the valid cache.

        k, v shape: ``[1, num_kv_heads, new_len, head_dim]``.
        Returns cache covering ``[0 : seq_len + new_len]`` for this slot.
        """
        new_len = k.shape[2]
        start = self.seq_len
        end = start + new_len
        self.pool.k[layer_idx, self.slot : self.slot + 1, :, start:end, :] = k
        self.pool.v[layer_idx, self.slot : self.slot + 1, :, start:end, :] = v
        return (
            self.pool.k[layer_idx, self.slot : self.slot + 1, :, :end, :],
            self.pool.v[layer_idx, self.slot : self.slot + 1, :, :end, :],
        )

    def advance(self, n: int) -> None:
        """Advance this slot's position counter."""
        self.seq_len += n
        self.pool.seq_lens[self.slot] = self.seq_len
```

**`DecodeCacheView`** wraps multiple active slots for batched decode. It writes new K/V to per-slot positions and gathers the full cache from the pool for attention:

```python
class DecodeCacheView:
    """Multi-slot cache view for batched decode.

    Provides KVCache-compatible interface over a subset of pool slots.
    Each slot's K/V is written at its own position.  The full cache is
    gathered from the pool for attention.
    """

    def __init__(self, pool: SlottedKVCache, active_slots: list[int]) -> None:
        self.pool = pool
        self.slots = active_slots
        self.slot_seq_lens = [pool.seq_lens[s] for s in active_slots]
        self._seq_len = max(self.slot_seq_lens) if active_slots else 0

    @property
    def seq_len(self) -> int:
        """Max of active slot seq_lens (used for mask width calculation)."""
        return self._seq_len

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write per-slot K/V and return gathered cache for attention.

        k, v shape: ``[active_batch, num_kv_heads, 1, head_dim]``.
        Writes each batch element's K/V to its slot's current position.
        Returns gathered cache ``[active_batch, num_kv_heads, max_len, head_dim]``.
        """
        for i, slot in enumerate(self.slots):
            pos = self.slot_seq_lens[i]
            self.pool.k[layer_idx, slot, :, pos, :] = k[i, :, 0, :]
            self.pool.v[layer_idx, slot, :, pos, :] = v[i, :, 0, :]

        max_len = self._seq_len + 1
        slot_idx = torch.tensor(self.slots, device=k.device)
        cached_k = self.pool.k[layer_idx, slot_idx, :, :max_len, :]
        cached_v = self.pool.v[layer_idx, slot_idx, :, :max_len, :]
        return cached_k, cached_v

    def advance(self, n: int) -> None:
        """Advance all active slots by n positions."""
        for i, slot in enumerate(self.slots):
            self.pool.seq_lens[slot] += n
            self.slot_seq_lens[i] += n
        self._seq_len += n
```

**`BatchedPrefillCacheView`** wraps multiple slots for batched prefill. It scatter-writes each batch element's K/V to its assigned pool slot and returns the input K/V directly (no gather needed during prefill):

```python
class BatchedPrefillCacheView:
    """Multi-slot cache view for batched prefill.

    Used when multiple requests arrive in the same step. Each batch
    element's K/V is scatter-written to its assigned pool slot.
    ``advance()`` sets per-slot seq_lens to actual prompt lengths,
    not the padded length.
    """

    def __init__(self, pool: SlottedKVCache, slots: list[int], prompt_lens: list[int]) -> None:
        self.pool = pool
        self.slots = slots
        self.prompt_lens = prompt_lens
        self.seq_len: int = 0

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Scatter-write K/V to per-slot positions, return input directly.

        k, v shape: ``[batch, num_kv_heads, padded_len, head_dim]``.
        Each batch element is written to its slot at the current position.
        Returns (k, v) unchanged — no gather needed during prefill.
        """
        padded_len = k.shape[2]
        start = self.seq_len
        end = start + padded_len
        for i, slot in enumerate(self.slots):
            self.pool.k[layer_idx, slot, :, start:end, :] = k[i]
            self.pool.v[layer_idx, slot, :, start:end, :] = v[i]
        return k, v

    def advance(self, n: int) -> None:
        """Set per-slot seq_lens to actual prompt lengths (not padded)."""
        self.seq_len += n
        for i, slot in enumerate(self.slots):
            self.pool.seq_lens[slot] = self.prompt_lens[i]
```

**Why return input K/V directly**: during prefill, attention runs over the full prompt. The model already has the correct K/V tensors from the current forward pass — it doesn't need to read back from the pool. The scatter-write is purely for populating the pool so subsequent decode steps can read from it. This avoids the gather overhead that `DecodeCacheView` incurs.

**Why `advance()` sets actual prompt lengths**: the padded forward pass processes `max_prompt_len` tokens, but shorter prompts only have `prompt_lens[i]` real tokens. Setting `pool.seq_lens[slot] = prompt_lens[i]` ensures decode starts reading from the correct position (right after the real prompt, not after padding).

**Gather cost during decode**: `DecodeCacheView.update()` gathers K/V from the pool via advanced indexing, which creates a copy. Per layer: `2 × active_batch × num_kv_heads × max_len × head_dim × dtype_bytes`. For Llama 3B (28 layers, 8 KV heads, 128 dim, bf16) with 8 active requests and max_len=4096: ~3.5 GB total across all layers. This roughly doubles the KV memory traffic compared to contiguous batching. Acceptable at the dev model scale; Phase 6 (paged attention) eliminates this overhead with block-based gather.

**Slot reuse**: when a slot is freed, its position counter resets to 0. Old K/V data remains in the cache but is ignored — the next request's prefill overwrites from position 0, and the padding mask only marks valid positions as True.

**Memory budget**: identical to Phase 4 (same total cache allocation), but allocated once at startup rather than per-batch. Formula: `2 (K+V) × num_layers × num_kv_heads × head_dim × max_seq_len × max_batch_size × dtype_bytes`.

| Model        | Layers | KV heads | head_dim | 8-slot pool (4K seq) | Weights |    Total |
|--------------|-------:|---------:|---------:|---------------------:|--------:|---------:|
| Llama 3.2 3B |     28 |        8 |      128 |              3.5 GB  |   ~6 GB |  ~9.5 GB |
| Qwen3 4B     |     36 |        8 |      128 |              4.5 GB  |   ~8 GB | ~12.5 GB |
| Gemma3 1B    |     26 |        1 |      256 |              0.8 GB  |   ~2 GB |  ~2.8 GB |

All three fit within 16 GB VRAM (dev tier). Qwen3-4B is the tightest fit — if batch size needs to increase beyond 8, `max_seq_len` can be reduced or a smaller Qwen model substituted.

### 2. Model `position_ids` support

For batched decode in continuous batching, sequences are at different KV positions and need different RoPE embeddings. The model forward methods gain an optional `position_ids` parameter.

**New signature** (all three models — `LlamaModel`, `Qwen3Model`, `Gemma3Model`):

```python
def forward(
    self,
    input_ids: Tensor,
    kv_cache: KVCacheProtocol | None = None,
    padding_mask: Tensor | None = None,
    position_ids: Tensor | None = None,  # NEW
) -> Tensor:
    """Forward pass.

    Args:
        input_ids: Token IDs, shape ``[batch, seq_len]``.
        kv_cache: Optional KV cache for incremental decoding.
        padding_mask: Optional boolean mask, shape ``[batch, total_kv_len]``.
        position_ids: Optional position indices, shape ``[batch, seq_len]``.
            When provided, RoPE cos/sin are looked up per-sequence instead
            of using ``kv_cache.seq_len``.  Used for continuous batching
            decode where sequences are at different positions.
            When ``None``, behavior is identical to Phase 4.
    """
```

**RoPE position lookup change** (inside `LlamaModel.forward` and `Qwen3Model.forward`):

```python
if position_ids is not None:
    # Continuous batching: per-sequence positions
    # position_ids: [batch, seq_len] (e.g., [batch, 1] for decode)
    cos = self.cos[position_ids]  # [batch, seq_len, head_dim]
    sin = self.sin[position_ids]  # [batch, seq_len, head_dim]
elif kv_cache is not None:
    pos = kv_cache.seq_len
    cos = self.cos[pos : pos + seq_len]  # [seq_len, head_dim]
    sin = self.sin[pos : pos + seq_len]
else:
    cos = self.cos[:seq_len]
    sin = self.sin[:seq_len]
```

**Gemma 3 dual RoPE tables**: `Gemma3Model` has two pairs of RoPE tables — `local_cos/sin` (for sliding-window layers, using `rope_local_base_freq`) and `global_cos/sin` (for full-attention layers, using `rope_theta`). The `position_ids` indexing must apply to both pairs:

```python
if position_ids is not None:
    local_cos = self.local_cos[position_ids]
    local_sin = self.local_sin[position_ids]
    global_cos = self.global_cos[position_ids]
    global_sin = self.global_sin[position_ids]
elif kv_cache is not None:
    pos = kv_cache.seq_len
    local_cos = self.local_cos[pos : pos + seq_len]
    local_sin = self.local_sin[pos : pos + seq_len]
    global_cos = self.global_cos[pos : pos + seq_len]
    global_sin = self.global_sin[pos : pos + seq_len]
# ... else branch unchanged ...
```

The per-layer dispatch (`local_cos/sin` for sliding-window layers, `global_cos/sin` for full-attention layers) remains unchanged — only the position lookup at the top of `forward` changes.

**Gemma 3 sliding-window mask limitation**: when `position_ids` is provided during batched decode, the sliding-window cutoff in `Gemma3Model.forward` is computed from `kv_len = kv_cache.seq_len + 1` where `kv_cache.seq_len = max(slot_seq_lens)`. This means the cutoff is based on the longest sequence, not per-sequence. For sequences at very different positions in the same batch, shorter sequences may have an overly conservative sliding window boundary. This is the same limitation as Phase 4's static batching (the cutoff is batch-wide, not per-sequence) and does not cause incorrect results — it only means some shorter sequences attend to slightly more positions than strictly necessary. Phase 6 (paged attention) enables per-sequence mask construction.

**Mask construction**: unchanged. When `position_ids` is provided, the model still uses `padding_mask` and `kv_cache.seq_len` for mask width (`kv_len = kv_cache.seq_len + seq_len`). The `DecodeCacheView.seq_len` property returns `max(slot_seq_lens)`, giving the correct mask width. Per-sequence valid lengths are handled by `padding_mask` as in Phase 4.

**`kv_cache.advance()` call**: unchanged. The model calls `kv_cache.advance(seq_len)` at the end of forward. For `PrefillCacheView`, this advances one slot. For `DecodeCacheView`, this advances all active slots.

**Last-position optimization**: when `position_ids` is provided with a `kv_cache`, skip the `x = x[:, -1:, :]` optimization (same as when `padding_mask` is provided). During decode (`seq_len=1`) this is a no-op anyway.

**Backward compatibility**: when `position_ids is None`, all behavior is identical to Phase 4. Existing tests pass without modification.

### 3. Triton RoPE kernel change (`src/infer/kernels/rope.py`)

The Triton kernel currently assumes `cos` and `sin` are 2D `[seq_len, head_dim]`. For batched decode with `position_ids`, cos/sin are 3D `[batch, seq_len, head_dim]`. A minimal kernel change handles both cases via stride.

**Kernel change** — add `stride_cos_batch` parameter to the kernel function signature and use it in offset calculation:

```python
# _rope_kernel signature gains one parameter:
@triton.jit
def _rope_kernel(
    Q, K, COS, SIN, Q_OUT, K_OUT,
    stride_q_batch, stride_q_head, stride_q_seq,
    stride_qo_batch, stride_qo_head, stride_qo_seq,
    stride_k_batch, stride_k_head, stride_k_seq,
    stride_ko_batch, stride_ko_head, stride_ko_seq,
    stride_cos_batch,  # NEW: 0 for 2D cos/sin, real stride for 3D
    stride_cos_seq,
    seq_len, num_q_heads, total_heads,
    HALF_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # ... (pid decomposition unchanged) ...

    # Before (Phase 4):
    # cos_offset = seq_idx * stride_cos_seq

    # After (Phase 5):
    cos_offset = batch_idx * stride_cos_batch + seq_idx * stride_cos_seq

    # ... (rest of kernel unchanged) ...
```

**Python wrapper change** — replace the 2D-only validation with 2D/3D dispatch:

```python
def triton_apply_rope(q, k, cos, sin):
    if cos.ndim == 2:
        # Phase 4 path: [seq_len, head_dim], broadcast across batch
        stride_cos_batch = 0
    elif cos.ndim == 3:
        # Phase 5 path: [batch, seq_len, head_dim], per-sequence positions
        stride_cos_batch = cos.stride(0)
    else:
        raise ValueError(
            f"cos must be 2-D [seq_len, head_dim] or 3-D [batch, seq_len, head_dim], "
            f"got shape {cos.shape}"
        )

    # Launch kernel with stride_cos_batch before stride_cos_seq:
    _rope_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        # ... q/k strides unchanged ...
        stride_cos_batch,      # NEW
        cos.stride(-2),        # stride_cos_seq (was cos.stride(0) for 2D)
        seq_len, num_heads, total_heads,
        HALF_DIM=half_dim, BLOCK_SIZE=BLOCK_SIZE,
    )
```

When `cos` is 2D, `stride_cos_batch = 0` makes `batch_idx * 0 = 0`, so cos is broadcast across the batch — identical to Phase 4. When cos is 3D, the kernel uses the real batch stride for per-sequence lookup. The kernel's grid and other strides remain unchanged. Note: `cos.stride(-2)` gives `stride_cos_seq` for both 2D (where dim -2 is dim 0) and 3D (where dim -2 is dim 1).

### 4. Request changes (`src/infer/engine/request.py`)

Add a slot index field for cache pool tracking:

```python
@dataclass
class Request:
    # ... existing fields ...

    # Cache pool slot — assigned during prefill, freed on retire.
    slot_idx: int | None = field(default=None, repr=False)
```

The `slot_idx` is set by `ContinuousRunner` during prefill and read by the engine to free the slot on retire.

### 5. ContinuousScheduler (`src/infer/engine/scheduler.py`)

Per-step scheduler that retires finished requests and admits new ones every step.

```python
@dataclass
class ScheduleOutput:
    """Result of one continuous scheduling step."""
    prefill: list[Request]    # newly admitted, need prefill
    decode: list[Request]     # active, need one decode step
    retired: list[Request]    # finished since last step, need slot freeing


class ContinuousScheduler:
    """Per-step scheduler for continuous batching.

    Each ``schedule()`` call:
    1. Retires finished/failed requests from the active set.
    2. Counts available capacity (max_batch_size - active count).
    3. Admits new requests from the waiting queue up to capacity.
    4. Returns the prefill list, decode list, and retired list.

    FCFS ordering: requests are admitted in arrival order (deque insertion order).
    Active requests run to completion. No bounded wait timer:
    requests are admitted immediately when slots are available.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []

    def add_request(self, request: Request) -> bool:
        """Add a request to the waiting queue.

        Returns ``False`` if the queue is full (caller should return 503).
        """
        if len(self.waiting) >= self.config.max_waiting_requests:
            return False
        self.waiting.append(request)
        return True

    def schedule(self) -> ScheduleOutput:
        """Run one scheduling step.

        1. Retire finished/failed requests from active set.
        2. Admit up to free capacity from waiting queue (FCFS).
        3. Return prefill, decode, and retired lists.
        """
        # 1. Retire finished.
        retired = [r for r in self.active if r.state in (RequestState.FINISHED, RequestState.FAILED)]
        self.active = [r for r in self.active if r.state not in (RequestState.FINISHED, RequestState.FAILED)]

        # 2. Admit new requests.
        capacity = self.config.max_batch_size - len(self.active)
        new: list[Request] = []
        while self.waiting and len(new) < capacity:
            new.append(self.waiting.popleft())
        self.active.extend(new)

        # 3. Partition active into decode and prefill.
        decode = [r for r in self.active if r.state == RequestState.DECODE]

        return ScheduleOutput(prefill=new, decode=decode, retired=retired)

    def has_work(self) -> bool:
        """True if there are active or waiting requests."""
        return bool(self.active) or bool(self.waiting)
```

**No bounded wait timer**: unlike `StaticScheduler`, the continuous scheduler admits requests immediately when slots are available. There's no batch formation to wait for — each step processes whatever is ready. Latency is minimized because a new request starts prefilling in the very next step after arrival.

### 6. ContinuousRunner (`src/infer/engine/continuous_runner.py`)

Manages the `SlottedKVCache` pool and executes two-phase forward passes.

```python
class ContinuousRunner:
    """Executes forward passes for continuous batching.

    Manages a pre-allocated SlottedKVCache pool. Each engine step runs:
    1. Batched decode for all active decode requests.
    2. Prefill for newly admitted requests — single request uses individual
       prefill (no padding overhead), multiple requests use batched prefill
       (right-padded, one forward pass via BatchedPrefillCacheView).

    Args:
        model: A loaded model with a ``.config`` attribute.
        tokenizer: Tokenizer for text decoding and EOS detection.
        config: Engine configuration.
    """

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

        # Pre-allocate cache pool at startup.
        model_config = getattr(model, "config", None)
        if model_config is None:
            raise TypeError("model must have a .config attribute")
        self.cache_pool = SlottedKVCache.from_model_config(
            model_config,
            max_seq_len=config.max_seq_len,
            max_batch_size=config.max_batch_size,
            dtype=self.dtype,
            device=config.device,
        )

        # Per-request state tracking.
        self._prev_text_lens: dict[str, int] = {}

    def step(
        self, prefill: list[Request], decode: list[Request],
    ) -> list[tuple[Request, StepOutput]]:
        """Run one engine step: batched decode then prefills.

        Dispatches prefill based on count: N=1 uses individual prefill
        (PrefillCacheView, no padding), N>1 uses batched prefill
        (BatchedPrefillCacheView, right-padded to longest prompt).
        """
        outputs: list[tuple[Request, StepOutput]] = []

        # Phase 1: Batched decode.
        if decode:
            decode_outputs = self._batched_decode(decode)
            outputs.extend(zip(decode, decode_outputs, strict=True))

        # Phase 2: Prefill — individual or batched.
        if len(prefill) == 1:
            output = self._prefill_one(prefill[0])
            outputs.append((prefill[0], output))
        elif len(prefill) > 1:
            prefill_outputs = self._prefill_batch(prefill)
            outputs.extend(zip(prefill, prefill_outputs, strict=True))

        return outputs

    def free_slot(self, slot_idx: int, request_id: str) -> None:
        """Release a cache slot and clean up tracking state."""

    def _prefill_one(self, req: Request) -> StepOutput:
        """Prefill a single request using PrefillCacheView."""

    def _prefill_batch(self, requests: list[Request]) -> list[StepOutput]:
        """Batched prefill for multiple requests using BatchedPrefillCacheView."""

    def _batched_decode(self, requests: list[Request]) -> list[StepOutput]:
        """Batched decode for all active requests using DecodeCacheView."""
```

**`_prefill_one` flow**:

```python
@torch.inference_mode()
def _prefill_one(self, req: Request) -> StepOutput:
    device = self.config.device

    # Allocate a cache slot.
    slot = self.cache_pool.allocate_slot()
    req.slot_idx = slot

    # Build input tensor [1, prompt_len].
    input_ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=device)

    # Create single-slot cache view.
    view = self.cache_pool.prefill_view(slot)

    # Forward pass — existing model code works unchanged.
    req.state = RequestState.PREFILL
    logits = self.model(input_ids, kv_cache=view)
    # logits: [1, 1, vocab_size] — padding_mask is None, so the model applies the
    # last-position optimization (x = x[:, -1:, :]) before the LM head.

    # Sample first token.
    next_logits = logits[0, -1, :]
    context = req.prompt_token_ids
    token = sample_token(next_logits, context, req.sampling_params, req.generator)
    req.generated_token_ids.append(token)
    req.state = RequestState.DECODE

    # Initialize text tracking.
    text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
    text_delta = text
    self._prev_text_lens[req.request_id] = len(text)

    # Check stop conditions.
    finished, reason = self._check_stop(req, token)
    if finished:
        req.state = RequestState.FINISHED
        req.finish_reason = reason
        if reason == "stop":
            text_delta = self._truncate_at_stop(text, 0, req)

    return self._make_step_output(req, token, text_delta, finished, reason)
```

Note: during prefill, `padding_mask` is not passed (single request, no padding needed). The model's existing Phase 3 path handles this: `padding_mask is None` → last-position optimization applied → logits shape `[1, 1, vocab_size]`.

**`_prefill_batch` flow** (used when multiple requests arrive in the same step):

```python
@torch.inference_mode()
def _prefill_batch(self, requests: list[Request]) -> list[StepOutput]:
    device = self.config.device
    batch_size = len(requests)

    # Allocate cache slots for all requests.
    slots: list[int] = []
    prompt_lens: list[int] = []
    for req in requests:
        slot = self.cache_pool.allocate_slot()
        req.slot_idx = slot
        slots.append(slot)
        prompt_lens.append(len(req.prompt_token_ids))

    # Right-pad prompts to longest and build input tensor.
    max_prompt_len = max(prompt_lens)
    padded = [
        req.prompt_token_ids + [0] * (max_prompt_len - len(req.prompt_token_ids))
        for req in requests
    ]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)  # [batch, max_prompt_len]

    # Build padding mask: True for real tokens, False for padding.
    padding_mask = torch.zeros(batch_size, max_prompt_len, dtype=torch.bool, device=device)
    for i, plen in enumerate(prompt_lens):
        padding_mask[i, :plen] = True

    # Create batched prefill cache view — scatter-writes to per-slot positions.
    view = self.cache_pool.batched_prefill_view(slots, prompt_lens)

    # Forward pass — one batched pass for all new requests.
    for req in requests:
        req.state = RequestState.PREFILL
    logits = self.model(input_ids, kv_cache=view, padding_mask=padding_mask)
    # logits: [batch, max_prompt_len, vocab_size] (padding_mask present, no last-position opt)

    # Sample per request at its actual last prompt position.
    outputs: list[StepOutput] = []
    for i, req in enumerate(requests):
        next_logits = logits[i, prompt_lens[i] - 1, :]
        context = req.prompt_token_ids
        token = sample_token(next_logits, context, req.sampling_params, req.generator)
        req.generated_token_ids.append(token)
        req.state = RequestState.DECODE

        # Initialize text tracking and check stop conditions.
        text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
        text_delta = text
        self._prev_text_lens[req.request_id] = len(text)

        finished, reason = self._check_stop(req, token)
        if finished:
            req.state = RequestState.FINISHED
            req.finish_reason = reason
            if reason == "stop":
                text_delta = self._truncate_at_stop(text, 0, req)

        outputs.append(self._make_step_output(req, token, text_delta, finished, reason))
    return outputs
```

Key differences from `_prefill_one`: (1) right-pads prompts and passes `padding_mask` so the model skips the last-position optimization, returning full logits `[batch, max_prompt_len, vocab_size]`; (2) samples at each request's actual last position (`logits[i, prompt_lens[i] - 1, :]`); (3) uses `BatchedPrefillCacheView` which scatter-writes each batch element's K/V to its assigned pool slot and sets per-slot `seq_lens` to actual prompt lengths (not padded length) on `advance()`.

**`_batched_decode` flow**:

```python
@torch.inference_mode()
def _batched_decode(self, requests: list[Request]) -> list[StepOutput]:
    device = self.config.device
    batch_size = len(requests)

    # Gather active slots and build inputs.
    active_slots = [req.slot_idx for req in requests]
    tokens = [req.generated_token_ids[-1] for req in requests]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)  # [batch, 1]

    # Build position_ids: each sequence's current position.
    positions = [self.cache_pool.seq_lens[slot] for slot in active_slots]
    position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(1)  # [batch, 1]

    # Build padding mask: True for valid positions per sequence.
    # Each slot has seq_lens[slot] cached tokens at positions [0, seq_lens[slot]).
    # The current decode token writes at position seq_lens[slot], so we mark
    # [0, seq_lens[slot]] inclusive as True (same pattern as Phase 4's runner,
    # which marks the current decode position before the forward pass).
    decode_view = self.cache_pool.decode_view(active_slots)
    max_kv_len = decode_view.seq_len + 1  # +1 for the token about to be written
    padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
    for i, slot in enumerate(active_slots):
        padding_mask[i, : self.cache_pool.seq_lens[slot] + 1] = True

    # Forward pass with position_ids for per-sequence RoPE.
    logits = self.model(
        input_ids,
        kv_cache=decode_view,
        padding_mask=padding_mask,
        position_ids=position_ids,
    )
    # logits: [batch, 1, vocab_size]

    # Sample per request.
    outputs: list[StepOutput] = []
    for i, req in enumerate(requests):
        context = req.prompt_token_ids + req.generated_token_ids
        token = sample_token(logits[i, -1, :], context, req.sampling_params, req.generator)
        req.generated_token_ids.append(token)

        # Compute text_delta.
        text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
        prev_len = self._prev_text_lens.get(req.request_id, 0)
        text_delta = text[prev_len:]

        finished, reason = self._check_stop(req, token)
        if finished:
            req.state = RequestState.FINISHED
            req.finish_reason = reason
            if reason == "stop":
                text_delta = self._truncate_at_stop(text, prev_len, req)
        self._prev_text_lens[req.request_id] = len(text)
        outputs.append(self._make_step_output(req, token, text_delta, finished, reason))

    return outputs
```

**Stop condition checking, text delta tracking, and StepOutput construction** reuse the same logic as `ModelRunner` (Phase 4). These are extracted into `src/infer/engine/runner_helpers.py` and imported by both runners.

### 7. Engine changes (`src/infer/engine/engine.py`)

The engine dispatches to the appropriate scheduler and runner based on `batching_mode`.

```python
class Engine:
    def __init__(self, config: EngineConfig) -> None:
        # ... load model, tokenizer (unchanged) ...

        if config.batching_mode == "continuous":
            self.scheduler = ContinuousScheduler(config)
            self.runner = ContinuousRunner(model, tokenizer, config)
        else:
            self.scheduler = StaticScheduler(config)
            self.runner = ModelRunner(model, tokenizer, config)

    def step(self) -> None:
        if isinstance(self.scheduler, ContinuousScheduler):
            self._step_continuous()
        else:
            self._step_static()

    def _step_continuous(self) -> None:
        """Engine step for continuous batching."""
        schedule = self.scheduler.schedule()

        # Free retired slots (free_slot handles both cache and _prev_text_lens cleanup).
        for req in schedule.retired:
            if req.slot_idx is not None:
                self.runner.free_slot(req.slot_idx, req.request_id)

        if not schedule.prefill and not schedule.decode:
            return

        try:
            outputs = self.runner.step(schedule.prefill, schedule.decode)
            for req, output in outputs:
                if req.output_queue is not None:
                    req.output_queue.put_nowait(output)
        except Exception as exc:
            for req in schedule.prefill + schedule.decode:
                req.state = RequestState.FAILED
                req.error = str(exc)
                if req.output_queue is not None:
                    req.output_queue.put_nowait(
                        StepOutput(
                            request_id=req.request_id,
                            token_id=None,
                            text_delta="",
                            finished=True,
                            finish_reason=None,
                            error=str(exc),
                        )
                    )

    def _step_static(self) -> None:
        """Engine step for static batching (Phase 4 behavior, unchanged)."""
        # ... existing Phase 4 step() logic ...
```

The `from_components` factory method is similarly updated to accept the `batching_mode` and create the appropriate scheduler/runner.

### 8. Config changes (`src/infer/engine/config.py`)

```python
_VALID_BATCHING_MODES = {"static", "continuous"}  # Phase 5: add "continuous"
```

No new config fields needed. The existing `batching_mode`, `scheduler_policy`, and `max_batch_size` fields are sufficient. `scheduler_policy` validation remains `{"fcfs"}` — additional policies are deferred.

### 9. Benchmark updates

Run the existing `bench_serving.py` workloads with `batching_mode="continuous"` and record results alongside the Phase 4 static batching baselines.

**Server launch for continuous batching**:
```bash
uv run python -m infer.server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --batching-mode continuous \
    --max-batch-size 8
```

**Key comparison**: the `continuous_batching` workload (32 requests, mixed prompt lengths [64-512], uniform 4 RPS) — this workload demonstrates the tail latency improvement from continuous batching, since short requests no longer wait for long ones.

**New metrics to highlight**:
- P99 TTFT reduction vs static batching (short requests served faster)
- Throughput comparison (should be similar or improved, since the batch stays fuller)
- P99 ITL comparison (may increase slightly due to prefill stalls — Phase 7 addresses this)

Results are appended to `benchmarks/log/SERVING_LOG.md` in a Phase 5 section.

---

## File Layout

New and modified files:

```
src/infer/
├── cache/
│   ├── __init__.py             # MODIFIED: export new types
│   ├── protocol.py             # NEW: KVCacheProtocol
│   ├── simple.py               # UNCHANGED: KVCache (still used for standalone generation)
│   └── slotted.py              # NEW: SlottedKVCache, PrefillCacheView, DecodeCacheView, BatchedPrefillCacheView
├── engine/
│   ├── __init__.py             # MODIFIED: export ContinuousScheduler, ContinuousRunner, ScheduleOutput
│   ├── config.py               # MODIFIED: add "continuous" to valid batching modes
│   ├── request.py              # MODIFIED: add slot_idx field
│   ├── scheduler.py            # MODIFIED: add ContinuousScheduler, ScheduleOutput
│   ├── runner.py               # MODIFIED: ModelRunner imports from runner_helpers
│   ├── runner_helpers.py       # NEW: shared _check_stop, _truncate_at_stop, _make_step_output
│   ├── continuous_runner.py    # NEW: ContinuousRunner
│   ├── engine.py               # MODIFIED: dispatch based on batching_mode
│   ├── sampler.py              # UNCHANGED
│   └── generate.py             # UNCHANGED
├── kernels/
│   └── rope.py                 # MODIFIED: add stride_cos_batch to kernel + wrapper
├── models/
│   ├── llama.py                # MODIFIED: add position_ids, KVCacheProtocol type hint
│   ├── qwen3.py                # MODIFIED: add position_ids, KVCacheProtocol type hint
│   ├── gemma3.py               # MODIFIED: add position_ids + dual RoPE, KVCacheProtocol type hint
│   └── common.py               # UNCHANGED
└── server/
    ├── api.py                  # UNCHANGED
    ├── __main__.py             # MODIFIED: add --batching-mode CLI arg
    └── __init__.py             # UNCHANGED

benchmarks/
├── bench_serving.py            # UNCHANGED (workloads already defined)
└── log/
    └── SERVING_LOG.md          # MODIFIED: add Phase 5 results section

tests/
├── unit/
│   ├── test_slotted_cache.py   # NEW: SlottedKVCache, views
│   ├── test_continuous_scheduler.py  # NEW: ContinuousScheduler
│   ├── test_continuous_runner.py     # NEW: ContinuousRunner
│   └── test_engine_config.py   # MODIFIED: test "continuous" mode validation
├── integration/
│   ├── test_api.py             # MODIFIED: add continuous batching API tests
│   └── test_logits_parity.py   # MODIFIED: verify position_ids doesn't break parity
└── stress/
    └── test_backpressure.py    # MODIFIED: add continuous batching stress tests
```

---

## Testing Plan

### Slotted KV cache tests (`tests/unit/test_slotted_cache.py`)

**Slot allocation**:
- Allocate all slots → `free_slot_count() == 0`.
- Allocate one more → `RuntimeError`.
- Free a slot → `free_slot_count() == 1`. Allocate again → succeeds.

**PrefillCacheView**:
- Create view for slot 3. Call `update()` with K/V of shape `[1, heads, prompt_len, dim]`.
- Verify data written to `pool.k[layer, 3, :, :prompt_len, :]`.
- Verify `view.seq_len == prompt_len` after `advance(prompt_len)`.
- Verify `pool.seq_lens[3] == prompt_len`.

**DecodeCacheView**:
- Set up pool with slots 0, 2, 5 at seq_lens [100, 50, 200].
- Create decode view. Verify `view.seq_len == 200`.
- Call `update()` with K/V of shape `[3, heads, 1, dim]`.
- Verify data written to correct per-slot positions (100, 50, 200).
- Verify gathered cache shape `[3, heads, 201, dim]`.
- Call `advance(1)`. Verify pool.seq_lens updated to [101, 51, 201].

**DecodeCacheView — single slot** (degenerate case):
- Set up pool with only slot 2 active at seq_len 75.
- Create decode view with `active_slots=[2]`. Verify `view.seq_len == 75`.
- Call `update()` with K/V of shape `[1, heads, 1, dim]`.
- Verify gathered cache shape `[1, heads, 76, dim]`.

**BatchedPrefillCacheView**:
- Create view for slots [0, 2] with prompt_lens [10, 20] and padded_len=20.
- Call `update()` with K/V of shape `[2, heads, 20, dim]`.
- Verify scatter-write: slot 0 gets K/V at positions 0-19, slot 2 gets K/V at positions 0-19.
- Verify `update()` returns the input K/V directly (no gather).
- Call `advance(20)`. Verify `pool.seq_lens[0] == 10` and `pool.seq_lens[2] == 20` (actual prompt lengths, not padded).
- Verify pool seq_lens overflow raises assertion error.
- Verify KVCacheProtocol satisfaction.

**Decode after batched prefill**:
- Batch-prefill slots [0, 1] with prompt_lens [5, 10] (padded to 10).
- Create decode view for [0, 1]. Verify `view.seq_len == 10`.
- Run decode update. Verify data written at correct positions (5 and 10 respectively).

**Slot reuse**:
- Prefill slot 0 with prompt_len=50. Free slot 0. Prefill slot 0 again with prompt_len=30.
- Verify seq_lens[0] == 30 (not 50+30).
- Verify data in positions 0-29 is from the new prefill.

### ContinuousScheduler tests (`tests/unit/test_continuous_scheduler.py`)

**Per-step admit/retire**:
- Add 3 requests. Schedule → `prefill=[A, B, C]`, `decode=[]`, `retired=[]`.
- Mark A, B, C as DECODE. Schedule → `prefill=[]`, `decode=[A, B, C]`, `retired=[]`.
- Mark A as FINISHED. Add D. Schedule → `prefill=[D]`, `decode=[B, C]`, `retired=[A]`.

**Capacity limit**:
- `max_batch_size=2`. Add A, B, C. Schedule → `prefill=[A, B]`, C stays in waiting.
- Mark A as FINISHED. Schedule → `retired=[A]`, `prefill=[C]`, `decode=[B]`.

**Queue overflow**:
- `max_waiting_requests=2`. Add A, B → success. Add C → returns `False`.

**FCFS ordering**:
- Add A, B, C in order. Schedule → `prefill=[A, B, C]` (same order).

**has_work**:
- Empty → `False`. Add request → `True`. Schedule and finish all → `False`.

**No bounded wait timer**:
- Add 1 request with `max_batch_size=8`. Schedule immediately → `prefill=[request]`.
  (Unlike StaticScheduler, no timer — admits immediately when slots available.)

### ContinuousRunner tests (`tests/unit/test_continuous_runner.py`)

Uses mock models (same pattern as `test_runner.py`).

**Single-request prefill**:
- Prefill one request. Verify slot allocated, `req.slot_idx` set, first token generated, state is DECODE.

**Batched prefill (multiple requests)**:
- Submit 3 requests simultaneously. Verify all get slots, first tokens, state is DECODE.
- Verify batched path is used (N>1 dispatches to `_prefill_batch`).
- Verify per-slot `seq_lens` match actual prompt lengths (not padded length).
- Verify correct logits are sampled at each request's last real token position.

**Batched prefill then decode**:
- Batch-prefill 2 requests. Run decode step.
- Verify both requests get correct second tokens with correct position_ids.

**Single prefill uses individual path**:
- Submit 1 request. Verify `_prefill_one` is used (no padding overhead).

**Batched decode**:
- Prefill 3 requests (individually). Run decode step.
- Verify all 3 get a second token, position_ids are correct per request.

**Mixed step (prefill + decode)**:
- Prefill A and B. Decode step for A and B. Admit C.
- Run `step(prefill=[C], decode=[A, B])`.
- Verify A and B get next token, C gets first token.

**Request completion and slot freeing**:
- Prefill A with `max_new_tokens=2`. After 2 decode steps, A finishes.
- Verify `free_slot(A.slot_idx)` works. Slot available for reuse.

**EOS during prefill**:
- Mock model returns EOS logits. Verify request finishes in PREFILL step with `finish_reason="eos"`.

**EOS during batched prefill**:
- Mock model returns EOS logits for one request in a batch. Verify that request finishes while others continue to DECODE.

### Model position_ids tests

**Parity with existing code**:
- Single-request decode with `position_ids=[[pos]]` vs Phase 4 path with `kv_cache.seq_len=pos`.
- Verify logits match exactly.

**Batched decode parity**:
- Two requests at positions 10 and 20. Run batched decode with `position_ids=[[10], [20]]`.
- Run each individually at their respective positions.
- Verify per-request logits match.

**Gemma 3 dual RoPE with position_ids**:
- Verify Gemma3Model applies `position_ids` indexing to both `local_cos/sin` and `global_cos/sin`.
- Single-request decode: `position_ids=[[pos]]` matches Phase 4 path for both layer types.

### RoPE kernel tests

**2D cos/sin (backward compat)**:
- Run existing RoPE tests — all pass unchanged.

**3D cos/sin**:
- Provide `cos` as `[batch, seq_len, dim]` with different values per batch element.
- Verify output matches element-wise rotation at per-batch positions.
- Compare against PyTorch reference for correctness.

### Integration tests (`tests/integration/test_api.py`)

**Continuous batching end-to-end**:
- Start engine with `batching_mode="continuous"`.
- Send request. Verify SSE stream with token events and done event.

**Concurrent requests with different lengths**:
- Send 4 requests: prompts of lengths [32, 128, 256, 512], generate [16, 32, 64, 128].
- Verify all complete correctly. Verify short requests finish first (don't wait for long ones).

### Stress tests (`tests/stress/test_backpressure.py`)

**No starvation under load**:
- Send `max_batch_size * 3` requests concurrently.
- Verify all complete within a bounded time (no infinite wait).

**Queue full with continuous batching**:
- `max_waiting_requests=4`, `max_batch_size=2`. Send 8 requests.
- At most 2 active + 4 waiting = 6 accepted. 2 get 503.

---

## Design Decisions

**Per-slot KV cache pool instead of per-batch allocation**: static batching allocates a new KV cache per batch and frees it when the batch completes. Continuous batching has no batch boundaries, so the cache must persist across steps. A pre-allocated pool with slot management is the natural extension. Pre-allocation also avoids GPU memory fragmentation from repeated alloc/free.

**Adaptive prefill (individual or batched)**: when a single request arrives, `_prefill_one` uses `PrefillCacheView` with no padding overhead. When multiple requests arrive in the same step, `_prefill_batch` right-pads them to the longest prompt and runs one batched forward pass via `BatchedPrefillCacheView`, amortizing weight loading. The dispatch threshold is N=1 (individual) vs N>1 (batched). Batched prefill matters most under high arrival rates where multiple requests queue up between decode steps — benchmarks show it recovers throughput that would otherwise collapse under bursty arrivals (e.g., Qwen3-4B chunked_prefill workload: 14.4 tok/s with individual prefill → 137.3 tok/s with batched). The `BatchedPrefillCacheView` scatter-writes each batch element's K/V to its assigned pool slot and returns the input K/V directly (no gather needed). Its `advance()` sets per-slot `seq_lens` to actual prompt lengths, not the padded length. This scatter-write primitive is also the foundation for Phase 7's chunked prefill, where partial prefill chunks are written to different slot positions.

**Decode-first step order**: running the decode batch before prefilling new arrivals prioritizes inter-token latency for in-flight requests. A newly arrived request waits one step before prefill begins — this adds ~10ms to TTFT (one decode step) but keeps ITL stable for existing requests. The alternative (prefill-first) risks stalling all decode requests when a long prompt arrives. Phase 7's chunked prefill eliminates this tradeoff entirely.

**Gather-based decode cache view**: `DecodeCacheView.update()` gathers K/V from the pool using advanced indexing, creating a copy for each layer. This roughly doubles KV memory traffic compared to static batching's contiguous cache. Acceptable at the dev model scale (~3.5 GB extra reads for Llama 3B at 4K sequence length, 8 active requests). The alternative — processing all `max_batch_size` slots including inactive ones — wastes MLP/LM-head compute on empty slots. Phase 6 (paged attention) eliminates the gather overhead entirely with block-level lookup.

**`position_ids` on model forward, not on Attention**: the position lookup (`cos = self.cos[position_ids]`) and the `position_ids` parameter are added to the model-level forward methods, not to `Attention` or `apply_rope`. This keeps the RoPE kernel changes minimal (just a stride parameter) and confines the new parameter to the same level as `padding_mask`. The `Attention` class and transformer blocks remain unchanged — they receive cos/sin as before and pass them through to `apply_rope` → `triton_apply_rope`. The only change in the data flow is that cos/sin may be 3D `[batch, seq_len, head_dim]` instead of 2D `[seq_len, head_dim]`, which the updated Triton kernel handles via stride.

**Triton kernel stride trick for batched RoPE**: instead of adding a separate Python fallback for batched cos/sin, the Triton kernel gains one stride parameter (`stride_cos_batch`). When cos is 2D (static batching), the stride is 0 and the kernel broadcasts across batch — identical behavior to Phase 4. When cos is 3D (continuous batching), the stride indexes per-batch-element. This is a single line change in the kernel offset calculation and requires no conditional branching.

**FCFS only for Phase 5**: the overall design mentions three scheduler policies (`fcfs`, `prefill_first`, `decode_round_robin`). Phase 5 implements only FCFS. The `OVERALL_DESIGN.md` open questions section says to revisit after Phase 5 policy benchmarking. Adding more policies to `ContinuousScheduler` is straightforward — the `schedule()` method just changes the admit/decode ordering — but is deferred to keep Phase 5 focused on the core continuous batching machinery.

**`KVCacheProtocol` for forward compatibility with paged attention**: Phase 5 introduces `PrefillCacheView` and `DecodeCacheView` alongside the original `KVCache`, all sharing the same `update`/`advance`/`seq_len` interface. A formal `Protocol` makes this contract explicit so Phase 6's paged cache views can be verified at type-check time without inheriting from a concrete class. Models type-hint `kv_cache: KVCacheProtocol | None` and never interact with the backing storage directly — only the view implementations change between phases. This means Phase 6 replaces `SlottedKVCache` with a `PagedKVCachePool` (block allocator + page tables) and provides new view classes, but `ContinuousScheduler`, `ContinuousRunner`'s structure, and all model code remain unchanged.

**Shared runner helpers**: `ContinuousRunner` and `ModelRunner` share the same `_check_stop`, `_truncate_at_stop`, `_make_step_output` logic. These are extracted into a shared module (`src/infer/engine/runner_helpers.py`) rather than duplicated, avoiding drift as both runners will coexist long-term. Both runners import from the shared module.

---

## Exit Criteria

1. Continuous batching produces correct output for all three benchmark models (verified by integration tests).
2. P99 TTFT improvement vs static batching on `continuous_batching` workload (short requests no longer wait for long ones).
3. No starvation in stress test: all requests complete within bounded time under sustained load.
4. Throughput on `continuous_batching` workload is comparable to or better than static batching.
5. All Phase 1-4 tests pass with `batching_mode="static"` (no regression).
6. New unit tests pass: slotted cache, continuous scheduler, continuous runner.
7. `position_ids` logits parity: batched decode with position_ids matches individual decode at same positions.
8. RoPE kernel handles both 2D and 3D cos/sin correctly (existing tests + new 3D test).
9. Benchmark results recorded in `SERVING_LOG.md` comparing static vs continuous batching.
10. `uv run ruff check .` and `uv run mypy .` pass cleanly.
11. v1 Definition of Done satisfied: continuous batching with streaming output, OpenAI-compatible endpoint, correctness tests pass.
