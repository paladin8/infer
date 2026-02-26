# Phase 7: Chunked Prefill

## Goal

Break long prefills into fixed-size chunks that interleave with decode steps, preventing decode stalls when new requests with long prompts arrive.

In Phase 5/6, when a new request arrives, the runner prefills the entire prompt in a single forward pass before any decode step runs. A 2048-token prompt occupies the GPU for the full prefill duration, during which all active decode requests are blocked --- they receive no decode step and emit no tokens. This causes inter-token latency (ITL) spikes that grow linearly with prompt length. For real-time applications (chat, streaming), these spikes are unacceptable.

Chunked prefill splits the prompt into fixed-size chunks (e.g., 512 tokens) and processes one chunk per request per engine step alongside the regular decode batch. All pending chunks --- regardless of each request's prefill progress --- are batched into a single padded forward pass, amortizing weight loading across all concurrent prefills. Each step's compute is bounded: `chunk_size * num_prefill_chunks + num_decode_requests * 1` tokens. Decode requests get serviced every step, keeping ITL stable regardless of incoming prompt length.

The tradeoff is clear: prefill throughput decreases (a 2048-token prompt now takes 4 steps instead of 1), but ITL stability improves dramatically under concurrent load.

Benchmark models: `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-4B`, `google/gemma-3-1b-it` (same as Phases 4-6).

---

## Architecture

```text
             Engine._step_continuous()
             +------------------------------------------------------+
             |  1. scheduler.retire()           retire finished      |
             |  2. free_slot() + cleanup()      free resources       |
             |  3. free_kv_tokens()             query budget         |
             |  4. scheduler.admit()            admit new requests   |
             |  5. scheduler.prefill_requests() <-- NEW: all         |
             |     OR scheduler.admit() result      prefill work     |
             |  6. scheduler.decode_requests()  active decodes       |
             |  7. runner.step(prefill, decode)  execute             |
             |  8. push StepOutputs             to client queues     |
             +------------------------------------------------------+

  Without chunked prefill (Phase 5/6):        With chunked prefill (Phase 7):
  +-----------------------------------+        +-------------------------------------+
  | Step 1:                           |        | Step 1:                             |
  |   prefill(A, 2048 tokens)         |        |   prefill_chunk(A, tokens 0-511)    |
  |   decode(B,C) BLOCKED             |        |   decode(B,C)                       |
  |                                   |        |                                     |
  | Step 2:                           |        | Step 2:                             |
  |   decode(A,B,C)                   |        |   prefill_chunk(A, tokens 512-1023) |
  |                                   |        |   decode(B,C)                       |
  |                                   |        |                                     |
  |                                   |        | Step 3:                             |
  |                                   |        |   prefill_chunk(A, 1024-1535)       |
  |                                   |        |   decode(B,C)                       |
  |                                   |        |                                     |
  |                                   |        | Step 4:                             |
  |                                   |        |   prefill_chunk(A, 1536-2047)       |
  |                                   |        |   decode(A,B,C) -- A joins          |
  +-----------------------------------+        +-------------------------------------+
    B,C see ~15ms ITL spike                      B,C see stable ~4ms ITL
```

**Key design decisions:**

1. **Separate forward passes for prefill chunks and decode.** The runner processes decode and prefill chunks as independent forward passes within a single `step()` call. This matches the existing Phase 5/6 architecture. Piggybacking prefill tokens into the decode batch (as in Sarathi/vLLM) requires variable-length attention support and is out of scope.

2. **Batched chunked prefill across progress levels.** All pending prefill chunks --- from requests at different stages of their prefill --- are batched into a single padded forward pass, just like Phase 5's batched prefill for new requests. Each chunk's Q tokens are right-padded to `max_chunk_len`, KV is padded to `max_kv_len`, and per-element attention masks (constructed from `position_ids`) handle the different causal boundaries. This amortizes weight loading across all concurrent prefills: N chunks = 1 forward pass, not N. An optional `max_prefill_chunks_per_step` config caps the batch size to control per-step compute.

3. **Full prompt block allocation on first chunk.** When a request's first chunk is processed, the runner allocates blocks for the entire prompt eagerly (same as Phase 6). This avoids mid-prefill allocation failures. The admission budget check uses `free_kv_tokens` to ensure capacity exists before admitting. No change to the allocator.

4. **Pass `position_ids` for chunked prefill.** The runner provides explicit position IDs for each chunk, matching the decode-mode convention. This reuses the existing RoPE lookup path (`cos[position_ids]`) with no model architecture changes.

5. **Position-ids-based attention mask.** When `position_ids` is provided and `seq_len > 1`, models construct per-element causal masks from `position_ids` instead of using the shared `causal_mask(seq_len)`. This naturally handles batched chunks at different progress levels: each batch element's causal boundary is determined by its own position IDs. No changes needed to the standalone `causal_mask()` / `sliding_window_causal_mask()` helper functions.

6. **Backward compatibility.** When `use_chunked_prefill=False` (default), the engine, scheduler, and runner use the existing Phase 5/6 code paths unchanged. Chunked prefill is a pure opt-in feature flag.

---

## Deliverables

### 1. EngineConfig extensions (`src/infer/engine/config.py`)

Add `use_chunked_prefill` and `prefill_chunk_size` fields with validation.

```python
# Chunked prefill.
use_chunked_prefill: bool = False
prefill_chunk_size: int = 512
max_prefill_chunks_per_step: int | None = None  # None = no cap (batch all)
```

Validation rules (in `validate()`):

```python
if self.use_chunked_prefill:
    if self.batching_mode != "continuous":
        raise ValueError("Chunked prefill requires batching_mode='continuous'")
    if self.prefill_chunk_size < 1:
        raise ValueError(f"prefill_chunk_size must be >= 1, got {self.prefill_chunk_size}")
    if self.max_prefill_chunks_per_step is not None and self.max_prefill_chunks_per_step < 1:
        raise ValueError(
            f"max_prefill_chunks_per_step must be >= 1 or None, "
            f"got {self.max_prefill_chunks_per_step}"
        )
```

These match the compatibility rules already specified in `OVERALL_DESIGN.md` section 8.

**Tests:** Config validation (chunked + static -> error, chunk_size < 1 -> error, max_chunks < 1 -> error).

### 2. Request extensions (`src/infer/engine/request.py`)

Add a `prefill_progress` field to track how many prompt tokens have been prefilled.

```python
@dataclass
class Request:
    ...
    # Chunked prefill progress --- number of prompt tokens prefilled so far.
    # 0 means not yet started. Equal to len(prompt_token_ids) means complete.
    prefill_progress: int = field(default=0, repr=False)
```

State transitions with chunked prefill:

```text
WAITING --(first chunk)--> PREFILL --(intermediate chunks)--> PREFILL --(last chunk, sample)--> DECODE --> FINISHED
                                                                  |
                                   prefill_progress increments each step
```

The request stays in `PREFILL` across multiple steps until `prefill_progress == len(prompt_token_ids)`, at which point the runner samples the first token and transitions to `DECODE`.

### 3. Position-ids-based mask construction (all three models)

When `position_ids is not None` and `seq_len > 1`, models construct per-element causal masks directly from `position_ids` instead of calling the shared `causal_mask()` helper. This handles batched chunks at different progress levels without requiring a `kv_len` parameter on the mask helpers.

**Why `position_ids` instead of `kv_len`?** A fixed `causal_mask(q_len, kv_len)` assumes all sequences share the same `offset = kv_len - q_len`. When batching chunks at different progress levels (e.g., one at `start_pos=0`, another at `start_pos=512`), the offset varies per sequence. `position_ids` encodes per-sequence absolute positions, from which the correct per-element causal boundary can be derived: query at position `p_q` attends to KV position `p_k` iff `p_k <= p_q`.

**Llama / Qwen3** (global causal mask only):

```python
# In the (padding_mask is not None or position_ids is not None) branch:
if seq_len > 1:
    if position_ids is not None:
        # Per-element causal mask from position_ids.
        # position_ids: [batch, q_len] with absolute positions.
        # KV positions: [0, 1, ..., kv_len - 1] (contiguous from cache start).
        kv_positions = torch.arange(kv_len, device=x.device)          # [kv_len]
        # Attend where kv_pos <= q_pos (standard causal constraint).
        mask = torch.where(
            kv_positions[None, None, :] <= position_ids[:, :, None],   # [batch, q_len, kv_len]
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            torch.tensor(float("-inf"), dtype=x.dtype, device=x.device),
        ).unsqueeze(1)  # [batch, 1, q_len, kv_len]
    else:
        # Existing path: all sequences start at pos=0 (Phase 5 batched prefill).
        mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)
        mask = mask.expand(batch_size, -1, -1, -1).clone()
```

**Gemma 3** (dual masks: global causal + local sliding window):

```python
if seq_len > 1:
    if position_ids is not None:
        kv_positions = torch.arange(kv_len, device=x.device)
        causal = kv_positions[None, None, :] <= position_ids[:, :, None]
        in_window = (position_ids[:, :, None] - kv_positions[None, None, :]) < self.sliding_window
        global_mask = torch.where(
            causal,
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            torch.tensor(float("-inf"), dtype=x.dtype, device=x.device),
        ).unsqueeze(1)  # [batch, 1, q_len, kv_len]
        local_mask = torch.where(
            causal & in_window,
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
            torch.tensor(float("-inf"), dtype=x.dtype, device=x.device),
        ).unsqueeze(1)
    else:
        local_mask = sliding_window_causal_mask(
            seq_len, self.sliding_window, dtype=x.dtype, device=x.device,
        ).expand(batch_size, -1, -1, -1).clone()
        global_mask = causal_mask(
            seq_len, dtype=x.dtype, device=x.device,
        ).expand(batch_size, -1, -1, -1).clone()
```

In both cases, `padding_mask` is applied on top as before to mask out padded KV positions:

```python
if padding_mask is not None:
    pad_mask = ~padding_mask[:, None, None, :kv_len]
    mask.masked_fill_(pad_mask, float("-inf"))
    # For Gemma 3: also local_mask.masked_fill_(...) and global_mask.masked_fill_(...)
```

**Verification:** The position_ids-based mask is equivalent to the existing code for all non-chunked cases:

- **Batched decode** (`seq_len == 1`): Uses the `seq_len == 1` branch (zeros). No change.
- **Batched prefill** (`position_ids is None`): Uses the `else` branch with `causal_mask(seq_len)`. No change.
- **Single chunked prefill** (`position_ids = [start_pos, ..., start_pos + chunk_len - 1]`): Produces the correct rectangular causal mask: prior cached positions are always visible, current chunk positions follow causal ordering.
- **Batched chunked prefill** (different `position_ids` per element): Each element gets its own causal boundary. Padding mask handles per-element KV length differences.

**Tests:**
- Single-element mask from position_ids matches `causal_mask(seq_len)` for the square case (start_pos=0).
- Rectangular case (start_pos > 0): correct per-element causal boundaries.
- Batched case with different start positions: each element has correct mask.
- Gemma 3 sliding window: positions outside window masked for both local and global layers.

### 4. Model forward pass changes (all three models)

Two changes per model (`LlamaModel`, `Qwen3Model`, `Gemma3Model`):

**a) Remove chunked prefill assert.** The existing `assert pos == 0, "Chunked prefill not supported"` guard in the `position_ids is None, seq_len > 1` branch is no longer needed. With Phase 7, chunked prefill always provides `position_ids`, so this branch is never hit during chunked prefill. We remove the assert for cleanliness --- the code path `cos = self.cos[pos : pos + seq_len]` already handles nonzero `pos` correctly.

**b) Add position_ids-based mask construction.** In the `padding_mask is not None or position_ids is not None` branch, when `position_ids is not None` and `seq_len > 1`, construct the mask from position_ids as described in Deliverable 3. The existing `else` path (used by Phase 5 batched prefill where `position_ids is None`) remains unchanged.

No other model changes needed. RoPE lookup, attention, MLP, and output slicing all work correctly with the existing `position_ids` code path. The `x[:, -1:, :]` optimization is already guarded by `padding_mask is None and position_ids is None`.

### 5. Cache views for batched chunked prefill

Add a `batched_chunked_prefill_view()` factory method to `CachePoolProtocol`:

```python
class CachePoolProtocol(Protocol):
    ...
    def batched_chunked_prefill_view(
        self,
        slots: list[int],
        start_positions: list[int],
        chunk_lens: list[int],
    ) -> KVCacheProtocol:
        """Return a multi-slot view for batched chunked prefill.

        Each slot writes K/V at ``[start_pos, start_pos + chunk_len)`` and
        the view returns gathered KV padded to ``max(start_pos + chunk_len)``
        for batched attention.
        """
        ...
```

**Slotted backend** (`src/infer/cache/slotted.py`):

A new `BatchedChunkedPrefillCacheView` class. For each batch element, writes the chunk K/V at `[start_pos, start_pos + chunk_len)` in the slot's contiguous cache, then reads back `[0, start_pos + chunk_len)` padded to `max_kv_len`.

```python
class BatchedChunkedPrefillCacheView:
    """Multi-slot view for batched chunked prefill (slotted backend).

    Handles sequences at different prefill progress levels. Each batch
    element's chunk is written to its slot, then the full KV
    [0, start_pos + chunk_len) is read back and zero-padded to max_kv_len.
    """

    def __init__(
        self,
        pool: SlottedKVCache,
        slots: list[int],
        start_positions: list[int],
        chunk_lens: list[int],
    ) -> None:
        self.pool = pool
        self.slots = slots
        self.start_positions = start_positions
        self.chunk_lens = chunk_lens
        self.kv_lens = [s + c for s, c in zip(start_positions, chunk_lens)]
        self.max_kv_len = max(self.kv_lens)
        self.max_chunk_len = max(self.chunk_lens)
        # Model computes kv_len = pos + seq_len. Input seq_len = max_chunk_len.
        # So pos = max_kv_len - max_chunk_len makes kv_len = max_kv_len.
        self._seq_len = self.max_kv_len - self.max_chunk_len

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def is_paged(self) -> bool:
        return False

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write chunk KV to slots, gather full KV padded to max_kv_len.

        k, v: [batch, num_kv_heads, max_chunk_len, head_dim].
        Returns: [batch, num_kv_heads, max_kv_len, head_dim] (zero-padded).
        """
        batch_size = len(self.slots)
        heads, dim = k.shape[1], k.shape[3]
        device = k.device

        # Write each element's chunk to its slot.
        for i, slot in enumerate(self.slots):
            chunk_len = self.chunk_lens[i]
            start = self.start_positions[i]
            self.pool.k[layer_idx, slot, :, start : start + chunk_len, :] = (
                k[i, :, :chunk_len, :]
            )
            self.pool.v[layer_idx, slot, :, start : start + chunk_len, :] = (
                v[i, :, :chunk_len, :]
            )

        # Gather full KV [0, kv_len_i) per element, padded to max_kv_len.
        # Use slot index tensor for batched read.
        slot_idx = torch.tensor(self.slots, device=device)
        cached_k = self.pool.k[layer_idx, slot_idx, :, : self.max_kv_len, :]
        cached_v = self.pool.v[layer_idx, slot_idx, :, : self.max_kv_len, :]
        # Positions beyond kv_lens[i] contain stale/zero data but are masked
        # out by padding_mask in the model's attention, so this is safe.
        return cached_k, cached_v

    def advance(self, n: int) -> None:
        """Set per-slot seq_lens to actual kv_len (not padded)."""
        for i, slot in enumerate(self.slots):
            self.pool.seq_lens[slot] = self.kv_lens[i]
```

The slotted cache is contiguous per slot, so the "gather" is a single indexed slice --- no scatter/gather overhead. Positions beyond each element's actual `kv_len` may contain stale data, but the padding mask ensures they are ignored by attention.

**Paged backend** (`src/infer/cache/paged.py`):

A new `PagedBatchedChunkedPrefillCacheView` class. Unlike the slotted backend, the paged backend stores KV in scattered blocks, requiring explicit scatter-write and gather operations.

```python
class PagedBatchedChunkedPrefillCacheView:
    """Multi-sequence paged view for batched chunked prefill.

    Handles sequences at different prefill progress levels. For each sequence:
    - Scatter-writes the chunk KV to blocks at [start_pos, start_pos + chunk_len).
    - Gathers all KV from [0, start_pos + chunk_len).
    - Pads to max_kv_len for batching.

    Returns [batch, heads, max_kv_len, dim] for SDPA.
    """

    def __init__(
        self,
        pool: PagedKVCachePool,
        seq_ids: list[int],
        start_positions: list[int],
        chunk_lens: list[int],
    ) -> None:
        self.pool = pool
        self.seq_ids = seq_ids
        self.start_positions = start_positions
        self.chunk_lens = chunk_lens
        self.kv_lens = [s + c for s, c in zip(start_positions, chunk_lens)]
        self.max_kv_len = max(self.kv_lens)
        self.max_chunk_len = max(self.chunk_lens)
        self._seq_len = self.max_kv_len - self.max_chunk_len

        # Lazy cached indices (built once on first update, reused across layers).
        self._indices_built = False
        self._scatter_block_ids: list[Tensor] = []
        self._scatter_offsets: list[Tensor] = []
        self._gather_block_ids: Tensor | None = None  # [batch, max_kv_len]
        self._gather_offsets: Tensor | None = None

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def is_paged(self) -> bool:
        return True

    def _build_indices(self, device: torch.device | str) -> None:
        """Build scatter and gather indices (once per view, reused across layers)."""
        if self._indices_built:
            return
        batch_size = len(self.seq_ids)

        # Per-sequence scatter indices.
        for i, seq_id in enumerate(self.seq_ids):
            blocks = self.pool.page_tables[seq_id]
            block_ids = torch.tensor(blocks, dtype=torch.long, device=device)
            positions = torch.arange(
                self.start_positions[i], self.kv_lens[i], device=device
            )
            self._scatter_block_ids.append(block_ids[positions // self.pool.block_size])
            self._scatter_offsets.append(positions % self.pool.block_size)

        # Batched gather indices: [batch, max_kv_len], zero-padded.
        # Positions beyond kv_lens[i] index block 0, offset 0 (zeros).
        # The padding_mask ensures these values are ignored by attention.
        gather_bids = torch.zeros(batch_size, self.max_kv_len, dtype=torch.long, device=device)
        gather_offs = torch.zeros(batch_size, self.max_kv_len, dtype=torch.long, device=device)
        for i, seq_id in enumerate(self.seq_ids):
            kv_len = self.kv_lens[i]
            blocks = self.pool.page_tables[seq_id]
            block_ids = torch.tensor(blocks, dtype=torch.long, device=device)
            positions = torch.arange(kv_len, device=device)
            gather_bids[i, :kv_len] = block_ids[positions // self.pool.block_size]
            gather_offs[i, :kv_len] = positions % self.pool.block_size

        self._gather_block_ids = gather_bids
        self._gather_offsets = gather_offs
        self._indices_built = True

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Scatter-write chunks to blocks, gather full KV padded to max_kv_len.

        k, v: [batch, num_kv_heads, max_chunk_len, head_dim].
        Returns: [batch, num_kv_heads, max_kv_len, head_dim] (zero-padded).
        """
        device = k.device
        self._build_indices(device)

        # Scatter-write per sequence.
        for i, seq_id in enumerate(self.seq_ids):
            chunk_len = self.chunk_lens[i]
            k_flat = k[i, :, :chunk_len, :].permute(1, 0, 2)  # [chunk_len, heads, dim]
            v_flat = v[i, :, :chunk_len, :].permute(1, 0, 2)
            self.pool.k[
                layer_idx, self._scatter_block_ids[i], :, self._scatter_offsets[i], :
            ] = k_flat
            self.pool.v[
                layer_idx, self._scatter_block_ids[i], :, self._scatter_offsets[i], :
            ] = v_flat

        # Batched gather: [batch, max_kv_len, heads, dim].
        assert self._gather_block_ids is not None
        cached_k = self.pool.k[
            layer_idx, self._gather_block_ids, :, self._gather_offsets, :
        ]
        cached_v = self.pool.v[
            layer_idx, self._gather_block_ids, :, self._gather_offsets, :
        ]

        # Permute to [batch, heads, max_kv_len, dim] for SDPA.
        return cached_k.permute(0, 2, 1, 3), cached_v.permute(0, 2, 1, 3)

    def advance(self, n: int) -> None:
        """Set per-sequence seq_lens to actual kv_len (not padded)."""
        for i, seq_id in enumerate(self.seq_ids):
            self.pool.seq_lens[seq_id] = self.kv_lens[i]
```

Factory methods:

```python
class PagedKVCachePool:
    ...
    def batched_chunked_prefill_view(
        self,
        slots: list[int],
        start_positions: list[int],
        chunk_lens: list[int],
    ) -> PagedBatchedChunkedPrefillCacheView:
        """Return a multi-sequence view for batched chunked prefill."""
        return PagedBatchedChunkedPrefillCacheView(
            self, slots, start_positions, chunk_lens
        )

class SlottedKVCache:
    ...
    def batched_chunked_prefill_view(
        self,
        slots: list[int],
        start_positions: list[int],
        chunk_lens: list[int],
    ) -> BatchedChunkedPrefillCacheView:
        """Return a multi-slot view for batched chunked prefill."""
        return BatchedChunkedPrefillCacheView(
            self, slots, start_positions, chunk_lens
        )
```

**`seq_len` property design.** The model computes `kv_len = pos + seq_len` where `pos = kv_cache.seq_len` (a property) and `seq_len = input_ids.shape[1] = max_chunk_len`. The view sets `pos = max_kv_len - max_chunk_len` so that `kv_len = max_kv_len`. This ensures mask and RoPE dimensions match the gathered KV.

**Performance note.** The paged backend gathers `start_pos + chunk_len` tokens from scattered blocks per sequence per layer, with padding to `max_kv_len`. For first chunks (`start_pos = 0`), the gather reads back exactly what was written --- redundant but correct. A potential optimization would return model KV directly for first chunks, but this complicates the batched path (mixed return strategies within a batch) for modest savings. The simpler always-gather approach is preferred for this educational codebase.

**Tests:**
- Write chunk 1 then chunk 2 (different start positions), verify gathered KV matches full prefill KV. Both backends.
- Batched view with mixed start positions returns correctly padded KV.
- `pool.seq_lens[slot]` correct after each view's `advance()`.
- Cache state after chunked prefill is correct for subsequent decode (`decode_view` gathers all KV).
- Gather index caching: same indices across all layers within a step.
- `seq_len` property returns correct value for model's kv_len computation.

### 6. Scheduler support for partial prefill (`src/infer/engine/scheduler.py`)

Add a `prefill_requests()` method that returns all active requests needing prefill work.

```python
class ContinuousScheduler:
    ...
    def prefill_requests(self, max_chunks: int | None = None) -> list[Request]:
        """Return active requests that need prefill work this step.

        Includes both newly admitted requests (WAITING state) and
        in-progress chunked prefills (PREFILL state with
        ``prefill_progress < len(prompt_token_ids)``).

        In-progress prefills are returned first (FIFO: resume before starting
        new), then newly admitted requests. The total is capped at
        ``max_chunks`` to limit the batched prefill size.

        Only meaningful when ``use_chunked_prefill=True``.
        """
        continuing: list[Request] = []
        new: list[Request] = []
        for r in self.active:
            if r.state == RequestState.PREFILL and r.prefill_progress < len(r.prompt_token_ids):
                continuing.append(r)
            elif r.state == RequestState.WAITING:
                new.append(r)
        result = continuing + new
        if max_chunks is not None:
            result = result[:max_chunks]
        return result
```

The existing `admit()` method is unchanged --- it still moves requests from the waiting queue to the active set and returns them. The engine calls `admit()` first (to populate the active set), then `prefill_requests(max_chunks)` to find all prefill work.

**Tests:**
- `prefill_requests()` and `decode_requests()` return disjoint sets --- PREFILL requests never appear in decode list.
- `max_prefill_chunks_per_step` cap is respected (N=3 requests, cap=2 -> only 2 get a chunk).
- `has_work()` returns `True` during multi-step prefill.

### 7. Runner batched chunked prefill (`src/infer/engine/continuous_runner.py`)

**a) New `_prefill_chunks_batched()` method:**

All pending chunks --- regardless of each request's prefill progress --- are batched into a single padded forward pass:

```python
@torch.inference_mode()
def _prefill_chunks_batched(self, requests: list[Request]) -> list[StepOutput | None]:
    """Process one chunk per request in a single batched forward pass.

    Returns StepOutput for each request that completes prefill (last chunk),
    None for requests with intermediate chunks.
    """
    device = self.config.device
    chunk_size = self.config.prefill_chunk_size

    # Compute per-request chunk bounds.
    start_positions: list[int] = []
    chunk_lens: list[int] = []
    chunk_ends: list[int] = []
    for req in requests:
        progress = req.prefill_progress
        prompt_len = len(req.prompt_token_ids)
        chunk_end = min(progress + chunk_size, prompt_len)
        start_positions.append(progress)
        chunk_lens.append(chunk_end - progress)
        chunk_ends.append(chunk_end)

    max_chunk_len = max(chunk_lens)
    max_kv_len = max(s + c for s, c in zip(start_positions, chunk_lens))
    batch_size = len(requests)

    # Allocate slots for first chunks.
    for req in requests:
        if req.prefill_progress == 0:
            slot = self.cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
            req.slot_idx = slot
            req.state = RequestState.PREFILL

    # Build padded input_ids [batch, max_chunk_len].
    padded_tokens: list[list[int]] = []
    for i, req in enumerate(requests):
        chunk_tokens = req.prompt_token_ids[start_positions[i] : chunk_ends[i]]
        padded_tokens.append(chunk_tokens + [0] * (max_chunk_len - len(chunk_tokens)))
    input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device)

    # Build position_ids [batch, max_chunk_len].
    # Real positions: [start_pos, start_pos + chunk_len). Padded positions: 0.
    position_ids = torch.zeros(batch_size, max_chunk_len, dtype=torch.long, device=device)
    for i in range(batch_size):
        position_ids[i, : chunk_lens[i]] = torch.arange(
            start_positions[i], start_positions[i] + chunk_lens[i], device=device
        )

    # Build padding_mask [batch, max_kv_len]: True for valid KV positions.
    padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        kv_len = start_positions[i] + chunk_lens[i]
        padding_mask[i, :kv_len] = True

    # Cache view.
    slots = [req.slot_idx for req in requests]
    assert all(s is not None for s in slots)
    view = self.cache_pool.batched_chunked_prefill_view(
        slots, start_positions, chunk_lens  # type: ignore[arg-type]
    )

    # Forward pass.
    logits = self.model(
        input_ids, kv_cache=view, padding_mask=padding_mask, position_ids=position_ids
    )
    # logits: [batch, max_chunk_len, vocab_size]

    # Update progress and handle last chunks.
    outputs: list[StepOutput | None] = []
    for i, req in enumerate(requests):
        req.prefill_progress = chunk_ends[i]
        is_last = chunk_ends[i] == len(req.prompt_token_ids)

        if not is_last:
            outputs.append(None)
            continue

        # Last chunk: sample first token at actual last position.
        last_pos = chunk_lens[i] - 1
        next_logits = logits[i, last_pos, :]
        context = req.prompt_token_ids
        token = sample_token(next_logits, context, req.sampling_params, req.generator)
        req.generated_token_ids.append(token)
        req.state = RequestState.DECODE

        # Initialize text tracking.
        text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
        text_delta = text
        self._prev_text_lens[req.request_id] = len(text)

        # Check stop conditions.
        finished, reason = check_stop(req, token, self.tokenizer)
        if finished:
            req.state = RequestState.FINISHED
            req.finish_reason = reason
            if reason == "stop":
                text_delta = truncate_at_stop(text, 0, req)

        outputs.append(make_step_output(req, token, text_delta, finished, reason))

    return outputs
```

**b) Updated `step()` method:**

```python
def step(
    self,
    prefill: list[Request],
    decode: list[Request],
) -> list[tuple[Request, StepOutput]]:
    """Run one engine step: decode first (prioritize ITL), then prefill."""
    outputs: list[tuple[Request, StepOutput]] = []

    # Phase 1: Batched decode (prioritize inter-token latency).
    if decode:
        decode_outputs = self._batched_decode(decode)
        outputs.extend(zip(decode, decode_outputs, strict=True))

    # Phase 2: Prefill.
    if self.config.use_chunked_prefill:
        if prefill:
            chunk_outputs = self._prefill_chunks_batched(prefill)
            for req, output in zip(prefill, chunk_outputs, strict=True):
                if output is not None:
                    outputs.append((req, output))
    else:
        # Existing Phase 5/6 prefill logic (unchanged).
        if len(prefill) == 1:
            output = self._prefill_one(prefill[0])
            outputs.append((prefill[0], output))
        elif len(prefill) > 1:
            prefill_outputs = self._prefill_batch(prefill)
            outputs.extend(zip(prefill, prefill_outputs, strict=True))

    return outputs
```

**Tests:**
- Multi-chunk prefill produces correct first token and correct subsequent tokens. State transitions verified.
- Output queue receives nothing during intermediate chunks, only `StepOutput` on final chunk.
- Boundary cases: `prompt_len == chunk_size`, `prompt_len == 1`, `chunk_size > prompt_len`.
- Logits parity: chunked prefill (2 chunks of 16) vs full prefill (32 tokens) produces identical logits (all three models, both backends).
- Multi-token generation (10+ tokens) after chunked prefill matches generation after full prefill.
- Batched chunked prefill with different progress levels produces identical per-request logits.
- 1-token final chunk: `prompt_len % chunk_size == 1` produces correct results.

### 8. Engine step integration (`src/infer/engine/engine.py`)

The engine's `_step_continuous()` method adds a branch for chunked prefill to gather all prefill work (newly admitted + continuing):

```python
def _step_continuous(self) -> None:
    assert isinstance(self.scheduler, ContinuousScheduler)
    assert isinstance(self.runner, ContinuousRunner)

    # Phase 1: Retire finished requests.
    retired = self.scheduler.retire()

    # Phase 2: Free cache resources.
    for req in retired:
        if req.slot_idx is not None:
            self.runner.free_slot(req.slot_idx)
        self.runner.cleanup_request(req.request_id)

    # Phase 3: Query available memory budget.
    free_kv_tokens = self.runner.free_kv_tokens()

    # Phase 4: Admit new requests.
    if self.config.use_chunked_prefill:
        self.scheduler.admit(free_kv_tokens=free_kv_tokens)
        # Phase 5a: Gather prefill work (all chunks batched together).
        prefill = self.scheduler.prefill_requests(
            max_chunks=self.config.max_prefill_chunks_per_step
        )
    else:
        # Phase 5b: Only newly admitted requests (Phase 5/6 behavior).
        prefill = self.scheduler.admit(free_kv_tokens=free_kv_tokens)

    # Phase 6: Identify decode requests.
    decode = self.scheduler.decode_requests()

    if not prefill and not decode:
        return

    # Phase 7: Execute forward passes.
    try:
        outputs = self.runner.step(prefill, decode)
        for req, output in outputs:
            if req.output_queue is not None:
                req.output_queue.put_nowait(output)

    except Exception as exc:
        all_requests = prefill + decode
        for req in all_requests:
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
```

**Tests:**
- Submit long-prompt request with `use_chunked_prefill=True`, verify it completes correctly over multiple steps and generates correct multi-token output.
- Concurrent prefill + decode: decode requests get output every step while prefill is chunked.
- Request arriving mid-chunk: second request admitted while first is mid-prefill; both batched in next step.
- Mid-prefill failure: exception during chunk 2/4 -> request fails, slot/blocks freed on next step, no leak.
- `use_chunked_prefill=False` produces identical results to previous phases (regression).

---

## Interaction with Existing Features

### Paged attention kernel

The Triton paged attention kernel auto-dispatches for decode (`seq_len == 1, mask is None, write_only`). Chunked prefill chunks have `seq_len > 1`, so they always use the SDPA path via the cache view's `update()` (gather + SDPA). No kernel changes needed.

### Batched prefill (Phase 5)

With `use_chunked_prefill=True`, multiple chunks are batched using `_prefill_chunks_batched()`, which extends the Phase 5 batched prefill concept to handle different `start_positions`. The `_prefill_batch()` path is only used when `use_chunked_prefill=False`. The two batching strategies share the same principles (right-padding, padding_mask, per-element logit extraction) but differ in mask construction: Phase 5 uses a shared square causal mask, Phase 7 uses position_ids-based per-element masks to handle different chunk offsets.

### Contiguous backend

Chunked prefill works with both backends. The slotted backend's `BatchedChunkedPrefillCacheView` leverages contiguous slot storage for efficient indexed reads. The paged backend's `PagedBatchedChunkedPrefillCacheView` scatter-writes to blocks and gathers, matching the existing paged pattern.

### Gemma 3 sliding window

The position_ids-based mask construction naturally handles Gemma 3's sliding window for chunked prefill. The sliding window constraint `(q_pos - kv_pos) < window_size` is applied per-element using each element's actual position IDs, correctly masking out-of-window positions regardless of chunk offset.

---

## Costs and Trade-offs

### Weight loading: batched vs serial

The central advantage of batched chunked prefill is weight-loading amortization. Regardless of how many chunks are in the batch, the model weights are loaded from HBM exactly once per forward pass.

```text
Approach                   N chunks   Forward passes   Weight reads (Llama 3B, ~6 GB)
-------------------------  --------   --------------   ------------------------------
Serial (one per chunk)          4               4   24 GB
Batched                         4               1    6 GB
```

Per step, the engine runs 2 forward passes total: 1 decode + 1 batched prefill. With serial processing, this would be 1 + N. The batched approach eliminates `N - 1` full weight reads.

For Llama 3.2 3B on RTX 5080 (960 GB/s memory bandwidth), each saved weight read is ~6.3 ms. With 4 concurrent prefills, batching saves ~19 ms per step.

### Padding waste

Batching chunks at different progress levels introduces padding waste in both Q and KV dimensions:

- **Q padding**: `batch * (max_chunk_len - avg_chunk_len)`. Usually minimal --- most chunks are `chunk_size` tokens, except the last chunk of each prompt which may be shorter.
- **KV padding**: `batch * (max_kv_len - avg_kv_len)`. Depends on the spread of start positions across concurrent prefills.

Example with 3 concurrent prefills:

```text
Req A: start_pos=0,    chunk_len=512, kv_len=512
Req B: start_pos=512,  chunk_len=512, kv_len=1024
Req C: start_pos=1024, chunk_len=512, kv_len=1536

max_chunk_len=512 (no Q padding), max_kv_len=1536
  Req A: 1024 wasted KV positions (padded from 512 to 1536)
  Req B: 512 wasted KV positions (padded from 1024 to 1536)
  Req C: 0 wasted
  Total: 1536 wasted KV positions across 3 sequences
```

At bf16 with 8 KV heads and 128 head_dim, 1536 wasted positions per layer = 1536 * 8 * 128 * 2 bytes = ~3 MB per layer. For a 28-layer model, ~84 MB total --- negligible compared to the ~18 GB saved by eliminating 3 weight reads.

**When padding waste matters:** If `max_prefill_chunks_per_step` is very large and prefill progress is widely spread (e.g., one request at chunk 0 and another at chunk 15 of a long prompt), KV padding grows. In practice, requests that arrive in the same time window tend to have similar progress, limiting spread. The `max_prefill_chunks_per_step` config can cap batch size if needed.

### Gather cost (paged backend)

The `PagedBatchedChunkedPrefillCacheView` gathers all prior KV from scattered blocks per sequence per layer. For the k-th chunk (of C total), the gather covers `k * chunk_size` tokens. Summed across all chunks: `C * (C+1) / 2 * chunk_size` total gathered tokens. For a 2048-token prompt with 512-token chunks (C=4): `4 * 5 / 2 * 512 = 5120` total gathered tokens over the entire prefill, vs 0 for full prefill. This is modest compared to decode gather traffic during generation.

### Memory

No additional memory overhead. Block allocation at admission is unchanged. The cache view gathers are computed on-the-fly with no persistent allocation. Padding tensors are temporary and freed after each forward pass.

### When to enable

Chunked prefill primarily benefits workloads with concurrent prefill and decode, where long prompts would otherwise stall active generations. For single-request workloads or batch inference without concurrent decode, chunked prefill adds overhead with no benefit --- leave it disabled.

### Chunk size selection

Per-step ITL impact depends on chunk batch size and compute: `total_tokens = sum(chunk_lens) + padding`. Weight loading dominates for small batches; compute dominates for large batches. The TTFT tradeoff scales with number of chunks:

```text
Chunk size   Chunks for 2048 prompt   Extra steps   TTFT increase (Llama 3B)
----------   ----------------------   -----------   ---------------------------------
       128                       16            15   ~15 extra steps of ~6.3 ms each
       256                        8             7   ~7 extra steps
       512                        4             3   ~3 extra steps  <-- default
      1024                        2             1   ~1 extra step
      2048                        1             0   Same as full prefill
```

Default: 512. This balances TTFT overhead (4 steps for a 2048-token prompt) with compute efficiency (512 tokens per chunk). Short prompts (<= 512 tokens) complete in one chunk with no overhead.

---

## Benchmark Plan

### Configuration

Single configuration: paged backend, chunked prefill enabled, `prefill_chunk_size=512`.

```text
kv_cache_backend = "paged"
use_chunked_prefill = True
prefill_chunk_size = 512
```

### Workloads

Run **all** existing workloads (`bench_serving.py --workload all`) for all three models to catch regressions and measure chunked prefill impact:

- **baseline** --- sequential single requests (overhead floor).
- **continuous_batching** --- staggered arrivals, varying lengths.
- **paged_attention** --- single burst, moderate lengths.
- **chunked_prefill** --- long prompts, Poisson arrivals. Primary workload for measuring ITL improvement: long-prompt arrivals create prefill pressure while decode requests are in-flight.
- **prefix_caching** --- shared system prompt.

Compare against Phase 6 results (paged, no chunked prefill) from existing benchmark reports.

### Metrics to report

- **P50/P99 ITL** across all workloads (primary metric --- should improve under `chunked_prefill` workload)
- **TTFT** (will increase for long prompts due to multi-step prefill)
- **Overall throughput** (tokens/sec --- may decrease slightly)
- **Regressions** on non-chunked workloads (baseline, continuous_batching) should be negligible

---

## Exit Criteria

1. **ITL stability**: P99 ITL for decode-active requests improves by at least 2x under concurrent long-prompt prefill workload, compared to unchunked baseline.
2. **No throughput regression when disabled**: With `use_chunked_prefill=False`, all Phase 5/6 benchmarks reproduce identical results (no code path changes).
3. **Correctness --- multi-token generation**: Chunked prefill produces identical generated sequence (at least 10 tokens, greedy decode) as full prefill for the same prompt. Not just the first token --- decode after chunked prefill must produce the same output, verifying the KV cache state is fully correct. Verified for all three models.
4. **Correctness --- logits parity**: Logits at the last prompt position are bitwise identical between chunked and full prefill under greedy decode. Verified for all three models, both backends.
5. **Config validation**: `use_chunked_prefill=True` with `batching_mode="static"` raises `ValueError`.
6. **No Phase 1-6 regressions**: All existing tests pass unchanged.
7. **Both backends**: Chunked prefill works with both `kv_cache_backend="contiguous"` and `"paged"`.
8. **Sliding window correctness**: Gemma 3's sliding window layers produce correct outputs during chunked prefill (verified via logits comparison with full prefill), including at the sliding window boundary.
9. **Chunk boundary correctness**: Prompts that don't divide evenly by `chunk_size` produce correct results (the last chunk is smaller). Includes 1-token final chunk edge case.
10. **Short prompt passthrough**: Prompts shorter than `chunk_size` complete in one chunk with no overhead.
11. **Boundary cases**: `prompt_len == chunk_size` (single chunk, is_first and is_last both true), `prompt_len == 1`, and `chunk_size > prompt_len` all produce correct results.
12. **Cache state consistency**: `pool.seq_lens[slot]` equals `prefill_progress` after every chunk step. `pool.seq_lens[slot] == len(prompt_token_ids)` after the final chunk.
13. **Output queue silence during intermediate chunks**: No `StepOutput` is pushed to the request's output queue during intermediate prefill chunks (only on the final chunk).
14. **Mid-prefill failure cleanup**: If an exception occurs during chunk N > 1, the request's slot and blocks are freed on the next retire cycle. No block leaks.
15. **Concurrent chunked prefills**: Multiple requests at different prefill progress levels are batched and processed correctly in the same step. `max_prefill_chunks_per_step` cap is respected when set.
16. **Benchmark results recorded**: All workloads run for all three models with `use_chunked_prefill=True, kv_cache_backend="paged"`. Results in benchmark report.

