# Phase 9: CUDA Graphs

## Goal

Eliminate CPU-side kernel launch overhead during decode by capturing the decode forward pass into a CUDA graph and replaying it each step.

During decode, each step executes the same sequence of GPU kernels with the same tensor shapes --- only the tensor contents change. Normally the CPU submits each kernel individually via the CUDA driver. For small, fast decode kernels, the GPU finishes one kernel before the CPU has submitted the next, leaving the GPU idle between launches. A Llama-3B decode step has ~100+ kernel launches at 5--15us each, adding 0.5--1.5ms of pure CPU overhead per step against ~10--15ms of GPU compute. CUDA graphs capture the full kernel sequence once, then replay it with a single CPU call, eliminating the inter-kernel gaps.

The key constraint: CUDA graphs require all tensor addresses and shapes to be fixed between replays. Only the tensor *contents* may change. This means no dynamic allocations, no variable-size tensors, and no Python control flow inside the captured region. The decode path is a natural fit because every step processes exactly one token per sequence (`seq_len == 1`), with fixed batch size (after padding).

Benchmark models: `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-4B`, `google/gemma-3-1b-it` (same as Phases 4--8).

---

## Architecture

```text
ContinuousRunner.step(prefill, decode)
│
├── Prefill: stays eager (variable shapes per request)
│
└── Decode: use_cuda_graphs?
    │
    ├── No → _batched_decode() [existing eager path]
    │
    └── Yes → CUDAGraphRunner.execute(requests)
              │
              ├── 1. Allocate blocks if needed (Python, before graph)
              │      pool.decode_view(slots)  →  _ensure_blocks_allocated()
              │
              ├── 2. Prepare static tensors (Python → GPU copies)
              │      ├── input_ids_buf.copy_(real_tokens)
              │      ├── position_ids_buf.copy_(real_positions)
              │      ├── view.prepare(seq_ids)
              │      │   ├── page_table_tensor[:] = current page tables
              │      │   ├── seq_lens_tensor[:] = current seq lens + 1
              │      │   ├── write_block_ids[:] = target block for each seq
              │      │   └── write_offsets[:] = offset within block
              │      └── (padding slots → scratch block)
              │
              ├── 3. graph.replay()  ← single CPU call
              │      ├── embed_tokens(input_ids_buf)
              │      ├── for each layer:
              │      │   ├── input_layernorm
              │      │   ├── Q/K/V projections
              │      │   ├── RoPE (from position_ids_buf)
              │      │   ├── write_only: scatter K/V to pool via GPU indices
              │      │   ├── triton_paged_attention(q, pool.k, pool.v,
              │      │   │       page_table_tensor, seq_lens_tensor)
              │      │   ├── output projection
              │      │   ├── fused_residual_rms_norm
              │      │   └── MLP
              │      ├── final norm
              │      └── lm_head → logits_buf
              │
              ├── 4. Post-replay CPU updates
              │      └── pool.seq_lens[seq_id] += 1  for each sequence
              │
              └── 5. Return logits_buf[:actual_batch]
```

**Warmup phase** (at server startup, before serving):

```text
for batch_size in [1, 2, 4, 8, 16, 32]:  # power-of-2 buckets
    if batch_size > max_batch_size: break
    1. Allocate scratch slots in pool
    2. Fill placeholder tensors with dummy data
    3. Run 3 warmup forward passes (non-captured, on side stream)
    4. Capture: torch.cuda.graph(g, pool=shared_mempool)
    5. Store CapturedGraph(graph=g, batch_size, placeholders)
    6. Free scratch slots
```

**Key design decisions:**

1. **Require paged backend.** CUDA graphs require the Triton paged attention path, which reads K/V directly from the pool via page tables (fixed-address tensors with updatable contents). The SDPA gather path creates variable-size tensors every step --- fundamentally incompatible. Config validation enforces `use_cuda_graphs=True` requires `kv_cache_backend="paged"`.

2. **Skip mask creation for graph-captured decode.** The model forward currently creates a `torch.zeros(batch, 1, 1, kv_len)` mask for every decode step, where `kv_len` grows each step --- a dynamic-shape allocation that breaks graph capture. Since the Triton paged attention kernel handles per-sequence lengths internally via `seq_lens_tensor`, no attention mask is needed. The model change: when `padding_mask is None` and `seq_len == 1`, set `mask = None` instead of creating a zero tensor. This enables both the Triton dispatch (which requires `mask is None`) and graph capture. For the SDPA path (non-paged backends), `padding_mask` is always provided, so this change has no effect.

3. **GPU-indexed KV scatter write.** The existing `_write_token()` uses Python-level indexing (`pool.k[layer, block_id, :, offset, :] = k[i, ...]`) where `block_id` and `offset` are computed in Python from `pool.seq_lens`. During graph capture, these Python-computed indices are baked into the graph and replayed verbatim --- producing incorrect writes on subsequent steps. The fix: pre-compute write positions as GPU tensors (`write_block_ids`, `write_offsets`) and use PyTorch advanced indexing (`pool.k[layer][write_block_ids, :, write_offsets, :] = k_new`). These GPU operations are captured in the graph; the tensor contents are updated before each replay.

4. **Batch size bucketing with power-of-2 sizes.** The decode batch size varies per step as requests finish and new ones start decoding. A separate graph is captured for each batch size encountered. Using power-of-2 buckets (1, 2, 4, 8, 16, 32) limits the total number of graphs to at most 6. The actual batch is padded to the nearest bucket; unused slots use dummy tokens with writes directed to a scratch block.

5. **Eager warmup at startup.** Graphs are captured for all power-of-2 buckets up to `max_batch_size` during server initialization, before any requests arrive. This avoids latency spikes during serving. The warmup allocates temporary cache slots, runs warmup forward passes, captures graphs, then frees the slots.

6. **Scratch block for padding.** One block is reserved as a "scratch block" during warmup. Padding slots' KV writes target this block, and their page tables point to it. The block's data is never read for real attention (padding slots have `seq_lens_tensor[i] = 0`, so the Triton kernel skips them). This ensures graph-captured writes for padding slots go to a safe location.

7. **Post-replay CPU state sync.** `kv_cache.advance(n)` inside the model forward updates `pool.seq_lens` (a Python dict) --- this is pure Python, not captured in the graph. The `CUDAGraphRunner` must call `pool.seq_lens[seq_id] += 1` for each real sequence after every replay to keep CPU state consistent with GPU reality.

8. **Shared CUDA memory pool.** All captured graphs share one `torch.cuda.graphs.graph_pool_handle()`. Since only one graph runs at a time, intermediate activation memory is reused across batch sizes. This keeps VRAM overhead close to that of a single (largest) graph, not the sum of all graphs.

9. **Triton paged attention extended with sliding window.** Gemma 3 uses sliding-window attention for some layers. Without sliding-window support in the Triton kernel, these layers would need SDPA with variable-size masks --- breaking graph capture. The kernel extension adds a `window_size` parameter: when nonzero, the kernel skips K/V blocks entirely before the window and masks out-of-window positions in the edge block. This keeps the entire Gemma 3 decode in the Triton path and graph-compatible.

10. **Answer to Open Question (OVERALL_DESIGN.md Section 15).** Eager warmup at startup for power-of-2 batch sizes. This avoids first-encounter latency during serving while keeping the number of captured graphs bounded. If a batch size exceeds the largest warmed bucket, that step falls back to eager mode.

---

## Deliverables

### 1. Triton paged attention sliding window extension (`src/infer/kernels/paged_attention.py`) --- MODIFIED

Add `window_size: int = 0` parameter to `triton_paged_attention` and the underlying Triton kernel.

When `window_size == 0` (default), behavior is unchanged (full attention). When `window_size > 0`:

- The kernel computes the attention window as `[seq_len - window_size, seq_len)`.
- Blocks whose entire range falls before the window start are skipped (no K/V loads, no computation).
- For the block straddling the window boundary, per-position masking sets out-of-window scores to `-inf` before softmax.

```python
def triton_paged_attention(
    q: Tensor,                    # [batch, num_heads, 1, head_dim]
    k_pool: Tensor,               # [total_blocks, num_kv_heads, block_size, head_dim]
    v_pool: Tensor,               # [total_blocks, num_kv_heads, block_size, head_dim]
    page_table: Tensor,           # [batch, max_num_blocks] int32
    seq_lens: Tensor,             # [batch] int32
    *,
    scale: float,
    max_num_blocks: int,
    window_size: int = 0,         # NEW: 0 = full attention
) -> Tensor:
    ...
```

**Tests:**

- Full attention (window_size=0): identical results to existing kernel (no regression).
- Windowed attention: for a sequence of length 100 with window_size=32, verify only positions 68--99 contribute to the output. Compare against SDPA with an explicit sliding-window mask.
- Window covers entire sequence (window_size >= seq_len): identical to full attention.
- Edge case: window_size=1 (attend only to current position).

### 2. Attention module changes (`src/infer/models/common.py`) --- MODIFIED

Add `sliding_window: int = 0` attribute to `Attention`. Pass it to `triton_paged_attention` in the Triton dispatch path.

```python
class Attention(nn.Module):
    def __init__(
        self,
        ...,
        sliding_window: int = 0,   # NEW: 0 = full attention
    ) -> None:
        ...
        self.sliding_window = sliding_window
```

Modified Triton dispatch in `Attention.forward`:

```python
if (
    kv_cache is not None
    and kv_cache.is_paged()
    and seq_len == 1
    and mask is None
    and hasattr(kv_cache, "write_only")
):
    paged_view: Any = kv_cache
    paged_view.write_only(layer_idx, k, v)
    out = triton_paged_attention(
        q,
        paged_view.pool.k[layer_idx],
        paged_view.pool.v[layer_idx],
        paged_view.page_table_tensor,
        paged_view.seq_lens_tensor,
        scale=self.scale,
        max_num_blocks=paged_view.page_table_tensor.shape[1],
        window_size=self.sliding_window,  # NEW
    )
```

When `sliding_window == 0`, behavior is identical to today. The Triton dispatch condition itself is unchanged --- it still requires `mask is None`.

**Note:** This change benefits ALL paged decode, not just CUDA graph mode. Currently, the batched decode path always creates a mask (because `position_ids is not None`), forcing the SDPA gather path even with the paged backend. After the model forward change (deliverable 4), paged decode uses the Triton kernel for all full-attention layers, avoiding the gather step. Sliding-window layers also use the Triton kernel (with `window_size`) instead of SDPA with a gathered mask. This is a free performance improvement independent of CUDA graphs.

**Tests:**

- Llama/Qwen3 attention: `sliding_window=0` → full Triton attention, output identical to existing.
- Gemma3 sliding-window attention: `sliding_window=4096` → windowed Triton attention, output matches SDPA with sliding-window mask.

### 3. Gemma3 model changes (`src/infer/models/gemma3.py`) --- MODIFIED

Add `sliding_window` and `layer_type` parameters to `Gemma3TransformerBlock` so the `Attention` module knows its window size. Currently, all blocks are constructed identically and the per-layer mask selection happens in `Gemma3Model.forward()`. After this change, the block carries the window size and the attention dispatch handles windowing internally via the Triton kernel.

**`Gemma3TransformerBlock` constructor change:**

```python
class Gemma3TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-5,
        query_pre_attn_scalar: float = 256.0,
        sliding_window: int = 0,              # NEW: 0 for full-attention layers
    ) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=False,
            qk_norm=False,
            scale=query_pre_attn_scalar**-0.5,
            sliding_window=sliding_window,    # NEW
        )
        # QK-norm assignment unchanged ...
```

**`Gemma3Model.__init__` change:**

```python
self.layers = nn.ModuleList(
    [
        Gemma3TransformerBlock(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.computed_head_dim,
            rms_norm_eps=config.rms_norm_eps,
            query_pre_attn_scalar=config.query_pre_attn_scalar or config.computed_head_dim,
            sliding_window=(                                      # NEW
                (config.sliding_window or 512)
                if self.layer_types[i] == "sliding_attention"
                else 0
            ),
        )
        for i in range(config.num_hidden_layers)
    ]
)
```

Note: `self.layer_types` must be initialized before the `nn.ModuleList` construction. It is already set at line 211 in the current code, but needs to be moved above the `self.layers` assignment (or duplicated).

Llama and Qwen3 `TransformerBlock` constructors already pass `sliding_window=0` by default (via the `Attention` default). No changes needed.

### 4. Model forward changes (`src/infer/models/llama.py`, `qwen3.py`, `gemma3.py`) --- MODIFIED

Skip mask creation during decode when `padding_mask is None`. This enables the Triton paged attention dispatch (which requires `mask is None`) and eliminates the dynamic-shape `torch.zeros` allocation that blocks graph capture.

**Llama and Qwen3** (identical change in both):

```python
# Current (llama.py lines 196-216):
if padding_mask is not None or position_ids is not None:
    kv_len = pos + seq_len
    if seq_len > 1:
        ...  # multi-token mask (unchanged)
    else:
        mask = torch.zeros(batch_size, 1, 1, kv_len, ...)  # ALWAYS created
    if padding_mask is not None:
        pad_mask = ~padding_mask[:, None, None, :kv_len]
        mask.masked_fill_(pad_mask, float("-inf"))
else:
    mask = None if seq_len == 1 else causal_mask(...)

# Changed to:
if padding_mask is not None or position_ids is not None:
    kv_len = pos + seq_len
    if seq_len > 1:
        ...  # multi-token mask (unchanged)
    else:
        if padding_mask is not None:
            mask = torch.zeros(batch_size, 1, 1, kv_len, ...)
            pad_mask = ~padding_mask[:, None, None, :kv_len]
            mask.masked_fill_(pad_mask, float("-inf"))
        else:
            # Decode without padding: no mask needed.
            # Triton paged attention handles per-sequence lengths
            # via seq_lens_tensor. The removed zero-valued mask had
            # no effect on attention scores.
            mask = None
    # Remove the post-loop padding_mask block (folded into the else above).
else:
    mask = None if seq_len == 1 else causal_mask(...)
```

**Gemma3** (analogous change, but for dual masks):

```python
# Current decode branch (padding_mask is not None or position_ids is not None, seq_len == 1):
global_mask = torch.zeros(batch_size, 1, 1, kv_len, ...)
local_mask = torch.zeros(batch_size, 1, 1, kv_len, ...)
cutoff = max(0, kv_len - self.sliding_window)
if cutoff > 0:
    local_mask[:, :, :, :cutoff] = float("-inf")
if padding_mask is not None:
    ...  # apply padding to both masks

# Changed to:
if padding_mask is not None:
    global_mask = torch.zeros(batch_size, 1, 1, kv_len, ...)
    local_mask = torch.zeros(batch_size, 1, 1, kv_len, ...)
    cutoff = max(0, kv_len - self.sliding_window)
    if cutoff > 0:
        local_mask[:, :, :, :cutoff] = float("-inf")
    pad_mask = ~padding_mask[:, None, None, :kv_len]
    global_mask.masked_fill_(pad_mask, float("-inf"))
    local_mask.masked_fill_(pad_mask, float("-inf"))
else:
    # Decode without padding: no masks needed.
    # Full-attention layers use Triton paged attention (mask=None).
    # Sliding-window layers use Triton paged attention with window_size.
    global_mask = None
    local_mask = None
```

**Correctness argument:** For decode (`seq_len == 1`), the current zero-valued mask has no effect on attention scores --- adding 0 is a no-op. The `padding_mask` fill (setting positions beyond each sequence's length to `-inf`) is only needed when different sequences in the batch have different lengths AND attention is computed via SDPA (which gathers K/V to a shared max-length tensor with garbage in padding positions). The Triton paged attention kernel handles per-sequence lengths internally via `seq_lens_tensor`, so no mask is needed.

For the SDPA path (non-paged backends), `_batched_decode` always passes `padding_mask`, so the mask is still created. No behavior change for contiguous/slotted backends.

**Tests:**

- Greedy decode output identical with and without the mask change (all three models).
- Triton paged attention dispatch fires for all decode layers when `padding_mask=None` (verify via hook or attribute check).

### 5. `GraphPagedDecodeCacheView` (`src/infer/cache/paged.py`) --- NEW

A decode cache view with pre-allocated, fixed-address GPU tensors for CUDA graph capture.

```python
class GraphPagedDecodeCacheView:
    """Paged decode cache view for CUDA graph capture and replay.

    All tensors are pre-allocated at construction time with fixed shapes and
    GPU addresses. Contents are updated via ``prepare()`` before each graph
    replay. Used exclusively by ``CUDAGraphRunner``.

    The view satisfies the ``KVCacheProtocol`` interface expected by the model
    forward pass, with graph-compatible implementations:

    - ``write_only()``: uses GPU-indexed advanced indexing instead of
      Python-level loop indexing.
    - ``advance()``: no-op (CPU state is updated by the graph runner after
      replay).
    - ``page_table_tensor`` / ``seq_lens_tensor``: pre-allocated at fixed
      addresses; contents updated before replay.

    Args:
        pool: The paged KV cache pool (provides ``k``, ``v`` storage tensors).
        max_batch_size: Maximum batch size (determines tensor first dimension).
        max_blocks_per_seq: Maximum blocks per sequence (``max_seq_len // block_size``).
        device: Target device.
    """

    def __init__(
        self,
        pool: PagedKVCachePool,
        max_batch_size: int,
        max_blocks_per_seq: int,
        device: str | torch.device = "cuda",
    ) -> None:
        self.pool = pool
        self._max_batch_size = max_batch_size

        # Static kernel tensors (fixed addresses, contents updated by prepare()).
        self.page_table_tensor = torch.zeros(
            max_batch_size, max_blocks_per_seq, dtype=torch.int32, device=device,
        )
        self.seq_lens_tensor = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device,
        )

        # Static write indices (for GPU-indexed KV scatter write).
        self._write_block_ids = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        self._write_offsets = torch.zeros(max_batch_size, dtype=torch.long, device=device)

        # Current max seq len (set by prepare()).
        self._seq_len: int = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def is_paged(self) -> bool:
        return True

    def write_only(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """Graph-compatible KV write using pre-computed GPU indices.

        k, v shape: ``[batch, num_kv_heads, 1, head_dim]``.

        Uses advanced indexing on the pool's K/V storage tensors. The
        ``_write_block_ids`` and ``_write_offsets`` tensors are at fixed GPU
        addresses; their contents were set by ``prepare()`` before the graph
        replay.
        """
        k_new = k[:, :, 0, :]  # [batch, kv_heads, head_dim]
        v_new = v[:, :, 0, :]
        self.pool.k[layer_idx][self._write_block_ids, :, self._write_offsets, :] = k_new
        self.pool.v[layer_idx][self._write_block_ids, :, self._write_offsets, :] = v_new

    def advance(self, n: int) -> None:
        """No-op. CPU state is updated by CUDAGraphRunner after replay."""
        pass

    def prepare(
        self,
        seq_ids: list[int],
        pool: PagedKVCachePool,
        scratch_block: int,
    ) -> None:
        """Update tensor contents before graph replay.

        Copies current page tables, sequence lengths, and write positions
        from CPU-side pool state to the pre-allocated GPU tensors. Padding
        slots (indices beyond ``len(seq_ids)``) are directed to the scratch
        block.

        Args:
            seq_ids: Active sequence IDs for this step.
            pool: The paged pool (source of page tables and seq_lens).
            scratch_block: Block ID reserved for padding slot writes.
        """
        batch = len(seq_ids)

        # Zero out all rows first (handles padding).
        self.page_table_tensor.zero_()
        self.seq_lens_tensor.zero_()
        self._write_block_ids.fill_(scratch_block)
        self._write_offsets.zero_()

        for i, seq_id in enumerate(seq_ids):
            pos = pool.seq_lens[seq_id]
            # Page table.
            page_table = pool.page_tables[seq_id]
            pt_tensor = torch.tensor(page_table, dtype=torch.int32, device=self.page_table_tensor.device)
            self.page_table_tensor[i, :len(page_table)] = pt_tensor
            # Seq len (includes +1 for the token being written).
            self.seq_lens_tensor[i] = pos + 1
            # Write position.
            block_idx = pos // pool.block_size
            offset = pos % pool.block_size
            self._write_block_ids[i] = page_table[block_idx]
            self._write_offsets[i] = offset

        self._seq_len = max((pool.seq_lens[sid] for sid in seq_ids), default=0)
```

**Tests:**

- `prepare()` correctly populates tensors from pool state.
- `write_only()` writes to correct positions (compare against `_write_token` reference).
- `advance()` is a no-op.
- Padding slots write to scratch block, not real sequence blocks.
- `page_table_tensor` and `seq_lens_tensor` at stable addresses across calls.

### 6. `CUDAGraphRunner` (`src/infer/engine/cuda_graph_runner.py`) --- NEW

Manages graph capture, replay, and batch size bucketing.

```python
@dataclass
class CapturedGraph:
    """A single captured CUDA graph for a fixed batch size."""

    graph: torch.cuda.CUDAGraph
    batch_size: int

    # Static input placeholders (fixed GPU addresses).
    input_ids: Tensor      # [batch_size, 1] long
    position_ids: Tensor   # [batch_size, 1] long

    # Static output placeholder.
    logits: Tensor         # [batch_size, 1, vocab_size] model dtype

    # Static cache view.
    view: GraphPagedDecodeCacheView
```

```python
class CUDAGraphRunner:
    """Captures and replays the decode forward pass as CUDA graphs.

    Pre-records a graph for each power-of-2 batch size up to
    ``max_batch_size`` during warmup. At runtime, pads the actual batch
    to the nearest bucket and replays the corresponding graph.

    Args:
        model: The loaded model.
        cache_pool: The paged KV cache pool.
        config: Engine configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        cache_pool: PagedKVCachePool,
        config: EngineConfig,
    ) -> None: ...

    def warmup(self) -> None:
        """Pre-capture graphs for all batch size buckets.

        Called once at server startup before serving. For each bucket:
        1. Allocate temporary cache slots.
        2. Run warmup forward passes (eager, on a side stream).
        3. Capture the graph.
        4. Free temporary slots.

        Uses a shared CUDA memory pool across all graphs so intermediate
        activation memory is reused (only one graph runs at a time).
        """
        ...

    def execute(
        self,
        requests: list[Request],
        cache_pool: PagedKVCachePool,
    ) -> Tensor:
        """Execute graph-captured decode for the given requests.

        1. Compute actual batch size and pad to nearest bucket.
        2. Allocate new blocks if needed (Python, before graph).
        3. Prepare static tensors (copy inputs, update cache view).
        4. Replay the captured graph.
        5. Advance CPU-side pool.seq_lens.
        6. Return logits sliced to actual batch size.

        If the actual batch size exceeds the largest captured bucket,
        returns ``None`` to signal fallback to eager mode.

        Args:
            requests: Active decode requests.
            cache_pool: The paged pool (for block allocation and state sync).

        Returns:
            Logits tensor ``[actual_batch, 1, vocab_size]``, or ``None``
            for eager fallback.
        """
        ...
```

**Bucket selection:**

```python
_BATCH_BUCKETS = [1, 2, 4, 8, 16, 32]

def _padded_batch_size(actual: int) -> int | None:
    """Return the smallest bucket >= actual, or None if too large."""
    for b in _BATCH_BUCKETS:
        if b >= actual:
            return b
    return None
```

**Warmup sequence per bucket:**

```python
def _capture_for_batch_size(self, batch_size: int) -> CapturedGraph:
    device = self.config.device
    max_blocks = self.config.max_seq_len // self.config.block_size

    # Pre-allocate static tensors.
    input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    view = GraphPagedDecodeCacheView(self.cache_pool, batch_size, max_blocks, device)

    # Allocate temporary cache slots for warmup.
    temp_slots = [self.cache_pool.allocate_slot(initial_tokens=1) for _ in range(batch_size)]
    temp_view = self.cache_pool.decode_view(temp_slots)
    view.prepare(temp_slots, self.cache_pool, self._scratch_block)

    # Warmup forward passes (eager, on side stream).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = self.model(input_ids, kv_cache=view, position_ids=position_ids)
    torch.cuda.current_stream().wait_stream(s)

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=self._mempool):
        logits = self.model(input_ids, kv_cache=view, position_ids=position_ids)

    # Free temporary slots.
    for slot in temp_slots:
        self.cache_pool.free_slot(slot)

    return CapturedGraph(
        graph=g,
        batch_size=batch_size,
        input_ids=input_ids,
        position_ids=position_ids,
        logits=logits,
        view=view,
    )
```

**Replay sequence:**

```python
def execute(self, requests, cache_pool):
    actual_batch = len(requests)
    padded = _padded_batch_size(actual_batch)
    if padded is None:
        return None  # fallback to eager

    captured = self._graphs[padded]
    slots = [req.slot_idx for req in requests]

    # 1. Block allocation (Python, before graph).
    _ = cache_pool.decode_view(slots)  # triggers _ensure_blocks_allocated

    # 2. Prepare inputs.
    tokens = [req.generated_token_ids[-1] for req in requests]
    positions = [cache_pool.get_seq_len(slot) for slot in slots]

    captured.input_ids[:actual_batch, 0] = torch.tensor(tokens, dtype=torch.long, device=device)
    captured.position_ids[:actual_batch, 0] = torch.tensor(positions, dtype=torch.long, device=device)
    # Padding slots: keep zeros from initialization.

    captured.view.prepare(slots, cache_pool, self._scratch_block)

    # 3. Replay.
    captured.graph.replay()

    # 4. Post-replay: advance pool state.
    for slot in slots:
        cache_pool.seq_lens[slot] += 1

    # 5. Return logits.
    return captured.logits[:actual_batch]
```

**Tests:**

- Warmup completes without error for all bucket sizes.
- Graph replay produces correct logits (compare against eager forward with same inputs).
- Batch padding: 3 requests → bucket 4, padding slot writes go to scratch block.
- `execute()` returns `None` for oversized batches (fallback to eager).
- Pool seq_lens correctly advanced after replay.
- Multiple consecutive replays produce correct results (no state drift).

### 7. `ContinuousRunner` integration (`src/infer/engine/continuous_runner.py`) --- MODIFIED

Create `CUDAGraphRunner` at init when enabled. Delegate decode to graph runner, with eager fallback.

```python
class ContinuousRunner:
    def __init__(self, model, tokenizer, config):
        ...
        # CUDA graph runner (Phase 9).
        self._cuda_graph_runner: CUDAGraphRunner | None = None
        if config.use_cuda_graphs:
            assert isinstance(self.cache_pool, PagedKVCachePool)
            self._cuda_graph_runner = CUDAGraphRunner(model, self.cache_pool, config)
```

Warmup is called from the engine or server before the first request:

```python
# In Engine.__init__ or server startup:
if config.use_cuda_graphs:
    runner.warmup_cuda_graphs()

# In ContinuousRunner:
def warmup_cuda_graphs(self) -> None:
    """Pre-capture CUDA graphs. Call before serving."""
    if self._cuda_graph_runner is not None:
        self._cuda_graph_runner.warmup()
```

Modified `_batched_decode`:

```python
@torch.inference_mode()
def _batched_decode(self, requests: list[Request]) -> list[StepOutput]:
    device = self.config.device

    # Try CUDA graph path.
    if self._cuda_graph_runner is not None:
        logits = self._cuda_graph_runner.execute(requests, self.cache_pool)
        if logits is not None:
            # Graph succeeded. Sample tokens from logits.
            return self._sample_decode(requests, logits)

    # Eager fallback (existing code, unchanged).
    ...
```

The sampling logic (currently inline in `_batched_decode`) is extracted into `_sample_decode()` so both graph and eager paths share it:

```python
def _sample_decode(self, requests: list[Request], logits: Tensor) -> list[StepOutput]:
    """Sample tokens and build StepOutputs from decode logits."""
    outputs: list[StepOutput] = []
    for i, req in enumerate(requests):
        context = req.prompt_token_ids + req.generated_token_ids
        token = sample_token(logits[i, -1, :], context, req.sampling_params, req.generator)
        req.generated_token_ids.append(token)

        text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
        prev_len = self._prev_text_lens.get(req.request_id, 0)
        text_delta = text[prev_len:]

        finished, reason = check_stop(req, token, self.tokenizer)
        if finished:
            req.state = RequestState.FINISHED
            req.finish_reason = reason
            if reason == "stop":
                text_delta = truncate_at_stop(text, prev_len, req)
        self._prev_text_lens[req.request_id] = len(text)
        outputs.append(make_step_output(req, token, text_delta, finished, reason))
    return outputs
```

**Note on `padding_mask`.** When using CUDA graphs, `_batched_decode` no longer passes `padding_mask` to the model --- the `CUDAGraphRunner` calls the model with `padding_mask=None` and `position_ids` only. When falling back to eager mode, the existing code (with `padding_mask`) runs unchanged.

**Tests:**

- `ContinuousRunner` creates `CUDAGraphRunner` when `use_cuda_graphs=True`.
- `ContinuousRunner` does NOT create `CUDAGraphRunner` when `use_cuda_graphs=False`.
- Decode uses graph path when available, falls back to eager for oversized batches.
- `warmup_cuda_graphs()` succeeds during init.

### 8. `EngineConfig` changes (`src/infer/engine/config.py`) --- MODIFIED

Add `use_cuda_graphs` field and validation.

**Field declaration:**

```python
@dataclass
class EngineConfig:
    ...
    # CUDA graphs (Phase 9).
    use_cuda_graphs: bool = False
```

**Docstring addition:**

```python
use_cuda_graphs: Capture decode forward pass as CUDA graphs (Phase 9).
    Requires ``kv_cache_backend="paged"`` and ``batching_mode="continuous"``.
```

**Validation rules (in `validate()`):**

```python
if self.use_cuda_graphs:
    if self.kv_cache_backend != "paged":
        raise ValueError("CUDA graphs require kv_cache_backend='paged'")
    if self.batching_mode != "continuous":
        raise ValueError("CUDA graphs require batching_mode='continuous'")
```

**Tests:** Config validation (cuda_graphs + contiguous -> error, cuda_graphs + static -> error, cuda_graphs + paged + continuous -> ok).

### 9. Server CLI (`src/infer/server/__main__.py`) --- MODIFIED

Add `--cuda-graphs` flag.

```python
parser.add_argument(
    "--cuda-graphs",
    action="store_true",
    default=False,
    help="enable CUDA graph capture for decode (requires --kv-cache-backend paged)",
)
```

Passed to `EngineConfig`:

```python
config = EngineConfig(
    ...
    use_cuda_graphs=args.cuda_graphs,
)
```

### 10. Engine warmup (`src/infer/engine/engine.py`) --- MODIFIED

Call CUDA graph warmup during engine initialization.

```python
# In Engine.__init__, after runner creation:
if config.use_cuda_graphs and isinstance(self.runner, ContinuousRunner):
    self.runner.warmup_cuda_graphs()
```

---

## Files NOT changed

- **`src/infer/cache/__init__.py`** --- `GraphPagedDecodeCacheView` is internal to `CUDAGraphRunner` and not part of the public cache API. No export needed.
- **`src/infer/cache/protocol.py`** --- `KVCacheProtocol` unchanged. `GraphPagedDecodeCacheView` satisfies the protocol (it has `seq_len`, `update` is not needed for Triton path, `write_only`/`advance`/`is_paged` are provided).
- **`src/infer/cache/slotted.py`** --- Contiguous/slotted backends unchanged. CUDA graphs require paged backend.
- **`src/infer/cache/prefix.py`** --- Prefix tree unchanged. CUDA graphs are orthogonal to prefix caching (prefix caching affects prefill; CUDA graphs affect decode).
- **`src/infer/engine/scheduler.py`** --- Scheduler unchanged. It has no knowledge of CUDA graphs.
- **`src/infer/engine/request.py`** --- No new request fields needed.
- **`src/infer/engine/sampler.py`** --- Sampler unchanged. Sampling happens after graph replay on CPU.
- **`src/infer/engine/runner.py`** --- Static-batch `ModelRunner` unchanged. CUDA graphs require continuous batching.
- **`src/infer/engine/runner_helpers.py`** --- Helpers unchanged. They run after the forward pass.
- **`src/infer/kernels/rms_norm.py`**, **`rope.py`**, **`fused_norm_residual.py`**, **`activation.py`** --- All Triton kernels unchanged. They are captured in the graph as-is.

---

## Invariants

1. **Fixed tensor addresses.** All tensors touched by the captured graph (placeholders, view tensors, pool K/V storage) remain at the same GPU memory addresses between replays. No reallocation of these tensors after capture.

2. **Correct write positions.** `GraphPagedDecodeCacheView.prepare()` computes write positions from current pool state. After replay, the KV data at position `pool.seq_lens[seq_id]` in the correct block is the new token's KV. The scratch block absorbs writes from padding slots.

3. **CPU/GPU state consistency.** After every replay, `pool.seq_lens[seq_id]` is incremented by 1 for each real sequence. This keeps CPU state (used by `prepare()`) in sync with GPU reality (where the KV write landed).

4. **Bucket coverage.** Every batch size from 1 to `max_batch_size` maps to a captured graph (padded to nearest power-of-2 bucket). Batch sizes exceeding the largest bucket fall back to eager mode.

5. **No graph for prefill.** Only the decode path is captured. Prefill shapes vary per request (different prompt lengths, chunk sizes) and cannot be captured in a fixed graph.

6. **Backward compatibility.** When `use_cuda_graphs=False` (default), no `CUDAGraphRunner` is created, and all behavior is identical to Phase 8. The model forward change (mask=None for decode without padding_mask) has no functional effect on existing paths: the eager batched decode path always passes `padding_mask`, and the single-request prefill path was already on the `mask=None` branch.

---

## Costs and Trade-offs

**VRAM overhead.** CUDA graph capture allocates memory for all intermediate activations in the decode forward pass. With a shared memory pool across all batch size buckets, the overhead is approximately that of the single largest bucket. Estimate for Llama 3B with max batch 32:

| Component | Size |
|---|---|
| Graph intermediates (shared pool) | ~50--100 MB |
| Logits placeholder (largest bucket) | ~8 MB |
| Page table + seq_lens + write index tensors | < 1 MB |
| Scratch block (1 block = block_size * head_dim * heads * layers * 2) | < 1 MB |
| **Total** | **~60--110 MB** |

This is roughly 0.5--1% of 16 GB VRAM. Acceptable.

**Warmup time.** Capturing 6 graphs (one per bucket) requires 6 * (3 warmup + 1 capture) = 24 forward passes at startup. For Llama 3B, each decode forward is ~10--15ms, so warmup takes ~250--400ms. This is a one-time cost at server startup.

**Padding waste.** A batch of 5 requests is padded to 8, running 3 dummy forward passes. The wasted compute is proportional to the padding ratio. Worst case: batch size 17 padded to 32 (47% waste). Average waste across uniform batch sizes 1--32: ~25%. This is acceptable because the graph replay saves more time from eliminated launch overhead than it loses from padding.

**Interaction with existing features:**

| Feature | Interaction |
|---|---|
| Prefix caching | Compatible. Prefix caching affects prefill (match/insert); CUDA graphs affect decode. Both can be enabled simultaneously. |
| Chunked prefill | Compatible. Prefill stays eager; decode uses graphs. |
| Paged attention | Required. Triton paged attention is the only graph-compatible attention path. |
| Contiguous backend | Incompatible. Config validation rejects this combination. |
| Static batching | Incompatible. Config validation rejects this combination. |

---

## Benchmark Plan

Run the `continuous_batching` and `baseline` workloads on all three benchmark models with CUDA graphs enabled.

**Configurations to compare:**

| Config | Backend | Chunked | Prefix | CUDA Graphs | Expected outcome |
|---|---|---|---|---|---|
| Baseline (Phase 8) | paged | on | off | off | Eager decode |
| +CUDA Graphs | paged | on | off | on | Reduced decode latency |

**Additional configuration (stacking with prefix caching):**

| Config | Backend | Chunked | Prefix | CUDA Graphs | Expected outcome |
|---|---|---|---|---|---|
| +Prefix+Graphs | paged | on | on | on | Both benefits combined |

**Metrics to report:**

- Throughput (tokens/sec --- expect 5--15% improvement from reduced launch overhead)
- ITL P50 / P99 (expect improvement --- each decode step is faster)
- TTFT P50 / P99 (unchanged --- prefill is not graphed)
- Request latency P50 / P99
- Graph memory overhead (VRAM before/after graph capture)

**Additional workloads for regression check:**

- Run all existing workloads (`baseline`, `continuous_batching`, `paged_attention`, `chunked_prefill`, `prefix_caching`) with `use_cuda_graphs=False` to verify no regression from the model forward change alone.
- Run with `use_cuda_graphs=True` on the non-prefix workloads to verify no interaction issues.

---

## Testing Plan

### Unit tests (`tests/unit/test_cuda_graph_runner.py`)

- **Bucket selection:** `_padded_batch_size(1)` → 1, `_padded_batch_size(3)` → 4, `_padded_batch_size(32)` → 32, `_padded_batch_size(33)` → `None`.
- **GraphPagedDecodeCacheView.prepare:** Verify page_table_tensor, seq_lens_tensor, write_block_ids, write_offsets match expected values from pool state.
- **GraphPagedDecodeCacheView.write_only:** Compare output against reference `_write_token` on same inputs.
- **GraphPagedDecodeCacheView.advance:** Verify no-op (pool state unchanged).
- **Scratch block:** Padding slot writes go to scratch block, not real blocks.

### Unit tests (`tests/unit/test_paged_attention_window.py`)

- **Full attention (window=0):** Output identical to existing kernel.
- **Sliding window:** Only last `window_size` positions contribute. Compare against SDPA with explicit sliding-window mask.
- **Window >= seq_len:** Equivalent to full attention.
- **Window = 1:** Attend only to current position.
- **Various block alignments:** Window boundary at block start, mid-block, end-of-block.

### Integration tests (`tests/unit/test_config.py`)

- **Config validation:** `use_cuda_graphs=True` + `contiguous` → error.
- **Config validation:** `use_cuda_graphs=True` + `static` → error.
- **Config validation:** `use_cuda_graphs=True` + `paged` + `continuous` → ok.

### End-to-end tests (`tests/integration/`)

- **Correctness parity:** Same prompt, same seed → identical output with and without CUDA graphs (greedy decode, all three models).
- **Multi-step decode:** Graph replay produces correct tokens over 32+ consecutive decode steps.
- **Batch size transitions:** Batch shrinks from 8 to 5 (request finishes) → bucket changes from 8 to 8, then to 4. Correctness maintained.
- **Eager fallback:** Batch of 33 requests → falls back to eager. Correct output.
- **Combined with prefix caching:** Prefix caching + CUDA graphs enabled together. Correct output.

### Benchmark

- **Decode throughput:** CUDA graphs vs eager on `continuous_batching` workload. Expect 5--15% throughput improvement.
- **No regression when disabled:** All Phase 1--8 tests pass with `use_cuda_graphs=False`.

---

## Exit Criteria

1. **Decode throughput improvement.** Measurable improvement (expect 5--15%) on the `continuous_batching` and `baseline` workloads for at least Llama 3B and Qwen3 4B.
2. **Correctness.** Greedy decode output identical with and without CUDA graphs (all three models, same seed).
3. **No regression when disabled.** All Phase 1--8 tests pass with `use_cuda_graphs=False`. All benchmark results unchanged.
4. **Graph memory overhead documented.** VRAM usage before and after graph capture reported in benchmark results.
5. **Config validation.** `use_cuda_graphs=True` with wrong backend or batching mode → `ValueError`.
6. **Eager fallback works.** Oversized batches fall back gracefully.

---

## References

- [CUDA Graphs (NVIDIA docs)](https://developer.nvidia.com/blog/cuda-graphs/)
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [PyTorch CUDA Graphs API](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [vLLM CUDA graph implementation](https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py)
- [SGLang CUDA graph runner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py)
