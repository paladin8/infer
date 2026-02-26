# Phase 6: Paged Attention

## Goal

Replace contiguous per-sequence KV cache allocation with block-based paged memory. In Phase 5, each cache slot pre-allocates `max_seq_len` positions regardless of how many tokens the sequence actually uses. A request that generates 200 tokens out of a 4096-position slot wastes 95% of its allocation. Paged attention allocates fixed-size blocks on demand as sequences grow, so unused positions can serve other sequences.

The primary benefit is higher concurrent sequence capacity at the same VRAM budget. With contiguous allocation, `max_batch_size` is tightly coupled to `max_seq_len` because every slot reserves worst-case memory. With paged allocation, the total block pool is shared and blocks flow to whichever sequences need them. When average sequence length is well below `max_seq_len`, the system can serve significantly more concurrent requests.

A key design principle is **protocol-driven abstraction**: the runner and engine interact with the cache pool through a `CachePoolProtocol`, eliminating `isinstance` checks and keeping backend-specific logic inside the cache module. The model code continues to use `KVCacheProtocol` for cache views, unchanged from Phase 5.

Benchmark models: `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-4B`, `google/gemma-3-1b-it` (same as Phases 4-5).

---

## Architecture

```text
               HTTP POST /v1/completions
                          │
                          ▼
             ┌─────────────────────────┐
             │      FastAPI Server     │  (unchanged)
             └────────────┬────────────┘
                          │ asyncio.Queue per request
                          ▼
             ┌──────────────────────────┐
             │      Engine.step()       │
             │                          │
             │  1. scheduler.retire()   │──► retire finished, free resources
             │  2. pool.free_slot()     │──► free blocks/slots
             │  3. pool.free_token_     │──► query available budget
             │     capacity()           │
             │  4. scheduler.admit()    │──► admit new (budget check)
             │  5. runner.step(...)     │──► prefill new + decode active
             │  6. push StepOutputs     │
             └─────────────┬────────────┘
                           │
           ┌───────────────┼─────────────────┐
           ▼               ▼                 ▼
   ┌───────────────┐ ┌────────────┐ ┌────────────────────────┐
   │ Continuous    │ │ Continuous │ │ CachePoolProtocol      │
   │ Scheduler     │ │ Runner     │ │                        │
   │               │ │            │ │  allocate_slot()       │
   │ waiting queue │ │ prefill()  │ │  free_slot()           │
   │ active set    │ │ decode()   │ │  get_seq_len()         │
   │ retire/admit  │ │            │ │  free_token_capacity() │
   │               │ │ cache_pool:│ │  prefill_view()        │
   │               │ │  CachePool │ │  decode_view()         │
   │               │ │  Protocol  │ │  batched_prefill_view()│
   └───────────────┘ └────────────┘ └────────────────────────┘
                                       │                │
                            ┌──────────┘                └──────────┐
                            ▼                                      ▼
                    SlottedKVCache                       PagedKVCachePool
                    (Phase 5, contiguous)                (Phase 6, paged)
                    [layers, slots,                      BlockAllocator
                     heads, max_seq,                     Page tables
                     dim]                                [layers, blocks,
                                                          heads, block_sz,
                                                          dim]
```

**Key difference from Phase 5**: the cache pool is no longer a `[layers, max_batch_size, heads, max_seq_len, dim]` tensor with fixed slot assignments. Instead, it is a `[layers, total_blocks, heads, block_size, dim]` tensor where blocks are dynamically allocated to sequences via page tables. A sequence's KV data may be scattered across non-contiguous blocks, but the page table provides the mapping from logical position to physical location.

**Protocol layering**:
- `KVCacheProtocol` (modified): `update()`, `advance()`, `seq_len`, `is_paged()` — used by model code (`Attention.forward()`). All cache views (slotted and paged) implement this. The new `is_paged()` method enables the Attention class to dispatch to the Triton paged kernel without `isinstance` checks.
- `CachePoolProtocol` (new): `allocate_slot()`, `free_slot()`, `get_seq_len()`, `free_token_capacity()`, `is_paged()`, view factories — used by the runner and engine. Both `SlottedKVCache` and `PagedKVCachePool` implement this. Eliminates all `isinstance` checks in the runner.

**Engine step order** (both backends use this split interface):
1. `scheduler.retire()` — identify and remove finished requests from active set
2. `runner.free_slot(slot_idx)` + `runner.cleanup_request(request_id)` for each retired request
3. `cache_pool.free_token_capacity()` — query available KV token budget
4. `scheduler.admit(free_kv_tokens)` — admit new requests, checking block budget
5. `scheduler.decode_requests()` — identify active decode requests
6. Batched decode for all active decode requests
7. Prefill new requests (individual or batched)
8. Push StepOutputs to per-request queues

The retire-then-free-then-admit ordering ensures blocks freed from completed sequences are immediately available for newly admitted requests in the same step. Phase 5's monolithic `schedule()` retired and admitted atomically, which was fine for contiguous allocation (slots are reused immediately via the free set) but would miss one step of block availability with paged allocation if the combined method were used with paged backend.

**No API changes**: the `POST /v1/completions` endpoint, SSE event format, and error responses are identical. Paged attention is selected via `kv_cache_backend="paged"` in `EngineConfig`. When `kv_cache_backend="contiguous"`, all Phase 5 behavior is preserved exactly.

---

## Deliverables

### 1. CachePoolProtocol (`src/infer/cache/protocol.py`)

A structural `Protocol` that abstracts the cache pool interface. Both `SlottedKVCache` (contiguous) and `PagedKVCachePool` (paged) satisfy this protocol, so the runner and engine interact with a single type without `isinstance` checks.

```python
class CachePoolProtocol(Protocol):
    """Interface for KV cache pools (slotted and paged).

    Abstracts allocation, deallocation, view creation, and capacity
    queries so the runner and engine operate identically across backends.
    """

    def allocate_slot(self, initial_tokens: int = 0) -> int:
        """Allocate a cache slot for a new sequence.

        Args:
            initial_tokens: Number of tokens to pre-allocate capacity for
                (prompt length). Paged backends use this to eagerly allocate
                blocks; contiguous backends ignore it (slots have fixed
                pre-allocated capacity).

        Returns:
            Integer slot/sequence ID, stored in ``request.slot_idx``.

        Raises:
            RuntimeError: If insufficient capacity is available.
        """
        ...

    def free_slot(self, slot: int) -> None:
        """Release a slot and all associated resources (blocks, seq_lens, page tables)."""
        ...

    def get_seq_len(self, slot: int) -> int:
        """Return the current sequence length for a slot."""
        ...

    def free_slot_count(self) -> int:
        """Number of available slots (contiguous) or a bound on allocatable sequences (paged)."""
        ...

    def free_token_capacity(self) -> int | None:
        """Number of tokens that can be stored in free capacity.

        Returns ``None`` for backends where token budget is not applicable
        (contiguous — capacity is per-slot, not token-granular).
        """
        ...

    def prefill_view(self, slot: int) -> KVCacheProtocol:
        """Return a single-slot view for prefill."""
        ...

    def decode_view(self, active_slots: list[int]) -> KVCacheProtocol:
        """Return a multi-slot view for batched decode."""
        ...

    def batched_prefill_view(
        self, slots: list[int], prompt_lens: list[int],
    ) -> KVCacheProtocol:
        """Return a multi-slot view for batched prefill."""
        ...

    def is_paged(self) -> bool:
        """Whether this pool uses paged block allocation.

        Used by the Attention class to dispatch to the Triton paged
        attention kernel when available. Returns ``False`` for
        contiguous backends, ``True`` for paged backends.
        """
        ...
```

**Required change to `KVCacheProtocol`** (`src/infer/cache/protocol.py`):

Add `is_paged()` so the Attention class can dispatch to the Triton kernel without `isinstance`:

```python
class KVCacheProtocol(Protocol):
    @property
    def seq_len(self) -> int:
        """Current sequence length (read by model for mask width)."""
        ...

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]: ...
    def advance(self, n: int) -> None: ...
    def is_paged(self) -> bool: ...  # NEW
```

Note: `seq_len` is declared as a read-only `@property` in the protocol rather than a plain `seq_len: int` attribute. This allows implementations to use either a regular attribute (`self.seq_len: int = 0` in prefill views) or a computed `@property` (`DecodeCacheView` computes it as the max of active slot seq_lens). A plain `seq_len: int` protocol member would require a writable attribute, which `@property` without a setter does not satisfy for mypy's structural checking.

All existing cache views (`PrefillCacheView`, `BatchedPrefillCacheView`, `DecodeCacheView`) must add `is_paged() -> bool` returning `False`. The paged views return `True`.

**Required changes to `SlottedKVCache`** (`src/infer/cache/slotted.py`):

All changes needed for `SlottedKVCache` to satisfy `CachePoolProtocol`, listed in one place. These are breaking interface changes that must be applied atomically with the runner changes (deliverable 7) — if the runner calls `allocate_slot(initial_tokens=N)` but `SlottedKVCache` doesn't accept that keyword, all contiguous-backend prefill calls will raise `TypeError`.

```python
# In slotted.py — new/modified methods:

# MODIFIED: add initial_tokens parameter (currently takes no args)
def allocate_slot(self, initial_tokens: int = 0) -> int:
    """Claim a free slot. Raises RuntimeError if none available.

    The ``initial_tokens`` parameter is accepted for protocol
    compatibility but ignored — slots have fixed pre-allocated capacity.
    """
    if not self._free_slots:
        raise RuntimeError("No free cache slots available")
    return self._free_slots.pop()

# NEW: protocol method replacing direct self.seq_lens[slot] access
def get_seq_len(self, slot: int) -> int:
    """Return the current sequence length for a slot."""
    return self.seq_lens[slot]

# NEW: protocol method (contiguous backend has no token-level budget)
def free_token_capacity(self) -> int | None:
    """Not applicable for contiguous backend."""
    return None
```

```python
# NEW: protocol method for Triton dispatch
def is_paged(self) -> bool:
    """Contiguous backend is not paged."""
    return False
```

Summary of all changes:
1. `allocate_slot()` → `allocate_slot(self, initial_tokens: int = 0)` — **modified** signature (ignored parameter)
2. `get_seq_len(self, slot: int) -> int` — **new** method
3. `free_token_capacity(self) -> int | None` — **new** method
4. `is_paged(self) -> bool` — **new** method (returns `False`)
5. `free_slot`, `free_slot_count`, `prefill_view`, `decode_view`, `batched_prefill_view` — **unchanged** (signatures already compatible)

### 2. Block allocator (`src/infer/cache/paged.py`)

Manages a pool of integer block IDs. Allocation and deallocation are O(1) using a set of free block IDs.

```python
class BlockAllocator:
    """Manages allocation and deallocation of fixed-size KV cache blocks.

    Each block holds ``block_size`` token positions worth of K/V data.
    Block IDs are integers in ``[0, total_blocks)``.

    Tracks block ownership for leak detection: every allocated block records
    the owner (sequence ID) that requested it. Blocks whose owner is no
    longer active are "leaked" and can be reclaimed.

    Attributes:
        total_blocks: Total number of blocks in the pool.
        _free_blocks: Set of currently unallocated block IDs.
        _allocated_blocks: Set of currently allocated block IDs (for
            double-free detection).
        _block_owners: Mapping of block ID to owner (sequence ID) for
            leak detection.
    """

    def __init__(self, total_blocks: int) -> None:
        self.total_blocks = total_blocks
        self._free_blocks: set[int] = set(range(total_blocks))
        self._allocated_blocks: set[int] = set()
        self._block_owners: dict[int, int] = {}

    def allocate(self, count: int, owner: int | None = None) -> list[int]:
        """Allocate ``count`` blocks. Returns list of block IDs.

        Args:
            count: Number of blocks to allocate.
            owner: Optional owner ID (sequence ID) for leak tracking.

        Raises ``RuntimeError`` if fewer than ``count`` blocks are free.
        """
        if count > len(self._free_blocks):
            raise RuntimeError(
                f"Cannot allocate {count} blocks: only {len(self._free_blocks)} free"
            )
        blocks = [self._free_blocks.pop() for _ in range(count)]
        self._allocated_blocks.update(blocks)
        if owner is not None:
            for bid in blocks:
                self._block_owners[bid] = owner
        return blocks

    def free(self, block_ids: list[int]) -> None:
        """Return blocks to the free pool.

        Raises ``ValueError`` if any block ID is not currently allocated
        (catches double-free bugs).
        """
        for bid in block_ids:
            if bid not in self._allocated_blocks:
                raise ValueError(
                    f"Cannot free block {bid}: not currently allocated "
                    f"(double-free or invalid block ID)"
                )
        self._allocated_blocks.difference_update(block_ids)
        self._free_blocks.update(block_ids)
        for bid in block_ids:
            self._block_owners.pop(bid, None)

    def num_free(self) -> int:
        """Number of currently free blocks."""
        return len(self._free_blocks)

    def num_allocated(self) -> int:
        """Number of currently allocated blocks."""
        return len(self._allocated_blocks)

    def can_allocate(self, count: int) -> bool:
        """Check whether ``count`` blocks can be allocated."""
        return len(self._free_blocks) >= count

    def find_leaked_blocks(self, active_owners: set[int]) -> dict[int, list[int]]:
        """Find blocks whose owner is not in the active set.

        Returns a dict mapping orphaned owner IDs to their block lists.
        An empty dict means no leaks detected.
        """
        leaked: dict[int, list[int]] = {}
        for bid, owner in self._block_owners.items():
            if owner not in active_owners:
                leaked.setdefault(owner, []).append(bid)
        return leaked

    def allocated_block_ids(self) -> set[int]:
        """Return a copy of the currently allocated block ID set."""
        return set(self._allocated_blocks)

    def find_unreferenced_blocks(self, referenced: set[int]) -> list[int]:
        """Find allocated blocks that are not in the referenced set.

        These are "leaked" blocks — allocated but not reachable via any
        external data structure (e.g., page tables). Returns sorted list.
        """
        return sorted(self._allocated_blocks - referenced)

    def force_free(self, block_ids: list[int]) -> None:
        """Free blocks without the normal double-free check.

        Used by leak reclamation paths where the blocks are known to be
        allocated but not tracked by normal ownership structures.
        """
        self._allocated_blocks.difference_update(block_ids)
        self._free_blocks.update(block_ids)
        for bid in block_ids:
            self._block_owners.pop(bid, None)
```

The allocator has no knowledge of tensor storage — it only tracks integer IDs. This separation makes it easy to test allocation logic without GPU memory.

**Safety features**:
- The `_allocated_blocks` set provides O(1) double-free detection, catching bugs that would otherwise silently corrupt state.
- The `_block_owners` dict tracks which sequence owns each block. `find_leaked_blocks()` compares owners against a set of known-active sequences to detect blocks that were allocated but never freed (e.g., because a sequence was abandoned mid-step or an error handler missed cleanup). This is an O(allocated_blocks) scan, suitable for periodic health checks or error recovery paths.
- `find_unreferenced_blocks(referenced)` detects blocks that are allocated but not reachable via any external data structure (e.g., page tables). `force_free()` reclaims them without the double-free check, since these blocks are known-allocated but not tracked by normal ownership paths.

### 3. Paged KV cache pool (`src/infer/cache/paged.py`)

The pool owns the block tensor storage, the block allocator, and per-sequence page tables. It implements `CachePoolProtocol`.

```python
class PagedKVCachePool:
    """Block-allocated KV cache pool for paged attention.

    Instead of pre-allocating ``max_seq_len`` positions per sequence, the pool
    maintains a shared tensor of fixed-size blocks and allocates them on demand.
    Each sequence has a page table mapping logical block indices to physical
    block IDs in the pool.

    Satisfies ``CachePoolProtocol``.

    Attributes:
        k: Key block pool, shape
            ``[num_layers, total_blocks, num_kv_heads, block_size, head_dim]``.
        v: Value block pool, same shape as ``k``.
        block_size: Tokens per block.
        allocator: Block allocator managing free/used block tracking.
        page_tables: Per-sequence page tables mapping seq_id to list of block IDs.
        seq_lens: Per-sequence position counters.
    """

    def __init__(
        self,
        k: Tensor,
        v: Tensor,
        total_blocks: int,
        block_size: int,
    ) -> None:
        self.k = k
        self.v = v
        self.block_size = block_size
        self.allocator = BlockAllocator(total_blocks)

        # Per-sequence state.
        self.page_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}
        self._next_seq_id: int = 0

    @staticmethod
    def from_model_config(
        config: ModelConfig,
        total_blocks: int,
        block_size: int = 16,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> PagedKVCachePool:
        """Allocate a block pool sized for the given model config.

        Args:
            config: Model configuration (provides layer count, KV head count,
                head dimension).
            total_blocks: Number of blocks in the pool.
            block_size: Tokens per block.
            dtype: Data type for cache tensors.
            device: Device for cache tensors.
        """
        shape = (
            config.num_hidden_layers,
            total_blocks,
            config.num_key_value_heads,
            block_size,
            config.computed_head_dim,
        )
        k = torch.zeros(shape, dtype=dtype, device=device)
        v = torch.zeros(shape, dtype=dtype, device=device)
        return PagedKVCachePool(k, v, total_blocks, block_size)

    def allocate_slot(self, initial_tokens: int = 0) -> int:
        """Allocate a new sequence and optionally pre-allocate blocks.

        Args:
            initial_tokens: Number of tokens to pre-allocate blocks for
                (typically the prompt length). Blocks are allocated eagerly
                to fail fast if the pool is exhausted.

        Returns:
            Integer sequence ID. Stored in ``request.slot_idx`` for
            compatibility with the runner interface.

        Raises:
            RuntimeError: If not enough blocks are available.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self.page_tables[seq_id] = []
        self.seq_lens[seq_id] = 0

        if initial_tokens > 0:
            blocks_needed = (initial_tokens + self.block_size - 1) // self.block_size
            try:
                blocks = self.allocator.allocate(blocks_needed, owner=seq_id)  # owner for leak tracking
            except RuntimeError:
                # Clean up partial state on failure.
                del self.page_tables[seq_id]
                del self.seq_lens[seq_id]
                raise
            self.page_tables[seq_id] = blocks

        return seq_id

    def free_slot(self, seq_id: int) -> None:
        """Free all blocks for a sequence and remove its tracking state.

        Raises ``KeyError`` if ``seq_id`` is not a valid active sequence.
        """
        blocks = self.page_tables.pop(seq_id)
        self.allocator.free(blocks)
        del self.seq_lens[seq_id]

    def get_seq_len(self, seq_id: int) -> int:
        """Return the current sequence length for a sequence."""
        return self.seq_lens[seq_id]

    def active_seq_count(self) -> int:
        """Number of currently active sequences."""
        return len(self.page_tables)

    def free_slot_count(self) -> int:
        """Number of available slots.

        For paged pools, returns the number of free blocks. This is an
        upper bound on new sequences (each needs at least 1 block).
        The scheduler uses ``free_token_capacity()`` for the actual
        admission budget, not this method.
        """
        return self.allocator.num_free()

    def free_token_capacity(self) -> int:
        """Number of tokens that can be stored in currently free blocks."""
        return self.allocator.num_free() * self.block_size

    def audit_blocks(self) -> list[int]:
        """Return block IDs that are allocated but not in any page table.

        These are "leaked" blocks — allocated but unreachable via any
        active sequence. Leaks can occur if an error handler fails to
        call ``free_slot()`` for a retired sequence, or if a crash
        interrupts the allocate → page-table-update sequence.

        Delegates to ``BlockAllocator.find_unreferenced_blocks()``.
        """
        referenced_blocks: set[int] = set()
        for blocks in self.page_tables.values():
            referenced_blocks.update(blocks)
        return self.allocator.find_unreferenced_blocks(referenced_blocks)

    def reclaim_leaked_blocks(self) -> int:
        """Free any leaked blocks (allocated but not in any page table).

        Returns the number of blocks reclaimed. Safe to call at any time
        (e.g., in the engine's error recovery path or a health check).

        Delegates to ``BlockAllocator.force_free()``.
        """
        orphaned = self.audit_blocks()
        if orphaned:
            self.allocator.force_free(orphaned)
        return len(orphaned)

    def is_paged(self) -> bool:
        """Paged backend."""
        return True

    def prefill_view(self, seq_id: int) -> PagedPrefillCacheView:
        """Return a single-sequence view for prefill."""
        return PagedPrefillCacheView(self, seq_id)

    def decode_view(self, seq_ids: list[int]) -> PagedDecodeCacheView:
        """Return a multi-sequence view for batched decode."""
        return PagedDecodeCacheView(self, seq_ids)

    def batched_prefill_view(
        self, seq_ids: list[int], prompt_lens: list[int],
    ) -> PagedBatchedPrefillCacheView:
        """Return a multi-sequence view for batched prefill."""
        return PagedBatchedPrefillCacheView(self, seq_ids, prompt_lens)
```

**Block pool tensor layout**: `[num_hidden_layers, total_blocks, num_key_value_heads, block_size, computed_head_dim]`. This mirrors the slotted layout `[num_hidden_layers, max_batch_size, num_key_value_heads, max_seq_len, computed_head_dim]` with `max_batch_size` replaced by `total_blocks` and `max_seq_len` replaced by `block_size`. Each block is contiguous in memory, enabling efficient block-level operations (scatter-write during prefill, gather during decode, and eventual block copy for Phase 8's prefix caching).

**`allocate_slot` compatibility**: returns an integer ID and stores per-sequence state in dicts keyed by that ID. The runner accesses `pool.get_seq_len(slot_idx)` through the protocol method. The `request.slot_idx` field is reused without modification.

**`initial_tokens` parameter**: when the runner calls `allocate_slot(initial_tokens=prompt_len)`, the pool pre-allocates enough blocks for the prompt. This is an eager allocation that fails fast if the pool is exhausted — better to reject the request before starting the forward pass than to fail mid-layer. If allocation fails, partial state is cleaned up before re-raising.

**Decoupled sizing**: the number of total blocks (`num_gpu_blocks` in `EngineConfig`) is independent of `max_batch_size`. For paged mode, `max_batch_size` controls compute concurrency (how many sequences run simultaneously), while `num_gpu_blocks` controls memory capacity (how many total KV token positions can be stored). These are independent concerns:

```
total_token_capacity = num_gpu_blocks * block_size
max_concurrent_sequences = max_batch_size  (compute budget only)
```

When `num_gpu_blocks` is not set, it defaults to `max_batch_size * max_seq_len // block_size`, matching the same total VRAM as the contiguous pool. Users can override `num_gpu_blocks` to allocate more or less cache memory independently of `max_batch_size`.

**Memory budget** (default auto-sizing with max_batch_size=8, block_size=16):

| Model        | Layers | KV heads | head_dim | Per-block bytes | Default blocks | Token capacity |  Cache VRAM |
|--------------|-------:|---------:|---------:|----------------:|---------------:|---------------:|------------:|
| Llama 3.2 3B |     28 |        8 |      128 |       1.12 KB   |          2,048 |         32,768 |    ~2.2 GB  |
| Qwen3 4B     |     36 |        8 |      128 |       1.44 KB   |          2,048 |         32,768 |    ~2.9 GB  |
| Gemma3 1B    |     26 |        1 |      256 |       0.26 KB   |          2,048 |         32,768 |    ~0.5 GB  |

Per-block VRAM = `2 (K+V) × num_layers × num_kv_heads × block_size × head_dim × 2 (bf16)`. This is the **total VRAM per logical block ID across all layers**, since each block ID occupies one position in every layer of the `[layers, blocks, heads, block_size, dim]` tensor. Cache VRAM = per-block VRAM × total blocks. Table values are approximate and should be recomputed from actual model configs during implementation.

**Capacity comparison** (same VRAM budget, varying average sequence length):

| Avg seq length | Phase 5 (contiguous, 8 slots) | Phase 6 (paged, max_batch_size=32) | Improvement |
|---------------:|------------------------------:|-----------------------------------:|------------:|
|            256 |                             8 |                                128 |         16x |
|            512 |                             8 |                                 64 |          8x |
|          1,024 |                             8 |                                 32 |          4x |
|          2,048 |                             8 |                                 16 |          2x |
|          4,096 |                             8 |                                  8 |          1x |

At `max_seq_len` (4096), paged and contiguous have identical capacity — all blocks are used. The benefit scales with the gap between average and maximum sequence length.

### 4. Paged cache views (`src/infer/cache/paged.py`)

Three cache view classes implement `KVCacheProtocol`, mirroring the Phase 5 slotted views. The model code calls `update()` and `advance()` without knowing whether the backing storage is contiguous or paged.

#### `PagedPrefillCacheView`

Wraps a single sequence for prefill. Writes K/V to blocks via vectorized scatter and returns the input K/V directly (same approach as Phase 5's `PrefillCacheView` — during prefill, the model already has the full K/V from the current forward pass and doesn't need to read from the cache).

```python
class PagedPrefillCacheView:
    """Single-sequence paged cache view for prefill.

    Scatter-writes K/V tokens to their assigned blocks. Returns the input
    K/V directly — during prefill, the model has the full K/V from the
    current forward pass and attention runs over it, not the cache. The
    scatter-write populates the cache for subsequent decode steps.
    """

    def __init__(self, pool: PagedKVCachePool, seq_id: int) -> None:
        self.pool = pool
        self.seq_id = seq_id
        self.seq_len: int = 0

    def is_paged(self) -> bool:
        return True

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Scatter-write K/V to blocks, return input directly.

        k, v shape: ``[1, num_kv_heads, prompt_len, head_dim]``.
        """
        prompt_len = k.shape[2]
        blocks = self.pool.page_tables[self.seq_id]
        device = k.device

        # Compute block ID and intra-block offset for each token position.
        positions = torch.arange(
            self.seq_len, self.seq_len + prompt_len, device=device,
        )
        block_indices = positions // self.pool.block_size
        offsets = positions % self.pool.block_size
        block_ids = torch.tensor(blocks, dtype=torch.long, device=device)
        physical_blocks = block_ids[block_indices]  # [prompt_len]

        # Vectorized scatter-write.
        # pool.k shape: [layers, total_blocks, kv_heads, block_size, head_dim]
        # k shape: [1, kv_heads, prompt_len, head_dim]
        # Rearrange k from [1, heads, prompt_len, dim] to [prompt_len, heads, dim]
        # to align with the [prompt_len]-indexed physical_blocks and offsets.
        k_flat = k[0].permute(1, 0, 2)  # [prompt_len, heads, dim]
        v_flat = v[0].permute(1, 0, 2)
        self.pool.k[layer_idx, physical_blocks, :, offsets, :] = k_flat
        self.pool.v[layer_idx, physical_blocks, :, offsets, :] = v_flat

        return k, v

    def advance(self, n: int) -> None:
        """Advance this sequence's position counter."""
        self.seq_len += n
        self.pool.seq_lens[self.seq_id] = self.seq_len
```

**Vectorized scatter-write**: instead of a Python loop over token positions, the write uses advanced indexing with `physical_blocks` and `offsets` tensors. The `k[0].permute(1, 0, 2)` rearranges from `[heads, prompt_len, dim]` to `[prompt_len, heads, dim]` to match the indexing dimensions `[prompt_len]` of `physical_blocks` and `offsets`. This is a single GPU operation per K and V, regardless of prompt length.

#### `PagedDecodeCacheView`

Wraps multiple active sequences for batched decode. Writes new K/V tokens to blocks and gathers the full cache from blocks for attention using a **vectorized flat-gather** that avoids per-sequence Python loops on every layer.

```python
class PagedDecodeCacheView:
    """Multi-sequence paged cache view for batched decode.

    Each sequence's new K/V token is written to its current block. The full
    K/V cache is gathered from scattered blocks for attention via SDPA.

    Gather indices are computed once at construction time and reused across
    all layers, reducing the per-layer cost to a single GPU advanced-index
    operation.
    """

    def __init__(self, pool: PagedKVCachePool, seq_ids: list[int]) -> None:
        self.pool = pool
        self.seq_ids = seq_ids
        self.slot_seq_lens = [pool.seq_lens[sid] for sid in seq_ids]
        self._seq_len = max(self.slot_seq_lens) if seq_ids else 0

        # Pre-computed gather indices (set on first update call).
        self._gather_block_ids: Tensor | None = None
        self._gather_offsets: Tensor | None = None

    def is_paged(self) -> bool:
        return True

    @property
    def seq_len(self) -> int:
        """Max of active sequence lengths (used for mask width)."""
        return self._seq_len

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write new tokens to blocks, gather full cache for attention.

        k, v shape: ``[batch, num_kv_heads, 1, head_dim]``.

        On the first layer (``layer_idx == 0``), allocates a new block for
        any sequence whose current block is full and computes gather indices
        for all layers. Subsequent layers reuse the cached indices.

        Block allocation failure raises ``RuntimeError``. The engine's
        error handler catches this and marks the entire batch as failed
        (same behavior as Phase 5). Per-sequence error isolation is
        deferred to future work.

        Returns gathered K/V of shape
        ``[batch, num_kv_heads, max_seq_len + 1, head_dim]``.
        """
        device = k.device

        if layer_idx == 0:
            # Allocate new blocks if needed (once per step).
            for i, seq_id in enumerate(self.seq_ids):
                pos = self.slot_seq_lens[i]
                block_idx = pos // self.pool.block_size
                if block_idx >= len(self.pool.page_tables[seq_id]):
                    new_block = self.pool.allocator.allocate(1, owner=seq_id)
                    self.pool.page_tables[seq_id].extend(new_block)

            # Write new token per sequence.
            for i, seq_id in enumerate(self.seq_ids):
                pos = self.slot_seq_lens[i]
                block_idx = pos // self.pool.block_size
                offset = pos % self.pool.block_size
                block_id = self.pool.page_tables[seq_id][block_idx]
                self.pool.k[layer_idx, block_id, :, offset, :] = k[i, :, 0, :]
                self.pool.v[layer_idx, block_id, :, offset, :] = v[i, :, 0, :]

            # Build flat gather indices for all sequences (reused across layers).
            max_len = self._seq_len + 1
            batch_size = len(self.seq_ids)
            gather_block_ids = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=device,
            )
            gather_offsets = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=device,
            )

            for i, seq_id in enumerate(self.seq_ids):
                seq_len_i = self.slot_seq_lens[i] + 1
                positions = torch.arange(seq_len_i, device=device)
                block_indices = positions // self.pool.block_size
                offsets = positions % self.pool.block_size
                blocks = torch.tensor(
                    self.pool.page_tables[seq_id], dtype=torch.long, device=device,
                )
                gather_block_ids[i, :seq_len_i] = blocks[block_indices]
                gather_offsets[i, :seq_len_i] = offsets

            self._gather_block_ids = gather_block_ids
            self._gather_offsets = gather_offsets
        else:
            # Write new token on layers > 0 (indices already computed).
            for i, seq_id in enumerate(self.seq_ids):
                pos = self.slot_seq_lens[i]
                block_idx = pos // self.pool.block_size
                offset = pos % self.pool.block_size
                block_id = self.pool.page_tables[seq_id][block_idx]
                self.pool.k[layer_idx, block_id, :, offset, :] = k[i, :, 0, :]
                self.pool.v[layer_idx, block_id, :, offset, :] = v[i, :, 0, :]

        # Vectorized flat gather: single GPU advanced-index operation.
        # pool.k[layer] shape: [total_blocks, kv_heads, block_size, head_dim]
        # gather_block_ids shape: [batch, max_len] — indexes dim 0 (blocks)
        # gather_offsets shape: [batch, max_len] — indexes dim 2 (block_size)
        # Result shape: [batch, max_len, kv_heads, head_dim] (advanced indices
        # are separated by a slice, so their dimensions come first).
        assert self._gather_block_ids is not None
        cached_k = self.pool.k[
            layer_idx, self._gather_block_ids, :, self._gather_offsets, :
        ]  # [batch, max_len, kv_heads, head_dim]
        cached_v = self.pool.v[
            layer_idx, self._gather_block_ids, :, self._gather_offsets, :
        ]

        # Permute to [batch, kv_heads, max_len, head_dim] for SDPA.
        return cached_k.permute(0, 2, 1, 3), cached_v.permute(0, 2, 1, 3)

    def advance(self, n: int) -> None:
        """Advance all active sequences by n positions."""
        for i, seq_id in enumerate(self.seq_ids):
            self.pool.seq_lens[seq_id] += n
            self.slot_seq_lens[i] += n
        self._seq_len += n
```

**Vectorized flat-gather**: the key performance optimization over the naive approach. Instead of a per-sequence Python loop on every layer (O(batch × layers) Python iterations), gather indices are computed once on layer 0 and reused for all subsequent layers. Each layer's gather is a single GPU advanced-index operation: `pool.k[layer_idx, block_ids, :, offsets, :]` where `block_ids` and `offsets` are `[batch, max_len]` tensors. PyTorch's advanced indexing with separated slice dimensions produces `[batch, max_len, kv_heads, head_dim]` in one kernel launch, then a permute gives the `[batch, kv_heads, max_len, head_dim]` layout SDPA expects.

**Per-layer cost breakdown**:
- Layer 0: O(batch) Python for block allocation + write + index construction, then 1 GPU gather
- Layers 1+: O(batch) Python for write only, then 1 GPU gather (indices cached)

The write loop (O(batch) per layer) is a single-token write per sequence — negligible compared to the gather. A fully vectorized write path is possible but would add complexity for minimal benefit (batch × 1 token vs batch × max_len for the gather).

**Lazy block allocation**: new blocks are allocated on `layer_idx == 0` when a sequence's current block is full (position falls beyond the allocated block count). This is transparent to the model and runner — the view handles it internally. If allocation fails (pool exhausted), a `RuntimeError` propagates to the engine's error handler, which marks the entire batch as failed (same behavior as Phase 5). Per-sequence error isolation is deferred.

**Padding mask**: the runner builds a padding mask `[batch, max_kv_len]` using `cache_pool.get_seq_len(slot)` for each sequence, exactly as in Phase 5. Positions beyond each sequence's actual length are masked out. The gathered K/V at those positions are zeros (from `torch.zeros` in the pool tensor init), and the mask ensures SDPA ignores them.

#### `PagedBatchedPrefillCacheView`

Wraps multiple sequences for batched prefill (when multiple requests arrive in the same step). Scatter-writes each batch element's K/V to its assigned sequence's blocks.

```python
class PagedBatchedPrefillCacheView:
    """Multi-sequence paged cache view for batched prefill.

    Used when multiple requests arrive in the same step. Each batch element's
    K/V is scatter-written to its assigned sequence's blocks. ``advance()``
    sets per-sequence ``seq_lens`` to actual prompt lengths, not padded length.
    """

    def __init__(
        self,
        pool: PagedKVCachePool,
        seq_ids: list[int],
        prompt_lens: list[int],
    ) -> None:
        self.pool = pool
        self.seq_ids = seq_ids
        self.prompt_lens = prompt_lens
        self.seq_len: int = 0

    def is_paged(self) -> bool:
        return True

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Scatter-write K/V to per-sequence blocks, return input directly.

        k, v shape: ``[batch, num_kv_heads, padded_len, head_dim]``.
        Each batch element is written to its sequence's blocks at the
        current position. Only real tokens (up to ``prompt_lens[i]``)
        are written; padding positions are skipped.
        """
        device = k.device
        for i, seq_id in enumerate(self.seq_ids):
            actual_len = self.prompt_lens[i]
            blocks = self.pool.page_tables[seq_id]
            positions = torch.arange(
                self.seq_len, self.seq_len + actual_len, device=device,
            )
            block_indices = positions // self.pool.block_size
            offsets = positions % self.pool.block_size
            block_ids = torch.tensor(blocks, dtype=torch.long, device=device)
            physical_blocks = block_ids[block_indices]

            k_flat = k[i, :, :actual_len, :].permute(1, 0, 2)  # [actual_len, heads, dim]
            v_flat = v[i, :, :actual_len, :].permute(1, 0, 2)
            self.pool.k[layer_idx, physical_blocks, :, offsets, :] = k_flat
            self.pool.v[layer_idx, physical_blocks, :, offsets, :] = v_flat

        return k, v

    def advance(self, n: int) -> None:
        """Set per-sequence seq_lens to actual prompt lengths (not padded)."""
        self.seq_len += n
        for i, seq_id in enumerate(self.seq_ids):
            self.pool.seq_lens[seq_id] = self.prompt_lens[i]
```

**Padding-aware writes**: unlike `PagedPrefillCacheView` which writes all tokens, the batched view writes only `prompt_lens[i]` tokens per sequence. Padding positions (from right-padding to `max_prompt_len`) are not written to blocks, which avoids wasting block capacity on padding data.

### 5. Scheduler changes (`src/infer/engine/scheduler.py`)

The `ContinuousScheduler` gains a split retire/admit/decode interface for the retire → free → admit ordering needed by paged allocation.

```python
class ContinuousScheduler:
    """Per-step scheduler for continuous batching.

    Phase 6 splits the monolithic ``schedule()`` into ``retire()``,
    ``admit()``, and ``decode_requests()`` to support the
    retire → free-blocks → admit-with-budget ordering needed for
    paged allocation.  Both backends use the split interface.
    The combined ``schedule()`` is retained for backward compatibility
    in existing Phase 5 tests.
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

    def retire(self) -> list[Request]:
        """Remove finished/failed requests from the active set.

        Returns the retired list so the engine can free their cache resources
        before admitting new requests.
        """
        retired = [r for r in self.active if r.state in _TERMINAL_STATES]
        self.active = [r for r in self.active if r.state not in _TERMINAL_STATES]
        return retired

    def admit(self, free_kv_tokens: int | None = None) -> list[Request]:
        """Admit new requests from the waiting queue.

        Checks two admission constraints:
        1. Compute budget: ``len(active) < max_batch_size``.
        2. Memory budget: cumulative prompt tokens of admitted requests
           must not exceed 80% of ``free_kv_tokens`` (when provided).
           The 20% reserve provides headroom for decode-time block
           allocation, reducing the likelihood of generation-time
           block exhaustion.

        When ``free_kv_tokens is None`` (contiguous backend), only the
        compute budget is checked — identical to Phase 5 behavior.

        Returns the list of newly admitted requests (need prefill).
        """
        capacity = self.config.max_batch_size - len(self.active)
        new: list[Request] = []
        # Reserve 20% of free tokens for decode-time block growth.
        remaining_tokens = int(free_kv_tokens * 0.8) if free_kv_tokens is not None else None

        while self.waiting and len(new) < capacity:
            req = self.waiting[0]
            if remaining_tokens is not None:
                prompt_len = len(req.prompt_token_ids)
                if prompt_len > remaining_tokens:
                    break  # not enough blocks for this request
                remaining_tokens -= prompt_len
            new.append(self.waiting.popleft())

        self.active.extend(new)
        return new

    def decode_requests(self) -> list[Request]:
        """Return active requests in DECODE state."""
        return [r for r in self.active if r.state == RequestState.DECODE]

    def schedule(self) -> ScheduleOutput:
        """Combined retire + admit + decode (backward-compatible convenience).

        Equivalent to ``retire()`` then ``admit(free_kv_tokens=None)`` then
        ``decode_requests()``. Does NOT pass a block budget — only safe for
        the contiguous backend where token budget is not applicable.

        Retained for backward compatibility with existing Phase 5 tests.
        The engine always uses the split interface.
        """
        retired = self.retire()
        prefill = self.admit(free_kv_tokens=None)
        decode = self.decode_requests()
        return ScheduleOutput(prefill=prefill, decode=decode, retired=retired)

    def has_work(self) -> bool:
        """True if there are active or waiting requests."""
        return bool(self.active) or bool(self.waiting)
```

**Engine always uses split interface**: both contiguous and paged backends use the `retire()` → `admit()` → `decode_requests()` path. For contiguous, `free_token_capacity()` returns `None` and `admit(free_kv_tokens=None)` checks only the compute budget — identical behavior to Phase 5. The combined `schedule()` is kept solely for backward compatibility in existing Phase 5 tests; the engine does not call it.

**Block budget as token count**: the scheduler's `admit()` takes `free_kv_tokens` (number of tokens that can be stored in free blocks) rather than a block count. This decouples the scheduler from block-size details and makes the budget check a simple integer comparison against prompt length. The engine computes `free_kv_tokens` from the cache pool before calling `admit()`.

**Admission reserve for generation headroom**: the admission check accounts only for prompt tokens, not the `max_new_tokens` each admitted request will eventually need for generation. To mitigate decode-time block exhaustion, `admit()` reserves 20% of free token capacity as headroom for generation growth. This means only 80% of free blocks are available for new prompt admission, leaving the remainder for in-flight sequences that need new blocks during decode. A more sophisticated approach (reserving `prompt_len + max_new_tokens` blocks per request) would provide stronger guarantees at the cost of lower utilization. The 20% reserve is a pragmatic middle ground — combined with lazy block allocation and the next step's admission check, it significantly reduces the likelihood of generation-time exhaustion without over-provisioning.

### 6. Engine changes (`src/infer/engine/engine.py`)

The engine step flow uses the split scheduler interface for both backends.

```python
def _step_continuous(self) -> None:
    """Engine step for continuous batching (contiguous and paged backends)."""
    assert isinstance(self.scheduler, ContinuousScheduler)
    assert isinstance(self.runner, ContinuousRunner)

    # Phase 1: Retire finished requests.
    retired = self.scheduler.retire()

    # Phase 2: Free cache resources for retired requests.
    for req in retired:
        if req.slot_idx is not None:
            self.runner.free_slot(req.slot_idx)
        self.runner.cleanup_request(req.request_id)

    # Phase 3: Query available memory budget (None for contiguous).
    free_kv_tokens = self.runner.free_kv_tokens()

    # Phase 4: Admit new requests with budget check.
    prefill = self.scheduler.admit(free_kv_tokens=free_kv_tokens)

    # Phase 5: Identify decode requests.
    decode = self.scheduler.decode_requests()

    if not prefill and not decode:
        return

    # Phase 6: Execute forward passes.
    try:
        outputs = self.runner.step(prefill, decode)
        for req, output in outputs:
            if req.output_queue is not None:
                req.output_queue.put_nowait(output)
    except Exception as exc:
        for req in prefill + decode:
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

This replaces the Phase 5 `_step_continuous` implementation. The key change: the monolithic `schedule()` call is replaced by the `retire` → `free` → `query budget` → `admit` → `decode` sequence. For the contiguous backend, `free_kv_tokens()` returns `None` and `admit(free_kv_tokens=None)` checks only the compute budget — identical behavior to Phase 5.

### 7. ContinuousRunner changes (`src/infer/engine/continuous_runner.py`)

The runner dispatches cache pool creation based on `kv_cache_backend` and types the pool as `CachePoolProtocol`. No `isinstance` checks in the hot path.

```python
from infer.cache.protocol import CachePoolProtocol

class ContinuousRunner:
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

        model_config = getattr(model, "config", None)
        if model_config is None:
            raise TypeError("model must have a .config attribute")

        # Dispatch cache pool creation based on backend.
        if config.kv_cache_backend == "paged":
            num_gpu_blocks = config.num_gpu_blocks
            if num_gpu_blocks is None:
                num_gpu_blocks = config.max_batch_size * config.max_seq_len // config.block_size
            self.cache_pool: CachePoolProtocol = PagedKVCachePool.from_model_config(
                model_config,
                total_blocks=num_gpu_blocks,
                block_size=config.block_size,
                dtype=self.dtype,
                device=config.device,
            )
        else:
            self.cache_pool = SlottedKVCache.from_model_config(
                model_config,
                max_seq_len=config.max_seq_len,
                max_batch_size=config.max_batch_size,
                dtype=self.dtype,
                device=config.device,
            )

        self._prev_text_lens: dict[str, int] = {}

    def free_slot(self, slot_idx: int) -> None:
        """Release a cache slot and clean up associated resources."""
        self.cache_pool.free_slot(slot_idx)

    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request tracking state."""
        self._prev_text_lens.pop(request_id, None)

    def free_kv_tokens(self) -> int | None:
        """Return available token capacity, or None for contiguous backend."""
        return self.cache_pool.free_token_capacity()
```

**Protocol-driven allocation**: the runner calls `self.cache_pool.allocate_slot(initial_tokens=...)` in `_prefill_one` and `_prefill_batch`:

```python
# In _prefill_one:
slot = self.cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
req.slot_idx = slot
view = self.cache_pool.prefill_view(slot)
# ... forward pass unchanged ...

# In _prefill_batch:
for req in requests:
    slot = self.cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
    req.slot_idx = slot
    slots.append(slot)
```

**Protocol-driven seq_len access** (breaking change): the current `_batched_decode` accesses `self.cache_pool.seq_lens[slot]` directly (a list for slotted, a dict for paged). This must switch to the protocol method `get_seq_len()` — the runner is typed as `CachePoolProtocol` which does not expose a `seq_lens` attribute:

```python
# In _batched_decode — BEFORE (Phase 5):
positions = [self.cache_pool.seq_lens[slot] for slot in slots]
padding_mask[i, : self.cache_pool.seq_lens[slot]] = True

# AFTER (Phase 6):
positions = [self.cache_pool.get_seq_len(slot) for slot in slots]
padding_mask[i, : self.cache_pool.get_seq_len(slot)] = True
```

The rest of `_prefill_one`, `_prefill_batch`, and `_batched_decode` are unchanged in structure — they call `cache_pool.prefill_view()`, `cache_pool.batched_prefill_view()`, and `cache_pool.decode_view()` as before. The returned view objects implement `KVCacheProtocol`, so the model forward code works without modification.

### 8. Config changes (`src/infer/engine/config.py`)

New fields and validation for paged attention:

```python
_VALID_KV_CACHE_BACKENDS = {"contiguous", "paged"}  # Phase 6: add "paged"

@dataclass
class EngineConfig:
    # ... existing fields ...

    # KV cache backend — Phase 6 adds "paged".
    kv_cache_backend: str = "contiguous"

    # Paged backend configuration.
    block_size: int = 16        # tokens per KV cache block (paged backend only)
    num_gpu_blocks: int | None = None  # total blocks; None = auto-compute
```

New validation rules:

```python
if self.kv_cache_backend == "paged":
    if self.batching_mode != "continuous":
        raise ValueError("Paged KV cache requires batching_mode='continuous'")
    if self.block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {self.block_size}")
    if self.num_gpu_blocks is not None and self.num_gpu_blocks < 1:
        raise ValueError(f"num_gpu_blocks must be >= 1, got {self.num_gpu_blocks}")
```

**`num_gpu_blocks` auto-computation**: when `num_gpu_blocks` is `None`, the runner computes `max_batch_size * max_seq_len // block_size`, giving the same total token capacity as the contiguous pool. Users can override this to independently size the block pool.

### 9. Triton paged attention kernel (`src/infer/kernels/paged_attention.py`)

The gather-based decode path (deliverable 4) copies block data into contiguous tensors before SDPA. The Triton kernel fuses the gather and attention computation, reading K/V directly from scattered blocks and computing attention in a single pass. This eliminates the O(batch × seq_len × head_dim) gather copy per layer.

```python
@triton.jit
def _paged_attention_kernel(
    OUT,  # [batch, num_q_heads, head_dim]
    Q,  # [batch, num_q_heads, 1, head_dim]
    K_POOL,  # [total_blocks, num_kv_heads, block_size, head_dim]
    V_POOL,  # [total_blocks, num_kv_heads, block_size, head_dim]
    PAGE_TABLE,  # [batch, max_num_blocks]  int32
    SEQ_LENS,  # [batch]  int32
    stride_out_batch, stride_out_head,
    stride_q_batch, stride_q_head,
    stride_k_block, stride_k_head, stride_k_pos,
    stride_v_block, stride_v_head, stride_v_pos,
    stride_pt_batch,
    GQA_GROUP_SIZE: tl.constexpr,  # num_q_heads // num_kv_heads
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
    BLOCK_HD: tl.constexpr,   # next_power_of_2(HEAD_DIM)
    BLOCK_BS: tl.constexpr,   # next_power_of_2(BLOCK_SIZE)
):
    """Paged attention for one (batch, q_head) pair.

    Grid: ``(batch, num_q_heads)``.

    Uses online softmax to iterate over KV blocks without materialising
    the full attention matrix.  Float32 accumulation for numerical stability.

    GQA is handled by mapping query heads to KV heads via a single
    ``GQA_GROUP_SIZE`` constexpr: ``kv_head = q_head // GQA_GROUP_SIZE``.
    """
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    # GQA: map query head to KV head.
    kv_head_idx = q_head_idx // GQA_GROUP_SIZE

    seq_len = tl.load(SEQ_LENS + batch_idx)

    # Load query vector [HEAD_DIM] → float32.
    # BLOCK_HD pads head_dim to the next power of 2 for efficient Triton loads.
    q_offset = batch_idx * stride_q_batch + q_head_idx * stride_q_head
    dim_offsets = tl.arange(0, BLOCK_HD)
    dim_mask = dim_offsets < HEAD_DIM
    q = tl.load(Q + q_offset + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    # Online softmax running state (explicit Triton scalars, not Python floats).
    m_prev = tl.full([], float("-inf"), dtype=tl.float32)  # running max
    l_prev = tl.full([], 0.0, dtype=tl.float32)            # running exp sum
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)

    pos_offsets = tl.arange(0, BLOCK_BS)  # padded to next_power_of_2(BLOCK_SIZE)

    for block_idx in range(MAX_NUM_BLOCKS):
        start_pos = block_idx * BLOCK_SIZE
        # No break in Triton for loops: use valid_len=0 to mask everything
        # for blocks past the sequence boundary (all scores become -inf,
        # contributing zero to the output via softmax).
        remaining = seq_len - start_pos
        valid_len = tl.maximum(0, tl.minimum(BLOCK_SIZE, remaining))

        # Physical block ID from page table.
        block_id = tl.load(
            PAGE_TABLE + batch_idx * stride_pt_batch + block_idx
        ).to(tl.int64)

        # --- Load K block [BLOCK_BS, BLOCK_HD] → float32 ---
        k_base = block_id * stride_k_block + kv_head_idx * stride_k_head
        k_ptrs = K_POOL + k_base + pos_offsets[:, None] * stride_k_pos + dim_offsets[None, :]
        k_mask = (pos_offsets[:, None] < valid_len) & dim_mask[None, :]
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Attention scores: q @ k^T * scale → [BLOCK_BS].
        scores = tl.sum(q[None, :] * k, axis=1) * SCALE
        scores = tl.where(pos_offsets < valid_len, scores, float("-inf"))

        # Online softmax update.
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        exp_scores = tl.exp(scores - m_new)
        correction = tl.exp(m_prev - m_new)
        l_new = correction * l_prev + tl.sum(exp_scores, axis=0)

        # --- Load V block [BLOCK_BS, BLOCK_HD] → float32 ---
        v_base = block_id * stride_v_block + kv_head_idx * stride_v_head
        v_ptrs = V_POOL + v_base + pos_offsets[:, None] * stride_v_pos + dim_offsets[None, :]
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Accumulate weighted V.
        acc = correction * acc + tl.sum(exp_scores[:, None] * v, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Final normalisation (guard against l_prev==0 from empty sequences).
    safe_l = tl.maximum(l_prev, 1e-6)
    out = tl.where(dim_mask, acc / safe_l, 0.0)

    # Store output.
    out_offset = batch_idx * stride_out_batch + q_head_idx * stride_out_head
    tl.store(OUT + out_offset + dim_offsets, out, mask=dim_mask)
```

**Key implementation details**:

- **No `break` in Triton loops**: Triton's `for` loop semantics do not reliably support `break` across all backends. Instead, blocks past the sequence boundary get `valid_len=0`, making all their scores `-inf`, which contributes zero to the output via softmax. This is defensive — the kernel is correct regardless of whether the backend executes predicated or truly-exiting loop iterations.

- **`BLOCK_HD` and `BLOCK_BS` padding**: Triton loads are most efficient at power-of-2 sizes. The wrapper computes `BLOCK_HD = next_power_of_2(head_dim)` and `BLOCK_BS = next_power_of_2(block_size)` at launch time. Loads use `dim_mask = dim_offsets < HEAD_DIM` and `pos_offsets < valid_len` to mask padded lanes.

- **`GQA_GROUP_SIZE` constexpr**: a single constexpr `GQA_GROUP_SIZE = num_q_heads // num_kv_heads` handles MHA (group=1), GQA (e.g. group=4), and MQA (group=num_q_heads). The kernel maps query heads to KV heads via `kv_head_idx = q_head_idx // GQA_GROUP_SIZE`.

- **Explicit Triton scalars**: the running softmax state (`m_prev`, `l_prev`) is initialised with `tl.full([], ...)` rather than Python floats, ensuring correct Triton intermediate types.

- **Division-by-zero guard**: `safe_l = tl.maximum(l_prev, 1e-6)` prevents NaN output for empty sequences (seq_len=0).

- **`num_warps` tuning**: the wrapper selects `num_warps=8` for `head_dim >= 128` (wider vectors need more parallelism) and `num_warps=4` otherwise.

**Python wrapper**:

```python
def triton_paged_attention(
    q: Tensor,            # [batch, num_q_heads, 1, head_dim]
    k_pool: Tensor,       # [total_blocks, num_kv_heads, block_size, head_dim]
    v_pool: Tensor,       # [total_blocks, num_kv_heads, block_size, head_dim]
    page_table: Tensor,   # [batch, max_num_blocks], int32
    seq_lens: Tensor,     # [batch], int32
    scale: float,
    max_num_blocks: int,
) -> Tensor:
    batch, num_q_heads, _, head_dim = q.shape
    _, num_kv_heads, block_size, _ = k_pool.shape
    gqa_group_size = num_q_heads // num_kv_heads

    out = q.new_empty(batch, num_q_heads, head_dim)

    BLOCK_HD = triton.next_power_of_2(head_dim)
    BLOCK_BS = triton.next_power_of_2(block_size)
    num_warps = 8 if BLOCK_HD >= 128 else 4

    grid = (batch, num_q_heads)
    _paged_attention_kernel[grid](
        out, q, k_pool, v_pool, page_table, seq_lens,
        out.stride(0), out.stride(1),
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
        page_table.stride(0),
        GQA_GROUP_SIZE=gqa_group_size,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        SCALE=scale,
        MAX_NUM_BLOCKS=max_num_blocks,
        BLOCK_HD=BLOCK_HD,
        BLOCK_BS=BLOCK_BS,
        num_warps=num_warps,
    )

    return out.unsqueeze(2)  # [batch, num_q_heads, 1, head_dim]
```

**Integration**: the Triton kernel is auto-dispatched during decode based on runtime conditions — no config flag or `attention_backend` setting needed. The `PagedDecodeCacheView` provides `write_only()` (writes K/V without gathering), and lazy `page_table_tensor` / `seq_lens_tensor` properties computed once per step via `_ensure_kernel_tensors()`:

```python
class PagedDecodeCacheView:
    # ... existing methods ...

    def write_only(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """Write new tokens to blocks without gathering (for Triton kernel path)."""
        # ... allocates new blocks on layer 0 if needed, writes K/V ...

    def _ensure_kernel_tensors(self) -> None:
        """Lazily compute page table and seq_lens tensors on first access."""
        if self._page_table_tensor is not None:
            return
        # Build [batch, max_blocks] int32 page table and [batch] int32 seq_lens
        # ...

    @property
    def page_table_tensor(self) -> Tensor:
        self._ensure_kernel_tensors()
        return self._page_table_tensor

    @property
    def seq_lens_tensor(self) -> Tensor:
        self._ensure_kernel_tensors()
        return self._seq_lens_tensor
```

**Lazy initialisation**: unlike the design's `build_kernel_tensors()` called explicitly by the runner, the actual implementation uses `_ensure_kernel_tensors()` which is called lazily on first access to `page_table_tensor` or `seq_lens_tensor`. This keeps the runner code simple (no Triton-specific calls) and the tensors are still computed at most once per step since the view is created fresh each step.

**Attention dispatch**: the `Attention` class auto-dispatches to the Triton kernel based on runtime conditions, avoiding any config flag:

```python
# In Attention.forward:
if (
    kv_cache is not None
    and kv_cache.is_paged()
    and seq_len == 1            # decode only (single query token)
    and mask is None            # full attention (not sliding window)
    and hasattr(kv_cache, "write_only")  # decode view, not prefill view
):
    paged_view: Any = kv_cache
    paged_view.write_only(layer_idx, k, v)
    out = triton_paged_attention(
        q, paged_view.pool.k[layer_idx], paged_view.pool.v[layer_idx],
        paged_view.page_table_tensor, paged_view.seq_lens_tensor,
        scale=self.scale,
        max_num_blocks=paged_view.page_table_tensor.shape[1],
    )
else:
    # Standard path: update cache (with gather), then SDPA.
    if kv_cache is not None:
        k, v = kv_cache.update(layer_idx, k, v)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
```

The dispatch uses four runtime checks instead of a config flag:
1. `is_paged()` — protocol method, avoids `isinstance` checks
2. `seq_len == 1` — decode only (prefill uses SDPA directly)
3. `mask is None` — full attention layers (Gemma 3 sliding-window layers fall back to gather+SDPA)
4. `hasattr(kv_cache, "write_only")` — distinguishes `PagedDecodeCacheView` from `PagedPrefillCacheView` (important when a single-token prompt produces `seq_len == 1` during prefill)

This approach eliminates the need for `attention_backend="triton_paged"` config, `self.use_triton_paged` flags on the Attention class, or any runner-level dispatch. The Triton kernel is always used when conditions are met, and the fallback is always safe.

**Scope**: the Triton kernel targets decode only (single query token per sequence) where the gather cost is highest. Prefill uses SDPA regardless (the full K/V is available without gather).

**Gemma 3 sliding window**: the Triton kernel and paged views do not handle per-layer sliding window attention masks. For Gemma 3's alternating full/sliding attention pattern, full-attention layers (mask=None) use the Triton kernel while sliding-window layers (mask provided) fall back to gather+SDPA. Adding sliding-window-aware block eviction (freeing blocks outside the window) is deferred to future work.

### 10. Benchmark updates

Run benchmarks with both contiguous and paged backends to demonstrate the capacity advantage.

**Why the existing `paged_attention` workload was insufficient**: the Phase 5 workload sent 48 requests in bursts of 8, run with `max_batch_size=8`. Both backends could handle this — contiguous pre-allocates 8 slots and requests queue in order. No requests fail due to VRAM, so the paged backend had nothing to rescue.

The real benefit of paged attention is **raising `max_batch_size`** while keeping the same VRAM budget. Contiguous allocation ties `max_batch_size` to worst-case per-slot memory (`max_seq_len` positions each). Paged allocation decouples them — short sequences use fewer blocks, leaving room for more concurrent sequences.

**Updated `paged_attention` workload**: sends **48 moderate-length requests** in a single burst (all at once, no stagger) with moderate prompts (128-384 tokens) and moderate generation (128-256 tokens). This creates a scenario where paged can admit many more concurrent requests than contiguous in the same memory budget.

```python
def _gen_paged_attention(n: int, rng: random.Random) -> list[RequestSpec]:
    """Single burst of moderate-length requests — saturates batch slots."""
    specs: list[RequestSpec] = []
    for _ in range(n):
        target_tokens = rng.randint(128, 384)
        max_tokens = rng.randint(128, 256)
        prompt = make_prompt(target_tokens, seed=rng.randint(0, 2**31))
        specs.append(RequestSpec(prompt=prompt, max_tokens=max_tokens, send_delay_s=0.0))
    return specs
```

With `default_num_requests=48`.

**Benchmark protocol** — same total KV memory budget across backends:

```bash
# Configuration A: Contiguous (Phase 4/5 baseline)
uv run python -m infer.server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --batching-mode continuous \
    --kv-cache-backend contiguous \
    --max-batch-size 8

# Configuration B: Paged (Phase 6) — same ~3.5 GB KV, 3x batch size
uv run python -m infer.server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --batching-mode continuous \
    --kv-cache-backend paged \
    --block-size 16 \
    --max-batch-size 24 \
    --num-gpu-blocks 2048
```

Note: `num_gpu_blocks=2048` with `block_size=16` gives `32,768` token positions — the same total VRAM as contiguous with `max_batch_size=8` and `max_seq_len=4096`.

**Per-model server configs** (tuned to 16 GB VRAM):

| Model | Batch | Blocks | KV Pool VRAM | Notes |
|-------|------:|-------:|-------------:|-------|
| Llama 3.2 3B | 24 | 2048 | ~3.5 GB | Fits comfortably |
| Qwen3-4B | 16 | 1024 | ~4.5 GB | Larger weights; batch=24 hits VRAM cliff |
| Gemma 3 1B | 24 | 2048 | ~3.3 GB | Fits comfortably |

Qwen3-4B required empirical batch size tuning. At batch=24/blocks=2048 (4.5 GB KV + 7.5 GB weights), throughput collapsed to 121 tok/s due to insufficient VRAM headroom for activations. A sweep found batch=16/blocks=1024 as the sweet spot (317 tok/s), with prefix_caching needing blocks=1536 to avoid OOM.

**Actual benchmark results** (paged_attention workload, 48 requests):

| Model | Contiguous (batch=8) | Paged (tuned) | Speedup |
|-------|---------------------:|--------------:|--------:|
| Llama 3.2 3B | 346.1 tok/s | 455.2 tok/s | +32% |
| Qwen3-4B | 271.0 tok/s | 317.4 tok/s | +17% |
| Gemma 3 1B | 289.1 tok/s | 325.2 tok/s | +12% |

All workloads are run and results recorded in `benchmarks/log/SERVING_LOG.md` in a Phase 6 section.

---

## File Layout

New and modified files:

```
src/infer/
├── cache/
│   ├── __init__.py             # MODIFIED: export CachePoolProtocol, PagedKVCachePool, paged views
│   ├── protocol.py             # MODIFIED: add CachePoolProtocol, add is_paged() to KVCacheProtocol
│   ├── simple.py               # MODIFIED: add is_paged() returning False
│   ├── slotted.py              # MODIFIED: add initial_tokens param, get_seq_len(),
│   │                           #           free_token_capacity(), is_paged(); add
│   │                           #           is_paged() to all slotted views
│   └── paged.py                # NEW: BlockAllocator, PagedKVCachePool, paged views
├── engine/
│   ├── __init__.py             # UNCHANGED
│   ├── config.py               # MODIFIED: add block_size, num_gpu_blocks, "paged" backend
│   ├── request.py              # UNCHANGED
│   ├── scheduler.py            # MODIFIED: add retire/admit/decode_requests methods
│   ├── runner.py               # UNCHANGED
│   ├── runner_helpers.py       # UNCHANGED
│   ├── continuous_runner.py    # MODIFIED: use CachePoolProtocol, dispatch pool creation
│   ├── engine.py               # MODIFIED: split step for retire/free/admit ordering
│   ├── sampler.py              # UNCHANGED
│   └── generate.py             # UNCHANGED
├── kernels/
│   ├── __init__.py             # MODIFIED: export triton_paged_attention (when present)
│   └── paged_attention.py      # NEW: Triton paged attention kernel
├── models/
│   ├── llama.py                # UNCHANGED
│   ├── qwen3.py                # UNCHANGED
│   ├── gemma3.py               # UNCHANGED
│   └── common.py               # MODIFIED: Attention auto-dispatches to Triton paged kernel
└── server/
    ├── api.py                  # UNCHANGED
    ├── __main__.py             # MODIFIED: add --kv-cache-backend, --block-size,
    │                           #           --num-gpu-blocks CLI args
    └── __init__.py             # UNCHANGED

benchmarks/
├── bench_serving.py            # MODIFIED: updated paged_attention workload (48 moderate-length burst)
└── log/
    └── SERVING_LOG.md          # MODIFIED: add Phase 6 results section

tests/
├── unit/
│   ├── test_block_allocator.py     # NEW
│   ├── test_paged_cache.py         # NEW: PagedKVCachePool, paged views
│   ├── test_cache_pool_protocol.py # NEW: verify both pools satisfy CachePoolProtocol
│   ├── test_paged_attention_kernel.py  # NEW: Triton kernel correctness (18 tests)
│   ├── test_continuous_scheduler.py    # MODIFIED: test retire/admit/decode_requests
│   ├── test_continuous_runner.py       # MODIFIED: test paged backend path
│   ├── test_engine_config.py   # MODIFIED: test paged backend validation
│   └── test_engine.py          # MODIFIED: test paged engine, mixed error handling
├── integration/
│   ├── test_api.py             # MODIFIED: add paged backend API tests
│   └── test_logits_parity.py   # MODIFIED: verify paged cache doesn't break parity
└── stress/
    └── test_backpressure.py    # MODIFIED: high-concurrency test with paged backend
```

---

## Testing Plan

### Block allocator tests (`tests/unit/test_block_allocator.py`)

**Basic allocation**:
- Allocate 5 blocks from a 10-block pool. Verify `num_free() == 5`.
- Allocate 5 more. Verify `num_free() == 0`.
- Allocate 1 more. Verify `RuntimeError`.

**Deallocation**:
- Allocate 3 blocks. Free them. Verify `num_free()` restored.
- Allocate again. Verify returned block IDs are from the freed set.

**Double-free detection**:
- Allocate 3 blocks. Free them. Free them again. Verify `ValueError`.
- Allocate 1 block. Free a block ID that was never allocated. Verify `ValueError`.

**`can_allocate` check**:
- 10-block pool, allocate 8. `can_allocate(2)` → True. `can_allocate(3)` → False.

**Zero allocation**:
- `allocate(0)` → returns empty list, no state change.

**Empty pool edge case**:
- 0-block pool. `allocate(1)` → RuntimeError. `num_free() == 0`. `can_allocate(0)` → True.

**Owner tracking**:
- Allocate 3 blocks with `owner=42`. Verify `_block_owners` maps all 3 to 42.
- Free 2 blocks. Verify only the freed blocks are removed from `_block_owners`.
- Allocate 2 blocks with `owner=99`. Verify mixed ownership: 1 block owned by 42, 2 by 99.

**Leak detection**:
- Allocate 5 blocks with `owner=1`, 3 blocks with `owner=2`.
- `find_leaked_blocks(active_owners={1})` → `{2: [block_ids]}`.
- `find_leaked_blocks(active_owners={1, 2})` → `{}` (no leaks).
- `find_leaked_blocks(active_owners=set())` → both owners reported as leaked.

**Unreferenced block detection**:
- Allocate 10 blocks. Pass 7 of them as the `referenced` set to `find_unreferenced_blocks()`. Verify the other 3 are returned.
- Pass all 10 as referenced. Verify empty result.
- Pass empty set as referenced. Verify all 10 returned.

**`force_free`**:
- Allocate 5 blocks. `force_free()` them. Verify `num_free()` restored, `_block_owners` cleared for those blocks.
- Verify `force_free()` does not raise on blocks that are already free (no double-free check).

**`allocated_block_ids`**:
- Allocate 5 blocks. Verify `allocated_block_ids()` returns a set matching the allocated block IDs.
- Free 2. Verify `allocated_block_ids()` reflects the change.
- Verify the returned set is a copy (mutating it does not affect allocator state).

### CachePoolProtocol conformance tests (`tests/unit/test_cache_pool_protocol.py`)

**Structural protocol check**:
- Verify `SlottedKVCache` satisfies `CachePoolProtocol` (via `isinstance` with `runtime_checkable` or manual attribute inspection).
- Verify `PagedKVCachePool` satisfies `CachePoolProtocol`.

**Interface parity**:
- For each pool type: `allocate_slot()`, `free_slot()`, `get_seq_len()`, `free_slot_count()`, `free_token_capacity()`, `is_paged()`, `prefill_view()`, `decode_view()`, `batched_prefill_view()` — all callable with correct signatures.
- Verify `SlottedKVCache.is_paged()` returns `False`, `PagedKVCachePool.is_paged()` returns `True`.
- Verify all slotted views have `is_paged()` returning `False`, all paged views return `True`.

**Cross-backend runner smoke test**:
- Create `ContinuousRunner` with `kv_cache_backend="contiguous"`, verify basic prefill/decode.
- Create `ContinuousRunner` with `kv_cache_backend="paged"`, verify same operations.

### Paged KV cache pool tests (`tests/unit/test_paged_cache.py`)

**Sequence allocation**:
- Allocate a sequence with `initial_tokens=50`, `block_size=16`. Verify 4 blocks allocated (ceil(50/16)).
- Verify `page_tables[seq_id]` has 4 entries. Verify `seq_lens[seq_id] == 0`.

**Sequence allocation with zero initial tokens**:
- `allocate_slot(initial_tokens=0)` → seq_id assigned, 0 blocks allocated, empty page table.

**Sequence deallocation**:
- Allocate sequence, then free it. Verify blocks returned to allocator.
- Verify `page_tables` and `seq_lens` entries removed.

**Free nonexistent sequence**:
- `free_slot(999)` → `KeyError`.

**Multiple sequences**:
- Allocate 3 sequences with different prompt lengths. Verify each has the correct number of blocks.
- Free the middle one. Verify only its blocks are freed.

**`free_token_capacity`**:
- 100-block pool, block_size=16. Allocate 30 blocks (across sequences). Verify `free_token_capacity() == 70 * 16 == 1120`.

**`get_seq_len`**:
- Allocate sequence. Verify `get_seq_len(seq_id) == 0`.
- Manually set `seq_lens[seq_id] = 42`. Verify `get_seq_len(seq_id) == 42`.

**Block exhaustion on allocate_slot**:
- Allocate sequences until the pool is exhausted. Verify `allocate_slot` raises `RuntimeError`.
- Verify no partial state is left (no orphaned page table or seq_lens entries).

**Block size 1 edge case**:
- Pool with `block_size=1`. Allocate sequence with `initial_tokens=5`. Verify 5 blocks allocated.
- Prefill and decode. Verify correct block-per-token behavior.

**Seq ID gap after failed allocation**:
- Fill pool to near capacity. Attempt `allocate_slot` that requires more blocks than available.
- Verify `RuntimeError` raised and `_next_seq_id` was still incremented.
- Verify next successful `allocate_slot` gets `_next_seq_id` (not the failed ID).

**`audit_blocks` — no leaks**:
- Allocate 3 sequences normally. `audit_blocks()` → empty list.

**`audit_blocks` — leaked blocks**:
- Allocate blocks directly via `allocator.allocate(5, owner=999)` without updating page tables.
- `audit_blocks()` → returns those 5 block IDs.

**`reclaim_leaked_blocks`**:
- Create leaked blocks as above. `reclaim_leaked_blocks()` → returns 5.
- Verify `allocator.num_free()` increased by 5.
- `audit_blocks()` → empty list (leaks reclaimed).

**`reclaim_leaked_blocks` after error recovery**:
- Simulate: `allocate_slot(initial_tokens=50)` succeeds, then sequence tracking is lost (manually delete `page_tables[seq_id]` and `seq_lens[seq_id]`).
- `audit_blocks()` → reports the orphaned blocks.
- `reclaim_leaked_blocks()` → reclaims them.

#### PagedPrefillCacheView tests

**Single-sequence prefill**:
- Allocate a sequence with 32 tokens (2 blocks of 16). Create prefill view.
- Call `update()` with K/V of shape `[1, heads, 32, dim]`.
- Verify data written to correct block positions:
  - Block 0, positions 0-15.
  - Block 1, positions 0-15.
- Verify `update()` returns input K/V unchanged.
- Call `advance(32)`. Verify `pool.seq_lens[seq_id] == 32`.

**Cross-block write correctness**:
- Allocate 3 blocks (block_size=16). Prefill 40 tokens.
- Verify first 16 tokens in block 0, next 16 in block 1, last 8 in block 2.
- Verify block 2 positions 8-15 are untouched (zeros).

**KVCacheProtocol satisfaction**:
- Verify `PagedPrefillCacheView` satisfies `KVCacheProtocol` (has `seq_len`, `update()`, `advance()`).

#### PagedDecodeCacheView tests

**Single decode step**:
- Prefill seq 0 with 20 tokens (2 blocks). Create decode view for [seq 0].
- Call `update()` with K/V of shape `[1, heads, 1, dim]`.
- Verify new token written at position 20 (block 1, offset 4).
- Verify gathered cache has shape `[1, heads, 21, dim]`.
- Verify gathered cache matches pool data for all 21 positions.

**Multi-sequence batched decode**:
- Prefill seq 0 with 10 tokens, seq 1 with 30 tokens. Create decode view for [seq 0, seq 1].
- Verify `view.seq_len == 30`.
- Call `update()`. Verify:
  - Seq 0: new token at position 10 (block 0, offset 10).
  - Seq 1: new token at position 30 (block 1, offset 14).
  - Gathered cache shape: `[2, heads, 31, dim]`.
  - Seq 0 data at positions 0-10, zeros at 11-30.
  - Seq 1 data at positions 0-30.

**Vectorized gather correctness**:
- Verify that the gather output from the vectorized path matches a naive per-sequence loop implementation for the same input data.

**Gather index caching across layers**:
- Call `update()` with `layer_idx=0`, then `layer_idx=1`. Verify `_gather_block_ids` is not recomputed on layer 1 (same tensor identity).

**Lazy block allocation during decode**:
- Prefill seq 0 with exactly 16 tokens (fills 1 block). Create decode view.
- Call `update()`. Verify a new block is allocated (position 16 → block 1, offset 0).
- Verify `allocator.num_free()` decreased by 1.

**Block allocation failure during decode**:
- Exhaust all blocks. Prefill a sequence that fills its last block completely. Create decode view.
- Call `update()` when a new block is needed. Verify `RuntimeError` raised.

**Decode at block boundaries**:
- Prefill exactly `block_size - 1` tokens. Decode 1 → fills last slot of block 0 (no new block needed).
- Prefill exactly `block_size` tokens. Decode 1 → requires new block 1, offset 0.
- Prefill exactly `2 * block_size` tokens. Decode 1 → requires new block 2, offset 0.
- Verify no off-by-one in block index calculation at each boundary.

**Decode after batched prefill**:
- Batch-prefill seq 0 (5 tokens) and seq 1 (10 tokens). Create decode view for [seq 0, seq 1].
- Verify `view.seq_len == 10`. Decode one step. Verify positions 5 and 10 respectively.

#### PagedBatchedPrefillCacheView tests

**Batched prefill with mixed lengths**:
- Allocate seq 0 (10 tokens, 1 block) and seq 1 (20 tokens, 2 blocks).
- Right-pad to 20 tokens. Create batched prefill view with `prompt_lens=[10, 20]`.
- Call `update()` with K/V of shape `[2, heads, 20, dim]`.
- Verify seq 0: only 10 tokens written to block 0 (not 20).
- Verify seq 1: 20 tokens written across 2 blocks.
- Call `advance(20)`. Verify `seq_lens[seq0] == 10`, `seq_lens[seq1] == 20`.

**Return value**:
- Verify `update()` returns input K/V unchanged (no gather).

### Scheduler tests (`tests/unit/test_continuous_scheduler.py` — extend existing)

**Split retire/admit interface**:
- Add 3 requests. `retire()` → empty (nothing to retire). `admit()` → [A, B, C].
- Mark A as FINISHED. `retire()` → [A]. `admit()` → [] (no new requests).
- Add D. `admit()` → [D]. `decode_requests()` → [B, C].

**Block budget admission (with 20% reserve)**:
- Add 3 requests with prompt lengths [100, 200, 300].
- `admit(free_kv_tokens=250)` → usable budget is `int(250 * 0.8) = 200`. Admits [A] (100 tokens; B needs 200 > remaining 100).
- Next step: `admit(free_kv_tokens=625)` → usable budget is `int(625 * 0.8) = 500`. Admits [B, C] (200 + 300 = 500).

**Block budget zero**:
- `admit(free_kv_tokens=0)` with requests waiting → usable budget is 0, admits none.

**Block budget None (contiguous mode)**:
- `admit(free_kv_tokens=None)` → admits up to `max_batch_size`, ignoring token budget. Same as Phase 5.

**Backward compatibility**:
- `schedule()` produces identical output to Phase 5's `schedule()` (calls `admit(free_kv_tokens=None)` internally).

### ContinuousRunner tests (`tests/unit/test_continuous_runner.py` — extend existing)

Uses mock models. Test both `kv_cache_backend="contiguous"` and `kv_cache_backend="paged"`.

**Paged backend — single prefill**:
- Create runner with `kv_cache_backend="paged"`. Prefill one request.
- Verify blocks allocated (prompt_len / block_size, rounded up).
- Verify `req.slot_idx` set. First token generated. State is DECODE.

**Paged backend — batched prefill**:
- Submit 3 requests with different prompt lengths. Verify all get sequence IDs, blocks, first tokens.
- Verify per-sequence `seq_lens` match actual prompt lengths.

**Paged backend — decode after prefill**:
- Prefill 2 requests. Run decode step.
- Verify position_ids correct (via `get_seq_len`), tokens generated, blocks allocated if needed.

**Paged backend — slot free on retire**:
- Prefill A. Complete A. `free_slot(A.slot_idx)`. `cleanup_request(A.request_id)`.
- Verify blocks returned to allocator. `free_kv_tokens()` increased.

**`free_kv_tokens()` dispatch**:
- Contiguous runner: `free_kv_tokens()` returns `None`.
- Paged runner: `free_kv_tokens()` returns `allocator.num_free() * block_size`.

**Contiguous backward compatibility**:
- All existing Phase 5 runner tests pass unchanged with `kv_cache_backend="contiguous"`.

### Engine step tests

**Retire-free-admit ordering**:
- Paged backend with limited blocks. Request A completes, freeing blocks.
- Request B is waiting and needs those blocks.
- Verify B is admitted in the same step (blocks freed before admission check).

**Generation-time block exhaustion**:
- Paged backend with limited blocks. Admit multiple requests whose prompts fit in the budget but whose combined `max_new_tokens` would exceed total capacity.
- Run decode steps until block exhaustion occurs. Verify the affected request is marked FAILED with an error, not a crash.
- Verify other requests in the batch are also handled (currently fail together; document this as expected Phase 6 behavior).

**Leak reclaim in error recovery**:
- After a forward pass failure in `_step_continuous`, verify `audit_blocks()` returns empty (no blocks leaked by the error handler).

**Contiguous backward compatibility**:
- All existing Phase 5 engine tests pass with `kv_cache_backend="contiguous"`.

### Config validation tests (`tests/unit/test_engine_config.py` — extend existing)

- `kv_cache_backend="paged"` with `batching_mode="static"` → `ValueError`.
- `kv_cache_backend="paged"` with `block_size=0` → `ValueError`.
- `kv_cache_backend="paged"` with `num_gpu_blocks=0` → `ValueError`.
- `kv_cache_backend="paged"` with `batching_mode="continuous"` → valid.
- `kv_cache_backend="contiguous"` ignores `block_size` and `num_gpu_blocks`.

### Model parity tests (`tests/integration/test_logits_parity.py` — extend existing)

**Paged vs contiguous decode parity**:
- Run same prompt through contiguous and paged backends with greedy decode.
- Verify generated tokens are identical.

**Paged prefill parity**:
- Single request: paged prefill logits match contiguous prefill logits.
- Batched requests: paged batched prefill logits match contiguous batched prefill.

### Triton paged attention kernel tests (`tests/unit/test_paged_attention_kernel.py`)

**Correctness vs gather + SDPA**:
- Set up a block pool with known data. Run gather-based attention (SDPA) and Triton kernel.
- Verify outputs match within tolerance (bf16: max abs diff < 1e-2, relative < 1e-3).

**GQA correctness**:
- Test with `num_q_heads != num_kv_heads` (e.g., 32 Q heads, 8 KV heads).
- Verify head mapping: Q head `i` attends to KV head `i // 4`.

**Variable sequence lengths in batch**:
- 4 sequences with lengths [16, 48, 100, 200]. Verify per-sequence output matches individual computation.

**Randomized page mappings**:
- Scatter blocks randomly (non-contiguous page tables). Verify correctness.
- This is the key test from the exit criteria.

**Single-block sequence**:
- Sequence with 1-15 tokens (less than one full block). Verify partial block handling.

**Edge case — exactly block-aligned length**:
- Sequence length exactly `N * block_size`. Verify no off-by-one in block iteration.

### Integration tests (`tests/integration/test_api.py` — extend existing)

**Paged backend end-to-end**:
- Start engine with `kv_cache_backend="paged"`. Send request. Verify correct SSE stream.

**High-concurrency with paged backend**:
- `max_batch_size=16` (higher than Phase 5's 8). Send 16 concurrent short requests.
- Verify all complete correctly. (Would fail with contiguous at batch_size=16 due to memory.)

### Stress tests (`tests/stress/test_backpressure.py` — extend existing)

**Block exhaustion under load**:
- Small block pool (100 blocks). Send many concurrent long-prompt requests.
- Verify admission control prevents block exhaustion. Excess requests wait in queue.
- All requests eventually complete (no starvation, no crashes).

---

## Design Decisions

**Protocol-driven abstraction**: the `CachePoolProtocol` is the central design choice. Rather than having the runner use `isinstance` checks to dispatch between slotted and paged pools, both pool types implement a shared protocol. The runner's type annotation `cache_pool: CachePoolProtocol` means all allocation, deallocation, view creation, and capacity queries go through protocol methods. The Triton kernel dispatch in `Attention.forward` uses `is_paged()` plus runtime feature detection (`hasattr(kv_cache, "write_only")`) rather than `isinstance` checks, keeping `common.py` decoupled from `paged.py` at the import level. There are no `isinstance` checks for cache types anywhere in the codebase.

**Block pool instead of per-request allocation**: all blocks live in a single pre-allocated tensor `[layers, total_blocks, heads, block_size, dim]`. This is the same approach as the Phase 5 slotted pool (single pre-allocated tensor) — the difference is that "slots" are replaced by variable-length sequences of smaller blocks. Pre-allocation avoids GPU memory fragmentation from repeated alloc/free and makes total VRAM usage predictable at engine startup.

**Decoupled `max_batch_size` and `num_gpu_blocks`**: for paged mode, `max_batch_size` is a compute budget (how many sequences can be in-flight), while `num_gpu_blocks` is a memory budget (how many KV positions can be stored). These are independent parameters. The auto-computed default (`max_batch_size × max_seq_len ÷ block_size`) matches contiguous VRAM, but users can override `num_gpu_blocks` to allocate more or less cache memory without changing `max_batch_size`.

**Eager block pre-allocation at prefill**: blocks for the full prompt are allocated before the forward pass, not lazily during `update()`. This fails fast if the pool is exhausted — the engine can handle the failure cleanly (return request to waiting queue) rather than failing mid-layer during a forward pass. The `allocate_slot` method cleans up partial state on failure. Decode-time allocation is lazy (one block at a time when the current block fills) because the allocation is small and predictable.

**Vectorized flat-gather for decode**: the paged decode view pre-computes `[batch, max_len]` index tensors (block IDs and offsets) once per step, then gathers with a single GPU advanced-index operation per layer. This replaces the naive O(batch × layers) Python loop with O(batch) Python on layer 0 plus O(layers) GPU operations. The index tensors are cached across layers within a single `update()` sequence.

**Double-free detection and leak tracking in BlockAllocator**: the allocator tracks free blocks, allocated blocks, and per-block ownership. Freeing an unallocated block raises `ValueError` immediately, catching bugs that would otherwise silently corrupt state. Owner tracking enables `find_leaked_blocks()` which compares block owners against a set of known-active sequences — blocks whose owner is no longer active are leaked. `PagedKVCachePool.audit_blocks()` cross-references the allocator against page tables to find blocks that are allocated but unreachable. `reclaim_leaked_blocks()` recovers them. These are called in error recovery paths and can be exposed via a health check endpoint.

**Split retire/admit used by both backends**: the engine always uses `retire()` → `free` → `admit()` → `decode_requests()`, even for the contiguous backend. This simplifies the engine to a single code path. For contiguous, `free_token_capacity()` returns `None` and `admit(free_kv_tokens=None)` checks only the compute budget — identical to Phase 5. The combined `schedule()` is kept only for backward compatibility in existing tests.

**Admission with 20% generation reserve**: the scheduler's `admit()` checks whether 80% of free token capacity can hold the prompt, reserving 20% as headroom for decode-time block growth. This is a pragmatic middle ground between exact reservation (`prompt_len + max_new_tokens`, which over-provisions because most sequences generate far fewer than `max_new_tokens` tokens) and no reservation at all (which risks generation-time exhaustion). The 20% reserve, combined with lazy block allocation and per-step admission throttling, provides adequate protection without significantly reducing utilization.

**Gather-based decode as primary path**: the paged decode view gathers blocks into contiguous tensors and feeds them to SDPA, exactly like Phase 5's `DecodeCacheView` gathers slot data. The gather cost is comparable to Phase 5 (same total data volume, slightly more scattered access pattern). This approach requires no changes to model code or the Attention class — only the cache views change. The Triton paged attention kernel is an optional optimization that eliminates the gather.

**Block size 16**: the default `block_size=16` follows vLLM's convention. It balances fragmentation (smaller blocks waste less internal padding but incur more metadata overhead and scattered access) against access efficiency (larger blocks are more contiguous in memory). 16 is also a natural Triton block size, aligning with the optional kernel's iteration granularity.

**No preemption or eviction**: when blocks are exhausted during decode, the engine raises an error and fails the affected requests. More sophisticated handling (preempting the longest sequence, swapping to CPU) is deferred to future work. With proper sizing of the block pool and admission control, block exhaustion during decode should be rare.

**Triton kernel is decode-only**: the paged attention kernel targets decode (single query token per sequence), where the gather cost is O(batch × max_seq_len × head_dim) per layer. During prefill, the model already has the full K/V and doesn't need to gather from the cache — SDPA runs directly on the forward pass's K/V tensors.

**Gemma 3 sliding window deferred**: the paged views and Triton kernel do not implement per-layer sliding window block eviction. For Gemma 3's alternating full/sliding attention pattern, the standard gather+SDPA path applies the sliding window mask via SDPA's `attn_mask` parameter, exactly as in Phase 5. Block-level sliding window optimization (freeing blocks that have fallen outside the window) is deferred.

---

## Exit Criteria

1. Paged attention produces correct output for all three benchmark models (verified by greedy-decode parity with contiguous backend).
2. Higher max concurrent sequence capacity than contiguous mode at the same VRAM budget (demonstrated by benchmark at higher `max_batch_size`).
3. Correctness tests pass with randomized page mappings (non-contiguous block assignments produce correct attention output).
4. `CachePoolProtocol` satisfied by both `SlottedKVCache` and `PagedKVCachePool` — runner has no `isinstance` checks in the standard path.
5. Block allocator tests pass: allocation, deallocation, exhaustion, reuse, double-free detection, owner tracking, leak detection.
6. Paged cache view tests pass: prefill scatter-write, decode vectorized gather, batched prefill, cross-block correctness.
7. Scheduler block-budget admission works: requests are rejected when insufficient blocks are available, admitted when blocks are freed.
8. All Phase 1-5 tests pass with `kv_cache_backend="contiguous"` (no regression).
9. Engine retire-free-admit ordering works: blocks freed from completed requests are available for new admissions in the same step.
10. Benchmark results recorded in `SERVING_LOG.md` comparing contiguous vs paged at matched and increased concurrency.
11. `audit_blocks()` returns empty after normal request lifecycle (no leaked blocks).
12. `uv run ruff check .` and `uv run mypy .` pass cleanly.
13. Triton paged attention kernel passes correctness tests against gather + SDPA reference, including GQA, variable-length sequences, and randomized page mappings (18 tests).
