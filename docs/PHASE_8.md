# Phase 8: Prefix Caching

## Goal

Eliminate redundant prefill computation for shared prompt prefixes by caching KV blocks in a radix tree and reusing them across requests.

In Phase 7, every request prefills its full prompt from scratch, even when many requests share a long common prefix (e.g., the same system prompt). The existing `prefix_caching` benchmark workload (added in Phase 7, defined in `bench_serving.py`) sends 48 requests that each re-prefill a ~1024-token shared system prompt with only 32-128 unique suffix tokens. Without prefix caching, the engine performs ~48x the same KV computation for the shared prefix --- wasting GPU cycles and inflating TTFT for later requests.

Prefix caching stores the KV data for completed prefill blocks in a radix tree keyed by token IDs. When a new request arrives with a prefix that matches cached blocks, the runner skips prefill for the matched portion and starts from the first unmatched token. The matched KV blocks are referenced directly in the new sequence's page table --- no copy needed. This trades a small amount of memory (retaining evictable blocks) for a large TTFT improvement on shared-prefix workloads.

The tradeoff: cached blocks consume pool capacity until evicted. Under memory pressure, an LRU eviction policy reclaims leaf nodes with zero references, cascading up the tree as branches become empty. Deep shared prefixes (many active references) are naturally resistant to eviction.

Benchmark models: `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-4B`, `google/gemma-3-1b-it` (same as Phases 4-7).

---

## Architecture

```text
Engine._step_continuous()
+-----------------------------------------------------------+
|  1. scheduler.retire()            retire finished         |
|  2. free_slot() + cleanup()       free resources          |
|     └── prefix-aware: release cached blocks,              |
|         free non-tree blocks                              |
|  3. free_kv_tokens()              query budget            |
|     └── includes evictable blocks                         |
|  4. scheduler.admit()             admit new requests      |
|  5. scheduler.prefill_requests()  prefill work            |
|  6. scheduler.decode_requests()   active decodes          |
|  7. runner.step(prefill, decode)  execute                 |
|     └── prefix-aware allocation:                          |
|         a. tree.match(token_ids) → matched_blocks         |
|         b. allocate suffix blocks only                    |
|         c. prefill from matched_tokens onward             |
|         d. tree.insert() after last chunk                 |
|  8. push StepOutputs              to client queues        |
+-----------------------------------------------------------+

Without prefix caching (Phase 7):        With prefix caching (Phase 8):
+--------------------------------------+ +--------------------------------------+
| Request A: [sys_prompt | suffix_A]   | | Request A: [sys_prompt | suffix_A]   |
|   prefill all 1024+64 tokens         | |   prefill all 1024+64 tokens         |
|                                      | |   insert 1024-token prefix to tree   |
| Request B: [sys_prompt | suffix_B]   | |                                      |
|   prefill all 1024+96 tokens         | | Request B: [sys_prompt | suffix_B]   |
|                                      | |   match() → 1024 cached tokens       |
| Request C: [sys_prompt | suffix_C]   | |   prefill only 96 suffix tokens      |
|   prefill all 1024+48 tokens         | |                                      |
|                                      | | Request C: [sys_prompt | suffix_C]   |
|                                      | |   match() → 1024 cached tokens       |
|                                      | |   prefill only 48 suffix tokens      |
+--------------------------------------+ +--------------------------------------+
 TTFT(C) ≈ TTFT(A) (no reuse)            TTFT(C) << TTFT(A) (prefix reused)
```

**Key design decisions:**

1. **Radix tree keyed by token IDs.** Each tree level corresponds to one KV cache block (`block_size` tokens). Edges are labeled with `tuple[int, ...]` of token IDs. A path from root to a node represents a token sequence whose KV data is stored in the blocks along that path. This is the standard approach used by SGLang and vLLM v2.

2. **Block-aligned matching only.** The tree only stores and matches complete blocks. A prompt of 1060 tokens with `block_size=16` matches up to 66 complete blocks (1056 tokens); the trailing 4 tokens are always re-prefilled. This simplifies refcounting and eviction --- every tree node owns exactly one block, and block lifetimes are managed entirely by the tree.

3. **Requires paged backend + chunked prefill.** Prefix caching requires `kv_cache_backend="paged"` (blocks are the unit of sharing) AND `use_chunked_prefill=True`. The chunked prefill view (`PagedBatchedChunkedPrefillCacheView`) gathers the full KV context (prefix + suffix) for attention, which is exactly what's needed when prefix blocks come from the cache. The non-chunked prefill views (`PagedPrefillCacheView`, `PagedBatchedPrefillCacheView`) return input K/V directly without gathering cached prefix data, so they cannot attend to a cached prefix. Requiring chunked prefill also simplifies the implementation: only one prefill code path needs prefix awareness.

4. **LRU eviction of leaf nodes.** When the block pool is exhausted during allocation, evict leaf nodes with `ref_count == 0` in LRU order. After evicting a leaf, its parent may become a new leaf with `ref_count == 0` --- cascading eviction continues until enough blocks are freed or no more evictable nodes exist. Deep prefix chains with active references are naturally protected: their `ref_count > 0` prevents eviction until all using sequences complete and their refcounts are decremented via `free_slot()`.

5. **Insert at prefill completion.** After a request's last prefill chunk, the runner inserts all complete blocks from that request's page table into the tree with `ref_count=1`. Blocks that were already matched (returned by `match()`) are skipped --- their ref was already set during matching. This ensures every block in the tree has a correct refcount reflecting the number of active sequences using it.

6. **No writes to shared blocks; full-hit handling.** The runner never overwrites data in matched prefix blocks. Prefill starts at `matched_tokens` and writes only to newly allocated suffix blocks. When all complete blocks are cached (full hit), the runner runs a single-token forward pass with just the last prompt token (`input_ids = [prompt[-1]]`, `position_ids = [len(prompt) - 1]`) to produce logits for sampling. The chunked prefill view gathers all cached KV for attention, so this last-token pass sees the full context. The KV write for this one token redundantly overwrites already-cached data, which is harmless.

7. **Conservative admission control (deviation from OVERALL_DESIGN.md).** The scheduler deducts the full `prompt_len` from the token budget when admitting a request, even when a prefix hit would reduce actual prefill work. This over-reserves but is safe and preserves the clean scheduler/runner separation: the scheduler has no knowledge of the prefix tree, and the runner handles all caching logic internally. Prefix hits appear to the scheduler as "fast prefills" rather than reduced allocations. Note: the original `OVERALL_DESIGN.md` Phase 8 outline listed "Scheduler hook for prefix-hit short-circuit in prefill" as a deliverable. This design deliberately omits that hook to avoid coupling the scheduler to cache internals. The OVERALL_DESIGN.md has been updated to reflect this decision. **Future optimization:** add a read-only `query_match_len(token_ids) -> int` method to PrefixTree (no refcount side effects), and pass a cost function to `scheduler.admit()` so it deducts `prompt_len - matched_tokens` instead of `prompt_len`. This would improve max concurrency on shared-prefix workloads. Deferred to a follow-up since the conservative approach is simpler and safe for initial benchmarking.

8. **`free_slot()` is prefix-aware via stored node references.** During `allocate_slot_with_prefix()`, the pool stores references to the matched `PrefixTreeNode` objects per sequence (`_seq_prefix_nodes`). When freeing a sequence's slot, the pool decrements each stored node's `ref_count` directly --- no tree traversal or token IDs needed. Non-tree blocks (the incomplete trailing block and any decode-phase blocks) are freed normally via the allocator. The tree node's block is only returned to the allocator when the node is evicted. This keeps all prefix state in the pool and out of the runner.

9. **`free_token_capacity()` includes evictable blocks.** The pool reports `allocator.num_free() * block_size + tree.evictable_count() * block_size` so admission control accounts for reclaimable memory. This prevents deadlock where all blocks are held by zero-refcount cached nodes while new requests are rejected.

---

## Deliverables

### 1. PrefixTree (`src/infer/cache/prefix.py`) --- NEW

A radix tree for caching KV block IDs, keyed by token ID tuples.

```python
@dataclass
class PrefixTreeNode:
    """A node in the prefix tree.

    Each node represents one KV cache block. The path from root to this
    node defines the token sequence whose KV data is stored in the
    blocks along that path.
    """

    tokens: tuple[int, ...]  # Edge label: token IDs for this block
    block_id: int            # Physical block ID in the cache pool
    ref_count: int           # Number of active sequences using this block
    last_access_time: int    # Monotonic counter for LRU eviction
    parent: PrefixTreeNode | None
    children: dict[tuple[int, ...], PrefixTreeNode]
```

```python
class PrefixTree:
    """Radix tree for prefix caching.

    Stores KV cache block IDs keyed by token ID sequences. Each tree
    level corresponds to one block (``block_size`` tokens). Supports
    matching, insertion, and LRU eviction. Refcount management is
    handled externally by the pool via stored node references.

    Args:
        block_size: Number of tokens per block (must match pool's block_size).
    """

    def __init__(self, block_size: int) -> None: ...

    def match(
        self, token_ids: list[int],
    ) -> tuple[list[int], list[PrefixTreeNode], int]:
        """Find the longest cached prefix matching the given token sequence.

        Walks the tree from root, matching ``block_size``-aligned chunks
        of token IDs. For each matched node, increments ``ref_count``
        and updates ``last_access_time``.

        Matches up to ``len(token_ids) // block_size`` complete blocks.
        A full match (all complete blocks cached) is allowed --- the
        runner handles this case with a single last-token forward pass
        (see design decision #6).

        Args:
            token_ids: Full prompt token IDs.

        Returns:
            Tuple of (matched_block_ids, matched_nodes, matched_tokens):
            - matched_block_ids: Physical block IDs for the cached prefix,
              in path order. These go directly into the sequence's page table.
            - matched_nodes: ``PrefixTreeNode`` references for the matched
              path. Stored by the pool per-sequence for refcount decrement
              at free time.
            - matched_tokens: Number of tokens covered by matched blocks
              (always a multiple of block_size).
        """
        ...

    def insert(
        self,
        token_ids: list[int],
        block_ids: list[int],
    ) -> None:
        """Insert completed blocks into the tree.

        Called after the last prefill chunk completes. Walks the token
        sequence in block_size-aligned chunks. For each chunk:
        - If a node already exists (from a prior ``match()``), skip it.
        - Otherwise, create a new node with ``ref_count=1``.

        Only inserts complete blocks (``len(token_ids) // block_size``
        blocks). The trailing partial block (if any) is not inserted.

        Args:
            token_ids: Full prompt token IDs.
            block_ids: Physical block IDs from the sequence's page table,
                in logical order. ``block_ids[i]`` corresponds to tokens
                ``[i * block_size, (i + 1) * block_size)``.
        """
        ...

    def evict(self, num_blocks: int) -> list[int]:
        """Evict up to ``num_blocks`` blocks using LRU policy.

        Evicts leaf nodes with ``ref_count == 0``, ordered by
        ``last_access_time`` (oldest first). After evicting a leaf, if
        its parent becomes a childless leaf with ``ref_count == 0``, the
        parent becomes eligible for eviction in the same pass (cascading).

        Returns:
            List of freed block IDs (to return to the allocator).
        """
        ...

    def contains_block(self, block_id: int) -> bool:
        """Check whether a block ID is currently in the tree.

        Used by ``free_slot()`` to distinguish tree-managed blocks
        (kept alive) from non-tree blocks (freed to allocator).
        Also used by ``audit_blocks()`` to exclude cached blocks
        from leak detection.
        """
        ...

    def evictable_count(self) -> int:
        """Number of blocks with ``ref_count == 0`` (eligible for eviction).

        Used by ``free_token_capacity()`` to report reclaimable memory.
        """
        ...

    def cached_block_count(self) -> int:
        """Total number of blocks currently in the tree."""
        ...
```

**Internal state:**

- `_root`: Sentinel root node (no block, no tokens).
- `_block_to_node: dict[int, PrefixTreeNode]`: Reverse lookup from block ID to node, for `contains_block()` and eviction.
- `_clock: int`: Monotonic counter incremented on each `match()` and `insert()`.

**Edge cases:**

- Empty prompt: `match()` returns `([], [], 0)`.
- Prompt shorter than `block_size`: `match()` returns `([], [], 0)`, `insert()` inserts nothing.
- Full match (all complete blocks cached): `match()` returns all blocks. Runner handles with single last-token forward pass.
- Prompt not block-aligned (e.g., 1060 tokens, block_size=16): `match()` returns up to 66 blocks (1056 tokens), trailing 4 tokens are always re-prefilled.
- Eviction when all nodes have `ref_count > 0`: `evict()` returns empty list, allocation fails normally.

**Tests:**

- Match on empty tree returns `([], [], 0)`.
- Insert then match returns correct blocks, nodes, and token count.
- Partial match: tokens diverge mid-path, only matching prefix returned.
- Full match: prompt fully cached (block-aligned), match returns all blocks.
- Non-block-aligned full match: 1060 tokens → matches 66 blocks (1056 tokens).
- Refcount lifecycle: match increments, direct `node.ref_count -= 1` decrements, evict only targets zero-refcount leaves.
- LRU ordering: oldest leaves evicted first.
- Cascading eviction: evicting a leaf exposes parent, parent evicted in same call.
- `contains_block()` consistency with insert/evict.
- `evictable_count()` tracks zero-refcount nodes correctly.

### 2. PagedKVCachePool changes (`src/infer/cache/paged.py`) --- MODIFIED

Add prefix tree integration to the paged cache pool.

```python
class PagedKVCachePool:
    def __init__(
        self,
        k: Tensor,
        v: Tensor,
        total_blocks: int,
        block_size: int,
        *,
        prefix_tree: PrefixTree | None = None,  # NEW
    ) -> None:
        ...
        self.prefix_tree = prefix_tree  # NEW

        # Per-sequence prefix node references for refcount management.
        # Populated by allocate_slot_with_prefix(), consumed by free_slot().
        self._seq_prefix_nodes: dict[int, list[PrefixTreeNode]] = {}  # NEW
```

**New method: `allocate_slot_with_prefix()`**

```python
def allocate_slot_with_prefix(
    self, token_ids: list[int],
) -> tuple[int, int]:
    """Allocate a slot with prefix-aware block reuse.

    1. Query the prefix tree for cached blocks matching token_ids.
    2. Place matched blocks directly in the new sequence's page table.
    3. Allocate fresh blocks for the remaining suffix.
    4. If the allocator is exhausted, evict from the tree and retry.

    Args:
        token_ids: Full prompt token IDs.

    Returns:
        Tuple of (seq_id, matched_tokens):
        - seq_id: Allocated sequence ID.
        - matched_tokens: Number of tokens covered by cached blocks
          (the runner starts prefill from this offset).

    Raises:
        RuntimeError: If not enough blocks available even after eviction.
    """
    ...
```

The allocation algorithm:

```text
1. seq_id = self._next_seq_id; self._next_seq_id += 1
2. matched_blocks, matched_nodes, matched_tokens = tree.match(token_ids)
3. total_blocks_needed = ceil(len(token_ids) / block_size)
4. suffix_blocks_needed = total_blocks_needed - len(matched_blocks)
5. if allocator.can_allocate(suffix_blocks_needed):
       suffix_blocks = allocator.allocate(suffix_blocks_needed, owner=seq_id)
   else:
       deficit = suffix_blocks_needed - allocator.num_free()
       evicted = tree.evict(deficit)
       allocator.free(evicted)
       suffix_blocks = allocator.allocate(suffix_blocks_needed, owner=seq_id)
6. page_tables[seq_id] = matched_blocks + suffix_blocks
7. seq_lens[seq_id] = 0
8. _seq_prefix_nodes[seq_id] = matched_nodes  # store for free_slot()
9. return seq_id, matched_tokens
```

Note on block ownership: matched blocks remain allocated in the `BlockAllocator` (they were allocated by the original sequence that produced them). They are not re-allocated or re-owned for the new sequence. The allocator's `_block_owners` still maps them to the original owner. This is acceptable because prefix-cached blocks have a different lifecycle: they are managed by the tree (refcount-based), not by the allocator's ownership tracking. The `audit_blocks()` method must be updated to exclude tree-managed blocks from its leak detection (see below).

**Modified `free_slot()`:**

```python
def free_slot(self, seq_id: int) -> None:
    """Free a sequence's resources, prefix-aware.

    1. Decrement ref_count on cached prefix nodes (via stored references).
    2. Free non-tree blocks to the allocator.
    3. Clean up per-sequence state.
    """
    # Decrement refcounts for prefix nodes.
    prefix_nodes = self._seq_prefix_nodes.pop(seq_id, [])
    for node in prefix_nodes:
        node.ref_count -= 1

    # Free blocks: skip tree-managed ones, free the rest.
    blocks = self.page_tables.pop(seq_id)
    if self.prefix_tree is not None:
        non_tree_blocks = [b for b in blocks if not self.prefix_tree.contains_block(b)]
        if non_tree_blocks:
            self.allocator.free(non_tree_blocks)
    else:
        if blocks:
            self.allocator.free(blocks)
    del self.seq_lens[seq_id]
```

**Modified `free_token_capacity()`:**

```python
def free_token_capacity(self) -> int:
    """Include evictable cached blocks in available capacity."""
    free = self.allocator.num_free() * self.block_size
    if self.prefix_tree is not None:
        free += self.prefix_tree.evictable_count() * self.block_size
    return free
```

**Modified `from_model_config()`:**

```python
@staticmethod
def from_model_config(
    config: ModelConfig,
    total_blocks: int,
    block_size: int = 16,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    use_prefix_caching: bool = False,  # NEW
) -> PagedKVCachePool:
    ...
    prefix_tree = PrefixTree(block_size) if use_prefix_caching else None
    return PagedKVCachePool(k, v, total_blocks, block_size, prefix_tree=prefix_tree)
```

**Modified `audit_blocks()`:**

When the prefix tree is active, blocks held by the tree are allocated but may not appear in any active page table (e.g., after a sequence that populated the cache is freed). These are not leaked --- they are intentionally cached. `audit_blocks()` must exclude tree-managed blocks:

```python
def audit_blocks(self) -> list[int]:
    referenced_blocks: set[int] = set()
    for blocks in self.page_tables.values():
        referenced_blocks.update(blocks)
    # Exclude blocks managed by the prefix tree from leak detection.
    if self.prefix_tree is not None:
        for block_id in self.allocator.allocated_block_ids():
            if self.prefix_tree.contains_block(block_id):
                referenced_blocks.add(block_id)
    return self.allocator.find_unreferenced_blocks(referenced_blocks)
```

**Tests:**

- `allocate_slot_with_prefix()` with empty tree (no match, allocates all blocks).
- `allocate_slot_with_prefix()` after inserting a prefix (matched blocks reused, only suffix allocated).
- `free_slot()` with prefix tree: tree blocks retained, non-tree blocks freed.
- `free_slot()` without prefix tree: all blocks freed (existing behavior preserved).
- `free_token_capacity()` includes evictable blocks.
- Eviction during allocation: pool exhausted, tree blocks evicted to make room.
- `audit_blocks()` excludes tree-managed blocks from leak reports.
- `audit_blocks()` returns empty after normal request lifecycle with prefix caching enabled.

### 3. EngineConfig extensions (`src/infer/engine/config.py`) --- MODIFIED

Add `use_prefix_caching` field with validation.

```python
@dataclass
class EngineConfig:
    ...
    # Prefix caching.
    use_prefix_caching: bool = False
```

Validation rules (in `validate()`):

```python
if self.use_prefix_caching:
    if self.kv_cache_backend != "paged":
        raise ValueError("Prefix caching requires kv_cache_backend='paged'")
    if not self.use_chunked_prefill:
        raise ValueError("Prefix caching requires use_chunked_prefill=True")
```

`OVERALL_DESIGN.md` section 8 has been updated to require both `kv_cache_backend="paged"` and `use_chunked_prefill=True` for prefix caching (see design decision #3 for rationale).

**Tests:** Config validation (prefix + contiguous -> error, prefix + no chunked -> error, prefix + paged + chunked -> ok).

### 4. ContinuousRunner changes (`src/infer/engine/continuous_runner.py`) --- MODIFIED

Integrate prefix-aware allocation, full-hit handling, and insertion into the chunked prefill path.

**Pool construction:**

```python
# In __init__, when creating the paged pool:
if config.kv_cache_backend == "paged":
    self.cache_pool = PagedKVCachePool.from_model_config(
        model_config,
        total_blocks=num_gpu_blocks,
        block_size=config.block_size,
        dtype=self.dtype,
        device=config.device,
        use_prefix_caching=config.use_prefix_caching,  # NEW
    )
```

**Prefix-aware allocation in `_prefill_chunks_batched()`:**

```python
# When allocating for first chunks (prefill_progress == 0):
for req in requests:
    if req.prefill_progress == 0:
        if self.config.use_prefix_caching:
            assert isinstance(self.cache_pool, PagedKVCachePool)
            slot, matched = self.cache_pool.allocate_slot_with_prefix(
                req.prompt_token_ids
            )
            req.slot_idx = slot
            req.prefill_progress = matched  # Skip matched prefix
        else:
            slot = self.cache_pool.allocate_slot(
                initial_tokens=len(req.prompt_token_ids)
            )
            req.slot_idx = slot
        req.state = RequestState.PREFILL
```

The `prefill_progress` is set to `matched_tokens`, so the next chunk starts after the cached prefix. The rest of the chunked prefill logic (chunk bounds, input_ids, position_ids, cache view) works unchanged because it reads `req.prefill_progress` to determine the start position.

**Full-hit handling in `_prefill_chunks_batched()`:**

When `prefill_progress == len(prompt_token_ids)` (all complete blocks were cached), the normal chunked prefill path would compute `chunk_len = 0`, which is invalid. Instead, the runner handles this as a special case --- a single-token forward pass with the last prompt token:

```python
# Separate full-hit requests from normal chunked prefill.
full_hit_requests = [
    req for req in requests
    if req.prefill_progress == len(req.prompt_token_ids)
]
chunk_requests = [
    req for req in requests
    if req.prefill_progress < len(req.prompt_token_ids)
]

# Handle full-hit requests: single last-token forward pass.
for req in full_hit_requests:
    slot = req.slot_idx
    prompt_len = len(req.prompt_token_ids)

    # input_ids: just the last prompt token.
    input_ids = torch.tensor(
        [[req.prompt_token_ids[-1]]], dtype=torch.long, device=device,
    )
    # position_ids: last prompt position.
    position_ids = torch.tensor(
        [[prompt_len - 1]], dtype=torch.long, device=device,
    )
    # Cache view: start_pos = prompt_len - 1, chunk_len = 1.
    # Gathers all cached KV from [0, prompt_len) for attention.
    view = self.cache_pool.batched_chunked_prefill_view(
        [slot], [prompt_len - 1], [1],
    )
    # Padding mask: all positions valid.
    padding_mask = torch.ones(1, prompt_len, dtype=torch.bool, device=device)

    logits = self.model(
        input_ids, kv_cache=view,
        padding_mask=padding_mask, position_ids=position_ids,
    )
    # Sample from logits, transition to DECODE (same as normal last-chunk path).
```

This avoids re-prefilling an entire chunk --- only one token's compute is spent. The chunked prefill view gathers the full cached KV context for the attention computation, so the model sees all prefix tokens.

**Block insertion after last chunk:**

```python
# After the last chunk completes (is_last == True), or after full-hit handling:
if self.config.use_prefix_caching:
    assert isinstance(self.cache_pool, PagedKVCachePool)
    tree = self.cache_pool.prefix_tree
    assert tree is not None
    tree.insert(
        token_ids=req.prompt_token_ids,
        block_ids=self.cache_pool.page_tables[req.slot_idx],
    )
```

**`free_slot()` is unchanged:**

```python
def free_slot(self, slot_idx: int) -> None:
    """Release a cache slot. Pool handles prefix refcounts internally."""
    self.cache_pool.free_slot(slot_idx)
```

All prefix state (node references, refcount decrement) is managed by the pool's `free_slot()` via `_seq_prefix_nodes`. The runner has no prefix-specific logic in `free_slot()`.

**Error recovery:** If a request fails during prefill (e.g., OOM on suffix block allocation after `allocate_slot_with_prefix()` succeeded), the pool's `free_slot()` decrements refcounts for any matched prefix nodes via `_seq_prefix_nodes.pop(seq_id, [])`. The empty-list default handles the case where allocation failed before nodes were stored (e.g., `match()` succeeded but suffix allocation raised).

**Tests:**

- Runner creates prefix tree when `use_prefix_caching=True`.
- Runner does NOT create prefix tree when `use_prefix_caching=False`.
- Slot allocation with prefix hit sets `prefill_progress` correctly.
- Full-hit: `prefill_progress == prompt_len` triggers single-token path, produces correct logits.
- Block insertion after last chunk populates tree.
- `free_slot()` decrements refcounts for tree blocks (via pool).
- Error recovery: failed prefill still decrements prefix refcounts correctly.

### 5. Server CLI (`src/infer/server/__main__.py`) --- MODIFIED

Add `--prefix-caching` flag.

```python
parser.add_argument(
    "--prefix-caching",
    action="store_true",
    default=False,
    help="enable prefix caching (requires --kv-cache-backend paged and --chunked-prefill)",
)
```

Passed to `EngineConfig`:

```python
config = EngineConfig(
    ...
    use_prefix_caching=args.prefix_caching,
)
```

### 6. Cache `__init__` (`src/infer/cache/__init__.py`) --- MODIFIED

Export the new types:

```python
from infer.cache.prefix import PrefixTree, PrefixTreeNode

__all__ = [
    ...
    "PrefixTree",
    "PrefixTreeNode",
]
```

---

## Files NOT changed

- **`src/infer/cache/protocol.py`** --- `CachePoolProtocol` is unchanged. `allocate_slot_with_prefix()` is a paged-specific method, not part of the protocol. The runner uses `isinstance(self.cache_pool, PagedKVCachePool)` only in the `use_prefix_caching=True` branch, which already requires paged backend.
- **`src/infer/engine/scheduler.py`** --- Scheduler unchanged. It has no knowledge of prefix caching. Admission control uses `free_kv_tokens` (which now includes evictable blocks) and deducts full `prompt_len`.
- **`src/infer/engine/request.py`** --- Uses the existing `prefill_progress` field (added in Phase 7). No new fields needed.
- **`src/infer/engine/engine.py`** --- `free_slot()` changes are internal to the pool and runner. Engine step order unchanged.
- **`src/infer/cache/slotted.py`** --- Contiguous backend unchanged. Prefix caching requires paged backend.
- **Model files** (`llama.py`, `qwen3.py`, `gemma3.py`) --- No changes. The chunked prefill attention mask and cache view handle prefix caching transparently.

---

## Invariants

1. **Block ownership.** Every block in the pool is in exactly one of: (a) the allocator free set, (b) a sequence's page table only (not yet inserted into tree), (c) the prefix tree only (cached, no active sequence using it, `ref_count == 0`), or (d) both a sequence's page table AND the prefix tree (`ref_count > 0`). The allocator considers all non-free blocks as "allocated" regardless of whether they are tree-managed. The `audit_blocks()` method excludes tree-managed blocks from leak detection.

2. **Refcount correctness.** `ref_count` equals the number of active sequences whose page table includes that block. `match()` increments, `free_slot()` decrements (via stored node references). A node with `ref_count > 0` is never evicted.

3. **No stale KV data.** The runner never writes to matched prefix blocks. Prefill starts at `matched_tokens` and writes only to suffix blocks. This ensures cached KV data is always consistent with the token sequence that produced it.

4. **At least one forward pass token.** The model always gets at least one token to produce logits. For partial matches, prefill starts at `matched_tokens` with at least one remaining token. For full matches (all complete blocks cached), the runner runs a single-token forward pass with the last prompt token.

5. **Eviction safety.** Only leaf nodes with `ref_count == 0` are evictable. Eviction removes the node from the tree and returns its block to the allocator. Cascading eviction may follow.

6. **Backward compatibility.** When `use_prefix_caching=False` (default), `prefix_tree` is `None`, and all pool/runner behavior is identical to Phase 7.

---

## Costs and Trade-offs

**Memory overhead of the radix tree.** Each `PrefixTreeNode` holds a tuple of token IDs (`block_size` ints), a block ID, refcount, timestamp, parent pointer, and children dict. For `block_size=16`, this is roughly ~200 bytes per node. A tree caching 1000 blocks uses ~200 KB of Python heap --- negligible compared to the GPU memory for the KV blocks themselves.

**Per-match/insert CPU cost.** `match()` walks one tree level per block in the prefix. For a 1024-token prefix with `block_size=16`, that is 64 dict lookups (tuple keys). `insert()` is similar. Both are O(prompt_len / block_size) with small constants. This is negligible relative to GPU forward pass time.

**Reduced effective pool capacity.** Cached blocks with `ref_count == 0` consume pool memory that could serve new sequences. The `free_token_capacity()` change (including evictable blocks) and on-demand eviction in `allocate_slot_with_prefix()` mitigate this: the pool can always reclaim cached blocks when needed. The worst case is eviction thrashing when the working set of unique prefixes exceeds pool capacity --- in this case, prefix caching degrades to the no-caching baseline (no correctness issue, just no TTFT benefit).

**Interaction with existing features:**

- **Chunked prefill.** Required. Prefix caching sets `prefill_progress` to skip cached tokens, and the chunked prefill path handles the rest. No interaction issues.
- **Paged attention.** Required. Prefix blocks are standard paged blocks --- the Triton paged attention kernel reads them via page tables like any other block.
- **Contiguous backend.** Not supported (prefix caching requires block-granular sharing). Config validation enforces this.
- **Static batching.** Not supported (requires continuous batching for chunked prefill). Config validation enforces this transitively.

---

## Benchmark Plan

Run the `prefix_caching` workload (already defined in `bench_serving.py`: 48 requests, ~1024-token shared system prompt, 32-128 unique suffix tokens, uniform 8 RPS arrivals) on all three benchmark models.

**Configurations to compare:**

| Config | Backend | Chunked | Prefix | Expected outcome |
|--------|---------|---------|--------|------------------|
| Baseline (Phase 7) | paged | on | off | Full re-prefill for every request |
| +Prefix caching | paged | on | on | TTFT improvement for requests 2-48 |

**Metrics to report:**

- TTFT P50 / P99 (primary metric --- expect large improvement)
- ITL P50 / P99 (should be unchanged or slightly improved)
- Throughput (tokens/sec --- may improve due to less prefill compute)
- Request latency P50 / P99

**Additional workloads for regression check:**

- Run `continuous_batching`, `paged_attention`, and `chunked_prefill` workloads with `use_prefix_caching=True` to verify no regression on workloads without shared prefixes.

---

## Testing Plan

### Unit tests (`tests/unit/test_prefix_tree.py`)

- **Empty tree:** `match()` returns `([], [], 0)`.
- **Insert and match:** Insert a 4-block sequence, match returns all 4 blocks.
- **Partial match:** Insert `[A, B, C, D]`, query `[A, B, X, Y]` → matches `[A, B]`.
- **Full match:** Insert 4 blocks for a 64-token (block-aligned) prompt, query same tokens → returns all 4 blocks and 64 matched tokens.
- **Refcount lifecycle:** `match()` sets `ref_count=1`, second `match()` sets `ref_count=2`, direct `node.ref_count -= 1` decrements, `evictable_count()` tracks correctly.
- **LRU eviction:** Insert two disjoint prefixes, access one, evict 1 → evicts the older one.
- **Cascading eviction:** Insert `[A, B]`, release all refs, evict `B` → `A` becomes leaf → evict `A` in same call.
- **`contains_block()`:** Returns `True` for inserted blocks, `False` after eviction.
- **`cached_block_count()`:** Tracks total nodes correctly across insert/evict.
- **Short prompt:** Prompt shorter than `block_size` → `match()` returns `([], [], 0)`, `insert()` is no-op.

### Integration tests (`tests/unit/test_paged_prefix.py`)

- **Prefix-aware allocation:** Allocate slot, insert prefix, allocate second slot → second slot reuses cached blocks.
- **Prefix-aware free:** Free slot with cached prefix blocks → tree refcounts decremented, non-tree blocks freed.
- **Eviction under pressure:** Fill the pool, then allocate → evicts cached blocks to make room.
- **`free_token_capacity()` accuracy:** Verify capacity includes evictable blocks.
- **`audit_blocks()` with prefix tree:** Tree-held blocks are correctly accounted for.

### End-to-end tests (`tests/integration/`)

- **Correctness parity:** Same prompt, same seed → identical output with and without prefix caching (greedy decode).
- **Logits parity:** Last prompt position logits match between cached and uncached prefill.
- **Multi-request shared prefix:** Two requests with shared prefix produce correct independent outputs.
- **Eviction correctness:** Under memory pressure, eviction occurs and subsequent requests still produce correct output.

### Benchmark

- **TTFT improvement:** Run `prefix_caching` workload with and without `--prefix-caching`. Expect significant TTFT improvement for requests after the first (which populates the cache).
- **No regression when disabled:** Run all existing workloads with `use_prefix_caching=False` and verify no performance or correctness regression.

---

## Exit Criteria

1. **TTFT improvement** on the `prefix_caching` workload (48 requests, ~1024-token shared prefix, 32-128 unique suffix tokens). Expect TTFT reduction proportional to prefix length for all requests after the first.
2. **Correctness:** Identical output with and without prefix caching under greedy decode (same seed).
3. **Refcount + eviction tests pass:** All unit tests for PrefixTree and integration tests for prefix-aware allocation/free.
4. **No regression when disabled:** All Phase 1-7 tests pass with `use_prefix_caching=False`. Benchmark results unchanged.
5. **Config validation:** `use_prefix_caching=True` with wrong backend or without chunked prefill → `ValueError`.

---

## References

- [SGLang: Efficient Execution of Structured Language Model Programs (radix tree prefix caching)](https://arxiv.org/abs/2312.07104)
- [vLLM: Easy, Fast, and Cheap LLM Serving (PagedAttention + prefix caching)](https://arxiv.org/abs/2309.06180)
- [Efficiently Programming Large Language Models using SGLang (RadixAttention)](https://lmsys.org/blog/2024-01-17-sglang/)
