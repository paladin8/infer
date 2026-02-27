"""Paged KV cache: block-allocated cache pool for paged attention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from infer.loader.config import ModelConfig

if TYPE_CHECKING:
    from infer.cache.prefix import PrefixTree, PrefixTreeNode


class BlockAllocator:
    """Manages allocation and deallocation of fixed-size KV cache blocks.

    Each block holds ``block_size`` token positions worth of K/V data.
    Block IDs are integers in ``[0, total_blocks)``.

    Tracks block ownership for leak detection: every allocated block records
    the owner (sequence ID) that requested it. Blocks whose owner is no
    longer active are "leaked" and can be reclaimed.

    Attributes:
        total_blocks: Total number of blocks in the pool.
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

        Raises:
            RuntimeError: If fewer than ``count`` blocks are free.
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
        *,
        prefix_tree: PrefixTree | None = None,
    ) -> None:
        self.k = k
        self.v = v
        self.block_size = block_size
        self.allocator = BlockAllocator(total_blocks)
        self.prefix_tree = prefix_tree

        # Per-sequence state.
        self.page_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}
        self._next_seq_id: int = 0

        # Per-sequence prefix node references for refcount management.
        # Populated by allocate_slot_with_prefix(), consumed by free_slot().
        self._seq_prefix_nodes: dict[int, list[PrefixTreeNode]] = {}

    @staticmethod
    def from_model_config(
        config: ModelConfig,
        total_blocks: int,
        block_size: int = 16,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        use_prefix_caching: bool = False,
    ) -> PagedKVCachePool:
        """Allocate a block pool sized for the given model config.

        Args:
            config: Model configuration (provides layer count, KV head count,
                head dimension).
            total_blocks: Number of blocks in the pool.
            block_size: Tokens per block.
            dtype: Data type for cache tensors.
            device: Device for cache tensors.
            use_prefix_caching: If True, create a PrefixTree for block reuse.
        """
        from infer.cache.prefix import PrefixTree

        shape = (
            config.num_hidden_layers,
            total_blocks,
            config.num_key_value_heads,
            block_size,
            config.computed_head_dim,
        )
        k = torch.zeros(shape, dtype=dtype, device=device)
        v = torch.zeros(shape, dtype=dtype, device=device)
        prefix_tree = PrefixTree(block_size) if use_prefix_caching else None
        return PagedKVCachePool(k, v, total_blocks, block_size, prefix_tree=prefix_tree)

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
                blocks = self.allocator.allocate(blocks_needed, owner=seq_id)
            except RuntimeError:
                # Clean up partial state on failure.
                del self.page_tables[seq_id]
                del self.seq_lens[seq_id]
                raise
            self.page_tables[seq_id] = blocks

        return seq_id

    def allocate_slot_with_prefix(self, token_ids: list[int]) -> tuple[int, int]:
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
        assert self.prefix_tree is not None
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        matched_blocks, matched_nodes, matched_tokens = self.prefix_tree.match(token_ids)
        total_blocks_needed = (len(token_ids) + self.block_size - 1) // self.block_size
        suffix_blocks_needed = total_blocks_needed - len(matched_blocks)

        try:
            if self.allocator.can_allocate(suffix_blocks_needed):
                suffix_blocks = self.allocator.allocate(suffix_blocks_needed, owner=seq_id)
            else:
                deficit = suffix_blocks_needed - self.allocator.num_free()
                evicted = self.prefix_tree.evict(deficit)
                if evicted:
                    self.allocator.free(evicted)
                suffix_blocks = self.allocator.allocate(suffix_blocks_needed, owner=seq_id)
        except RuntimeError:
            # Allocation failed even after eviction. Undo the match refcounts.
            for node in matched_nodes:
                node.ref_count -= 1
            raise

        self.page_tables[seq_id] = matched_blocks + suffix_blocks
        self.seq_lens[seq_id] = 0
        self._seq_prefix_nodes[seq_id] = matched_nodes
        return seq_id, matched_tokens

    def insert_prefix(self, seq_id: int, token_ids: list[int]) -> None:
        """Insert a sequence's completed blocks into the prefix tree.

        Called by the runner after the last prefill chunk completes.
        Newly created tree nodes are tracked in ``_seq_prefix_nodes``
        so that ``free_slot()`` decrements their refcounts.

        Args:
            seq_id: The sequence that just finished prefill.
            token_ids: Full prompt token IDs.
        """
        assert self.prefix_tree is not None
        new_nodes = self.prefix_tree.insert(token_ids, self.page_tables[seq_id])
        self._seq_prefix_nodes.setdefault(seq_id, []).extend(new_nodes)

    def free_slot(self, seq_id: int) -> None:
        """Free a sequence's resources, prefix-aware.

        1. Decrement ref_count on cached prefix nodes (via stored references).
        2. Free non-tree blocks to the allocator.
        3. Clean up per-sequence state.

        Raises ``KeyError`` if ``seq_id`` is not a valid active sequence.
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

    def get_seq_len(self, seq_id: int) -> int:
        """Return the current sequence length for a sequence."""
        return self.seq_lens[seq_id]

    def active_seq_count(self) -> int:
        """Number of currently active sequences."""
        return len(self.page_tables)

    def free_slot_count(self) -> int:
        """Number of free blocks (upper bound on new sequences)."""
        return self.allocator.num_free()

    def free_token_capacity(self) -> int:
        """Number of tokens that can be stored in free + evictable blocks."""
        free = self.allocator.num_free() * self.block_size
        if self.prefix_tree is not None:
            free += self.prefix_tree.evictable_count() * self.block_size
        return free

    def audit_blocks(self) -> list[int]:
        """Return block IDs that are allocated but not in any page table.

        These are "leaked" blocks — allocated but unreachable via any
        active sequence. When the prefix tree is active, tree-managed
        blocks are excluded (they are intentionally cached, not leaked).

        Delegates to ``BlockAllocator.find_unreferenced_blocks()``.
        """
        referenced_blocks: set[int] = set()
        for blocks in self.page_tables.values():
            referenced_blocks.update(blocks)
        # Exclude blocks managed by the prefix tree from leak detection.
        if self.prefix_tree is not None:
            for block_id in self.allocator.allocated_block_ids():
                if self.prefix_tree.contains_block(block_id):
                    referenced_blocks.add(block_id)
        return self.allocator.find_unreferenced_blocks(referenced_blocks)

    def reclaim_leaked_blocks(self) -> int:
        """Free any leaked blocks (allocated but not in any page table).

        Returns the number of blocks reclaimed. Safe to call at any time.

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
        self,
        seq_ids: list[int],
        prompt_lens: list[int],
    ) -> PagedBatchedPrefillCacheView:
        """Return a multi-sequence view for batched prefill."""
        return PagedBatchedPrefillCacheView(self, seq_ids, prompt_lens)

    def batched_chunked_prefill_view(
        self,
        seq_ids: list[int],
        start_positions: list[int],
        chunk_lens: list[int],
    ) -> PagedBatchedChunkedPrefillCacheView:
        """Return a multi-sequence view for batched chunked prefill."""
        return PagedBatchedChunkedPrefillCacheView(self, seq_ids, start_positions, chunk_lens)


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
            self.seq_len,
            self.seq_len + prompt_len,
            device=device,
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


class PagedDecodeCacheView:
    """Multi-sequence paged cache view for batched decode.

    Each sequence's new K/V token is written to its current block. The full
    K/V cache is either gathered for SDPA (``update()``) or read directly
    by the Triton paged attention kernel (``write_only()``).

    Supports mixed-layer dispatch: some layers can use the Triton kernel
    (``write_only()`` + ``triton_paged_attention``) while others use
    gather + SDPA (``update()``).  Block allocation, gather indices, and
    kernel tensors are each computed lazily on first use and reused across
    all layers.
    """

    def __init__(self, pool: PagedKVCachePool, seq_ids: list[int]) -> None:
        self.pool = pool
        self.seq_ids = seq_ids
        self.slot_seq_lens = [pool.seq_lens[sid] for sid in seq_ids]
        self._seq_len = max(self.slot_seq_lens) if seq_ids else 0

        self._blocks_allocated = False

        # Lazy gather indices (for update / SDPA path).
        self._gather_block_ids: Tensor | None = None
        self._gather_offsets: Tensor | None = None

        # Lazy kernel tensors (for write_only / Triton path).
        self._page_table_tensor: Tensor | None = None
        self._seq_lens_tensor: Tensor | None = None

    def is_paged(self) -> bool:
        return True

    @property
    def seq_len(self) -> int:
        """Max of active sequence lengths (used for mask width)."""
        return self._seq_len

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_blocks_allocated(self) -> None:
        """Allocate new blocks for sequences that need them (once per step).

        When prefix caching is active, evicts from the prefix tree if the
        allocator is exhausted.

        NOTE: if allocation succeeds for sequences 0..k but fails at k+1,
        sequences 0..k have new blocks in their page tables that haven't
        been written to yet.  The engine error handler frees all slots on
        failure, so no leak occurs, but the blocks contain stale data.
        """
        if self._blocks_allocated:
            return
        for i, seq_id in enumerate(self.seq_ids):
            pos = self.slot_seq_lens[i]
            block_idx = pos // self.pool.block_size
            if block_idx >= len(self.pool.page_tables[seq_id]):
                if not self.pool.allocator.can_allocate(1) and self.pool.prefix_tree is not None:
                    evicted = self.pool.prefix_tree.evict(1)
                    if evicted:
                        self.pool.allocator.free(evicted)
                new_block = self.pool.allocator.allocate(1, owner=seq_id)
                self.pool.page_tables[seq_id].extend(new_block)
        self._blocks_allocated = True

    def _write_token(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """Write one new decode token per sequence to the pool."""
        for i, seq_id in enumerate(self.seq_ids):
            pos = self.slot_seq_lens[i]
            block_idx = pos // self.pool.block_size
            offset = pos % self.pool.block_size
            block_id = self.pool.page_tables[seq_id][block_idx]
            self.pool.k[layer_idx, block_id, :, offset, :] = k[i, :, 0, :]
            self.pool.v[layer_idx, block_id, :, offset, :] = v[i, :, 0, :]

    def _ensure_gather_indices(self, device: torch.device | str) -> None:
        """Build flat gather indices for SDPA path (lazy, once per step).

        Zero-initialized, so positions beyond a sequence's actual length
        gather from block 0, offset 0.  The attention padding mask ensures
        these values are ignored by SDPA.
        """
        if self._gather_block_ids is not None:
            return
        max_len = self._seq_len + 1
        batch_size = len(self.seq_ids)
        gather_block_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        gather_offsets = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

        for i, seq_id in enumerate(self.seq_ids):
            seq_len_i = self.slot_seq_lens[i] + 1
            positions = torch.arange(seq_len_i, device=device)
            block_indices = positions // self.pool.block_size
            offsets = positions % self.pool.block_size
            blocks = torch.tensor(
                self.pool.page_tables[seq_id],
                dtype=torch.long,
                device=device,
            )
            gather_block_ids[i, :seq_len_i] = blocks[block_indices]
            gather_offsets[i, :seq_len_i] = offsets

        self._gather_block_ids = gather_block_ids
        self._gather_offsets = gather_offsets

    def _ensure_kernel_tensors(self, device: torch.device | str) -> None:
        """Build page-table and seq-lens GPU tensors for the Triton kernel.

        Lazy, once per step.  ``seq_lens`` includes +1 for the token being
        written in the current step.
        """
        if self._page_table_tensor is not None:
            return
        batch_size = len(self.seq_ids)
        max_blocks = max(
            (len(self.pool.page_tables[sid]) for sid in self.seq_ids),
            default=0,
        )
        page_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
        seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

        for i, seq_id in enumerate(self.seq_ids):
            blocks = self.pool.page_tables[seq_id]
            page_table[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
            seq_lens[i] = self.slot_seq_lens[i] + 1

        self._page_table_tensor = page_table
        self._seq_lens_tensor = seq_lens

    # ------------------------------------------------------------------
    # Public API — SDPA path (gather + F.scaled_dot_product_attention)
    # ------------------------------------------------------------------

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write new tokens to blocks, gather full cache for attention.

        k, v shape: ``[batch, num_kv_heads, 1, head_dim]``.

        Returns gathered K/V of shape
        ``[batch, num_kv_heads, max_seq_len + 1, head_dim]``.
        """
        self._ensure_blocks_allocated()
        self._write_token(layer_idx, k, v)
        self._ensure_gather_indices(k.device)

        assert self._gather_block_ids is not None
        cached_k = self.pool.k[
            layer_idx, self._gather_block_ids, :, self._gather_offsets, :
        ]  # [batch, max_len, kv_heads, head_dim]
        cached_v = self.pool.v[layer_idx, self._gather_block_ids, :, self._gather_offsets, :]

        # Permute to [batch, kv_heads, max_len, head_dim] for SDPA.
        return cached_k.permute(0, 2, 1, 3), cached_v.permute(0, 2, 1, 3)

    # ------------------------------------------------------------------
    # Public API — Triton paged attention path
    # ------------------------------------------------------------------

    def write_only(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """Write new tokens to blocks without gathering.

        Used with ``triton_paged_attention`` which reads K/V directly from
        the pool via the page table.  Block allocation and kernel tensor
        construction happen lazily on first call.

        k, v shape: ``[batch, num_kv_heads, 1, head_dim]``.
        """
        self._ensure_blocks_allocated()
        self._write_token(layer_idx, k, v)
        self._ensure_kernel_tensors(k.device)

    @property
    def page_table_tensor(self) -> Tensor:
        """Padded page table ``[batch, max_blocks]``, int32."""
        assert self._page_table_tensor is not None, "Call write_only() first"
        return self._page_table_tensor

    @property
    def seq_lens_tensor(self) -> Tensor:
        """Per-sequence token counts ``[batch]``, int32 (includes +1 for new token)."""
        assert self._seq_lens_tensor is not None, "Call write_only() first"
        return self._seq_lens_tensor

    # ------------------------------------------------------------------

    def advance(self, n: int) -> None:
        """Advance all active sequences by n positions."""
        for i, seq_id in enumerate(self.seq_ids):
            self.pool.seq_lens[seq_id] += n
            self.slot_seq_lens[i] += n
        self._seq_len += n


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
                self.seq_len,
                self.seq_len + actual_len,
                device=device,
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
        """Set per-sequence seq_lens to actual prompt lengths (not padded).

        NOTE: ``n`` is used only for the internal ``self.seq_len`` counter.
        Per-sequence seq_lens are always set to ``prompt_lens[i]``, not
        accumulated.  This is correct for single-call usage (the model calls
        ``advance(padded_len)`` once after all layers).  If advance were
        called multiple times (e.g. chunked prefill), per-sequence tracking
        would need to accumulate instead.
        """
        self.seq_len += n
        for i, seq_id in enumerate(self.seq_ids):
            self.pool.seq_lens[seq_id] = self.prompt_lens[i]


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
        self.kv_lens = [s + c for s, c in zip(start_positions, chunk_lens, strict=True)]
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
            positions = torch.arange(self.start_positions[i], self.kv_lens[i], device=device)
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
        for i, _seq_id in enumerate(self.seq_ids):
            chunk_len = self.chunk_lens[i]
            k_flat = k[i, :, :chunk_len, :].permute(1, 0, 2)  # [chunk_len, heads, dim]
            v_flat = v[i, :, :chunk_len, :].permute(1, 0, 2)
            self.pool.k[layer_idx, self._scatter_block_ids[i], :, self._scatter_offsets[i], :] = (
                k_flat
            )
            self.pool.v[layer_idx, self._scatter_block_ids[i], :, self._scatter_offsets[i], :] = (
                v_flat
            )

        # Batched gather: [batch, max_kv_len, heads, dim].
        assert self._gather_block_ids is not None
        cached_k = self.pool.k[layer_idx, self._gather_block_ids, :, self._gather_offsets, :]
        cached_v = self.pool.v[layer_idx, self._gather_block_ids, :, self._gather_offsets, :]

        # Permute to [batch, heads, max_kv_len, dim] for SDPA.
        return cached_k.permute(0, 2, 1, 3), cached_v.permute(0, 2, 1, 3)

    def advance(self, n: int) -> None:
        """Set per-sequence seq_lens to actual kv_len (not padded).

        The ``n`` parameter is accepted for KVCacheProtocol conformance but
        not used; per-sequence seq_lens are set to the precomputed ``kv_lens[i]``.
        """
        for i, seq_id in enumerate(self.seq_ids):
            self.pool.seq_lens[seq_id] = self.kv_lens[i]
