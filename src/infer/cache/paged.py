"""Paged KV cache: block-allocated cache pool for paged attention."""

from __future__ import annotations

import torch
from torch import Tensor

from infer.loader.config import ModelConfig


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
                blocks = self.allocator.allocate(blocks_needed, owner=seq_id)
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
        """Number of tokens that can be stored in currently free blocks."""
        return self.allocator.num_free() * self.block_size

    def audit_blocks(self) -> list[int]:
        """Return block IDs that are allocated but not in any page table.

        These are "leaked" blocks — allocated but unreachable via any
        active sequence.

        Delegates to ``BlockAllocator.find_unreferenced_blocks()``.
        """
        referenced_blocks: set[int] = set()
        for blocks in self.page_tables.values():
            referenced_blocks.update(blocks)
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

        Returns gathered K/V of shape
        ``[batch, num_kv_heads, max_seq_len + 1, head_dim]``.
        """
        device = k.device

        if layer_idx == 0:
            # Allocate new blocks if needed (once per step).
            # NOTE: if allocation succeeds for sequences 0..k but fails at k+1,
            # sequences 0..k have new blocks in their page tables that haven't
            # been written to yet. The engine error handler frees all slots on
            # failure, so no leak occurs, but the blocks contain stale data.
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
            # NOTE: zero-initialized, so positions beyond a sequence's actual
            # length gather from block 0, offset 0.  The attention padding mask
            # ensures these values are ignored by SDPA.
            max_len = self._seq_len + 1
            batch_size = len(self.seq_ids)
            gather_block_ids = torch.zeros(
                batch_size,
                max_len,
                dtype=torch.long,
                device=device,
            )
            gather_offsets = torch.zeros(
                batch_size,
                max_len,
                dtype=torch.long,
                device=device,
            )

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
        cached_v = self.pool.v[layer_idx, self._gather_block_ids, :, self._gather_offsets, :]

        # Permute to [batch, kv_heads, max_len, head_dim] for SDPA.
        return cached_k.permute(0, 2, 1, 3), cached_v.permute(0, 2, 1, 3)

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
