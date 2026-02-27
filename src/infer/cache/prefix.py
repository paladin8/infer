"""Radix tree for prefix caching of KV cache blocks."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass
class PrefixTreeNode:
    """A node in the prefix tree.

    Each node represents one KV cache block. The path from root to this
    node defines the token sequence whose KV data is stored in the
    blocks along that path.
    """

    tokens: tuple[int, ...]
    block_id: int
    ref_count: int
    last_access_time: int
    parent: PrefixTreeNode | None
    children: dict[tuple[int, ...], PrefixTreeNode] = field(default_factory=dict)


class PrefixTree:
    """Radix tree for prefix caching.

    Stores KV cache block IDs keyed by token ID sequences. Each tree
    level corresponds to one block (``block_size`` tokens). Supports
    matching, insertion, and LRU eviction. Refcount management is
    handled externally by the pool via stored node references.

    Args:
        block_size: Number of tokens per block (must match pool's block_size).
    """

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._root = PrefixTreeNode(
            tokens=(),
            block_id=-1,
            ref_count=0,
            last_access_time=0,
            parent=None,
        )
        self._block_to_node: dict[int, PrefixTreeNode] = {}
        self._clock: int = 0

    def match(
        self,
        token_ids: list[int],
    ) -> tuple[list[int], list[PrefixTreeNode], int]:
        """Find the longest cached prefix matching the given token sequence.

        Walks the tree from root, matching ``block_size``-aligned chunks
        of token IDs. For each matched node, increments ``ref_count``
        and updates ``last_access_time``.

        Matches up to ``len(token_ids) // block_size`` complete blocks.
        A full match (all complete blocks cached) is allowed --- the
        runner handles this case with a single last-token forward pass.

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
        num_complete_blocks = len(token_ids) // self.block_size
        if num_complete_blocks == 0:
            return [], [], 0

        self._clock += 1
        matched_block_ids: list[int] = []
        matched_nodes: list[PrefixTreeNode] = []
        node = self._root

        for i in range(num_complete_blocks):
            start = i * self.block_size
            end = start + self.block_size
            chunk = tuple(token_ids[start:end])

            child = node.children.get(chunk)
            if child is None:
                break

            child.ref_count += 1
            child.last_access_time = self._clock
            matched_block_ids.append(child.block_id)
            matched_nodes.append(child)
            node = child

        matched_tokens = len(matched_block_ids) * self.block_size
        return matched_block_ids, matched_nodes, matched_tokens

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
        num_complete_blocks = len(token_ids) // self.block_size
        if num_complete_blocks == 0:
            return
        assert len(block_ids) >= num_complete_blocks, (
            f"block_ids has {len(block_ids)} entries but need {num_complete_blocks}"
        )

        self._clock += 1
        node = self._root

        for i in range(num_complete_blocks):
            start = i * self.block_size
            end = start + self.block_size
            chunk = tuple(token_ids[start:end])

            child = node.children.get(chunk)
            if child is not None:
                # Already cached (from a prior match or insert). Skip.
                node = child
                continue

            # Create new node for this block.
            new_node = PrefixTreeNode(
                tokens=chunk,
                block_id=block_ids[i],
                ref_count=1,
                last_access_time=self._clock,
                parent=node,
            )
            node.children[chunk] = new_node
            self._block_to_node[block_ids[i]] = new_node
            node = new_node

    def evict(self, num_blocks: int) -> list[int]:
        """Evict up to ``num_blocks`` blocks using LRU policy.

        Evicts leaf nodes with ``ref_count == 0``, ordered by
        ``last_access_time`` (oldest first). After evicting a leaf, if
        its parent becomes a childless leaf with ``ref_count == 0``, the
        parent becomes eligible for eviction in the same pass (cascading).

        Returns:
            List of freed block IDs (to return to the allocator).
        """
        freed: list[int] = []
        if num_blocks <= 0:
            return freed

        # Build a min-heap of evictable leaves keyed by (last_access_time, block_id).
        heap: list[tuple[int, int, PrefixTreeNode]] = []
        for node in self._block_to_node.values():
            if not node.children and node.ref_count == 0:
                heap.append((node.last_access_time, node.block_id, node))
        heapq.heapify(heap)

        while len(freed) < num_blocks and heap:
            _, _, node = heapq.heappop(heap)

            # Node may have been evicted already (cascading) or gained refs.
            if node.block_id not in self._block_to_node:
                continue
            if node.children or node.ref_count > 0:
                continue

            # Evict this node.
            freed.append(node.block_id)
            del self._block_to_node[node.block_id]

            parent = node.parent
            if parent is not None:
                del parent.children[node.tokens]

                # Cascading: if parent becomes an evictable leaf, add it.
                if parent is not self._root and not parent.children and parent.ref_count == 0:
                    heapq.heappush(heap, (parent.last_access_time, parent.block_id, parent))

        return freed

    def contains_block(self, block_id: int) -> bool:
        """Check whether a block ID is currently in the tree.

        Used by ``free_slot()`` to distinguish tree-managed blocks
        (kept alive) from non-tree blocks (freed to allocator).
        Also used by ``audit_blocks()`` to exclude cached blocks
        from leak detection.
        """
        return block_id in self._block_to_node

    def evictable_count(self) -> int:
        """Number of blocks reclaimable via eviction (including cascading).

        A node is reclaimable if ``ref_count == 0`` and all of its
        descendants also have ``ref_count == 0``. This accounts for
        cascading eviction where freeing leaves exposes parents.

        Used by ``free_token_capacity()`` to report reclaimable memory.
        """
        count, _ = self._count_evictable(self._root)
        return count

    def cached_block_count(self) -> int:
        """Total number of blocks currently in the tree."""
        return len(self._block_to_node)

    def _count_evictable(self, node: PrefixTreeNode) -> tuple[int, bool]:
        """Count reclaimable nodes in a subtree.

        Returns ``(count, is_fully_evictable)`` where ``is_fully_evictable``
        is True when the node and all its descendants have ``ref_count == 0``.
        """
        if node is self._root:
            total = 0
            for child in node.children.values():
                child_count, _ = self._count_evictable(child)
                total += child_count
            return total, False

        if not node.children:
            evictable = node.ref_count == 0
            return (1 if evictable else 0), evictable

        total = 0
        all_evictable = True
        for child in node.children.values():
            child_count, child_evictable = self._count_evictable(child)
            total += child_count
            if not child_evictable:
                all_evictable = False

        self_evictable = node.ref_count == 0 and all_evictable
        if self_evictable:
            total += 1

        return total, self_evictable
