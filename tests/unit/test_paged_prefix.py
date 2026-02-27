"""Integration tests for PagedKVCachePool with prefix caching."""

from __future__ import annotations

import pytest
import torch

from infer.cache.paged import PagedKVCachePool
from infer.cache.prefix import PrefixTree
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
BLOCK_SIZE = 4
TOTAL_BLOCKS = 16
DTYPE = torch.float32
DEVICE = "cpu"


def _make_pool(
    total_blocks: int = TOTAL_BLOCKS,
    *,
    prefix_caching: bool = True,
) -> PagedKVCachePool:
    """Create a small PagedKVCachePool for testing, optionally with prefix tree."""
    shape = (NUM_LAYERS, total_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    k = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    v = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    tree = PrefixTree(BLOCK_SIZE) if prefix_caching else None
    return PagedKVCachePool(k, v, total_blocks, BLOCK_SIZE, prefix_tree=tree)


@pytest.fixture()
def pool() -> PagedKVCachePool:
    """Pool with prefix tree enabled."""
    return _make_pool()


@pytest.fixture()
def pool_no_prefix() -> PagedKVCachePool:
    """Pool without prefix tree (Phase 7 behavior)."""
    return _make_pool(prefix_caching=False)


# ---------------------------------------------------------------------------
# allocate_slot_with_prefix
# ---------------------------------------------------------------------------


class TestAllocateSlotWithPrefix:
    def test_empty_tree_allocates_all(self, pool: PagedKVCachePool) -> None:
        """No prefix match -> allocates all blocks fresh."""
        tokens = list(range(8))  # 2 blocks
        seq_id, matched = pool.allocate_slot_with_prefix(tokens)

        assert matched == 0
        assert len(pool.page_tables[seq_id]) == 2
        assert pool.seq_lens[seq_id] == 0
        assert pool.allocator.num_allocated() == 2

    def test_prefix_reuse_after_insert(self, pool: PagedKVCachePool) -> None:
        """Insert prefix into tree, then second allocation reuses cached blocks."""
        tokens = list(range(12))  # 3 blocks

        # First request: no match, allocates 3 blocks.
        seq_id_a, matched_a = pool.allocate_slot_with_prefix(tokens)
        assert matched_a == 0
        blocks_a = list(pool.page_tables[seq_id_a])
        assert len(blocks_a) == 3

        # Insert into tree (via pool to track nodes for refcount management).
        pool.insert_prefix(seq_id_a, tokens)

        # Simulate request completion: free slot (decrements refcounts).
        pool.free_slot(seq_id_a)

        # Second request: same prefix -> reuses cached blocks.
        seq_id_b, matched_b = pool.allocate_slot_with_prefix(tokens)
        assert matched_b == 12  # all 3 blocks matched (3 * 4 = 12 tokens)
        blocks_b = pool.page_tables[seq_id_b]
        # Matched blocks are the same physical blocks.
        assert blocks_b == blocks_a
        # No suffix blocks needed.
        assert pool.allocator.num_allocated() == 3  # same 3 blocks

    def test_partial_match_allocates_suffix(self, pool: PagedKVCachePool) -> None:
        """Partial prefix match: matched blocks reused, suffix freshly allocated."""
        shared = list(range(8))  # 2 blocks
        tokens_a = [*shared, 100, 101, 102, 103]  # 3 blocks
        tokens_b = [*shared, 200, 201, 202, 203]  # 3 blocks, different suffix

        # Insert first sequence's prefix.
        seq_id_a, _ = pool.allocate_slot_with_prefix(tokens_a)
        blocks_a = list(pool.page_tables[seq_id_a])
        pool.insert_prefix(seq_id_a, tokens_a)

        # Second sequence: shares 2-block prefix, needs 1 suffix block.
        seq_id_b, matched_b = pool.allocate_slot_with_prefix(tokens_b)
        assert matched_b == 8  # 2 blocks matched
        blocks_b = pool.page_tables[seq_id_b]
        assert blocks_b[:2] == blocks_a[:2]  # shared prefix
        assert blocks_b[2] != blocks_a[2]  # different suffix block

    def test_concurrent_prefix_sharing(self, pool: PagedKVCachePool) -> None:
        """Two active sequences sharing the same prefix simultaneously."""
        tokens = list(range(8))  # 2 blocks

        # Seq A: populate the tree.
        seq_a, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_a, tokens)

        # Seq B: matches A's prefix while A is still active.
        seq_b, matched_b = pool.allocate_slot_with_prefix(tokens)
        assert matched_b == 8
        assert pool.page_tables[seq_b] == pool.page_tables[seq_a]

        # Both sequences' nodes have refcounts > 0.
        assert pool.prefix_tree is not None
        _, nodes, _ = pool.prefix_tree.match(tokens)
        for node in nodes:
            # insert(ref=1) + match_b(ref=1) + our_match(ref=1) = 3
            assert node.ref_count == 3
            node.ref_count -= 1  # undo our verification match

        # Free A: blocks stay (B still holds refs).
        pool.free_slot(seq_a)
        for node in nodes:
            assert node.ref_count == 1  # only B's match ref remains

        # Free B: refcounts drop to 0, blocks remain in tree but evictable.
        pool.free_slot(seq_b)
        for node in nodes:
            assert node.ref_count == 0
        assert pool.prefix_tree.evictable_count() == 2

    def test_non_block_aligned_prompt(self, pool: PagedKVCachePool) -> None:
        """Prompt with trailing tokens: match covers complete blocks only."""
        tokens = list(range(10))  # 2 complete blocks + 2 trailing -> 3 blocks total

        # First request: 3 blocks allocated.
        seq_id_a, _ = pool.allocate_slot_with_prefix(tokens)
        blocks_a = list(pool.page_tables[seq_id_a])
        assert len(blocks_a) == 3
        pool.insert_prefix(seq_id_a, tokens)
        pool.free_slot(seq_id_a)

        # Second request: 2 complete blocks matched, 1 suffix block for trailing.
        seq_id_b, matched_b = pool.allocate_slot_with_prefix(tokens)
        assert matched_b == 8  # 2 blocks * 4 tokens
        blocks_b = pool.page_tables[seq_id_b]
        assert blocks_b[:2] == blocks_a[:2]
        # Third block is new (suffix containing the trailing tokens).
        assert len(blocks_b) == 3

    def test_seq_prefix_nodes_stored(self, pool: PagedKVCachePool) -> None:
        """allocate_slot_with_prefix stores node references for free_slot."""
        tokens = list(range(8))
        seq_id_a, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_id_a, tokens)
        pool.free_slot(seq_id_a)

        seq_id_b, _ = pool.allocate_slot_with_prefix(tokens)
        assert seq_id_b in pool._seq_prefix_nodes
        assert len(pool._seq_prefix_nodes[seq_id_b]) == 2  # 2 matched nodes

    def test_allocation_failure_undoes_refcounts(self, pool: PagedKVCachePool) -> None:
        """If suffix allocation fails, matched refcounts are rolled back."""
        tokens = list(range(8))
        # Fill tree with 2 blocks.
        seq_id_a, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_id_a, tokens)
        pool.free_slot(seq_id_a)

        # Exhaust pool: allocate all remaining blocks.
        remaining = pool.allocator.num_free()
        if remaining > 0:
            pool.allocator.allocate(remaining)

        # New request with different suffix: match succeeds but suffix allocation
        # will fail (0 blocks free, tree blocks have ref_count > 0 from match).
        new_tokens = [*list(range(8)), 99, 99, 99, 99]  # needs 1 suffix block
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            pool.allocate_slot_with_prefix(new_tokens)

        # Refcounts should have been rolled back.
        assert pool.prefix_tree is not None
        _, nodes, _ = pool.prefix_tree.match(tokens)
        for node in nodes:
            # Each node: insert gave ref=1, first free_slot decremented to 0,
            # failed alloc matched (ref=1) then rolled back (ref=0),
            # our match() just incremented again (ref=1).
            assert node.ref_count == 1
            node.ref_count -= 1  # clean up our match


# ---------------------------------------------------------------------------
# free_slot with prefix tree
# ---------------------------------------------------------------------------


class TestFreeSlotWithPrefix:
    def test_tree_blocks_retained(self, pool: PagedKVCachePool) -> None:
        """Freeing a slot with prefix blocks: tree blocks stay allocated."""
        tokens = list(range(8))
        seq_id, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_id, tokens)

        allocated_before = pool.allocator.num_allocated()
        pool.free_slot(seq_id)

        # Tree blocks (2) are still allocated. No blocks were freed since
        # all blocks belong to the tree.
        assert pool.allocator.num_allocated() == allocated_before

    def test_non_tree_blocks_freed(self, pool: PagedKVCachePool) -> None:
        """Non-tree blocks (suffix, trailing) are freed on free_slot."""
        tokens_a = list(range(8))
        seq_id_a, _ = pool.allocate_slot_with_prefix(tokens_a)
        pool.insert_prefix(seq_id_a, tokens_a)
        pool.free_slot(seq_id_a)

        # Second request: matches 2 prefix blocks, allocates 1 suffix block.
        tokens_b = [*list(range(8)), 99, 99, 99, 99]
        seq_id_b, matched = pool.allocate_slot_with_prefix(tokens_b)
        assert matched == 8
        blocks_b = pool.page_tables[seq_id_b]
        assert len(blocks_b) == 3  # 2 matched + 1 suffix

        # Insert the full sequence so blocks 0-2 are in tree,
        # but block 2 is the suffix block we just allocated.
        pool.insert_prefix(seq_id_b, tokens_b)

        allocated_before = pool.allocator.num_allocated()
        pool.free_slot(seq_id_b)

        # All 3 blocks are now in the tree, so none freed to allocator.
        assert pool.allocator.num_allocated() == allocated_before

    def test_refcounts_decremented(self, pool: PagedKVCachePool) -> None:
        """free_slot decrements refcounts on cached prefix nodes."""
        tokens = list(range(8))
        seq_id_a, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_id_a, tokens)
        pool.free_slot(seq_id_a)

        # Match gives ref_count = 1 for each node.
        seq_id_b, _ = pool.allocate_slot_with_prefix(tokens)
        assert pool.prefix_tree is not None
        _, nodes, _ = pool.prefix_tree.match(tokens)
        # ref_count = 1 (from alloc_b match) + 1 (from our match) = 2
        for node in nodes:
            assert node.ref_count == 2
            node.ref_count -= 1  # undo our match

        pool.free_slot(seq_id_b)
        # After free, refcounts should be 0.
        for node in nodes:
            assert node.ref_count == 0

    def test_without_prefix_tree_frees_all(self, pool_no_prefix: PagedKVCachePool) -> None:
        """Without prefix tree, free_slot frees all blocks (existing behavior)."""
        seq_id = pool_no_prefix.allocate_slot(initial_tokens=8)
        pool_no_prefix.free_slot(seq_id)
        assert pool_no_prefix.allocator.num_allocated() == 0


# ---------------------------------------------------------------------------
# free_token_capacity
# ---------------------------------------------------------------------------


class TestFreeTokenCapacityWithPrefix:
    def test_includes_evictable_blocks(self, pool: PagedKVCachePool) -> None:
        """free_token_capacity includes evictable cached blocks."""
        tokens = list(range(8))
        seq_id, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_id, tokens)
        pool.free_slot(seq_id)

        # 2 blocks in tree with ref_count=0, rest are free.
        assert pool.prefix_tree is not None
        free_blocks = pool.allocator.num_free()
        evictable = pool.prefix_tree.evictable_count()
        assert evictable == 2
        assert pool.free_token_capacity() == (free_blocks + evictable) * BLOCK_SIZE

    def test_no_prefix_tree(self, pool_no_prefix: PagedKVCachePool) -> None:
        """Without prefix tree, capacity is just free blocks."""
        assert pool_no_prefix.free_token_capacity() == TOTAL_BLOCKS * BLOCK_SIZE
        pool_no_prefix.allocate_slot(initial_tokens=8)
        expected = (TOTAL_BLOCKS - 2) * BLOCK_SIZE
        assert pool_no_prefix.free_token_capacity() == expected


# ---------------------------------------------------------------------------
# Eviction during allocation
# ---------------------------------------------------------------------------


class TestEvictionDuringAllocation:
    def test_evicts_to_make_room(self) -> None:
        """Pool exhausted, tree blocks evicted to satisfy new allocation."""
        pool = _make_pool(total_blocks=4)  # very small pool

        # Fill with 4 blocks (1 sequence of 16 tokens).
        tokens_a = list(range(16))
        seq_id_a, _ = pool.allocate_slot_with_prefix(tokens_a)
        pool.insert_prefix(seq_id_a, tokens_a)
        pool.free_slot(seq_id_a)

        # Pool: 0 free blocks, 4 in tree with ref_count=0.
        assert pool.prefix_tree is not None
        assert pool.allocator.num_free() == 0
        assert pool.prefix_tree.evictable_count() == 4

        # New request needs 2 blocks: should evict 2 from tree.
        tokens_b = list(range(100, 108))
        seq_id_b, matched = pool.allocate_slot_with_prefix(tokens_b)
        assert matched == 0  # different tokens, no match
        assert len(pool.page_tables[seq_id_b]) == 2
        assert pool.prefix_tree.cached_block_count() == 2  # 2 remaining in tree


# ---------------------------------------------------------------------------
# audit_blocks with prefix tree
# ---------------------------------------------------------------------------


class TestAuditBlocksWithPrefix:
    def test_tree_blocks_not_leaked(self, pool: PagedKVCachePool) -> None:
        """Tree-managed blocks are not reported as leaked."""
        tokens = list(range(8))
        seq_id, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_id, tokens)
        pool.free_slot(seq_id)

        # Blocks are in tree but not in any page table. Should not be leaked.
        assert pool.audit_blocks() == []

    def test_real_leaks_still_detected(self, pool: PagedKVCachePool) -> None:
        """Actually leaked blocks are still caught even with prefix tree."""
        pool.allocator.allocate(3, owner=999)
        assert len(pool.audit_blocks()) == 3

    def test_no_prefix_tree_unchanged(self, pool_no_prefix: PagedKVCachePool) -> None:
        """Without prefix tree, audit_blocks works as before."""
        pool_no_prefix.allocate_slot(initial_tokens=8)
        assert pool_no_prefix.audit_blocks() == []

    def test_clean_after_full_lifecycle(self, pool: PagedKVCachePool) -> None:
        """No leaks after alloc -> insert -> free -> alloc -> free cycle."""
        tokens = list(range(8))

        # First request.
        seq_a, _ = pool.allocate_slot_with_prefix(tokens)
        pool.insert_prefix(seq_a, tokens)
        pool.free_slot(seq_a)

        # Second request reusing prefix.
        seq_b, _ = pool.allocate_slot_with_prefix(tokens)
        pool.free_slot(seq_b)

        assert pool.audit_blocks() == []


# ---------------------------------------------------------------------------
# from_model_config with prefix caching
# ---------------------------------------------------------------------------


class TestFromModelConfigPrefixCaching:
    @pytest.fixture()
    def config(self) -> ModelConfig:
        return ModelConfig(
            model_type="llama",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=4,
            num_key_value_heads=NUM_KV_HEADS,
            vocab_size=100,
            max_position_embeddings=128,
            head_dim=HEAD_DIM,
        )

    def test_creates_prefix_tree(self, config: ModelConfig) -> None:
        pool = PagedKVCachePool.from_model_config(
            config,
            total_blocks=16,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            device=DEVICE,
            use_prefix_caching=True,
        )
        assert pool.prefix_tree is not None
        assert pool.prefix_tree.block_size == BLOCK_SIZE

    def test_no_prefix_tree_by_default(self, config: ModelConfig) -> None:
        pool = PagedKVCachePool.from_model_config(
            config,
            total_blocks=16,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            device=DEVICE,
        )
        assert pool.prefix_tree is None
