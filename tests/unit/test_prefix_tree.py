"""Unit tests for PrefixTree."""

from __future__ import annotations

import pytest

from infer.cache.prefix import PrefixTree

BLOCK_SIZE = 4


@pytest.fixture()
def tree() -> PrefixTree:
    return PrefixTree(block_size=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# match() on empty tree
# ---------------------------------------------------------------------------


class TestMatchEmpty:
    def test_empty_tree_returns_empty(self, tree: PrefixTree) -> None:
        block_ids, nodes, matched = tree.match([1, 2, 3, 4, 5, 6, 7, 8])
        assert block_ids == []
        assert nodes == []
        assert matched == 0

    def test_empty_prompt(self, tree: PrefixTree) -> None:
        block_ids, nodes, matched = tree.match([])
        assert block_ids == []
        assert nodes == []
        assert matched == 0

    def test_short_prompt(self, tree: PrefixTree) -> None:
        """Prompt shorter than block_size returns no match."""
        block_ids, nodes, matched = tree.match([1, 2])
        assert block_ids == []
        assert nodes == []
        assert matched == 0


# ---------------------------------------------------------------------------
# insert() + match()
# ---------------------------------------------------------------------------


class TestInsertAndMatch:
    def test_insert_then_full_match(self, tree: PrefixTree) -> None:
        """Insert 4 blocks, match the same tokens -> returns all 4 blocks."""
        tokens = list(range(16))  # 4 blocks of 4 tokens
        blocks = [10, 20, 30, 40]
        tree.insert(tokens, blocks)

        assert tree.cached_block_count() == 4

        block_ids, nodes, matched = tree.match(tokens)
        assert block_ids == [10, 20, 30, 40]
        assert len(nodes) == 4
        assert matched == 16

    def test_partial_match(self, tree: PrefixTree) -> None:
        """Tokens diverge mid-path, only matching prefix returned."""
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        tree.insert(tokens_a, [10, 20, 30, 40])

        # Query with same first 2 blocks, different third block.
        tokens_b = [1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, 13, 14, 15, 16]
        block_ids, nodes, matched = tree.match(tokens_b)
        assert block_ids == [10, 20]
        assert len(nodes) == 2
        assert matched == 8

    def test_non_block_aligned_prompt(self, tree: PrefixTree) -> None:
        """Prompt not aligned to block_size: trailing tokens ignored."""
        tokens = list(range(10))  # 2 complete blocks + 2 trailing
        blocks = [10, 20, 30]  # page table has 3 blocks (ceil(10/4))
        tree.insert(tokens, blocks)

        # Only 2 complete blocks inserted.
        assert tree.cached_block_count() == 2

        block_ids, _nodes, matched = tree.match(tokens)
        assert block_ids == [10, 20]
        assert matched == 8

    def test_insert_short_prompt_is_noop(self, tree: PrefixTree) -> None:
        """Prompt shorter than block_size: insert does nothing."""
        tree.insert([1, 2], [10])
        assert tree.cached_block_count() == 0

    def test_insert_idempotent(self, tree: PrefixTree) -> None:
        """Inserting the same prefix twice doesn't create duplicate nodes."""
        tokens = list(range(8))
        blocks = [10, 20]
        tree.insert(tokens, blocks)
        tree.insert(tokens, blocks)
        assert tree.cached_block_count() == 2

    def test_branching_prefixes(self, tree: PrefixTree) -> None:
        """Two sequences sharing a prefix then diverging."""
        shared = [1, 2, 3, 4]
        tokens_a = [*shared, 5, 6, 7, 8]
        tokens_b = [*shared, 9, 10, 11, 12]

        tree.insert(tokens_a, [10, 20])
        tree.insert(tokens_b, [10, 30])  # block 10 is the shared root

        assert tree.cached_block_count() == 3  # [10, 20, 30]

        ids_a, _, matched_a = tree.match(tokens_a)
        assert ids_a == [10, 20]
        assert matched_a == 8

        ids_b, _, matched_b = tree.match(tokens_b)
        assert ids_b == [10, 30]
        assert matched_b == 8

    def test_full_match_block_aligned(self, tree: PrefixTree) -> None:
        """Full match when prompt is exactly block-aligned."""
        tokens = list(range(8))  # exactly 2 blocks
        tree.insert(tokens, [10, 20])

        block_ids, _nodes, matched = tree.match(tokens)
        assert block_ids == [10, 20]
        assert matched == 8  # all tokens matched

    def test_full_match_non_block_aligned(self, tree: PrefixTree) -> None:
        """Full match when prompt has trailing tokens beyond last complete block."""
        # 10 tokens = 2 complete blocks + 2 trailing.
        tokens = list(range(10))
        tree.insert(tokens, [10, 20, 30])  # 3 blocks in page table

        # Only 2 complete blocks cached. match() matches up to 2 blocks.
        block_ids, _nodes, matched = tree.match(tokens)
        assert block_ids == [10, 20]
        assert matched == 8

    def test_insert_too_few_block_ids_raises(self, tree: PrefixTree) -> None:
        """block_ids shorter than num_complete_blocks raises AssertionError."""
        tokens = list(range(8))  # 2 complete blocks
        with pytest.raises(AssertionError, match="block_ids has 1 entries but need 2"):
            tree.insert(tokens, [10])  # only 1 block ID


# ---------------------------------------------------------------------------
# Refcount lifecycle
# ---------------------------------------------------------------------------


class TestRefcount:
    def test_match_increments_refcount(self, tree: PrefixTree) -> None:
        tokens = list(range(8))
        tree.insert(tokens, [10, 20])

        # First match: ref_count goes to 2 (1 from insert + 1 from match).
        _, nodes, _ = tree.match(tokens)
        assert nodes[0].ref_count == 2
        assert nodes[1].ref_count == 2

    def test_multiple_matches_increment(self, tree: PrefixTree) -> None:
        tokens = list(range(4))
        tree.insert(tokens, [10])

        tree.match(tokens)
        tree.match(tokens)
        _, nodes, _ = tree.match(tokens)
        # 1 (insert) + 3 (matches) = 4.
        assert nodes[0].ref_count == 4

    def test_direct_decrement(self, tree: PrefixTree) -> None:
        """Pool decrements ref_count directly on stored node references."""
        tokens = list(range(4))
        tree.insert(tokens, [10])

        _, nodes, _ = tree.match(tokens)
        assert nodes[0].ref_count == 2

        # Simulate pool free_slot decrement.
        nodes[0].ref_count -= 1
        assert nodes[0].ref_count == 1

    def test_evictable_count_tracks_zero_refcount(self, tree: PrefixTree) -> None:
        tokens = list(range(4))
        tree.insert(tokens, [10])  # ref_count = 1

        assert tree.evictable_count() == 0

        # Decrement to 0.
        _, nodes, _ = tree.match(tokens)  # ref_count = 2
        nodes[0].ref_count -= 1  # back to 1
        assert tree.evictable_count() == 0

        nodes[0].ref_count -= 1  # now 0
        assert tree.evictable_count() == 1

    def test_evictable_count_includes_cascadable_chain(self, tree: PrefixTree) -> None:
        """A chain of zero-refcount nodes: all are reclaimable via cascading."""
        tokens = list(range(12))  # 3 blocks
        tree.insert(tokens, [10, 20, 30])

        _, nodes, _ = tree.match(tokens)
        for n in nodes:
            n.ref_count = 0

        # All 3 are reclaimable (leaf first, then cascading frees parents).
        assert tree.evictable_count() == 3

    def test_evictable_count_excludes_blocked_ancestors(self, tree: PrefixTree) -> None:
        """Ancestor with ref_count > 0 blocks cascading: only descendants count."""
        tokens = list(range(12))  # 3 blocks: A -> B -> C
        tree.insert(tokens, [10, 20, 30])

        _, nodes, _ = tree.match(tokens)
        nodes[0].ref_count = 1  # A: protected
        nodes[1].ref_count = 0  # B: blocked by child C being present
        nodes[2].ref_count = 0  # C: leaf, evictable

        # C is evictable. B becomes leaf after C evicted, then B is evictable.
        # But A has ref_count > 0, so A is not evictable.
        assert tree.evictable_count() == 2


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_evict_oldest_leaf(self, tree: PrefixTree) -> None:
        """Two disjoint prefixes, evict 1 -> evicts the older one."""
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tree.insert(tokens_a, [10])
        tree.insert(tokens_b, [20])

        # Make both evictable (ref_count = 0).
        _, nodes_a, _ = tree.match(tokens_a)
        _, nodes_b, _ = tree.match(tokens_b)
        # insert sets ref_count=1, match adds 1 -> ref_count=2
        nodes_a[0].ref_count = 0
        nodes_b[0].ref_count = 0

        # tokens_a was inserted first (lower last_access_time from insert),
        # but match updated last_access_time. Let's set explicit times.
        nodes_a[0].last_access_time = 1  # older
        nodes_b[0].last_access_time = 5  # newer

        freed = tree.evict(1)
        assert freed == [10]  # evicts the older one
        assert not tree.contains_block(10)
        assert tree.contains_block(20)

    def test_evict_does_not_evict_referenced(self, tree: PrefixTree) -> None:
        """Nodes with ref_count > 0 are protected."""
        tokens = list(range(4))
        tree.insert(tokens, [10])  # ref_count = 1

        freed = tree.evict(1)
        assert freed == []
        assert tree.contains_block(10)

    def test_cascading_eviction(self, tree: PrefixTree) -> None:
        """Evicting a leaf exposes parent, parent evicted in same call."""
        tokens = list(range(8))  # 2 blocks: [0..3] -> [4..7]
        tree.insert(tokens, [10, 20])

        # Set both to ref_count = 0.
        _, nodes, _ = tree.match(tokens)
        for n in nodes:
            n.ref_count = 0

        # Evict 2: should cascade (evict leaf 20 first, then 10 becomes leaf).
        freed = tree.evict(2)
        assert sorted(freed) == [10, 20]
        assert tree.cached_block_count() == 0

    def test_cascading_stops_at_referenced_parent(self, tree: PrefixTree) -> None:
        """Cascading doesn't evict parent with ref_count > 0."""
        tokens = list(range(8))
        tree.insert(tokens, [10, 20])

        _, nodes, _ = tree.match(tokens)
        nodes[0].ref_count = 0  # parent (block 10): will try to cascade
        # Actually, match increments both. Let's be explicit.
        # After insert: both ref_count=1. After match: both ref_count=2.
        nodes[0].ref_count = 1  # parent: protected
        nodes[1].ref_count = 0  # leaf: evictable

        freed = tree.evict(2)
        # Only the leaf should be evicted. Parent has ref_count=1.
        assert freed == [20]
        assert tree.contains_block(10)
        assert not tree.contains_block(20)

    def test_evict_zero_returns_empty(self, tree: PrefixTree) -> None:
        freed = tree.evict(0)
        assert freed == []

    def test_evict_more_than_available(self, tree: PrefixTree) -> None:
        """Requesting more blocks than evictable returns what's available."""
        tokens = list(range(4))
        tree.insert(tokens, [10])

        _, nodes, _ = tree.match(tokens)
        nodes[0].ref_count = 0

        freed = tree.evict(5)
        assert freed == [10]

    def test_evict_from_branching_tree(self, tree: PrefixTree) -> None:
        """Evict leaves from a branching tree, shared root protected."""
        shared = [1, 2, 3, 4]
        tokens_a = [*shared, 5, 6, 7, 8]
        tokens_b = [*shared, 9, 10, 11, 12]

        tree.insert(tokens_a, [10, 20])
        tree.insert(tokens_b, [10, 30])

        # Match both paths so we have node references.
        _, nodes_a, _ = tree.match(tokens_a)
        _, nodes_b, _ = tree.match(tokens_b)

        # Root node (block 10) is shared, has ref_count from both matches + both inserts.
        # Leaf A (block 20): set to 0.
        # Leaf B (block 30): set to 0.
        # Root (block 10): keep ref_count > 0 (still in use).
        nodes_a[1].ref_count = 0  # leaf of branch A
        nodes_b[1].ref_count = 0  # leaf of branch B
        # Root: nodes_a[0] and nodes_b[0] are the same node.
        assert nodes_a[0] is nodes_b[0]
        nodes_a[0].ref_count = 1  # protected

        freed = tree.evict(2)
        assert sorted(freed) == [20, 30]
        # Root still exists.
        assert tree.contains_block(10)
        assert tree.cached_block_count() == 1


# ---------------------------------------------------------------------------
# contains_block
# ---------------------------------------------------------------------------


class TestContainsBlock:
    def test_contains_after_insert(self, tree: PrefixTree) -> None:
        tokens = list(range(8))
        tree.insert(tokens, [10, 20])
        assert tree.contains_block(10)
        assert tree.contains_block(20)
        assert not tree.contains_block(30)

    def test_not_contains_after_evict(self, tree: PrefixTree) -> None:
        tokens = list(range(4))
        tree.insert(tokens, [10])

        _, nodes, _ = tree.match(tokens)
        nodes[0].ref_count = 0

        tree.evict(1)
        assert not tree.contains_block(10)


# ---------------------------------------------------------------------------
# cached_block_count
# ---------------------------------------------------------------------------


class TestCachedBlockCount:
    def test_empty(self, tree: PrefixTree) -> None:
        assert tree.cached_block_count() == 0

    def test_after_insert(self, tree: PrefixTree) -> None:
        tree.insert(list(range(8)), [10, 20])
        assert tree.cached_block_count() == 2

    def test_after_evict(self, tree: PrefixTree) -> None:
        tokens = list(range(4))
        tree.insert(tokens, [10])

        _, nodes, _ = tree.match(tokens)
        nodes[0].ref_count = 0
        tree.evict(1)

        assert tree.cached_block_count() == 0


# ---------------------------------------------------------------------------
# Clock / last_access_time
# ---------------------------------------------------------------------------


class TestClock:
    def test_match_updates_access_time(self, tree: PrefixTree) -> None:
        tokens = list(range(4))
        tree.insert(tokens, [10])

        initial_time = tree._clock
        tree.match(tokens)
        assert tree._clock > initial_time

    def test_insert_advances_clock(self, tree: PrefixTree) -> None:
        clock_before = tree._clock
        tree.insert(list(range(4)), [10])
        assert tree._clock > clock_before

    def test_lru_ordering_respects_access_time(self, tree: PrefixTree) -> None:
        """More recently accessed nodes survive eviction."""
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tree.insert(tokens_a, [10])
        tree.insert(tokens_b, [20])

        # Access A more recently than B.
        _, nodes_b, _ = tree.match(tokens_b)
        _, nodes_a, _ = tree.match(tokens_a)  # A accessed last

        nodes_a[0].ref_count = 0
        nodes_b[0].ref_count = 0

        # Evict 1: should evict B (older access time).
        freed = tree.evict(1)
        assert freed == [20]
