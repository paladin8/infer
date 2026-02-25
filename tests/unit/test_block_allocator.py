"""Unit tests for BlockAllocator."""

from __future__ import annotations

import pytest

from infer.cache.paged import BlockAllocator


class TestBasicAllocation:
    def test_initial_state(self) -> None:
        alloc = BlockAllocator(10)
        assert alloc.num_free() == 10
        assert alloc.num_allocated() == 0

    def test_allocate_some(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(5)
        assert len(blocks) == 5
        assert len(set(blocks)) == 5  # unique IDs
        assert alloc.num_free() == 5
        assert alloc.num_allocated() == 5

    def test_allocate_all(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(10)
        assert len(blocks) == 10
        assert alloc.num_free() == 0
        assert alloc.num_allocated() == 10

    def test_allocate_when_exhausted(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(10)
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            alloc.allocate(1)

    def test_allocate_zero(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(0)
        assert blocks == []
        assert alloc.num_free() == 10

    def test_empty_pool(self) -> None:
        alloc = BlockAllocator(0)
        assert alloc.num_free() == 0
        assert alloc.can_allocate(0)
        with pytest.raises(RuntimeError):
            alloc.allocate(1)


class TestDeallocation:
    def test_free_restores_capacity(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(3)
        alloc.free(blocks)
        assert alloc.num_free() == 10
        assert alloc.num_allocated() == 0

    def test_freed_blocks_are_reusable(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(10)
        freed = blocks[:3]
        alloc.free(freed)
        new_blocks = alloc.allocate(3)
        assert set(new_blocks) == set(freed)

    def test_double_free_raises(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(3)
        alloc.free(blocks)
        with pytest.raises(ValueError, match="not currently allocated"):
            alloc.free(blocks)

    def test_free_invalid_block_raises(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(1)
        with pytest.raises(ValueError, match="not currently allocated"):
            alloc.free([999])


class TestCanAllocate:
    def test_within_capacity(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(8)
        assert alloc.can_allocate(2) is True
        assert alloc.can_allocate(3) is False

    def test_zero_count(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(10)
        assert alloc.can_allocate(0) is True


class TestOwnerTracking:
    def test_allocate_with_owner(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(3, owner=42)
        for bid in blocks:
            assert alloc._block_owners[bid] == 42

    def test_free_clears_ownership(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(3, owner=42)
        freed = blocks[:2]
        alloc.free(freed)
        for bid in freed:
            assert bid not in alloc._block_owners
        assert alloc._block_owners[blocks[2]] == 42

    def test_mixed_owners(self) -> None:
        alloc = BlockAllocator(10)
        blocks_a = alloc.allocate(3, owner=42)
        alloc.free(blocks_a[:2])
        alloc.allocate(2, owner=99)
        # 1 block owned by 42, 2 by 99
        owner_counts: dict[int, int] = {}
        for owner in alloc._block_owners.values():
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
        assert owner_counts[42] == 1
        assert owner_counts[99] == 2

    def test_allocate_without_owner(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(3)
        for bid in blocks:
            assert bid not in alloc._block_owners


class TestLeakDetection:
    def test_no_leaks(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(5, owner=1)
        alloc.allocate(3, owner=2)
        leaked = alloc.find_leaked_blocks(active_owners={1, 2})
        assert leaked == {}

    def test_one_leaked_owner(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(5, owner=1)
        blocks_2 = alloc.allocate(3, owner=2)
        leaked = alloc.find_leaked_blocks(active_owners={1})
        assert set(leaked.keys()) == {2}
        assert set(leaked[2]) == set(blocks_2)

    def test_all_leaked(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(5, owner=1)
        alloc.allocate(3, owner=2)
        leaked = alloc.find_leaked_blocks(active_owners=set())
        assert set(leaked.keys()) == {1, 2}


class TestUnreferencedBlocks:
    def test_all_referenced(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(10)
        assert alloc.find_unreferenced_blocks(set(blocks)) == []

    def test_some_unreferenced(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(10)
        referenced = set(blocks[:7])
        unreferenced = alloc.find_unreferenced_blocks(referenced)
        assert set(unreferenced) == set(blocks[7:])

    def test_none_referenced(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(10)
        unreferenced = alloc.find_unreferenced_blocks(set())
        assert set(unreferenced) == set(blocks)

    def test_returns_sorted(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(10)
        result = alloc.find_unreferenced_blocks(set())
        assert result == sorted(result)


class TestForceFree:
    def test_force_free_restores_capacity(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(5, owner=1)
        alloc.force_free(blocks)
        assert alloc.num_free() == 10
        assert alloc.num_allocated() == 0
        for bid in blocks:
            assert bid not in alloc._block_owners

    def test_force_free_already_free_is_noop(self) -> None:
        """force_free does not raise on already-free blocks."""
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(5)
        alloc.free(blocks)
        # Should not raise.
        alloc.force_free(blocks)
        assert alloc.num_free() == 10


class TestAllocatedBlockIds:
    def test_reflects_allocations(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(5)
        assert alloc.allocated_block_ids() == set(blocks)

    def test_reflects_frees(self) -> None:
        alloc = BlockAllocator(10)
        blocks = alloc.allocate(5)
        alloc.free(blocks[:2])
        assert alloc.allocated_block_ids() == set(blocks[2:])

    def test_returns_copy(self) -> None:
        alloc = BlockAllocator(10)
        alloc.allocate(5)
        ids = alloc.allocated_block_ids()
        ids.clear()
        assert alloc.num_allocated() == 5
