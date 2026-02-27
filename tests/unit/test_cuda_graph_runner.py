"""Unit tests for CUDAGraphRunner and GraphPagedDecodeCacheView."""

from __future__ import annotations

import pytest
import torch

from infer.cache.paged import PagedKVCachePool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class TestPaddedBatchSize:
    """Tests for _padded_batch_size bucket selection."""

    def test_exact_match(self) -> None:
        from infer.engine.cuda_graph_runner import _padded_batch_size

        assert _padded_batch_size(1) == 1
        assert _padded_batch_size(2) == 2
        assert _padded_batch_size(4) == 4
        assert _padded_batch_size(8) == 8
        assert _padded_batch_size(16) == 16
        assert _padded_batch_size(32) == 32

    def test_round_up(self) -> None:
        from infer.engine.cuda_graph_runner import _padded_batch_size

        assert _padded_batch_size(3) == 4
        assert _padded_batch_size(5) == 8
        assert _padded_batch_size(7) == 8
        assert _padded_batch_size(9) == 16
        assert _padded_batch_size(17) == 32

    def test_too_large(self) -> None:
        from infer.engine.cuda_graph_runner import _padded_batch_size

        assert _padded_batch_size(33) is None
        assert _padded_batch_size(64) is None


class TestGraphPagedDecodeCacheView:
    """Tests for the graph-compatible cache view."""

    def _make_pool(
        self,
        num_layers: int = 2,
        total_blocks: int = 20,
        num_kv_heads: int = 2,
        block_size: int = 4,
        head_dim: int = 8,
    ) -> PagedKVCachePool:
        return PagedKVCachePool(
            k=torch.zeros(
                num_layers, total_blocks, num_kv_heads, block_size, head_dim, device="cuda"
            ),
            v=torch.zeros(
                num_layers, total_blocks, num_kv_heads, block_size, head_dim, device="cuda"
            ),
            total_blocks=total_blocks,
            block_size=block_size,
        )

    def test_prepare_populates_tensors(self) -> None:
        """prepare() correctly populates page_table_tensor and seq_lens_tensor."""
        from infer.cache.paged import GraphPagedDecodeCacheView

        pool = self._make_pool()
        s0 = pool.allocate_slot(initial_tokens=8)  # 2 blocks
        pool.seq_lens[s0] = 5
        s1 = pool.allocate_slot(initial_tokens=4)  # 1 block
        pool.seq_lens[s1] = 3

        view = GraphPagedDecodeCacheView(
            pool, max_batch_size=4, max_blocks_per_seq=3, device="cuda"
        )
        scratch = 19  # arbitrary scratch block
        view.prepare([s0, s1], pool, scratch)

        # Seq lens: 5+1=6, 3+1=4.
        assert view.seq_lens_tensor[0].item() == 6
        assert view.seq_lens_tensor[1].item() == 4
        # Padding slots should be zero.
        assert view.seq_lens_tensor[2].item() == 0
        assert view.seq_lens_tensor[3].item() == 0

        # Page table rows should match pool state.
        pt0 = pool.page_tables[s0]
        for j, bid in enumerate(pt0):
            assert view.page_table_tensor[0, j].item() == bid

        # Write positions.
        assert view._write_block_ids[0].item() == pt0[5 // pool.block_size]
        assert view._write_offsets[0].item() == 5 % pool.block_size

    def test_padding_writes_to_scratch(self) -> None:
        """Padding slots' write targets go to the scratch block."""
        from infer.cache.paged import GraphPagedDecodeCacheView

        pool = self._make_pool()
        s0 = pool.allocate_slot(initial_tokens=4)
        pool.seq_lens[s0] = 2
        scratch = 15

        view = GraphPagedDecodeCacheView(
            pool, max_batch_size=4, max_blocks_per_seq=2, device="cuda"
        )
        view.prepare([s0], pool, scratch)

        # Slots 1-3 are padding → should target scratch.
        assert view._write_block_ids[1].item() == scratch
        assert view._write_block_ids[2].item() == scratch
        assert view._write_block_ids[3].item() == scratch

    def test_write_only_writes_to_pool(self) -> None:
        """write_only() writes K/V to correct pool positions."""
        from infer.cache.paged import GraphPagedDecodeCacheView

        pool = self._make_pool()
        s0 = pool.allocate_slot(initial_tokens=4)
        pool.seq_lens[s0] = 2  # write position = 2 → block 0, offset 2

        view = GraphPagedDecodeCacheView(
            pool, max_batch_size=1, max_blocks_per_seq=2, device="cuda"
        )
        view.prepare([s0], pool, scratch_block=19)

        k = torch.ones(1, 2, 1, 8, device="cuda") * 3.0
        v = torch.ones(1, 2, 1, 8, device="cuda") * 7.0
        view.write_only(0, k, v)

        block_id = pool.page_tables[s0][0]  # block for position 2
        assert pool.k[0, block_id, 0, 2, 0].item() == 3.0
        assert pool.v[0, block_id, 0, 2, 0].item() == 7.0

    def test_advance_is_noop(self) -> None:
        """advance() should not modify pool state."""
        from infer.cache.paged import GraphPagedDecodeCacheView

        pool = self._make_pool()
        s0 = pool.allocate_slot(initial_tokens=4)
        pool.seq_lens[s0] = 5

        view = GraphPagedDecodeCacheView(
            pool, max_batch_size=1, max_blocks_per_seq=2, device="cuda"
        )
        view.advance(1)
        assert pool.seq_lens[s0] == 5

    def test_stable_tensor_addresses(self) -> None:
        """Tensors should maintain the same GPU addresses across prepare() calls."""
        from infer.cache.paged import GraphPagedDecodeCacheView

        pool = self._make_pool()
        s0 = pool.allocate_slot(initial_tokens=4)
        pool.seq_lens[s0] = 2

        view = GraphPagedDecodeCacheView(
            pool, max_batch_size=2, max_blocks_per_seq=2, device="cuda"
        )

        view.prepare([s0], pool, scratch_block=19)
        pt_ptr = view.page_table_tensor.data_ptr()
        sl_ptr = view.seq_lens_tensor.data_ptr()
        wb_ptr = view._write_block_ids.data_ptr()
        wo_ptr = view._write_offsets.data_ptr()

        pool.seq_lens[s0] = 3
        view.prepare([s0], pool, scratch_block=19)

        assert view.page_table_tensor.data_ptr() == pt_ptr
        assert view.seq_lens_tensor.data_ptr() == sl_ptr
        assert view._write_block_ids.data_ptr() == wb_ptr
        assert view._write_offsets.data_ptr() == wo_ptr

    def test_is_paged(self) -> None:
        from infer.cache.paged import GraphPagedDecodeCacheView

        pool = self._make_pool()
        view = GraphPagedDecodeCacheView(
            pool, max_batch_size=1, max_blocks_per_seq=1, device="cuda"
        )
        assert view.is_paged() is True
