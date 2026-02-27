"""CUDA graph capture and replay for the decode forward pass.

Captures the decode forward pass into CUDA graphs for power-of-2 batch sizes
during warmup, then replays them during serving to eliminate CPU-side kernel
launch overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from infer.cache.paged import GraphPagedDecodeCacheView, PagedKVCachePool
from infer.engine.config import EngineConfig
from infer.engine.request import Request

_BATCH_BUCKETS = [1, 2, 4, 8, 16, 32]


def _padded_batch_size(actual: int) -> int | None:
    """Return the smallest bucket >= actual, or None if too large."""
    for b in _BATCH_BUCKETS:
        if b >= actual:
            return b
    return None


@dataclass
class CapturedGraph:
    """A single captured CUDA graph for a fixed batch size."""

    graph: torch.cuda.CUDAGraph
    batch_size: int

    # Static input placeholders (fixed GPU addresses).
    input_ids: Tensor  # [batch_size, 1] long
    position_ids: Tensor  # [batch_size, 1] long

    # Static output placeholder.
    logits: Tensor  # [batch_size, 1, vocab_size] model dtype

    # Static cache view.
    view: GraphPagedDecodeCacheView


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
    ) -> None:
        self.model = model
        self.cache_pool = cache_pool
        self.config = config
        self._graphs: dict[int, CapturedGraph] = {}
        self._mempool = torch.cuda.graph_pool_handle()
        self._scratch_block: int = -1

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
        # Reserve a scratch block for padding slot writes.
        scratch_blocks = self.cache_pool.allocator.allocate(1)
        self._scratch_block = scratch_blocks[0]

        for bucket in _BATCH_BUCKETS:
            if bucket > self.config.max_batch_size:
                break
            self._graphs[bucket] = self._capture_for_batch_size(bucket)

    def _capture_for_batch_size(self, batch_size: int) -> CapturedGraph:
        """Capture a CUDA graph for the given batch size."""
        device = self.config.device
        max_blocks = self.config.max_seq_len // self.cache_pool.block_size

        # Pre-allocate static tensors.
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        view = GraphPagedDecodeCacheView(self.cache_pool, batch_size, max_blocks, device)

        # Allocate temporary cache slots for warmup.
        temp_slots = [self.cache_pool.allocate_slot(initial_tokens=1) for _ in range(batch_size)]
        # Set seq_lens to 1 so prepare() finds valid page table entries.
        for slot in temp_slots:
            self.cache_pool.seq_lens[slot] = 1
        view.prepare(temp_slots, self.cache_pool, self._scratch_block)

        # Warmup forward passes (eager, on side stream).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.inference_mode():
                    self.model(input_ids, kv_cache=view, position_ids=position_ids)
        torch.cuda.current_stream().wait_stream(s)

        # Capture.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._mempool), torch.inference_mode():
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

    def execute(
        self,
        requests: list[Request],
        cache_pool: PagedKVCachePool,
    ) -> Tensor | None:
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
        actual_batch = len(requests)
        padded = _padded_batch_size(actual_batch)
        if padded is None or padded not in self._graphs:
            return None  # fallback to eager

        captured = self._graphs[padded]
        device = self.config.device

        slots = [req.slot_idx for req in requests]
        assert all(s is not None for s in slots)
        int_slots: list[int] = slots  # type: ignore[assignment]

        # 1. Block allocation (Python, before graph).
        _ = cache_pool.decode_view(int_slots)  # triggers _ensure_blocks_allocated

        # 2. Prepare inputs.
        tokens = [req.generated_token_ids[-1] for req in requests]
        positions = [cache_pool.get_seq_len(slot) for slot in int_slots]

        captured.input_ids.zero_()
        captured.position_ids.zero_()
        captured.input_ids[:actual_batch, 0] = torch.tensor(tokens, dtype=torch.long, device=device)
        captured.position_ids[:actual_batch, 0] = torch.tensor(
            positions, dtype=torch.long, device=device
        )

        captured.view.prepare(int_slots, cache_pool, self._scratch_block)

        # 3. Replay.
        captured.graph.replay()

        # 4. Post-replay: advance pool state.
        for slot in int_slots:
            cache_pool.seq_lens[slot] += 1

        # 5. Return logits.
        return captured.logits[:actual_batch]
