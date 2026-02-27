"""Profile CUDA graph decode vs eager decode to identify bottlenecks.

Measures wall-clock time for each phase of a decode step:
  - Graph path: block alloc, prepare inputs, prepare cache view, replay, post-replay
  - Eager path: input build, padding mask, forward pass

Usage:
    uv run python benchmarks/profile_cuda_graph.py \
        --model meta-llama/Llama-3.2-3B-Instruct --batch-size 1
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
from torch import nn

from infer.cache.paged import PagedKVCachePool
from infer.engine.config import EngineConfig
from infer.engine.cuda_graph_runner import CUDAGraphRunner
from infer.loader.model_loader import load_model


def profile_decode(
    model: nn.Module,
    pool: PagedKVCachePool,
    config: EngineConfig,
    runner: CUDAGraphRunner,
    batch_size: int,
    num_steps: int = 50,
) -> None:
    """Run decode steps and time each phase."""
    device = config.device

    # Allocate slots and prefill one token each.
    slots: list[int] = []
    for _ in range(batch_size):
        slot = pool.allocate_slot(initial_tokens=1)
        slots.append(slot)
        pool.seq_lens[slot] = 1

    # Ensure at least one block allocated per slot.
    tmp = pool.decode_view(slots)
    tmp._ensure_blocks_allocated()

    # Build a fake Request-like object for the graph runner.
    class FakeReq:
        def __init__(self, slot: int) -> None:
            self.slot_idx = slot
            self.generated_token_ids = [1]  # dummy token

    requests: list[Any] = [FakeReq(s) for s in slots]

    # --- Profile CUDA graph path ---
    print(f"\n=== CUDA Graph path (batch={batch_size}, {num_steps} steps) ===")

    # Warmup
    for _ in range(5):
        runner.execute(requests, pool)
        for s in slots:
            pool.seq_lens[s] -= 1  # undo advance

    torch.cuda.synchronize()

    t_alloc = 0.0
    t_prepare_inputs = 0.0
    t_prepare_view = 0.0
    t_replay = 0.0
    t_post = 0.0
    t_total_graph = 0.0

    from infer.engine.cuda_graph_runner import _padded_batch_size

    for _ in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        actual_batch = len(requests)
        padded = _padded_batch_size(actual_batch)
        assert padded is not None
        captured = runner._graphs[padded]

        int_slots = [r.slot_idx for r in requests]

        # 1. Block allocation
        t1 = time.perf_counter()
        tmp_view = pool.decode_view(int_slots)
        tmp_view._ensure_blocks_allocated()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_alloc += t2 - t1

        # 2. Prepare inputs
        tokens = [r.generated_token_ids[-1] for r in requests]
        positions = [pool.get_seq_len(s) for s in int_slots]
        captured.input_ids.zero_()
        captured.position_ids.zero_()
        captured.input_ids[:actual_batch, 0] = torch.tensor(tokens, dtype=torch.long, device=device)
        captured.position_ids[:actual_batch, 0] = torch.tensor(
            positions, dtype=torch.long, device=device
        )
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_prepare_inputs += t3 - t2

        # 3. Prepare cache view
        captured.view.prepare(int_slots, pool, runner._scratch_block)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        t_prepare_view += t4 - t3

        # 4. Replay
        captured.graph.replay()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        t_replay += t5 - t4

        # 5. Post-replay
        for s in int_slots:
            pool.seq_lens[s] += 1
        t6 = time.perf_counter()
        t_post += t6 - t5

        t_total_graph += t6 - t0

    print(f"  Total:          {t_total_graph / num_steps * 1000:8.3f} ms/step")
    print(f"  Block alloc:    {t_alloc / num_steps * 1000:8.3f} ms/step")
    print(f"  Prepare inputs: {t_prepare_inputs / num_steps * 1000:8.3f} ms/step")
    print(f"  Prepare view:   {t_prepare_view / num_steps * 1000:8.3f} ms/step")
    print(f"  Replay:         {t_replay / num_steps * 1000:8.3f} ms/step")
    print(f"  Post-replay:    {t_post / num_steps * 1000:8.3f} ms/step")

    # Reset seq_lens for eager profiling.
    for s in slots:
        pool.seq_lens[s] = 1

    # --- Profile eager path ---
    print(f"\n=== Eager path (batch={batch_size}, {num_steps} steps) ===")

    # Warmup
    for _ in range(5):
        decode_view = pool.decode_view(slots)
        max_kv_len = decode_view.seq_len + 1
        input_ids = torch.tensor([1] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        position_ids = torch.tensor(
            [pool.get_seq_len(s) for s in slots], dtype=torch.long, device=device
        ).unsqueeze(1)
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for i, s in enumerate(slots):
            padding_mask[i, : pool.get_seq_len(s)] = True
        with torch.inference_mode():
            model(
                input_ids,
                kv_cache=decode_view,
                padding_mask=padding_mask,
                position_ids=position_ids,
            )
        for s in slots:
            pool.seq_lens[s] -= 1

    torch.cuda.synchronize()

    t_input_build = 0.0
    t_view_build = 0.0
    t_mask_build = 0.0
    t_forward = 0.0
    t_total_eager = 0.0

    for _ in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. Build inputs
        input_ids = torch.tensor([1] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        positions = [pool.get_seq_len(s) for s in slots]
        position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(1)
        t1 = time.perf_counter()
        t_input_build += t1 - t0

        # 2. Build decode view
        decode_view = pool.decode_view(slots)
        t2 = time.perf_counter()
        t_view_build += t2 - t1

        # 3. Build padding mask
        max_kv_len = decode_view.seq_len + 1
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for i, s in enumerate(slots):
            padding_mask[i, : pool.get_seq_len(s)] = True
        t3 = time.perf_counter()
        t_mask_build += t3 - t2

        # 4. Forward
        with torch.inference_mode():
            model(
                input_ids,
                kv_cache=decode_view,
                padding_mask=padding_mask,
                position_ids=position_ids,
            )
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        t_forward += t4 - t3

        t_total_eager += t4 - t0

        # Undo advance done inside model forward.
        for s in slots:
            pool.seq_lens[s] -= 1

    print(f"  Total:          {t_total_eager / num_steps * 1000:8.3f} ms/step")
    print(f"  Input build:    {t_input_build / num_steps * 1000:8.3f} ms/step")
    print(f"  View build:     {t_view_build / num_steps * 1000:8.3f} ms/step")
    print(f"  Mask build:     {t_mask_build / num_steps * 1000:8.3f} ms/step")
    print(f"  Forward:        {t_forward / num_steps * 1000:8.3f} ms/step")

    # Reset seq_lens for Triton eager profiling.
    for s in slots:
        pool.seq_lens[s] = 1

    # --- Profile eager Triton path (no padding_mask, mask=None → Triton dispatch) ---
    print(f"\n=== Eager Triton path (batch={batch_size}, {num_steps} steps) ===")

    # Warmup
    for _ in range(5):
        decode_view = pool.decode_view(slots)
        input_ids = torch.tensor([1] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        position_ids = torch.tensor(
            [pool.get_seq_len(s) for s in slots], dtype=torch.long, device=device
        ).unsqueeze(1)
        with torch.inference_mode():
            model(input_ids, kv_cache=decode_view, position_ids=position_ids)
        for s in slots:
            pool.seq_lens[s] -= 1

    torch.cuda.synchronize()

    t_total_triton_eager = 0.0
    t_triton_forward = 0.0

    for _ in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        decode_view = pool.decode_view(slots)
        input_ids = torch.tensor([1] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        position_ids = torch.tensor(
            [pool.get_seq_len(s) for s in slots], dtype=torch.long, device=device
        ).unsqueeze(1)

        with torch.inference_mode():
            model(input_ids, kv_cache=decode_view, position_ids=position_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        t_triton_forward += t1 - t0
        t_total_triton_eager += t1 - t0

        for s in slots:
            pool.seq_lens[s] -= 1

    print(f"  Total:          {t_total_triton_eager / num_steps * 1000:8.3f} ms/step")

    print("\n=== Comparison ===")
    graph_ms = t_total_graph / num_steps * 1000
    eager_ms = t_total_eager / num_steps * 1000
    triton_eager_ms = t_total_triton_eager / num_steps * 1000
    print(f"  Graph replay:     {graph_ms:.3f} ms/step")
    print(
        f"  Eager (SDPA):     {eager_ms:.3f} ms/step  ({(graph_ms - eager_ms) / eager_ms * 100:+.1f}% vs graph)"
    )
    print(
        f"  Eager (Triton):   {triton_eager_ms:.3f} ms/step  ({(graph_ms - triton_eager_ms) / triton_eager_ms * 100:+.1f}% vs graph)"
    )

    # Cleanup
    for s in slots:
        pool.free_slot(s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CUDA graph vs eager decode")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-gpu-blocks", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=24)
    args = parser.parse_args()

    config = EngineConfig(
        model=args.model,
        dtype="bfloat16",
        device="cuda",
        max_seq_len=4096,
        max_batch_size=args.max_batch_size,
        batching_mode="continuous",
        kv_cache_backend="paged",
        use_chunked_prefill=True,
        use_prefix_caching=True,
        use_cuda_graphs=True,
        num_gpu_blocks=args.num_gpu_blocks,
    )

    model, model_config = load_model(config.model, dtype=torch.bfloat16, device=config.device)

    num_gpu_blocks = args.num_gpu_blocks
    pool = PagedKVCachePool.from_model_config(
        model_config,
        total_blocks=num_gpu_blocks,
        block_size=config.block_size,
        dtype=torch.bfloat16,
        device=config.device,
        use_prefix_caching=config.use_prefix_caching,
    )
    runner = CUDAGraphRunner(model, pool, config)

    print("Warming up CUDA graphs...")
    runner.warmup()
    print("Warmup complete.")

    for bs in args.batch_size:
        if bs > args.max_batch_size:
            print(f"\nSkipping batch_size={bs} (exceeds max_batch_size={args.max_batch_size})")
            continue
        profile_decode(model, pool, config, runner, bs, num_steps=args.num_steps)


if __name__ == "__main__":
    main()
