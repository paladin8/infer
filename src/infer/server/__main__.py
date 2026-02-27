"""CLI entry point: ``python -m infer.server``."""

from __future__ import annotations

import argparse

import uvicorn

from infer.engine.config import EngineConfig
from infer.server.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="infer serving engine")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-waiting-requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-wait-timeout", type=float, default=0.05)
    parser.add_argument(
        "--batching-mode",
        default="static",
        choices=["static", "continuous"],
        help="batching strategy (default: static)",
    )
    parser.add_argument(
        "--kv-cache-backend",
        default="contiguous",
        choices=["contiguous", "paged"],
        help="KV cache backend (default: contiguous). Paged requires --batching-mode continuous.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="tokens per KV cache block, paged backend only (default: 16)",
    )
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=None,
        help="total KV cache blocks, paged backend only (default: auto-compute)",
    )
    parser.add_argument(
        "--chunked-prefill",
        action="store_true",
        default=False,
        help="enable chunked prefill (requires --batching-mode continuous)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="tokens per prefill chunk (default: 512)",
    )
    parser.add_argument(
        "--max-prefill-chunks-per-step",
        type=int,
        default=None,
        help="cap on prefill chunks per step (default: no cap)",
    )
    parser.add_argument(
        "--prefix-caching",
        action="store_true",
        default=False,
        help="enable prefix caching (requires --kv-cache-backend paged and --chunked-prefill)",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        default=False,
        help="enable CUDA graph capture for decode (requires --kv-cache-backend paged). "
        "Not recommended: Triton kernels replay slower inside CUDA graphs than eagerly.",
    )
    args = parser.parse_args()

    config = EngineConfig(
        model=args.model,
        dtype=args.dtype,
        device=args.device,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_waiting_requests=args.max_waiting_requests,
        seed=args.seed,
        batch_wait_timeout_s=args.batch_wait_timeout,
        batching_mode=args.batching_mode,
        kv_cache_backend=args.kv_cache_backend,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        use_chunked_prefill=args.chunked_prefill,
        prefill_chunk_size=args.prefill_chunk_size,
        max_prefill_chunks_per_step=args.max_prefill_chunks_per_step,
        use_prefix_caching=args.prefix_caching,
        use_cuda_graphs=args.cuda_graphs,
    )

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
