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
    )

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
