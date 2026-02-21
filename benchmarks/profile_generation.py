"""Profile a generation run and produce CUDA kernel summaries + Chrome traces.

Usage:
    uv run python benchmarks/profile_generation.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --prompt-tokens 256 --decode-tokens 64

Outputs:
    - Chrome trace JSON in benchmarks/profiles/ (viewable in chrome://tracing or Perfetto)
    - Text summary table: top-N CUDA kernels by total GPU time
    - Separate prefill and decode breakdown
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from infer.engine.generate import generate
from infer.engine.sampler import SamplingParams
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer

PROFILES_DIR = Path(__file__).parent / "profiles"


def make_synthetic_prompt(tokenizer: Tokenizer, target_tokens: int) -> list[int]:
    """Create a prompt of exactly *target_tokens* length."""
    phrase = "The quick brown fox jumps over the lazy dog. "
    text = phrase * ((target_tokens // 5) + 10)
    ids = tokenizer.encode(text, add_special_tokens=True)
    return ids[:target_tokens]


def _print_kernel_table(
    events: list[torch.autograd.profiler_util.FunctionEvent], top_n: int
) -> None:
    """Print a summary table of CUDA kernels sorted by total GPU time."""
    # Filter to CUDA kernel events only.
    cuda_events = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    if not cuda_events:
        print("  (no CUDA kernel events found)")
        return

    # Aggregate by kernel name.
    kernel_stats: dict[str, dict[str, float | int]] = {}
    for evt in cuda_events:
        name = evt.name
        if name not in kernel_stats:
            kernel_stats[name] = {"total_us": 0.0, "count": 0, "avg_us": 0.0}
        kernel_stats[name]["total_us"] += evt.cuda_time_total
        kernel_stats[name]["count"] += 1  # type: ignore[operator]

    for stats in kernel_stats.values():
        count = stats["count"]
        if isinstance(count, (int, float)) and count > 0:
            stats["avg_us"] = stats["total_us"] / count  # type: ignore[operator]

    # Sort by total time descending.
    sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)

    total_gpu_us = sum(s["total_us"] for _, s in sorted_kernels)  # type: ignore[arg-type]

    # Print table.
    print(f"  {'Kernel':<70s}  {'Total(ms)':>10s}  {'Count':>6s}  {'Avg(us)':>9s}  {'%GPU':>6s}")
    print(f"  {'-' * 70}  {'-' * 10}  {'-' * 6}  {'-' * 9}  {'-' * 6}")
    for name, stats in sorted_kernels[:top_n]:
        total_ms = stats["total_us"] / 1000  # type: ignore[operator]
        pct = (stats["total_us"] / total_gpu_us * 100) if total_gpu_us > 0 else 0.0  # type: ignore[operator]
        print(
            f"  {name[:70]:<70s}  {total_ms:>10.2f}  {int(stats['count']):>6d}"
            f"  {stats['avg_us']:>9.1f}  {pct:>5.1f}%"
        )

    if len(sorted_kernels) > top_n:
        rest_us = sum(s["total_us"] for _, s in sorted_kernels[top_n:])  # type: ignore[arg-type]
        rest_count = sum(int(s["count"]) for _, s in sorted_kernels[top_n:])
        pct = (rest_us / total_gpu_us * 100) if total_gpu_us > 0 else 0.0
        print(
            f"  {'... (remaining ' + str(len(sorted_kernels) - top_n) + ' kernels)':<70s}"
            f"  {rest_us / 1000:>10.2f}  {rest_count:>6d}  {'':>9s}  {pct:>5.1f}%"
        )

    print(
        f"\n  Total GPU time: {total_gpu_us / 1000:.2f} ms across {len(cuda_events)} kernel launches"
    )


def _classify_op(name: str) -> str:
    """Classify a CUDA kernel name into an operation category."""
    name_lower = name.lower()
    if any(k in name_lower for k in ["gemm", "cutlass", "cublas", "matmul"]):
        return "matmul"
    if any(k in name_lower for k in ["rms", "norm", "layer_norm", "layernorm"]):
        return "norm"
    if any(k in name_lower for k in ["rope", "rotary"]):
        return "rope"
    if any(k in name_lower for k in ["attention", "sdpa", "flash", "fmha"]):
        return "attention"
    if any(k in name_lower for k in ["silu", "gelu", "swiglu", "geglu", "activation"]):
        return "activation"
    if any(k in name_lower for k in ["softmax"]):
        return "attention"
    if any(k in name_lower for k in ["embedding", "index_select", "gather"]):
        return "embedding"
    if any(k in name_lower for k in ["copy", "cast", "convert", "to("]):
        return "memory/cast"
    if any(k in name_lower for k in ["elementwise", "add", "mul", "pow", "rsqrt", "mean"]):
        return "elementwise"
    if any(k in name_lower for k in ["cat", "slice", "split", "narrow", "view"]):
        return "reshape"
    if any(k in name_lower for k in ["triton"]):
        return "triton"
    return "other"


def _print_category_summary(events: list[torch.autograd.profiler_util.FunctionEvent]) -> None:
    """Print a summary table grouped by operation category."""
    cuda_events = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    if not cuda_events:
        return

    categories: dict[str, dict[str, float | int]] = {}
    for evt in cuda_events:
        cat = _classify_op(evt.name)
        if cat not in categories:
            categories[cat] = {"total_us": 0.0, "count": 0}
        categories[cat]["total_us"] += evt.cuda_time_total
        categories[cat]["count"] += 1  # type: ignore[operator]

    total_us = sum(c["total_us"] for c in categories.values())  # type: ignore[arg-type]
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["total_us"], reverse=True)

    print(f"\n  {'Category':<20s}  {'Total(ms)':>10s}  {'Count':>6s}  {'%GPU':>6s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 6}  {'-' * 6}")
    for cat, stats in sorted_cats:
        total_ms = stats["total_us"] / 1000  # type: ignore[operator]
        pct = (stats["total_us"] / total_us * 100) if total_us > 0 else 0.0  # type: ignore[operator]
        print(f"  {cat:<20s}  {total_ms:>10.2f}  {int(stats['count']):>6d}  {pct:>5.1f}%")


def profile_generation(
    model_id: str,
    prompt_tokens: int,
    decode_tokens: int,
    dtype_str: str = "bfloat16",
    top_n: int = 25,
    save_trace: bool = True,
) -> None:
    """Profile a generation run."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available, profiling will be limited")

    # Load model.
    print(f"Loading model: {model_id} ({dtype_str}) ...")
    t0 = time.perf_counter()
    model, _config = load_model(model_id, dtype=dtype, device=device)
    tokenizer = Tokenizer(model_id)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Create prompt.
    prompt_ids = make_synthetic_prompt(tokenizer, prompt_tokens)
    actual_prompt_tokens = len(prompt_ids)
    params = SamplingParams(temperature=0.0, max_new_tokens=decode_tokens)

    print(f"\nProfiling: prompt={actual_prompt_tokens} tokens, decode={decode_tokens} tokens")
    print(f"Device: {device}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name()})")
    else:
        print()

    # Warmup run (not profiled).
    print("\nWarmup run...")
    generate(model, tokenizer, prompt_ids, params, device=device, use_kv_cache=True)

    # Profiled run.
    print("Profiled run...")

    # Determine trace path.
    model_slug = model_id.replace("/", "_")
    trace_path = PROFILES_DIR / f"{model_slug}_{dtype_str}_p{actual_prompt_tokens}_d{decode_tokens}"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=(
            torch.profiler.tensorboard_trace_handler(str(trace_path)) if save_trace else None
        ),
    ) as prof:
        result = generate(model, tokenizer, prompt_ids, params, device=device, use_kv_cache=True)
        prof.step()

    gen_tokens = result.generated_tokens
    print(f"Generated {gen_tokens} tokens")
    print(f"  Prefill: {result.timing.prefill_time_s * 1000:.1f}ms")
    if result.timing.decode_time_s > 0:
        decode_tps = gen_tokens / result.timing.decode_time_s
        print(f"  Decode:  {result.timing.decode_time_s * 1000:.1f}ms ({decode_tps:.1f} tok/s)")

    # Export Chrome trace.
    if save_trace:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        # The tensorboard handler already writes traces, but also write a standalone one.
        standalone_trace = PROFILES_DIR / f"{model_slug}_{dtype_str}_trace.json"
        prof.export_chrome_trace(str(standalone_trace))
        print(f"\nChrome trace saved to: {standalone_trace}")
        print(f"Tensorboard trace dir: {trace_path}")

    # Print summary tables.
    events = prof.key_averages()

    print(f"\n{'=' * 110}")
    print("Top CUDA Kernels by Total GPU Time")
    print(f"{'=' * 110}")
    _print_kernel_table(prof.events(), top_n)

    print(f"\n{'=' * 110}")
    print("GPU Time by Operation Category")
    print(f"{'=' * 110}")
    _print_category_summary(prof.events())

    # Print key averages table.
    print(f"\n{'=' * 110}")
    print("Key Averages (CPU + CUDA)")
    print(f"{'=' * 110}")
    print(events.table(sort_by="cuda_time_total", row_limit=top_n))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile LLM generation and produce CUDA kernel summaries",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=256,
        help="Number of prompt tokens (synthetic)",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=64,
        help="Number of tokens to decode",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top kernels to display",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Don't save Chrome trace files",
    )
    args = parser.parse_args()

    profile_generation(
        model_id=args.model,
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        dtype_str=args.dtype,
        top_n=args.top_n,
        save_trace=not args.no_trace,
    )


if __name__ == "__main__":
    main()
