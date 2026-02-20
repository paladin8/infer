"""Phase 2 generation benchmark: measure throughput and latency without KV cache.

Usage (ad-hoc single run):
    uv run python benchmarks/bench_generation.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --prompt-tokens 256 --max-new-tokens 256

Usage (canonical suite):
    uv run python benchmarks/bench_generation.py --suite quick
    uv run python benchmarks/bench_generation.py --suite standard
    uv run python benchmarks/bench_generation.py --suite full
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from infer.engine.generate import GenerationResult, generate
from infer.engine.sampler import SamplingParams
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer

REPORTS_DIR = Path(__file__).parent / "reports"


# ---------------------------------------------------------------------------
# Canonical benchmark configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """A single benchmark workload configuration."""

    name: str
    prompt_tokens: int
    max_new_tokens: int
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42

    def to_sampling_params(self) -> SamplingParams:
        """Build a ``SamplingParams`` from this config."""
        seed = self.seed if self.temperature > 0.0 else None
        return SamplingParams(
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            seed=seed,
        )


CANONICAL_CONFIGS: list[BenchmarkConfig] = [
    BenchmarkConfig("short-greedy", prompt_tokens=64, max_new_tokens=64),
    BenchmarkConfig("medium-greedy", prompt_tokens=256, max_new_tokens=256),
    BenchmarkConfig("prefill-heavy", prompt_tokens=1024, max_new_tokens=64),
    BenchmarkConfig("decode-heavy", prompt_tokens=64, max_new_tokens=512),
    BenchmarkConfig("long-context", prompt_tokens=1024, max_new_tokens=512),
    BenchmarkConfig(
        "medium-sampled",
        prompt_tokens=256,
        max_new_tokens=256,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    ),
    BenchmarkConfig(
        "medium-full-pipeline",
        prompt_tokens=256,
        max_new_tokens=256,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
    ),
]

SUITE_TIERS: dict[str, list[str]] = {
    "quick": ["short-greedy", "medium-greedy"],
    "standard": [
        "short-greedy",
        "medium-greedy",
        "prefill-heavy",
        "decode-heavy",
        "medium-sampled",
    ],
    "full": [c.name for c in CANONICAL_CONFIGS],
}

_CONFIGS_BY_NAME: dict[str, BenchmarkConfig] = {c.name: c for c in CANONICAL_CONFIGS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_prompt(tokenizer: Tokenizer, target_tokens: int) -> list[int]:
    """Create a prompt of exactly *target_tokens* length.

    Encodes a repeated phrase (including BOS) and truncates to the exact
    target length.
    """
    phrase = "The quick brown fox jumps over the lazy dog. "
    text = phrase * ((target_tokens // 5) + 10)
    ids = tokenizer.encode(text, add_special_tokens=True)
    return ids[:target_tokens]


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) using linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def median(vals: list[float]) -> float:
    """Compute the median of a list of floats."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------------------
# Single-config benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class ConfigReport:
    """Metrics for a single benchmark config run."""

    config_name: str
    prompt_tokens: int
    generated_tokens: int
    ttft_median_ms: float
    prompt_throughput_tps: float
    decode_throughput_tps: float
    e2e_throughput_tps: float
    wall_time_median_s: float
    step_mean_ms: float
    step_p50_ms: float
    step_p95_ms: float
    step_p99_ms: float
    step_min_ms: float
    step_max_ms: float
    end_gpu_memory_gb: float
    end_gpu_reserved_gb: float
    peak_gpu_memory_gb: float
    peak_gpu_reserved_gb: float


def run_config(
    model: nn.Module,
    tokenizer: Tokenizer,
    config: BenchmarkConfig,
    *,
    device: str,
    warmup_runs: int = 2,
    trials: int = 3,
) -> ConfigReport:
    """Run a single benchmark config and return computed metrics."""
    prompt_ids = make_synthetic_prompt(tokenizer, config.prompt_tokens)
    actual_prompt_tokens = len(prompt_ids)
    params = config.to_sampling_params()

    # Release cached allocator blocks from previous configs.
    if device == "cuda":
        torch.cuda.empty_cache()

    # Warmup.
    for _ in range(warmup_runs):
        generate(model, tokenizer, prompt_ids, params, device=device)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Timed trials.
    results: list[GenerationResult] = []
    for i in range(trials):
        r = generate(model, tokenizer, prompt_ids, params, device=device)
        results.append(r)
        line = (
            f"    Trial {i + 1}: {r.generated_tokens} tok, "
            f"prefill={r.timing.prefill_time_s * 1000:.1f}ms, "
            f"decode={r.timing.decode_time_s * 1000:.1f}ms"
        )
        if device == "cuda":
            alloc_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
            line += (
                f", alloc={alloc_gb:.2f}GB, reserved={reserved_gb:.2f}GB, "
                f"peak_alloc={peak_alloc_gb:.2f}GB, peak_reserved={peak_reserved_gb:.2f}GB"
            )
        print(line)

    end_gpu_mem_gb = 0.0
    end_gpu_reserved_gb = 0.0
    peak_gpu_mem_gb = 0.0
    peak_gpu_reserved_gb = 0.0
    if device == "cuda":
        end_gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
        end_gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_gpu_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

    # Compute metrics.
    ttft_med = median([r.timing.prefill_time_s * 1000 for r in results])
    prompt_tp_med = median([actual_prompt_tokens / r.timing.prefill_time_s for r in results])

    decode_tps = [
        r.generated_tokens / r.timing.decode_time_s for r in results if r.timing.decode_time_s > 0
    ]
    decode_tp_med = median(decode_tps) if decode_tps else 0.0

    e2e_tps = [
        r.generated_tokens / r.timing.total_time_s for r in results if r.timing.total_time_s > 0
    ]
    e2e_tp_med = median(e2e_tps) if e2e_tps else 0.0

    total_med = median([r.timing.total_time_s for r in results])

    gen_med = int(median([float(r.generated_tokens) for r in results]))

    all_step_ms = [t * 1000 for r in results for t in r.timing.decode_times_s]
    step_mean_v = sum(all_step_ms) / len(all_step_ms) if all_step_ms else 0.0

    return ConfigReport(
        config_name=config.name,
        prompt_tokens=actual_prompt_tokens,
        generated_tokens=gen_med,
        ttft_median_ms=ttft_med,
        prompt_throughput_tps=prompt_tp_med,
        decode_throughput_tps=decode_tp_med,
        e2e_throughput_tps=e2e_tp_med,
        wall_time_median_s=total_med,
        step_mean_ms=step_mean_v,
        step_p50_ms=percentile(all_step_ms, 50),
        step_p95_ms=percentile(all_step_ms, 95),
        step_p99_ms=percentile(all_step_ms, 99),
        step_min_ms=min(all_step_ms) if all_step_ms else 0.0,
        step_max_ms=max(all_step_ms) if all_step_ms else 0.0,
        end_gpu_memory_gb=end_gpu_mem_gb,
        end_gpu_reserved_gb=end_gpu_reserved_gb,
        peak_gpu_memory_gb=peak_gpu_mem_gb,
        peak_gpu_reserved_gb=peak_gpu_reserved_gb,
    )


# ---------------------------------------------------------------------------
# Detailed single-config report (for ad-hoc runs)
# ---------------------------------------------------------------------------


def print_detailed_report(
    report: ConfigReport,
    *,
    model_id: str,
    dtype_str: str,
    device: str,
    gpu_name: str,
    cuda_version: str,
    post_load_mem_gb: float,
    post_load_reserved_gb: float,
    config: BenchmarkConfig,
    trials: int,
) -> None:
    """Print the full detailed report for a single config run."""
    print()
    print("=== Phase 2 Generation Benchmark ===")
    print(f"Model:            {model_id}")
    print(f"Dtype:            {dtype_str}")
    print(f"Device:           {device}" + (f" ({gpu_name})" if gpu_name else ""))
    if cuda_version:
        print(f"CUDA version:     {cuda_version}")
    print(f"PyTorch version:  {torch.__version__}")
    print(f"Prompt tokens:    {report.prompt_tokens}")
    print(f"Generated tokens: {report.generated_tokens} (median across {trials} trials)")
    print(f"Temperature:      {config.temperature}")
    if config.temperature > 0.0:
        print(f"Seed:             {config.seed}")
    if config.top_k is not None:
        print(f"Top-k:            {config.top_k}")
    if config.top_p < 1.0:
        print(f"Top-p:            {config.top_p}")
    if config.repetition_penalty != 1.0:
        print(f"Rep. penalty:     {config.repetition_penalty}")
    print()
    print("--- Prefill ---")
    print(f"TTFT (median):      {report.ttft_median_ms:.1f} ms")
    print(f"Prompt throughput:  {report.prompt_throughput_tps:.0f} tok/s")
    print()
    print("--- Decode ---")
    if report.decode_throughput_tps > 0:
        print(
            f"Total decode time (median): {report.wall_time_median_s - report.ttft_median_ms / 1000:.1f} s"
        )
        print(f"Decode throughput (median): {report.decode_throughput_tps:.1f} tok/s")
    if report.step_p50_ms > 0:
        print("Per-step latency:")
        print(f"  Mean:           {report.step_mean_ms:.1f} ms")
        print(f"  P50:            {report.step_p50_ms:.1f} ms")
        print(f"  P95:            {report.step_p95_ms:.1f} ms")
        print(f"  P99:            {report.step_p99_ms:.1f} ms")
        print(f"  Min:            {report.step_min_ms:.1f} ms")
        print(f"  Max:            {report.step_max_ms:.1f} ms")
    else:
        print("Per-step latency: (no decode steps)")
    print()
    print("--- Total ---")
    print(f"Wall time (median):       {report.wall_time_median_s:.1f} s")
    print(f"End-to-end throughput:    {report.e2e_throughput_tps:.1f} tok/s")
    print()
    print("--- Memory ---")
    if device == "cuda":
        print(f"Post-load alloc:  {post_load_mem_gb:.2f} GB")
        print(f"Post-load reserve:{post_load_reserved_gb:.2f} GB")
        print(f"End alloc:        {report.end_gpu_memory_gb:.2f} GB")
        print(f"End reserve:      {report.end_gpu_reserved_gb:.2f} GB")
        print(f"Peak alloc:       {report.peak_gpu_memory_gb:.2f} GB")
        print(f"Peak reserve:     {report.peak_gpu_reserved_gb:.2f} GB")
    else:
        print("(CPU mode - GPU memory not available)")
    print()


# ---------------------------------------------------------------------------
# Suite summary table
# ---------------------------------------------------------------------------


def print_suite_summary(
    reports: list[ConfigReport],
    *,
    model_id: str,
    dtype_str: str,
    device: str,
    gpu_name: str,
    suite_name: str,
) -> None:
    """Print a compact comparison table across all configs in a suite."""
    print()
    device_label = f"{device} ({gpu_name})" if gpu_name else device
    print(f"=== Suite '{suite_name}' Summary: {model_id} ({dtype_str}, {device_label}) ===")
    print()

    # Header.
    header = (
        f"{'Config':<24s}  {'Prompt':>6s}  {'Gen':>5s}  "
        f"{'TTFT(ms)':>8s}  {'Decode(t/s)':>11s}  {'E2E(t/s)':>9s}  "
        f"{'Wall(s)':>8s}  {'PeakAlloc':>9s}  {'PeakResv':>9s}"
    )
    print(header)
    print("-" * len(header))

    for r in reports:
        peak_alloc_str = f"{r.peak_gpu_memory_gb:.1f} GB" if r.peak_gpu_memory_gb > 0 else "N/A"
        peak_resv_str = f"{r.peak_gpu_reserved_gb:.1f} GB" if r.peak_gpu_reserved_gb > 0 else "N/A"
        decode_str = f"{r.decode_throughput_tps:.1f}" if r.decode_throughput_tps > 0 else "N/A"
        print(
            f"{r.config_name:<24s}  {r.prompt_tokens:>6d}  {r.generated_tokens:>5d}  "
            f"{r.ttft_median_ms:>8.1f}  {decode_str:>11s}  "
            f"{r.e2e_throughput_tps:>9.1f}  {r.wall_time_median_s:>8.1f}  "
            f"{peak_alloc_str:>9s}  {peak_resv_str:>9s}"
        )

    print()


# ---------------------------------------------------------------------------
# Model loading (shared across suite configs)
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """A loaded model with associated metadata."""

    model: nn.Module
    tokenizer: Tokenizer
    device: str
    gpu_name: str
    cuda_version: str
    post_load_mem_gb: float
    post_load_reserved_gb: float


def load_benchmark_model(model_id: str, dtype_str: str) -> LoadedModel:
    """Load a model and return it with device metadata."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_id} ({dtype_str}) ...")
    t0 = time.perf_counter()
    model, _config = load_model(model_id, dtype=dtype, device=device)
    tokenizer = Tokenizer(model_id)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    post_load_mem_gb = 0.0
    post_load_reserved_gb = 0.0
    gpu_name = ""
    cuda_version = ""
    if device == "cuda":
        post_load_mem_gb = torch.cuda.memory_allocated() / (1024**3)
        post_load_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        gpu_name = torch.cuda.get_device_name()
        cuda_version = torch.version.cuda or "N/A"

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        post_load_mem_gb=post_load_mem_gb,
        post_load_reserved_gb=post_load_reserved_gb,
    )


# ---------------------------------------------------------------------------
# JSON report building
# ---------------------------------------------------------------------------


def build_report_json(
    report: ConfigReport,
    *,
    model_id: str,
    dtype_str: str,
    loaded: LoadedModel,
    config: BenchmarkConfig,
    trials: int,
) -> dict[str, object]:
    """Build a JSON-serializable report dict for a single config."""
    return {
        "phase": 2,
        "config_name": config.name,
        "model": model_id,
        "dtype": dtype_str,
        "device": loaded.device,
        "gpu_name": loaded.gpu_name,
        "prompt_tokens": report.prompt_tokens,
        "generated_tokens": report.generated_tokens,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "seed": config.seed,
        "trials": trials,
        "ttft_median_ms": round(report.ttft_median_ms, 2),
        "prompt_throughput_median_tps": round(report.prompt_throughput_tps, 1),
        "decode_throughput_median_tps": round(report.decode_throughput_tps, 1),
        "e2e_throughput_tps": round(report.e2e_throughput_tps, 1),
        "per_step_latency_mean_ms": round(report.step_mean_ms, 2),
        "per_step_latency_p50_ms": round(report.step_p50_ms, 2),
        "per_step_latency_p95_ms": round(report.step_p95_ms, 2),
        "per_step_latency_p99_ms": round(report.step_p99_ms, 2),
        "per_step_latency_min_ms": round(report.step_min_ms, 2),
        "per_step_latency_max_ms": round(report.step_max_ms, 2),
        "wall_time_median_s": round(report.wall_time_median_s, 3),
        "post_load_gpu_gb": round(loaded.post_load_mem_gb, 2),
        "post_load_gpu_reserved_gb": round(loaded.post_load_reserved_gb, 2),
        "end_gpu_memory_gb": round(report.end_gpu_memory_gb, 2),
        "end_gpu_reserved_gb": round(report.end_gpu_reserved_gb, 2),
        "peak_gpu_memory_gb": round(report.peak_gpu_memory_gb, 2),
        "peak_gpu_reserved_gb": round(report.peak_gpu_reserved_gb, 2),
        "pytorch_version": torch.__version__,
        "cuda_version": loaded.cuda_version,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 2 generation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "suite tiers:\n"
            "  quick     short-greedy, medium-greedy\n"
            "  standard  quick + prefill-heavy, decode-heavy, medium-sampled\n"
            "  full      standard + long-context, medium-full-pipeline\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--trials", type=int, default=3, help="Number of timed trials")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save JSON report to disk",
    )

    # Suite mode.
    parser.add_argument(
        "--suite",
        type=str,
        choices=list(SUITE_TIERS.keys()),
        default=None,
        help="Run a canonical suite of configs instead of a single ad-hoc run",
    )

    # Ad-hoc single-run params (ignored when --suite is set).
    adhoc = parser.add_argument_group("ad-hoc run parameters (ignored with --suite)")
    adhoc.add_argument(
        "--prompt-tokens", type=int, default=256, help="Number of prompt tokens (synthetic)"
    )
    adhoc.add_argument(
        "--prompt", type=str, default=None, help="Custom prompt text (overrides --prompt-tokens)"
    )
    adhoc.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate")
    adhoc.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)"
    )
    adhoc.add_argument("--seed", type=int, default=42, help="Random seed for non-greedy sampling")

    args = parser.parse_args()

    if args.suite:
        _run_suite(args)
    else:
        _run_adhoc(args)


def _run_suite(args: argparse.Namespace) -> None:
    """Run a canonical suite of benchmark configs."""
    suite_name: str = args.suite
    config_names = SUITE_TIERS[suite_name]
    configs = [_CONFIGS_BY_NAME[name] for name in config_names]

    loaded = load_benchmark_model(args.model, args.dtype)

    reports: list[ConfigReport] = []
    all_json: list[dict[str, object]] = []

    for i, config in enumerate(configs):
        print(
            f"\n[{i + 1}/{len(configs)}] Running config: {config.name} "
            f"(prompt={config.prompt_tokens}, gen={config.max_new_tokens}) ..."
        )
        report = run_config(
            loaded.model,
            loaded.tokenizer,
            config,
            device=loaded.device,
            warmup_runs=args.warmup_runs,
            trials=args.trials,
        )
        reports.append(report)
        all_json.append(
            build_report_json(
                report,
                model_id=args.model,
                dtype_str=args.dtype,
                loaded=loaded,
                config=config,
                trials=args.trials,
            )
        )

    print_suite_summary(
        reports,
        model_id=args.model,
        dtype_str=args.dtype,
        device=loaded.device,
        gpu_name=loaded.gpu_name,
        suite_name=suite_name,
    )

    if not args.no_save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        model_slug = args.model.replace("/", "_")
        report_path = REPORTS_DIR / f"phase2_{model_slug}_{args.dtype}_{suite_name}.json"
        suite_report: dict[str, object] = {
            "phase": 2,
            "suite": suite_name,
            "model": args.model,
            "dtype": args.dtype,
            "device": loaded.device,
            "gpu_name": loaded.gpu_name,
            "trials": args.trials,
            "warmup_runs": args.warmup_runs,
            "pytorch_version": torch.__version__,
            "cuda_version": loaded.cuda_version,
            "post_load_gpu_gb": round(loaded.post_load_mem_gb, 2),
            "post_load_gpu_reserved_gb": round(loaded.post_load_reserved_gb, 2),
            "configs": all_json,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(report_path, "w") as f:
            json.dump(suite_report, f, indent=2)
        print(f"Report saved to {report_path}")


def _run_adhoc(args: argparse.Namespace) -> None:
    """Run a single ad-hoc benchmark."""
    loaded = load_benchmark_model(args.model, args.dtype)

    # Build config from CLI args.
    if args.prompt:
        prompt_ids = loaded.tokenizer.encode(args.prompt, add_special_tokens=True)
        prompt_tokens = len(prompt_ids)
    else:
        prompt_tokens = args.prompt_tokens

    config = BenchmarkConfig(
        name="adhoc",
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print(f"\nRunning: prompt={config.prompt_tokens}, gen={config.max_new_tokens} ...")
    report = run_config(
        loaded.model,
        loaded.tokenizer,
        config,
        device=loaded.device,
        warmup_runs=args.warmup_runs,
        trials=args.trials,
    )

    print_detailed_report(
        report,
        model_id=args.model,
        dtype_str=args.dtype,
        device=loaded.device,
        gpu_name=loaded.gpu_name,
        cuda_version=loaded.cuda_version,
        post_load_mem_gb=loaded.post_load_mem_gb,
        post_load_reserved_gb=loaded.post_load_reserved_gb,
        config=config,
        trials=args.trials,
    )

    if not args.no_save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        model_slug = args.model.replace("/", "_")
        report_path = REPORTS_DIR / f"phase2_{model_slug}_{args.dtype}.json"
        report_json = build_report_json(
            report,
            model_id=args.model,
            dtype_str=args.dtype,
            loaded=loaded,
            config=config,
            trials=args.trials,
        )
        with open(report_path, "w") as f:
            json.dump(report_json, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
