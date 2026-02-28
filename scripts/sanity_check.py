"""Sanity check: run each supported model on a simple prompt and print the output.

Usage:
    uv run python scripts/sanity_check.py
    uv run python scripts/sanity_check.py --models llama qwen3
    uv run python scripts/sanity_check.py --prompt "Explain gravity in one sentence."
"""

from __future__ import annotations

import argparse
import sys

import torch

from infer.engine.generate import generate
from infer.engine.sampler import SamplingParams
from infer.loader.chat_template import render_chat_template
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer

MODELS: dict[str, str] = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3": "Qwen/Qwen3-1.7B",
    "gemma3": "google/gemma-3-1b-it",
    "qwen3-8b-fp8": "Qwen/Qwen3-8B-FP8",
    "qwen3-8b-int8": "nytopop/Qwen3-8B.w8a8",
}

DEFAULT_PROMPT = "List the top 5 cities in Japan."


def run_model(name: str, model_id: str, prompt: str, max_new_tokens: int) -> bool:
    """Load a model, generate a response, and print results. Returns True on success."""
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  {name.upper()} ({model_id})")
    print(separator)

    try:
        model, config = load_model(model_id, dtype=torch.bfloat16, device="cuda")
        tokenizer = Tokenizer(model_id)
    except Exception as exc:
        print(f"  SKIP: could not load model — {exc}")
        return False

    messages = [{"role": "user", "content": prompt}]
    prompt_text = render_chat_template(messages, model_type=config.model_type)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
    result = generate(model, tokenizer, prompt_ids, params, device="cuda")

    print(f"\n  Prompt:     {prompt}")
    print(f"  Tokens:     {result.prompt_tokens} prompt + {result.generated_tokens} generated")
    print(f"  Finish:     {result.finish_reason}")
    print(f"  Prefill:    {result.timing.prefill_time_s * 1000:.1f} ms")
    if result.generated_tokens > 0 and result.timing.decode_time_s > 0:
        tok_s = result.generated_tokens / result.timing.decode_time_s
        print(f"  Decode:     {tok_s:.1f} tok/s ({result.timing.decode_time_s * 1000:.1f} ms)")
    print(f"  Total:      {result.timing.total_time_s:.2f} s")
    print("\n  Response:\n")
    for line in result.text.strip().splitlines():
        print(f"    {line}")
    print()

    # Free GPU memory before loading the next model.
    del model
    torch.cuda.empty_cache()

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check generation for each model.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which models to run (default: all).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt to send to each model (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for inference.", file=sys.stderr)
        sys.exit(1)

    results: dict[str, bool] = {}
    for name in args.models:
        results[name] = run_model(name, MODELS[name], args.prompt, args.max_new_tokens)

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        status = "OK" if ok else "SKIP"
        print(f"  {name:10s}  {status}")
    print()

    if not any(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
