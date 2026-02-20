"""Autoregressive generation loop (no KV cache)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from infer.engine.sampler import SamplingParams, sample_token
from infer.loader.tokenizer import Tokenizer


@dataclass
class GenerationTiming:
    """Timing breakdown for a single generation."""

    prefill_time_s: float
    """Time for the first forward pass (full prompt)."""

    decode_times_s: list[float] = field(default_factory=list)
    """Time for each individual decode step."""

    @property
    def decode_time_s(self) -> float:
        """Total decode time (sum of all decode steps)."""
        return sum(self.decode_times_s)

    @property
    def total_time_s(self) -> float:
        """Wall clock time (prefill + decode)."""
        return self.prefill_time_s + self.decode_time_s


@dataclass
class GenerationResult:
    """Result of a single generation request."""

    token_ids: list[int]
    """Generated token IDs (excluding the prompt)."""

    text: str
    """Decoded generated text (truncated at stop string if applicable)."""

    finish_reason: str
    """Why generation stopped: ``"eos"``, ``"stop"``, or ``"length"``."""

    prompt_tokens: int
    """Number of tokens in the input prompt."""

    generated_tokens: int
    """Number of tokens actually generated (may be less than max_new_tokens)."""

    timing: GenerationTiming
    """Per-step and aggregate timing measurements."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sync_device(device: torch.device) -> None:
    """Synchronize CUDA device if applicable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_stop_strings(
    generated_ids: list[int],
    tokenizer: Tokenizer,
    stop_strings: list[str],
) -> str | None:
    """Check if any stop string appears in the decoded text.

    Returns the matched stop string, or ``None``.
    """
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    for s in stop_strings:
        if s in text:
            return s
    return None


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt_token_ids: list[int],
    params: SamplingParams,
    *,
    device: str | torch.device = "cuda",
) -> GenerationResult:
    """Generate tokens autoregressively from a prompt.

    This is the naive (no KV cache) implementation: each decode step
    runs a full forward pass over the entire sequence (prompt + all
    generated tokens so far).  This gives O(n^2) compute for *n*
    generated tokens.

    Args:
        model: A loaded model (LlamaModel, Qwen3Model, or Gemma3Model).
        tokenizer: Tokenizer for the model.
        prompt_token_ids: Pre-tokenized prompt (list of token IDs).
        params: Sampling parameters.
        device: Device to run on.

    Returns:
        A ``GenerationResult`` with the generated text, token IDs,
        finish reason, and timing breakdown.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")

    device = torch.device(device)

    # Create RNG generator from seed.
    generator: torch.Generator | None = None
    if params.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(params.seed)

    # Resolve EOS token IDs.
    eos_ids: set[int] = tokenizer.eos_token_ids

    tokens = list(prompt_token_ids)
    generated_ids: list[int] = []
    finish_reason = "length"
    decode_times: list[float] = []

    # --- Prefill: first forward pass (full prompt) + first token sampling ---
    _sync_device(device)
    t0 = time.perf_counter()

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits: Tensor = model(input_ids)
    next_logits = logits[0, -1, :]
    token = sample_token(next_logits, tokens, params, generator)

    _sync_device(device)
    prefill_time = time.perf_counter() - t0

    tokens.append(token)
    generated_ids.append(token)

    # Check stop conditions after first token.
    if token in eos_ids:
        finish_reason = "eos"
    elif params.stop and _check_stop_strings(generated_ids, tokenizer, params.stop):
        finish_reason = "stop"

    # --- Decode loop ---
    if finish_reason == "length":
        for _step in range(1, params.max_new_tokens):
            _sync_device(device)
            t0 = time.perf_counter()

            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_logits = logits[0, -1, :]

            _sync_device(device)
            step_time = time.perf_counter() - t0
            decode_times.append(step_time)

            token = sample_token(next_logits, tokens, params, generator)
            tokens.append(token)
            generated_ids.append(token)

            # Check stop conditions (EOS takes priority over stop strings).
            if token in eos_ids:
                finish_reason = "eos"
                break
            if params.stop and _check_stop_strings(generated_ids, tokenizer, params.stop):
                finish_reason = "stop"
                break

    # Decode final text, truncating at the earliest stop string occurrence.
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if finish_reason == "stop" and params.stop:
        earliest = len(text)
        for s in params.stop:
            idx = text.find(s)
            if idx != -1 and idx < earliest:
                earliest = idx
        text = text[:earliest]

    return GenerationResult(
        token_ids=generated_ids,
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=len(prompt_token_ids),
        generated_tokens=len(generated_ids),
        timing=GenerationTiming(
            prefill_time_s=prefill_time,
            decode_times_s=decode_times,
        ),
    )
