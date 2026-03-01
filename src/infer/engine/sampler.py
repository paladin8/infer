"""Sampling parameters and token sampling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from infer.structured.logit_mask import StructuredOutputState

from infer.structured.logit_mask import apply_structured_output_mask


@dataclass
class SamplingParams:
    """Parameters controlling token sampling during generation.

    Attributes:
        temperature: Scales logits before softmax.  ``0.0`` is greedy (argmax).
        top_p: Nucleus sampling threshold.  ``1.0`` disables.
        top_k: Keep only top-k tokens.  ``None`` disables.
        repetition_penalty: CTRL-paper penalty for repeated tokens.  ``1.0`` disables.
        max_new_tokens: Maximum tokens to generate (excluding prompt).
        stop: Text strings that trigger early stopping.
        seed: Random seed for reproducible sampling.  ``None`` = non-deterministic.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    max_new_tokens: int = 128
    stop: list[str] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate parameter ranges, raising ``ValueError`` on invalid values."""
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0.0, got {self.repetition_penalty}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")


# ---------------------------------------------------------------------------
# Sampling transforms (applied in fixed order)
# ---------------------------------------------------------------------------


def apply_repetition_penalty(
    logits: Tensor,
    token_ids: list[int],
    penalty: float,
) -> Tensor:
    """Penalize tokens that appear in the context.

    For each unique token in *token_ids*:
    - Positive logits are divided by *penalty*.
    - Negative logits are multiplied by *penalty*.

    This makes repeated tokens less likely regardless of their original
    logit sign (CTRL-paper formulation).

    Args:
        logits: Raw logits, shape ``[vocab_size]``.
        token_ids: All token IDs seen so far (prompt + generated).
        penalty: Penalty factor.  ``1.0`` is a no-op.

    Returns:
        Penalized logits (same shape and dtype).
    """
    if penalty == 1.0 or len(token_ids) == 0:
        return logits

    unique_ids = torch.tensor(list(set(token_ids)), dtype=torch.long, device=logits.device)
    penalized = logits.clone()
    scores = penalized[unique_ids]
    penalized[unique_ids] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return penalized


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits by temperature.

    Returns logits unchanged when ``temperature == 1.0``.
    This function should not be called when ``temperature == 0.0``
    (greedy mode is handled at the sampling step).

    Args:
        logits: Raw logits, shape ``[vocab_size]``.
        temperature: Scaling factor.

    Returns:
        Scaled logits (same shape and dtype).
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep only the top-k logits, setting the rest to ``-inf``.

    Args:
        logits: Logits, shape ``[vocab_size]``.
        k: Number of top entries to keep.

    Returns:
        Filtered logits (same shape and dtype).
    """
    if k >= logits.shape[-1]:
        return logits
    top_values, top_indices = torch.topk(logits, k)
    result = torch.full_like(logits, float("-inf"))
    result.scatter_(0, top_indices, top_values)
    return result


def apply_top_p(logits: Tensor, p: float) -> Tensor:
    """Nucleus sampling: keep the smallest set of tokens with cumulative probability >= *p*.

    Args:
        logits: Logits, shape ``[vocab_size]``.
        p: Cumulative probability threshold.  ``1.0`` is a no-op.

    Returns:
        Filtered logits (same shape and dtype).
    """
    if p == 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens whose cumulative probability exceeds p (keep at least the top token).
    # The "- sorted_probs" excludes each token's own probability so the first token
    # whose cumulative probability crosses p is still kept.
    sorted_mask = (cumulative_probs - sorted_probs) >= p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to original order into a fresh tensor.
    result = torch.full_like(logits, float("-inf"))
    result.scatter_(0, sorted_indices, sorted_logits)
    return result


def sample_token(
    logits: Tensor,
    context_token_ids: list[int],
    params: SamplingParams,
    generator: torch.Generator | None = None,
    structured_state: StructuredOutputState | None = None,
) -> int:
    """Sample a single token from logits using the full sampling pipeline.

    Transform order: structured output mask -> repetition penalty ->
    temperature -> top-k -> top-p -> sample.

    When ``structured_state`` is provided, applies FSM-based logit masking
    before all other transforms. This ensures only grammar-valid tokens
    can be sampled.

    Args:
        logits: Raw logits for a single position, shape ``[vocab_size]``.
        context_token_ids: All token IDs seen so far (prompt + generated).
        params: Sampling parameters.
        generator: Optional RNG for reproducible sampling.
        structured_state: Optional structured output state for constrained generation.

    Returns:
        The sampled token ID.
    """
    # Apply structured output mask first (Phase 12).
    if structured_state is not None:
        logits = apply_structured_output_mask(logits, structured_state)

    # Greedy: skip all transforms and return argmax.
    if params.temperature == 0.0:
        logits = apply_repetition_penalty(logits, context_token_ids, params.repetition_penalty)
        return int(torch.argmax(logits).item())

    logits = apply_repetition_penalty(logits, context_token_ids, params.repetition_penalty)
    logits = apply_temperature(logits, params.temperature)
    if params.top_k is not None:
        logits = apply_top_k(logits, params.top_k)
    if params.top_p < 1.0:
        logits = apply_top_p(logits, params.top_p)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs.unsqueeze(0), num_samples=1, generator=generator).item())
