"""Logit masking for structured output generation.

Provides ``StructuredOutputState`` to track per-request FSM state and
``apply_structured_output_mask`` to apply -inf logit masking for tokens
that violate the grammar constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from infer.structured.token_fsm import TokenFSM


@dataclass
class StructuredOutputState:
    """Per-request state for structured output generation.

    Each request with a ``response_format`` carries one of these.
    It tracks the current DFA state and provides convenience methods
    for advancing state and checking terminal conditions.

    Attributes:
        fsm: The compiled TokenFSM for this request's schema/pattern.
        current_state: Current DFA state.
    """

    fsm: TokenFSM
    current_state: int

    def advance(self, token_id: int) -> None:
        """Advance FSM state after a token is accepted.

        Args:
            token_id: The token ID that was sampled.
        """
        self.current_state = self.fsm.next_state(self.current_state, token_id)

    def is_terminal(self) -> bool:
        """Whether the current state is a valid completion point.

        Returns:
            True if generation can validly stop at the current state.
        """
        return self.fsm.is_terminal(self.current_state)

    def allowed_tokens(self) -> set[int]:
        """Return valid token IDs from the current state.

        Returns:
            Set of token IDs that are allowed by the grammar.
        """
        return self.fsm.allowed_tokens(self.current_state)


def apply_structured_output_mask(
    logits: Tensor,
    state: StructuredOutputState,
) -> Tensor:
    """Apply -inf mask to logits for tokens not allowed by the FSM.

    Sets logits of all tokens not in the FSM's allowed set to -inf,
    ensuring only grammar-valid tokens can be sampled.

    Args:
        logits: Raw logits, shape ``[vocab_size]``.
        state: Current structured output state with FSM reference.

    Returns:
        Masked logits (same shape and dtype). Disallowed tokens set to -inf.
    """
    allowed = state.allowed_tokens()

    if not allowed:
        # No tokens allowed — return all -inf (edge case).
        return torch.full_like(logits, float("-inf"))

    # Create mask: True for allowed tokens.
    mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    allowed_indices = torch.tensor(list(allowed), dtype=torch.long, device=logits.device)
    mask[allowed_indices] = True

    # Apply mask: set disallowed tokens to -inf.
    result = logits.clone()
    result[~mask] = float("-inf")
    return result
