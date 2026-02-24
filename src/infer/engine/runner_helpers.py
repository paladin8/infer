"""Shared helpers for model runners (static and continuous)."""

from __future__ import annotations

from infer.engine.request import Request, StepOutput
from infer.loader.tokenizer import Tokenizer


def check_stop(req: Request, token: int, tokenizer: Tokenizer) -> tuple[bool, str | None]:
    """Check stop conditions for a request after appending a token.

    Returns ``(finished, finish_reason)``.
    """
    # EOS takes priority.
    if token in tokenizer.eos_token_ids:
        return True, "eos"

    # Stop strings.
    if req.sampling_params.stop:
        text = tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
        for s in req.sampling_params.stop:
            if s in text:
                return True, "stop"

    # Max tokens.
    if len(req.generated_token_ids) >= req.sampling_params.max_new_tokens:
        return True, "length"

    return False, None


def truncate_at_stop(full_text: str, prev_len: int, req: Request) -> str:
    """Truncate text_delta at the earliest stop string, excluding the stop string."""
    stop_strings = req.sampling_params.stop or []
    earliest = len(full_text)
    for s in stop_strings:
        idx = full_text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return full_text[prev_len:earliest]


def make_step_output(
    req: Request,
    token: int,
    text_delta: str,
    finished: bool,
    reason: str | None,
) -> StepOutput:
    """Create a StepOutput for a normal (active) request."""
    return StepOutput(
        request_id=req.request_id,
        token_id=token,
        text_delta=text_delta,
        finished=finished,
        finish_reason=reason,
        prompt_tokens=len(req.prompt_token_ids) if finished else 0,
        completion_tokens=len(req.generated_token_ids) if finished else 0,
    )


def make_finished_noop(req: Request) -> StepOutput:
    """Create a no-op StepOutput for an already-finished request."""
    return StepOutput(
        request_id=req.request_id,
        token_id=None,
        text_delta="",
        finished=True,
        finish_reason=req.finish_reason,
        prompt_tokens=len(req.prompt_token_ids),
        completion_tokens=len(req.generated_token_ids),
    )
