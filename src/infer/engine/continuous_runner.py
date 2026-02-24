"""Model runner for continuous batching with slotted KV cache."""

from __future__ import annotations

import torch
from torch import nn

from infer.cache.slotted import SlottedKVCache
from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.runner_helpers import (
    check_stop,
    make_step_output,
    truncate_at_stop,
)
from infer.engine.sampler import sample_token
from infer.loader.tokenizer import Tokenizer


class ContinuousRunner:
    """Executes forward passes for continuous batching.

    Manages a pre-allocated :class:`SlottedKVCache` pool.  Each engine step
    runs:

    1. Batched decode for all active decode requests.
    2. Individual prefill for each newly admitted request.

    Args:
        model: A loaded model with a ``.config`` attribute.
        tokenizer: Tokenizer for text decoding and EOS detection.
        config: Engine configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        config: EngineConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]

        # Pre-allocate cache pool at startup.
        model_config = getattr(model, "config", None)
        if model_config is None:
            raise TypeError("model must have a .config attribute")
        self.cache_pool = SlottedKVCache.from_model_config(
            model_config,
            max_seq_len=config.max_seq_len,
            max_batch_size=config.max_batch_size,
            dtype=self.dtype,
            device=config.device,
        )

        # Per-request text tracking, keyed by request_id.
        self._prev_text_lens: dict[str, int] = {}

    def step(
        self,
        prefill: list[Request],
        decode: list[Request],
    ) -> list[tuple[Request, StepOutput]]:
        """Run one engine step: batched decode then individual prefills.

        Returns a list of ``(request, step_output)`` pairs.
        """
        outputs: list[tuple[Request, StepOutput]] = []

        # Phase 1: Batched decode (prioritize inter-token latency).
        if decode:
            decode_outputs = self._batched_decode(decode)
            outputs.extend(zip(decode, decode_outputs, strict=True))

        # Phase 2: Individual prefills.
        for req in prefill:
            output = self._prefill_one(req)
            outputs.append((req, output))

        return outputs

    def free_slot(self, slot_idx: int) -> None:
        """Release a cache slot and clean up tracking state."""
        self.cache_pool.free_slot(slot_idx)

    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request tracking state."""
        self._prev_text_lens.pop(request_id, None)

    @torch.inference_mode()
    def _prefill_one(self, req: Request) -> StepOutput:
        """Prefill a single request using PrefillCacheView."""
        device = self.config.device

        # Allocate a cache slot.
        slot = self.cache_pool.allocate_slot()
        req.slot_idx = slot

        # Build input tensor [1, prompt_len].
        input_ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=device)

        # Create single-slot cache view.
        view = self.cache_pool.prefill_view(slot)

        # Forward pass — single request, no padding_mask needed.
        req.state = RequestState.PREFILL
        logits = self.model(input_ids, kv_cache=view)
        # logits: [1, 1, vocab_size] (last-position optimization applied)

        # Sample first token.
        next_logits = logits[0, -1, :]
        context = req.prompt_token_ids
        token = sample_token(next_logits, context, req.sampling_params, req.generator)
        req.generated_token_ids.append(token)
        req.state = RequestState.DECODE

        # Initialize text tracking.
        text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
        text_delta = text
        self._prev_text_lens[req.request_id] = len(text)

        # Check stop conditions.
        finished, reason = check_stop(req, token, self.tokenizer)
        if finished:
            req.state = RequestState.FINISHED
            req.finish_reason = reason
            if reason == "stop":
                text_delta = truncate_at_stop(text, 0, req)

        return make_step_output(req, token, text_delta, finished, reason)

    @torch.inference_mode()
    def _batched_decode(self, requests: list[Request]) -> list[StepOutput]:
        """Batched decode for all active requests using DecodeCacheView."""
        device = self.config.device
        batch_size = len(requests)

        # Gather active slots and build inputs.
        active_slots = [req.slot_idx for req in requests]
        assert all(s is not None for s in active_slots), "All decode requests must have a slot"
        slots: list[int] = active_slots  # type: ignore[assignment]

        tokens = [req.generated_token_ids[-1] for req in requests]
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)  # [batch, 1]

        # Build position_ids: each sequence's current position.
        positions = [self.cache_pool.seq_lens[slot] for slot in slots]
        position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(
            1
        )  # [batch, 1]

        # Build padding mask: True for valid positions per sequence.
        decode_view = self.cache_pool.decode_view(slots)
        max_kv_len = decode_view.seq_len + 1  # +1 for the token about to be written
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for i, slot in enumerate(slots):
            padding_mask[i, : self.cache_pool.seq_lens[slot]] = True

        # Forward pass with position_ids for per-sequence RoPE.
        logits = self.model(
            input_ids,
            kv_cache=decode_view,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )
        # logits: [batch, 1, vocab_size]

        # Sample per request.
        outputs: list[StepOutput] = []
        for i, req in enumerate(requests):
            context = req.prompt_token_ids + req.generated_token_ids
            token = sample_token(logits[i, -1, :], context, req.sampling_params, req.generator)
            req.generated_token_ids.append(token)

            # Compute text_delta.
            text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
            prev_len = self._prev_text_lens.get(req.request_id, 0)
            text_delta = text[prev_len:]

            finished, reason = check_stop(req, token, self.tokenizer)
            if finished:
                req.state = RequestState.FINISHED
                req.finish_reason = reason
                if reason == "stop":
                    text_delta = truncate_at_stop(text, prev_len, req)
            self._prev_text_lens[req.request_id] = len(text)
            outputs.append(make_step_output(req, token, text_delta, finished, reason))

        return outputs
