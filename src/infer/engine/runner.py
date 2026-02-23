"""Model runner: batched prefill and decode for static batching."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from infer.cache.simple import KVCache
from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.sampler import sample_token
from infer.loader.tokenizer import Tokenizer


class ModelRunner:
    """Executes batched forward passes for a batch of requests.

    Manages right-padding, mask construction, KV cache allocation,
    and per-request sampling.  The runner is stateful: ``prefill()``
    sets up batch-level state that ``decode_step()`` uses, and
    ``clear_batch()`` frees it.

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

        # Active batch state (set during prefill, cleared when batch completes).
        self._kv_cache: KVCache | None = None
        self._prompt_lens: list[int] = []
        self._max_prompt_len: int = 0
        self._padding_mask: Tensor | None = None
        self._prev_text_lens: list[int] = []

    @torch.inference_mode()
    def prefill(self, requests: list[Request]) -> list[StepOutput]:
        """Run batched prefill for a batch of requests.

        Right-pads all prompts to the longest prompt length, allocates a
        single batched KV cache, runs one forward pass, and samples the
        first token per request.

        Returns one ``StepOutput`` per request.
        """
        if not requests:
            return []

        batch_size = len(requests)
        device = self.config.device

        # Record per-request prompt lengths.
        self._prompt_lens = [len(req.prompt_token_ids) for req in requests]
        self._max_prompt_len = max(self._prompt_lens)

        # Right-pad all prompts to max_prompt_len (pad with 0).
        padded = []
        for req in requests:
            tokens = req.prompt_token_ids
            padded.append(tokens + [0] * (self._max_prompt_len - len(tokens)))
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # Build padding mask: True for real tokens, False for padding.
        max_decode = max(req.sampling_params.max_new_tokens for req in requests)
        max_total = min(self._max_prompt_len + max_decode, self.config.max_seq_len)
        self._padding_mask = torch.zeros(batch_size, max_total, dtype=torch.bool, device=device)
        for i, plen in enumerate(self._prompt_lens):
            self._padding_mask[i, :plen] = True

        # Allocate batched KV cache.
        model_config = getattr(self.model, "config", None)
        if model_config is None:
            raise TypeError("model must have a .config attribute for KV cache allocation")
        self._kv_cache = KVCache.from_model_config(
            model_config,
            max_seq_len=max_total,
            batch_size=batch_size,
            dtype=self.dtype,
            device=device,
        )

        # Batched forward pass.
        for req in requests:
            req.state = RequestState.PREFILL
        logits = self.model(input_ids, kv_cache=self._kv_cache, padding_mask=self._padding_mask)
        # logits: [batch, max_prompt_len, vocab_size] (full, no last-position opt)

        # Gather logits at each request's last real token position.
        last_positions = torch.tensor(
            [plen - 1 for plen in self._prompt_lens], dtype=torch.long, device=device
        )
        next_logits = logits[torch.arange(batch_size, device=device), last_positions, :]

        # Initialize text tracking.
        self._prev_text_lens = [0] * batch_size

        # Sample first token per request.
        outputs: list[StepOutput] = []
        for i, req in enumerate(requests):
            context = req.prompt_token_ids
            token = sample_token(next_logits[i], context, req.sampling_params, req.generator)
            req.generated_token_ids.append(token)
            req.state = RequestState.DECODE

            # Mark the generated token's position as valid in the padding mask.
            self._padding_mask[i, self._max_prompt_len] = True

            # Compute text_delta via incremental decode.
            text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
            text_delta = text[self._prev_text_lens[i] :]

            finished, reason = self._check_stop(req, token)
            if finished:
                req.state = RequestState.FINISHED
                req.finish_reason = reason
                if reason == "stop":
                    text_delta = self._truncate_at_stop(text, self._prev_text_lens[i], req)
            self._prev_text_lens[i] = len(text)
            outputs.append(self._make_step_output(req, token, text_delta, finished, reason))

        return outputs

    @torch.inference_mode()
    def decode_step(self, requests: list[Request]) -> list[StepOutput]:
        """Run one batched decode step for active requests.

        Feeds each request's last generated token (or a dummy token for
        finished requests) as a ``[batch, 1]`` input, runs one forward pass,
        and samples the next token per active request.

        Returns one ``StepOutput`` per request.
        """
        device = self.config.device

        # Build input: last generated token for active requests, dummy (0) for finished.
        tokens = []
        for req in requests:
            if req.state == RequestState.DECODE:
                tokens.append(req.generated_token_ids[-1])
            else:
                tokens.append(0)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)

        # Forward pass.
        logits = self.model(input_ids, kv_cache=self._kv_cache, padding_mask=self._padding_mask)
        # logits: [batch, 1, vocab_size]

        # Sample per request.
        outputs: list[StepOutput] = []
        for i, req in enumerate(requests):
            if req.state != RequestState.DECODE:
                # Already finished — emit no-op output.
                outputs.append(self._make_finished_noop(req))
                continue

            context = req.prompt_token_ids + req.generated_token_ids
            token = sample_token(logits[i, -1, :], context, req.sampling_params, req.generator)
            req.generated_token_ids.append(token)

            # Mark this decode position as valid in padding mask.
            current_pos = self._max_prompt_len + len(req.generated_token_ids) - 1
            assert self._padding_mask is not None
            if current_pos < self._padding_mask.shape[1]:
                self._padding_mask[i, current_pos] = True

            # Compute text_delta via incremental decode.
            text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
            text_delta = text[self._prev_text_lens[i] :]

            finished, reason = self._check_stop(req, token)
            if finished:
                req.state = RequestState.FINISHED
                req.finish_reason = reason
                if reason == "stop":
                    text_delta = self._truncate_at_stop(text, self._prev_text_lens[i], req)
            self._prev_text_lens[i] = len(text)
            outputs.append(self._make_step_output(req, token, text_delta, finished, reason))

        return outputs

    def clear_batch(self) -> None:
        """Free batch-level state (KV cache, masks)."""
        self._kv_cache = None
        self._prompt_lens = []
        self._max_prompt_len = 0
        self._padding_mask = None
        self._prev_text_lens = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate_at_stop(self, full_text: str, prev_len: int, req: Request) -> str:
        """Truncate text_delta at the earliest stop string, excluding the stop string."""
        stop_strings = req.sampling_params.stop or []
        earliest = len(full_text)
        for s in stop_strings:
            idx = full_text.find(s)
            if idx != -1 and idx < earliest:
                earliest = idx
        return full_text[prev_len:earliest]

    def _check_stop(self, req: Request, token: int) -> tuple[bool, str | None]:
        """Check stop conditions for a request after appending a token.

        Returns ``(finished, finish_reason)``.
        """
        # EOS takes priority.
        eos_ids: set[int] = self.tokenizer.eos_token_ids
        if token in eos_ids:
            return True, "eos"

        # Stop strings.
        if req.sampling_params.stop:
            text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
            for s in req.sampling_params.stop:
                if s in text:
                    return True, "stop"

        # Max tokens.
        if len(req.generated_token_ids) >= req.sampling_params.max_new_tokens:
            return True, "length"

        return False, None

    def _make_step_output(
        self,
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

    def _make_finished_noop(self, req: Request) -> StepOutput:
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
