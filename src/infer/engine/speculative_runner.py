"""Speculative decoding runner: draft-then-verify decode loop (Phase 11).

Uses a small draft model to propose K candidate tokens per step, verified in
a single forward pass of the target model. The output distribution is
mathematically identical to pure target-model sampling (lossless).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from infer.cache.paged import PagedKVCachePool
from infer.cache.protocol import CachePoolProtocol
from infer.cache.slotted import SlottedKVCache
from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.runner_helpers import (
    check_stop,
    make_step_output,
    truncate_at_stop,
)
from infer.engine.sampler import (
    SamplingParams,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_token,
)
from infer.loader.tokenizer import Tokenizer


def _create_cache_pool(
    model: nn.Module,
    config: EngineConfig,
    dtype: torch.dtype,
) -> CachePoolProtocol:
    """Create a KV cache pool for a model based on engine config.

    Args:
        model: Model with a ``.config`` attribute.
        config: Engine configuration (provides backend, sizing parameters).
        dtype: Cache tensor dtype.

    Returns:
        A cache pool satisfying ``CachePoolProtocol``.
    """
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise TypeError("model must have a .config attribute")

    if config.kv_cache_backend == "paged":
        num_gpu_blocks = config.num_gpu_blocks
        if num_gpu_blocks is None:
            num_gpu_blocks = config.max_batch_size * config.max_seq_len // config.block_size
        return PagedKVCachePool.from_model_config(
            model_config,
            total_blocks=num_gpu_blocks,
            block_size=config.block_size,
            dtype=dtype,
            device=config.device,
        )
    else:
        return SlottedKVCache.from_model_config(
            model_config,
            max_seq_len=config.max_seq_len,
            max_batch_size=config.max_batch_size,
            dtype=dtype,
            device=config.device,
        )


def _apply_sampling_transforms(
    logits: Tensor,
    context_token_ids: list[int],
    params: SamplingParams,
) -> Tensor:
    """Apply the full sampling transform pipeline to logits.

    Transform order: repetition penalty -> temperature -> top-k -> top-p.
    Returns transformed logits (not probabilities).

    Args:
        logits: Raw logits for a single position, shape ``[vocab_size]``.
        context_token_ids: All token IDs seen so far (prompt + generated).
        params: Sampling parameters.

    Returns:
        Transformed logits.
    """
    logits = apply_repetition_penalty(logits, context_token_ids, params.repetition_penalty)
    if params.temperature > 0.0:
        logits = apply_temperature(logits, params.temperature)
    if params.top_k is not None:
        logits = apply_top_k(logits, params.top_k)
    if params.top_p < 1.0:
        logits = apply_top_p(logits, params.top_p)
    return logits


class SpeculativeRunner:
    """Executes forward passes with speculative decoding.

    Uses a draft model to propose ``spec_length`` candidate tokens, then
    verifies them in a single forward pass of the target model. Produces
    1 to ``spec_length + 1`` tokens per decode step per request.

    Manages two cache pools (target and draft) with synchronized slot
    lifecycle. Delegates prefill to the target model's cache pool and
    forward pass.

    Args:
        target_model: The main model for verification.
        draft_model: The smaller model for drafting.
        tokenizer: Tokenizer (shared between models).
        config: Engine configuration.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer: Tokenizer,
        config: EngineConfig,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]

        # Create two separate cache pools.
        self.target_cache_pool: CachePoolProtocol = _create_cache_pool(
            target_model, config, self.dtype
        )
        self.draft_cache_pool: CachePoolProtocol = _create_cache_pool(
            draft_model, config, self.dtype
        )

        # Per-request text tracking, keyed by request_id.
        self._prev_text_lens: dict[str, int] = {}

        # Draft cache slot mapping: target_slot -> draft_slot.
        self._draft_slots: dict[int, int] = {}

        # Track which requests have had their draft model KV cache populated.
        self._draft_prefilled: set[str] = set()

    def step(
        self,
        prefill: list[Request],
        decode: list[Request],
    ) -> list[tuple[Request, StepOutput]]:
        """Run one engine step: decode first (prioritize ITL), then prefill.

        Returns a list of ``(request, step_output)`` pairs. Speculative
        decoding may produce multiple outputs per request per step.
        """
        outputs: list[tuple[Request, StepOutput]] = []

        # Phase 1: Speculative decode for active requests.
        if decode:
            decode_outputs = self._speculative_decode(decode)
            outputs.extend(decode_outputs)

        # Phase 2: Prefill (target model only; draft prefill is lazy).
        if self.config.use_chunked_prefill:
            if prefill:
                chunk_outputs = self._prefill_chunks_batched(prefill)
                for req, output in zip(prefill, chunk_outputs, strict=True):
                    if output is not None:
                        outputs.append((req, output))
        else:
            if len(prefill) == 1:
                output = self._prefill_one(prefill[0])
                outputs.append((prefill[0], output))
            elif len(prefill) > 1:
                prefill_outputs = self._prefill_batch(prefill)
                outputs.extend(zip(prefill, prefill_outputs, strict=True))

        return outputs

    def free_slot(self, slot_idx: int) -> None:
        """Release cache slots in both target and draft pools.

        Args:
            slot_idx: The target cache slot to free.
        """
        self.target_cache_pool.free_slot(slot_idx)
        draft_slot = self._draft_slots.pop(slot_idx, None)
        if draft_slot is not None:
            self.draft_cache_pool.free_slot(draft_slot)

    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request tracking state.

        Args:
            request_id: The request ID to clean up.
        """
        self._prev_text_lens.pop(request_id, None)
        self._draft_prefilled.discard(request_id)

    def free_kv_tokens(self) -> int | None:
        """Return available token capacity, accounting for both pools.

        Returns the minimum of both pools' free token capacity. If either
        pool returns ``None`` (contiguous backend), the other pool's value
        is used.
        """
        target_free = self.target_cache_pool.free_token_capacity()
        draft_free = self.draft_cache_pool.free_token_capacity()
        if target_free is None and draft_free is None:
            return None
        if target_free is None:
            return draft_free
        if draft_free is None:
            return target_free
        return min(target_free, draft_free)

    # ------------------------------------------------------------------
    # Prefill (target model only; draft prefill is lazy)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _prefill_one(self, req: Request) -> StepOutput:
        """Prefill a single request using the target model.

        Also allocates a draft cache slot (but does NOT run draft prefill).
        """
        device = self.config.device

        # Allocate target cache slot.
        target_slot = self.target_cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
        req.slot_idx = target_slot

        # Allocate draft cache slot.
        draft_slot = self.draft_cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
        self._draft_slots[target_slot] = draft_slot

        # Build input tensor [1, prompt_len].
        input_ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=device)

        # Create single-slot cache view and run forward pass.
        view = self.target_cache_pool.prefill_view(target_slot)
        req.state = RequestState.PREFILL
        logits = self.target_model(input_ids, kv_cache=view)

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
    def _prefill_batch(self, requests: list[Request]) -> list[StepOutput]:
        """Prefill multiple requests in one batched forward pass (target model).

        Also allocates draft cache slots.
        """
        device = self.config.device

        # Allocate cache slots for all requests.
        for req in requests:
            target_slot = self.target_cache_pool.allocate_slot(
                initial_tokens=len(req.prompt_token_ids)
            )
            req.slot_idx = target_slot
            draft_slot = self.draft_cache_pool.allocate_slot(
                initial_tokens=len(req.prompt_token_ids)
            )
            self._draft_slots[target_slot] = draft_slot

        # Right-pad prompts to max length.
        prompt_lens = [len(req.prompt_token_ids) for req in requests]
        max_len = max(prompt_lens)
        padded = [
            req.prompt_token_ids + [0] * (max_len - len(req.prompt_token_ids)) for req in requests
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # Padding mask.
        batch_size = len(requests)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        for i, plen in enumerate(prompt_lens):
            padding_mask[i, :plen] = True

        # Create batched cache view and run forward pass.
        slots = [req.slot_idx for req in requests]
        assert all(s is not None for s in slots)
        view = self.target_cache_pool.batched_prefill_view(
            slots,  # type: ignore[arg-type]
            prompt_lens,
        )
        for req in requests:
            req.state = RequestState.PREFILL
        logits = self.target_model(input_ids, kv_cache=view, padding_mask=padding_mask)

        # Sample first token per request.
        outputs: list[StepOutput] = []
        for i, req in enumerate(requests):
            next_logits = logits[i, prompt_lens[i] - 1, :]
            context = req.prompt_token_ids
            token = sample_token(next_logits, context, req.sampling_params, req.generator)
            req.generated_token_ids.append(token)
            req.state = RequestState.DECODE

            text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
            text_delta = text
            self._prev_text_lens[req.request_id] = len(text)

            finished, reason = check_stop(req, token, self.tokenizer)
            if finished:
                req.state = RequestState.FINISHED
                req.finish_reason = reason
                if reason == "stop":
                    text_delta = truncate_at_stop(text, 0, req)

            outputs.append(make_step_output(req, token, text_delta, finished, reason))

        return outputs

    @torch.inference_mode()
    def _prefill_chunks_batched(self, requests: list[Request]) -> list[StepOutput | None]:
        """Process one chunk per request (target model) with draft slot allocation.

        Returns StepOutput for requests completing prefill, None for intermediate chunks.
        """
        device = self.config.device
        chunk_size = self.config.prefill_chunk_size

        # Allocate slots for first chunks.
        for req in requests:
            if req.prefill_progress == 0:
                target_slot = self.target_cache_pool.allocate_slot(
                    initial_tokens=len(req.prompt_token_ids)
                )
                req.slot_idx = target_slot
                draft_slot = self.draft_cache_pool.allocate_slot(
                    initial_tokens=len(req.prompt_token_ids)
                )
                self._draft_slots[target_slot] = draft_slot
                req.state = RequestState.PREFILL

        # Compute per-request chunk bounds.
        outputs: list[StepOutput | None] = [None] * len(requests)
        chunk_indices: list[int] = []

        for i, req in enumerate(requests):
            if req.prefill_progress == len(req.prompt_token_ids):
                # Full hit (prefix caching) -- not typical for speculative runner.
                pass
            else:
                chunk_indices.append(i)

        if not chunk_indices:
            return outputs

        chunk_requests = [requests[i] for i in chunk_indices]

        start_positions: list[int] = []
        chunk_lens: list[int] = []
        chunk_ends: list[int] = []
        for req in chunk_requests:
            progress = req.prefill_progress
            prompt_len = len(req.prompt_token_ids)
            chunk_end = min(progress + chunk_size, prompt_len)
            start_positions.append(progress)
            chunk_lens.append(chunk_end - progress)
            chunk_ends.append(chunk_end)

        max_chunk_len = max(chunk_lens)
        max_kv_len = max(s + c for s, c in zip(start_positions, chunk_lens, strict=True))
        batch_size = len(chunk_requests)

        # Build padded input_ids.
        padded_tokens: list[list[int]] = []
        for j, req in enumerate(chunk_requests):
            chunk_tokens = req.prompt_token_ids[start_positions[j] : chunk_ends[j]]
            padded_tokens.append(chunk_tokens + [0] * (max_chunk_len - len(chunk_tokens)))
        input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device)

        # Build position_ids.
        position_ids = torch.zeros(batch_size, max_chunk_len, dtype=torch.long, device=device)
        for j in range(batch_size):
            position_ids[j, : chunk_lens[j]] = torch.arange(
                start_positions[j],
                start_positions[j] + chunk_lens[j],
                device=device,
            )

        # Build padding_mask.
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for j in range(batch_size):
            kv_len = start_positions[j] + chunk_lens[j]
            padding_mask[j, :kv_len] = True

        # Cache view.
        slots: list[int] = [req.slot_idx for req in chunk_requests]  # type: ignore[misc]
        view = self.target_cache_pool.batched_chunked_prefill_view(
            slots, start_positions, chunk_lens
        )

        # Forward pass.
        logits = self.target_model(
            input_ids,
            kv_cache=view,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )

        # Update progress and handle last chunks.
        for j, req in enumerate(chunk_requests):
            req.prefill_progress = chunk_ends[j]
            is_last = chunk_ends[j] == len(req.prompt_token_ids)

            if not is_last:
                continue

            # Last chunk: sample first token.
            last_pos = chunk_lens[j] - 1
            next_logits = logits[j, last_pos, :]
            context = req.prompt_token_ids
            token = sample_token(next_logits, context, req.sampling_params, req.generator)
            req.generated_token_ids.append(token)
            req.state = RequestState.DECODE

            text = self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)
            text_delta = text
            self._prev_text_lens[req.request_id] = len(text)

            finished, reason = check_stop(req, token, self.tokenizer)
            if finished:
                req.state = RequestState.FINISHED
                req.finish_reason = reason
                if reason == "stop":
                    text_delta = truncate_at_stop(text, 0, req)

            outputs[chunk_indices[j]] = make_step_output(req, token, text_delta, finished, reason)

        return outputs

    # ------------------------------------------------------------------
    # Speculative decode (core loop)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _speculative_decode(self, requests: list[Request]) -> list[tuple[Request, StepOutput]]:
        """Run draft-then-verify speculative decode for a batch of requests.

        Returns a list of ``(request, step_output)`` pairs. Each request may
        produce 1 to ``spec_length + 1`` outputs.
        """
        if not requests:
            return []

        spec_length = self.config.spec_length

        # Compute effective spec length per request (overflow guard).
        effective_spec: dict[str, int] = {}
        for req in requests:
            assert req.slot_idx is not None
            current_len = self.target_cache_pool.get_seq_len(req.slot_idx)
            remaining = self.config.max_seq_len - current_len
            # Need K+1 positions in the target cache for verification.
            if remaining <= 1:
                effective_spec[req.request_id] = 0
            else:
                effective_spec[req.request_id] = min(spec_length, remaining - 1)

        # Separate requests that can speculate from those that fall back.
        spec_requests = [r for r in requests if effective_spec[r.request_id] > 0]
        fallback_requests = [r for r in requests if effective_spec[r.request_id] == 0]

        outputs: list[tuple[Request, StepOutput]] = []

        # Handle fallback requests with single-token decode.
        if fallback_requests:
            fallback_outputs = self._single_token_decode(fallback_requests)
            outputs.extend(fallback_outputs)

        if not spec_requests:
            return outputs

        # Record start positions for rollback.
        draft_start_positions: dict[str, int] = {}
        target_start_positions: dict[str, int] = {}
        for req in spec_requests:
            assert req.slot_idx is not None
            draft_slot = self._draft_slots[req.slot_idx]
            target_start_positions[req.request_id] = self.target_cache_pool.get_seq_len(
                req.slot_idx
            )
            draft_start_positions[req.request_id] = self.draft_cache_pool.get_seq_len(draft_slot)

        # Phase 1: Draft generation.
        draft_tokens, draft_log_probs = self._draft_generate(spec_requests, effective_spec)

        # Phase 2: Target verification.
        target_logits = self._verify(spec_requests, draft_tokens, effective_spec)

        # Phase 3: Accept/reject.
        accepted_tokens = self._accept_reject(
            spec_requests, draft_tokens, draft_log_probs, target_logits, effective_spec
        )

        # Phase 4: KV cache rollback.
        self._rollback_kv_caches(
            spec_requests,
            accepted_tokens,
            draft_start_positions,
            target_start_positions,
            effective_spec,
        )

        # Phase 5: Emit StepOutputs and update request state.
        for req in spec_requests:
            tokens = accepted_tokens[req.request_id]
            num_accepted_draft = (
                len(tokens) - 1
                if len(tokens) <= effective_spec[req.request_id]
                else effective_spec[req.request_id]
            )
            # Record acceptance rate.
            k = effective_spec[req.request_id]
            if k > 0:
                req.speculation_acceptance_rates.append(num_accepted_draft / k)

            for token in tokens:
                req.generated_token_ids.append(token)

                # Compute text delta.
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

                # Add acceptance rate on final output.
                step_out = make_step_output(req, token, text_delta, finished, reason)
                if finished and req.speculation_acceptance_rates:
                    step_out.acceptance_rate = sum(req.speculation_acceptance_rates) / len(
                        req.speculation_acceptance_rates
                    )

                outputs.append((req, step_out))

                # Stop emitting further tokens if this request is done.
                if finished:
                    break

        return outputs

    # ------------------------------------------------------------------
    # Draft generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _draft_generate(
        self,
        requests: list[Request],
        effective_spec: dict[str, int],
    ) -> tuple[dict[str, list[int]], dict[str, list[Tensor]]]:
        """Run K autoregressive steps on the draft model.

        Returns per-request draft tokens and draft log-probability tensors.

        Args:
            requests: Decode-phase requests.
            effective_spec: Per-request effective spec length.

        Returns:
            Tuple of (draft_tokens, draft_log_probs) dicts keyed by request_id.
        """
        device = self.config.device
        max_k = max(effective_spec[r.request_id] for r in requests)

        # Lazy draft prefill: populate draft KV cache for new requests.
        needs_prefill = [r for r in requests if r.request_id not in self._draft_prefilled]
        if needs_prefill:
            self._draft_prefill(needs_prefill)

        draft_tokens: dict[str, list[int]] = {r.request_id: [] for r in requests}
        draft_log_probs: dict[str, list[Tensor]] = {r.request_id: [] for r in requests}

        # Start with each request's last generated token.
        current_tokens: dict[str, int] = {}
        for req in requests:
            current_tokens[req.request_id] = req.generated_token_ids[-1]

        for step_k in range(max_k):
            # Filter to requests that still need drafting at this step.
            active = [r for r in requests if step_k < effective_spec[r.request_id]]
            if not active:
                break

            # Build input_ids [batch, 1].
            tokens = [current_tokens[r.request_id] for r in active]
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)

            # Build draft decode view.
            draft_slots = [self._draft_slots[r.slot_idx] for r in active]  # type: ignore[index]
            draft_view = self.draft_cache_pool.decode_view(draft_slots)

            # Build position_ids and padding_mask.
            positions = [self.draft_cache_pool.get_seq_len(s) for s in draft_slots]
            position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(1)

            max_kv_len = draft_view.seq_len + 1
            batch_size = len(active)
            padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
            for i, slot in enumerate(draft_slots):
                padding_mask[i, : self.draft_cache_pool.get_seq_len(slot)] = True

            # Forward pass on draft model.
            logits = self.draft_model(
                input_ids,
                kv_cache=draft_view,
                padding_mask=padding_mask,
                position_ids=position_ids,
            )
            # logits: [batch, 1, vocab_size]

            # Sample draft tokens and store log-probs.
            for i, req in enumerate(active):
                raw_logits = logits[i, -1, :]

                if req.sampling_params.temperature == 0.0:
                    # Greedy: just pick argmax, no log-probs needed.
                    token = int(torch.argmax(raw_logits).item())
                    draft_tokens[req.request_id].append(token)
                    # Store empty tensor for greedy (not used in accept/reject).
                    draft_log_probs[req.request_id].append(torch.tensor([]))
                else:
                    # Apply sampling transforms to get adjusted logits.
                    context = (
                        req.prompt_token_ids
                        + req.generated_token_ids
                        + draft_tokens[req.request_id]
                    )
                    adjusted = _apply_sampling_transforms(raw_logits, context, req.sampling_params)
                    # Store full log-prob distribution for rejection sampling.
                    log_probs = F.log_softmax(adjusted, dim=-1)
                    draft_log_probs[req.request_id].append(log_probs)

                    # Sample from the adjusted distribution.
                    probs = F.softmax(adjusted, dim=-1)
                    token = int(
                        torch.multinomial(
                            probs.unsqueeze(0), num_samples=1, generator=req.generator
                        ).item()
                    )
                    draft_tokens[req.request_id].append(token)

                current_tokens[req.request_id] = token

        return draft_tokens, draft_log_probs

    @torch.inference_mode()
    def _draft_prefill(self, requests: list[Request]) -> None:
        """Populate draft KV cache for requests that haven't been draft-prefilled.

        Uses batched prefill on the draft model. Output logits are discarded.

        Args:
            requests: Requests needing draft prefill.
        """
        device = self.config.device

        if len(requests) == 1:
            req = requests[0]
            assert req.slot_idx is not None
            draft_slot = self._draft_slots[req.slot_idx]
            input_ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=device)
            view = self.draft_cache_pool.prefill_view(draft_slot)
            self.draft_model(input_ids, kv_cache=view)
            self._draft_prefilled.add(req.request_id)
        else:
            # Batched draft prefill.
            prompt_lens = [len(r.prompt_token_ids) for r in requests]
            max_len = max(prompt_lens)
            padded = [
                r.prompt_token_ids + [0] * (max_len - len(r.prompt_token_ids)) for r in requests
            ]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)

            batch_size = len(requests)
            padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
            for i, plen in enumerate(prompt_lens):
                padding_mask[i, :plen] = True

            draft_slots = [
                self._draft_slots[r.slot_idx]  # type: ignore[index]
                for r in requests
            ]
            view = self.draft_cache_pool.batched_prefill_view(draft_slots, prompt_lens)
            self.draft_model(input_ids, kv_cache=view, padding_mask=padding_mask)

            for req in requests:
                self._draft_prefilled.add(req.request_id)

    # ------------------------------------------------------------------
    # Target verification
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _verify(
        self,
        requests: list[Request],
        draft_tokens: dict[str, list[int]],
        effective_spec: dict[str, int],
    ) -> Tensor:
        """Run a single forward pass of the target model to verify draft tokens.

        The verification input is ``[g, t_1, ..., t_K]`` per request (K+1 tokens),
        where ``g`` is the last generated token (not yet fed through target) and
        ``t_i`` are the draft tokens.

        Args:
            requests: Decode-phase requests.
            draft_tokens: Per-request draft tokens.
            effective_spec: Per-request effective spec length.

        Returns:
            Target logits tensor of shape ``[batch, max_K+1, vocab_size]``.
        """
        device = self.config.device
        batch_size = len(requests)
        max_k = max(effective_spec[r.request_id] for r in requests)
        verify_len = max_k + 1  # g + K draft tokens

        # Build input_ids [batch, verify_len] with right-padding.
        padded_tokens: list[list[int]] = []
        chunk_lens: list[int] = []
        start_positions: list[int] = []

        for req in requests:
            k = effective_spec[req.request_id]
            # g = last generated token, not yet fed through target.
            g = req.generated_token_ids[-1]
            req_tokens = [g, *draft_tokens[req.request_id][:k]]
            actual_len = len(req_tokens)
            chunk_lens.append(actual_len)
            # Pad to verify_len.
            padded_tokens.append(req_tokens + [0] * (verify_len - actual_len))

            assert req.slot_idx is not None
            start_pos = self.target_cache_pool.get_seq_len(req.slot_idx)
            start_positions.append(start_pos)

        input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device)

        # Build position_ids [batch, verify_len].
        position_ids = torch.zeros(batch_size, verify_len, dtype=torch.long, device=device)
        for i, _req in enumerate(requests):
            actual_len = chunk_lens[i]
            position_ids[i, :actual_len] = torch.arange(
                start_positions[i],
                start_positions[i] + actual_len,
                device=device,
            )

        # Build padding_mask [batch, max_kv_len].
        max_kv_len = max(sp + cl for sp, cl in zip(start_positions, chunk_lens, strict=True))
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            kv_len = start_positions[i] + chunk_lens[i]
            padding_mask[i, :kv_len] = True

        # Cache view: chunked prefill-style for K+1 positions.
        target_slots = [req.slot_idx for req in requests]
        assert all(s is not None for s in target_slots)
        view = self.target_cache_pool.batched_chunked_prefill_view(
            target_slots,  # type: ignore[arg-type]
            start_positions,
            chunk_lens,
        )

        # Forward pass.
        target_logits = self.target_model(
            input_ids,
            kv_cache=view,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )
        # target_logits: [batch, verify_len, vocab_size]

        return target_logits

    # ------------------------------------------------------------------
    # Accept/reject
    # ------------------------------------------------------------------

    def _accept_reject(
        self,
        requests: list[Request],
        draft_tokens: dict[str, list[int]],
        draft_log_probs: dict[str, list[Tensor]],
        target_logits: Tensor,
        effective_spec: dict[str, int],
    ) -> dict[str, list[int]]:
        """Run accept/reject for each request.

        Args:
            requests: Decode-phase requests.
            draft_tokens: Per-request draft tokens.
            draft_log_probs: Per-request draft log-prob distributions.
            target_logits: Target logits [batch, max_K+1, vocab_size].
            effective_spec: Per-request effective spec length.

        Returns:
            Dict of request_id -> list of accepted token IDs (1 to K+1).
        """
        accepted_tokens: dict[str, list[int]] = {}

        for i, req in enumerate(requests):
            k = effective_spec[req.request_id]
            req_draft_tokens = draft_tokens[req.request_id]
            req_target_logits = target_logits[i]  # [verify_len, vocab_size]

            if req.sampling_params.temperature == 0.0:
                accepted = self._accept_reject_greedy(req_draft_tokens, req_target_logits, k)
            else:
                accepted = self._accept_reject_sampling(
                    req,
                    req_draft_tokens,
                    draft_log_probs[req.request_id],
                    req_target_logits,
                    k,
                )

            accepted_tokens[req.request_id] = accepted

        return accepted_tokens

    def _accept_reject_greedy(
        self,
        draft_tokens: list[int],
        target_logits: Tensor,
        k: int,
    ) -> list[int]:
        """Greedy accept/reject: accept while target argmax matches draft.

        Args:
            draft_tokens: K draft token IDs.
            target_logits: [verify_len, vocab_size] target logits.
            k: Number of draft tokens.

        Returns:
            List of accepted token IDs (1 to K+1).
        """
        accepted: list[int] = []
        for pos in range(k):
            target_token = int(torch.argmax(target_logits[pos]).item())
            if target_token == draft_tokens[pos]:
                accepted.append(draft_tokens[pos])
            else:
                # Reject: use target's choice as the correction.
                accepted.append(target_token)
                return accepted

        # All K accepted: sample bonus token from position K.
        bonus = int(torch.argmax(target_logits[k]).item())
        accepted.append(bonus)
        return accepted

    def _accept_reject_sampling(
        self,
        req: Request,
        draft_tokens: list[int],
        draft_log_probs: list[Tensor],
        target_logits: Tensor,
        k: int,
    ) -> list[int]:
        """Sampling-mode accept/reject with rejection sampling.

        Args:
            req: The request (provides sampling params and generator).
            draft_tokens: K draft token IDs.
            draft_log_probs: K draft log-prob distributions [vocab_size].
            target_logits: [verify_len, vocab_size] target logits.
            k: Number of draft tokens.

        Returns:
            List of accepted token IDs (1 to K+1).
        """
        accepted: list[int] = []
        params = req.sampling_params

        for pos in range(k):
            # Build context for this verification position.
            context = req.prompt_token_ids + req.generated_token_ids + accepted

            # Apply sampling transforms to target logits at this position.
            target_adjusted = _apply_sampling_transforms(target_logits[pos], context, params)
            p_target = F.softmax(target_adjusted, dim=-1)

            # Get draft probabilities (already adjusted during drafting).
            q_draft = torch.exp(draft_log_probs[pos])

            draft_tok = draft_tokens[pos]

            # Compute acceptance ratio.
            p_tok = p_target[draft_tok].item()
            q_tok = q_draft[draft_tok].item()

            if q_tok <= 0:
                # Draft assigned zero probability -- reject and sample from target.
                token = int(
                    torch.multinomial(p_target.unsqueeze(0), 1, generator=req.generator).item()
                )
                accepted.append(token)
                return accepted

            ratio = min(1.0, p_tok / q_tok)

            # Draw uniform random for acceptance test.
            u = torch.rand(1, generator=req.generator).item()

            if u < ratio:
                accepted.append(draft_tok)
            else:
                # Reject: sample from correction distribution.
                diff = torch.clamp(p_target - q_draft, min=0.0)
                total = diff.sum()
                if total <= 0:
                    # Fallback to target distribution.
                    token = int(
                        torch.multinomial(p_target.unsqueeze(0), 1, generator=req.generator).item()
                    )
                else:
                    correction = diff / total
                    token = int(
                        torch.multinomial(
                            correction.unsqueeze(0), 1, generator=req.generator
                        ).item()
                    )
                accepted.append(token)
                return accepted

        # All K accepted: sample bonus token from position K.
        context = req.prompt_token_ids + req.generated_token_ids + accepted
        target_adjusted = _apply_sampling_transforms(target_logits[k], context, params)
        bonus = sample_token(target_logits[k], context, params, req.generator)
        accepted.append(bonus)
        return accepted

    # ------------------------------------------------------------------
    # KV cache rollback
    # ------------------------------------------------------------------

    def _rollback_kv_caches(
        self,
        requests: list[Request],
        accepted_tokens: dict[str, list[int]],
        draft_start_positions: dict[str, int],
        target_start_positions: dict[str, int],
        effective_spec: dict[str, int],
    ) -> None:
        """Roll back draft and target KV caches after accept/reject.

        After the speculation round:
        - Draft cache advanced by K positions (K draft steps).
        - Target cache advanced by K+1 positions (verification pass).
        - We need to truncate both to start_pos + num_accepted + 1
          (keeping entries for the input token g plus accepted tokens).

        Args:
            requests: Speculative decode requests.
            accepted_tokens: Per-request accepted token lists.
            draft_start_positions: Per-request draft cache start positions.
            target_start_positions: Per-request target cache start positions.
            effective_spec: Per-request effective spec length.
        """
        for req in requests:
            assert req.slot_idx is not None
            k = effective_spec[req.request_id]
            n_accepted_tokens = len(accepted_tokens[req.request_id])

            # Count how many draft tokens were accepted (excluding bonus/correction).
            # All K accepted + bonus -> K; otherwise last token is correction.
            num_accepted_draft = k if n_accepted_tokens == k + 1 else n_accepted_tokens - 1

            # Valid cache entries: g (the input token) + num_accepted_draft accepted tokens.
            # = num_accepted_draft + 1 new entries from this round.
            new_valid = num_accepted_draft + 1

            # Draft cache: was at draft_start, advanced by K.
            draft_slot = self._draft_slots[req.slot_idx]
            draft_target_len = draft_start_positions[req.request_id] + new_valid
            self.draft_cache_pool.truncate_to(draft_slot, draft_target_len)

            # Target cache: was at target_start, advanced by K+1.
            target_target_len = target_start_positions[req.request_id] + new_valid
            self.target_cache_pool.truncate_to(req.slot_idx, target_target_len)

    # ------------------------------------------------------------------
    # Single-token decode fallback
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _single_token_decode(self, requests: list[Request]) -> list[tuple[Request, StepOutput]]:
        """Fallback: single-token decode when speculation is not possible.

        Used when a request has insufficient remaining capacity for speculation.

        Args:
            requests: Requests to decode with a single token.

        Returns:
            List of (request, step_output) pairs.
        """
        device = self.config.device
        batch_size = len(requests)

        # Build input_ids from last generated token.
        tokens = [req.generated_token_ids[-1] for req in requests]
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)

        # Build decode view on target cache.
        target_slots = [req.slot_idx for req in requests]
        assert all(s is not None for s in target_slots)
        slots: list[int] = target_slots  # type: ignore[assignment]

        positions = [self.target_cache_pool.get_seq_len(s) for s in slots]
        position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(1)

        decode_view = self.target_cache_pool.decode_view(slots)
        max_kv_len = decode_view.seq_len + 1
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for i, slot in enumerate(slots):
            padding_mask[i, : self.target_cache_pool.get_seq_len(slot)] = True

        logits = self.target_model(
            input_ids,
            kv_cache=decode_view,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )

        outputs: list[tuple[Request, StepOutput]] = []
        for i, req in enumerate(requests):
            context = req.prompt_token_ids + req.generated_token_ids
            token = sample_token(logits[i, -1, :], context, req.sampling_params, req.generator)
            req.generated_token_ids.append(token)

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
            outputs.append((req, make_step_output(req, token, text_delta, finished, reason)))

        return outputs
