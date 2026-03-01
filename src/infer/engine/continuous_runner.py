"""Model runner for continuous batching (slotted and paged KV cache)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from infer.cache.paged import PagedKVCachePool
from infer.cache.protocol import CachePoolProtocol
from infer.cache.slotted import SlottedKVCache
from infer.engine.config import EngineConfig
from infer.engine.cuda_graph_runner import CUDAGraphRunner
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

    Manages a :class:`CachePoolProtocol`-typed pool (slotted or paged).
    Each engine step runs:

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

        model_config = getattr(model, "config", None)
        if model_config is None:
            raise TypeError("model must have a .config attribute")

        # Dispatch cache pool creation based on backend.
        if config.kv_cache_backend == "paged":
            num_gpu_blocks = config.num_gpu_blocks
            if num_gpu_blocks is None:
                num_gpu_blocks = config.max_batch_size * config.max_seq_len // config.block_size
            self.cache_pool: CachePoolProtocol = PagedKVCachePool.from_model_config(
                model_config,
                total_blocks=num_gpu_blocks,
                block_size=config.block_size,
                dtype=self.dtype,
                device=config.device,
                use_prefix_caching=config.use_prefix_caching,
            )
        else:
            self.cache_pool = SlottedKVCache.from_model_config(
                model_config,
                max_seq_len=config.max_seq_len,
                max_batch_size=config.max_batch_size,
                dtype=self.dtype,
                device=config.device,
            )

        # Per-request text tracking, keyed by request_id.
        self._prev_text_lens: dict[str, int] = {}

        # CUDA graph runner (Phase 9).
        self._cuda_graph_runner: CUDAGraphRunner | None = None
        if config.use_cuda_graphs:
            assert isinstance(self.cache_pool, PagedKVCachePool)
            self._cuda_graph_runner = CUDAGraphRunner(model, self.cache_pool, config)

    def step(
        self,
        prefill: list[Request],
        decode: list[Request],
    ) -> list[tuple[Request, StepOutput]]:
        """Run one engine step: decode first (prioritize ITL), then prefill.

        Returns a list of ``(request, step_output)`` pairs.
        """
        outputs: list[tuple[Request, StepOutput]] = []

        # Phase 1: Batched decode (prioritize inter-token latency).
        if decode:
            decode_outputs = self._batched_decode(decode)
            outputs.extend(zip(decode, decode_outputs, strict=True))

        # Phase 2: Prefill.
        if self.config.use_chunked_prefill:
            if prefill:
                chunk_outputs = self._prefill_chunks_batched(prefill)
                for req, output in zip(prefill, chunk_outputs, strict=True):
                    if output is not None:
                        outputs.append((req, output))
        else:
            # Existing Phase 5/6 prefill logic (unchanged).
            if len(prefill) == 1:
                output = self._prefill_one(prefill[0])
                outputs.append((prefill[0], output))
            elif len(prefill) > 1:
                prefill_outputs = self._prefill_batch(prefill)
                outputs.extend(zip(prefill, prefill_outputs, strict=True))

        return outputs

    def free_slot(self, slot_idx: int) -> None:
        """Release a cache slot and clean up associated resources."""
        self.cache_pool.free_slot(slot_idx)

    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request tracking state."""
        self._prev_text_lens.pop(request_id, None)

    def warmup_cuda_graphs(self) -> None:
        """Pre-capture CUDA graphs. Call before serving."""
        if self._cuda_graph_runner is not None:
            self._cuda_graph_runner.warmup()

    def free_kv_tokens(self) -> int | None:
        """Return available token capacity, or None for contiguous backend."""
        return self.cache_pool.free_token_capacity()

    @torch.inference_mode()
    def _prefill_one(self, req: Request) -> StepOutput:
        """Prefill a single request using PrefillCacheView."""
        device = self.config.device

        # Allocate a cache slot.
        slot = self.cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
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
        token = sample_token(
            next_logits,
            context,
            req.sampling_params,
            req.generator,
            structured_state=req.structured_output_state,
        )
        req.generated_token_ids.append(token)
        if req.structured_output_state is not None:
            req.structured_output_state.advance(token)
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
        """Prefill multiple requests in one batched forward pass.

        Right-pads prompts to the longest length and runs a single forward
        pass, amortizing weight loading across all new requests.
        """
        device = self.config.device

        # Allocate cache slots for all requests.
        slots: list[int] = []
        for req in requests:
            slot = self.cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
            req.slot_idx = slot
            slots.append(slot)

        # Right-pad prompts to max length.
        prompt_lens = [len(req.prompt_token_ids) for req in requests]
        max_len = max(prompt_lens)
        padded = [
            req.prompt_token_ids + [0] * (max_len - len(req.prompt_token_ids)) for req in requests
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)  # [batch, max_len]

        # Padding mask: True for valid positions.
        batch_size = len(requests)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        for i, plen in enumerate(prompt_lens):
            padding_mask[i, :plen] = True

        # Create batched cache view and run forward pass.
        view = self.cache_pool.batched_prefill_view(slots, prompt_lens)
        for req in requests:
            req.state = RequestState.PREFILL
        logits = self.model(input_ids, kv_cache=view, padding_mask=padding_mask)
        # logits: [batch, max_len, vocab_size] (last-pos opt skipped due to padding_mask)

        # Sample first token per request at its actual last position.
        outputs: list[StepOutput] = []
        for i, req in enumerate(requests):
            next_logits = logits[i, prompt_lens[i] - 1, :]
            context = req.prompt_token_ids
            token = sample_token(
                next_logits,
                context,
                req.sampling_params,
                req.generator,
                structured_state=req.structured_output_state,
            )
            req.generated_token_ids.append(token)
            if req.structured_output_state is not None:
                req.structured_output_state.advance(token)
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

            outputs.append(make_step_output(req, token, text_delta, finished, reason))

        return outputs

    @torch.inference_mode()
    def _prefill_chunks_batched(self, requests: list[Request]) -> list[StepOutput | None]:
        """Process one chunk per request in a single batched forward pass.

        Returns StepOutput for each request that completes prefill (last chunk),
        None for requests with intermediate chunks.

        When prefix caching is enabled, newly admitted requests use
        ``allocate_slot_with_prefix()`` to reuse cached KV blocks. If all
        complete blocks are cached (full hit on a block-aligned prompt),
        the request is handled via a single last-token forward pass instead
        of the normal chunked path.
        """
        device = self.config.device
        chunk_size = self.config.prefill_chunk_size

        # Allocate slots for first chunks.
        for req in requests:
            if req.prefill_progress == 0:
                if self.config.use_prefix_caching:
                    assert isinstance(self.cache_pool, PagedKVCachePool)
                    slot, matched = self.cache_pool.allocate_slot_with_prefix(req.prompt_token_ids)
                    req.slot_idx = slot
                    req.prefill_progress = matched  # Skip cached prefix
                else:
                    slot = self.cache_pool.allocate_slot(initial_tokens=len(req.prompt_token_ids))
                    req.slot_idx = slot
                req.state = RequestState.PREFILL

        # Separate full-hit requests (all complete blocks cached, block-aligned
        # prompt) from requests that need normal chunked prefill.
        outputs: list[StepOutput | None] = [None] * len(requests)
        chunk_indices: list[int] = []

        for i, req in enumerate(requests):
            if req.prefill_progress == len(req.prompt_token_ids):
                # Full hit: single last-token forward pass.
                outputs[i] = self._prefill_full_hit(req)
            else:
                assert req.prefill_progress < len(req.prompt_token_ids), (
                    f"prefill_progress ({req.prefill_progress}) exceeds "
                    f"prompt_len ({len(req.prompt_token_ids)})"
                )
                chunk_indices.append(i)

        if not chunk_indices:
            return outputs

        # Normal chunked prefill for remaining requests.
        chunk_requests = [requests[i] for i in chunk_indices]

        # Compute per-request chunk bounds.
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

        # Build padded input_ids [batch, max_chunk_len].
        padded_tokens: list[list[int]] = []
        for j, req in enumerate(chunk_requests):
            chunk_tokens = req.prompt_token_ids[start_positions[j] : chunk_ends[j]]
            padded_tokens.append(chunk_tokens + [0] * (max_chunk_len - len(chunk_tokens)))
        input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device)

        # Build position_ids [batch, max_chunk_len].
        # Real positions: [start_pos, start_pos + chunk_len). Padded positions: 0.
        position_ids = torch.zeros(batch_size, max_chunk_len, dtype=torch.long, device=device)
        for j in range(batch_size):
            position_ids[j, : chunk_lens[j]] = torch.arange(
                start_positions[j],
                start_positions[j] + chunk_lens[j],
                device=device,
            )

        # Build padding_mask [batch, max_kv_len]: True for valid KV positions.
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for j in range(batch_size):
            kv_len = start_positions[j] + chunk_lens[j]
            padding_mask[j, :kv_len] = True

        # Cache view.
        slots: list[int] = [req.slot_idx for req in chunk_requests]  # type: ignore[misc]
        assert all(s is not None for s in slots)
        view = self.cache_pool.batched_chunked_prefill_view(slots, start_positions, chunk_lens)

        # Forward pass.
        logits = self.model(
            input_ids,
            kv_cache=view,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )
        # logits: [batch, max_chunk_len, vocab_size]

        # Update progress and handle last chunks.
        for j, req in enumerate(chunk_requests):
            req.prefill_progress = chunk_ends[j]
            is_last = chunk_ends[j] == len(req.prompt_token_ids)

            if not is_last:
                continue

            # Insert prefix blocks into tree after last chunk.
            if self.config.use_prefix_caching:
                assert isinstance(self.cache_pool, PagedKVCachePool)
                assert req.slot_idx is not None
                self.cache_pool.insert_prefix(req.slot_idx, req.prompt_token_ids)

            # Last chunk: sample first token at actual last position.
            last_pos = chunk_lens[j] - 1
            next_logits = logits[j, last_pos, :]
            context = req.prompt_token_ids
            token = sample_token(
                next_logits,
                context,
                req.sampling_params,
                req.generator,
                structured_state=req.structured_output_state,
            )
            req.generated_token_ids.append(token)
            if req.structured_output_state is not None:
                req.structured_output_state.advance(token)
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

            outputs[chunk_indices[j]] = make_step_output(req, token, text_delta, finished, reason)

        return outputs

    @torch.inference_mode()
    def _prefill_full_hit(self, req: Request) -> StepOutput:
        """Handle a full prefix cache hit with a single last-token forward pass.

        When all complete blocks of a block-aligned prompt are cached,
        only one forward pass with the last prompt token is needed to
        produce logits. The chunked prefill view gathers the full cached
        KV for attention.
        """
        device = self.config.device
        assert isinstance(self.cache_pool, PagedKVCachePool)
        assert req.slot_idx is not None
        slot = req.slot_idx
        prompt_len = len(req.prompt_token_ids)

        # input_ids: just the last prompt token.
        input_ids = torch.tensor([[req.prompt_token_ids[-1]]], dtype=torch.long, device=device)
        # position_ids: last prompt position.
        position_ids = torch.tensor([[prompt_len - 1]], dtype=torch.long, device=device)
        # Cache view: start_pos = prompt_len - 1, chunk_len = 1.
        # Gathers all cached KV from [0, prompt_len) for attention.
        view = self.cache_pool.batched_chunked_prefill_view([slot], [prompt_len - 1], [1])
        # Padding mask: all positions valid.
        padding_mask = torch.ones(1, prompt_len, dtype=torch.bool, device=device)

        logits = self.model(
            input_ids,
            kv_cache=view,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )

        # Insert prefix blocks into tree (no-op for full hit since all
        # blocks were already in the tree from the prior match).
        self.cache_pool.insert_prefix(slot, req.prompt_token_ids)

        # Sample first token.
        next_logits = logits[0, -1, :]
        context = req.prompt_token_ids
        token = sample_token(
            next_logits,
            context,
            req.sampling_params,
            req.generator,
            structured_state=req.structured_output_state,
        )
        req.generated_token_ids.append(token)
        if req.structured_output_state is not None:
            req.structured_output_state.advance(token)
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
        # Try CUDA graph path.
        if self._cuda_graph_runner is not None:
            assert isinstance(self.cache_pool, PagedKVCachePool)
            logits = self._cuda_graph_runner.execute(requests, self.cache_pool)
            if logits is not None:
                return self._sample_decode(requests, logits)

        # Eager fallback (existing code).
        device = self.config.device
        batch_size = len(requests)

        # Gather active slots and build inputs.
        active_slots = [req.slot_idx for req in requests]
        assert all(s is not None for s in active_slots), "All decode requests must have a slot"
        slots: list[int] = active_slots  # type: ignore[assignment]

        tokens = [req.generated_token_ids[-1] for req in requests]
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)  # [batch, 1]

        # Build position_ids: each sequence's current position.
        positions = [self.cache_pool.get_seq_len(slot) for slot in slots]
        position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(
            1
        )  # [batch, 1]

        # Build padding mask: True for valid positions per sequence.
        decode_view = self.cache_pool.decode_view(slots)
        max_kv_len = decode_view.seq_len + 1  # +1 for the token about to be written
        padding_mask = torch.zeros(batch_size, max_kv_len, dtype=torch.bool, device=device)
        for i, slot in enumerate(slots):
            padding_mask[i, : self.cache_pool.get_seq_len(slot)] = True

        # Forward pass with position_ids for per-sequence RoPE.
        logits = self.model(
            input_ids,
            kv_cache=decode_view,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )
        # logits: [batch, 1, vocab_size]

        return self._sample_decode(requests, logits)

    def _sample_decode(self, requests: list[Request], logits: Tensor) -> list[StepOutput]:
        """Sample tokens and build StepOutputs from decode logits."""
        outputs: list[StepOutput] = []
        for i, req in enumerate(requests):
            context = req.prompt_token_ids + req.generated_token_ids
            token = sample_token(
                logits[i, -1, :],
                context,
                req.sampling_params,
                req.generator,
                structured_state=req.structured_output_state,
            )
            req.generated_token_ids.append(token)
            if req.structured_output_state is not None:
                req.structured_output_state.advance(token)

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
