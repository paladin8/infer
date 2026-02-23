"""Engine: orchestrates scheduler, model runner, and request lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import Self

import torch

from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.runner import ModelRunner
from infer.engine.sampler import SamplingParams
from infer.engine.scheduler import StaticScheduler
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer


class Engine:
    """Top-level orchestrator that ties scheduler and runner together.

    The engine owns the model, tokenizer, scheduler, and runner.  The API
    layer calls :meth:`add_request` and the engine loop calls :meth:`step`.
    """

    def __init__(self, config: EngineConfig) -> None:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
        model, _model_config = load_model(config.model, dtype=dtype, device=config.device)
        tokenizer = Tokenizer(config.model)

        self.config = config
        self.model_id = config.model
        self.tokenizer = tokenizer
        self.scheduler = StaticScheduler(config)
        self.runner = ModelRunner(model, tokenizer, config)

    @classmethod
    def from_components(
        cls,
        config: EngineConfig,
        model: torch.nn.Module,
        tokenizer: object,
    ) -> Self:
        """Create an engine from pre-built components (for testing)."""
        engine = object.__new__(cls)
        engine.config = config
        engine.model_id = config.model
        engine.tokenizer = tokenizer  # type: ignore[assignment]
        engine.scheduler = StaticScheduler(config)
        engine.runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        output_queue: asyncio.Queue[StepOutput],
    ) -> bool:
        """Enqueue a new completion request.

        Returns ``False`` if the waiting queue is full (caller should 503).
        Raises ``ValueError`` if the prompt + max_new_tokens exceeds max_seq_len.
        """
        prompt_token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt

        if len(prompt_token_ids) + sampling_params.max_new_tokens > self.config.max_seq_len:
            raise ValueError(
                f"prompt ({len(prompt_token_ids)} tokens) + max_new_tokens "
                f"({sampling_params.max_new_tokens}) exceeds max_seq_len ({self.config.max_seq_len})"
            )

        # Per-request RNG: request seed > global seed > None.
        generator: torch.Generator | None = None
        seed = sampling_params.seed if sampling_params.seed is not None else self.config.seed
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time_s=time.perf_counter(),
            generator=generator,
            output_queue=output_queue,
        )
        return self.scheduler.add_request(request)

    def step(self) -> None:
        """Run one engine step: schedule a batch and execute a forward pass."""
        batch = self.scheduler.schedule()
        if not batch:
            return

        try:
            is_new_batch = any(r.state == RequestState.WAITING for r in batch)
            outputs = self.runner.prefill(batch) if is_new_batch else self.runner.decode_step(batch)

            for req, output in zip(batch, outputs, strict=True):
                if req.output_queue is not None:
                    req.output_queue.put_nowait(output)

            if all(r.state in (RequestState.FINISHED, RequestState.FAILED) for r in batch):
                self.runner.clear_batch()

        except Exception as exc:
            for req in batch:
                req.state = RequestState.FAILED
                req.error = str(exc)
                if req.output_queue is not None:
                    req.output_queue.put_nowait(
                        StepOutput(
                            request_id=req.request_id,
                            token_id=None,
                            text_delta="",
                            finished=True,
                            finish_reason=None,
                            error=str(exc),
                        )
                    )
            self.runner.clear_batch()

    def has_work(self) -> bool:
        """True if the scheduler has active or waiting requests."""
        return self.scheduler.has_work()
