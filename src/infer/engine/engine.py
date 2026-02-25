"""Engine: orchestrates scheduler, model runner, and request lifecycle."""

from __future__ import annotations

import asyncio
import time
from typing import Self

import torch

from infer.engine.config import EngineConfig
from infer.engine.continuous_runner import ContinuousRunner
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.runner import ModelRunner
from infer.engine.sampler import SamplingParams
from infer.engine.scheduler import ContinuousScheduler, StaticScheduler
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer


class Engine:
    """Top-level orchestrator that ties scheduler and runner together.

    The engine owns the model, tokenizer, scheduler, and runner.  The API
    layer calls :meth:`add_request` and the engine loop calls :meth:`step`.

    The ``batching_mode`` config selects the scheduler and runner:

    - ``"static"``: :class:`StaticScheduler` + :class:`ModelRunner` (Phase 4).
    - ``"continuous"``: :class:`ContinuousScheduler` + :class:`ContinuousRunner` (Phase 5).
    """

    def __init__(self, config: EngineConfig) -> None:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
        model, _model_config = load_model(config.model, dtype=dtype, device=config.device)
        tokenizer = Tokenizer(config.model)
        self._init_components(config, model, tokenizer)

    @classmethod
    def from_components(
        cls,
        config: EngineConfig,
        model: torch.nn.Module,
        tokenizer: object,
    ) -> Self:
        """Create an engine from pre-built components (for testing)."""
        engine = object.__new__(cls)
        engine._init_components(config, model, tokenizer)  # type: ignore[arg-type]
        return engine

    def _init_components(
        self,
        config: EngineConfig,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
    ) -> None:
        """Initialize engine components based on batching mode."""
        self.config = config
        self.model_id = config.model
        self.tokenizer = tokenizer

        if config.batching_mode == "continuous":
            self.scheduler: StaticScheduler | ContinuousScheduler = ContinuousScheduler(config)
            self.runner: ModelRunner | ContinuousRunner = ContinuousRunner(model, tokenizer, config)
        else:
            self.scheduler = StaticScheduler(config)
            self.runner = ModelRunner(model, tokenizer, config)

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
        """Run one engine step: schedule and execute forward pass(es)."""
        if isinstance(self.scheduler, ContinuousScheduler):
            self._step_continuous()
        else:
            self._step_static()

    def has_work(self) -> bool:
        """True if the scheduler has active or waiting requests."""
        return self.scheduler.has_work()

    # ------------------------------------------------------------------
    # Static batching step (Phase 4)
    # ------------------------------------------------------------------

    def _step_static(self) -> None:
        assert isinstance(self.scheduler, StaticScheduler)
        assert isinstance(self.runner, ModelRunner)

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

    # ------------------------------------------------------------------
    # Continuous batching step (Phase 5+6)
    # ------------------------------------------------------------------

    def _step_continuous(self) -> None:
        """Engine step for continuous batching (contiguous and paged backends)."""
        assert isinstance(self.scheduler, ContinuousScheduler)
        assert isinstance(self.runner, ContinuousRunner)

        # Phase 1: Retire finished requests.
        retired = self.scheduler.retire()

        # Phase 2: Free cache resources for retired requests.
        for req in retired:
            if req.slot_idx is not None:
                self.runner.free_slot(req.slot_idx)
            self.runner.cleanup_request(req.request_id)

        # Phase 3: Query available memory budget (None for contiguous).
        free_kv_tokens = self.runner.free_kv_tokens()

        # Phase 4: Admit new requests with budget check.
        prefill = self.scheduler.admit(free_kv_tokens=free_kv_tokens)

        # Phase 5: Identify decode requests.
        decode = self.scheduler.decode_requests()

        if not prefill and not decode:
            return

        # Phase 6: Execute forward passes.
        try:
            outputs = self.runner.step(prefill, decode)
            for req, output in outputs:
                if req.output_queue is not None:
                    req.output_queue.put_nowait(output)

        except Exception as exc:
            for req in prefill + decode:
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
