"""End-to-end integration tests for speculative decoding.

Tests load real Llama-3.2-1B (draft) and Llama-3.2-3B (target) models and
verify correctness of speculative decoding with both greedy and sampling modes.

Marked ``@pytest.mark.slow`` — skip with ``pytest -m "not slow"``.
Requires CUDA and both models available locally.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from infer.engine.config import EngineConfig
from infer.engine.engine import Engine
from infer.engine.request import StepOutput
from infer.engine.sampler import SamplingParams
from infer.loader.chat_template import render_chat_template
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer

_TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

_PROMPTS = [
    [{"role": "user", "content": "What is 2+2? Answer in one word."}],
]


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for speculative decoding integration tests")


def _load_models(device: str, dtype: torch.dtype) -> tuple:
    """Load target and draft models, skipping if unavailable."""
    try:
        target_model, target_config = load_model(_TARGET_MODEL, dtype=dtype, device=device)
    except Exception as exc:
        pytest.skip(f"Could not load target model {_TARGET_MODEL}: {exc}")

    try:
        draft_model, _draft_config = load_model(_DRAFT_MODEL, dtype=dtype, device=device)
    except Exception as exc:
        pytest.skip(f"Could not load draft model {_DRAFT_MODEL}: {exc}")

    try:
        tokenizer = Tokenizer(_TARGET_MODEL)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {_TARGET_MODEL}: {exc}")

    return target_model, draft_model, tokenizer, target_config


def _make_speculative_engine(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    tokenizer: object,
    *,
    spec_length: int = 3,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    kv_cache_backend: str = "contiguous",
) -> Engine:
    """Create an Engine with speculative decoding enabled."""
    config = EngineConfig(
        model=_TARGET_MODEL,
        dtype="bfloat16",
        device="cuda",
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        batching_mode="continuous",
        kv_cache_backend=kv_cache_backend,
        use_speculative_decoding=True,
        draft_model=_DRAFT_MODEL,
        spec_length=spec_length,
        batch_wait_timeout_s=0.0,
    )
    return Engine.from_components(config, target_model, tokenizer, draft_model=draft_model)


def _make_normal_engine(
    target_model: torch.nn.Module,
    tokenizer: object,
    *,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    kv_cache_backend: str = "contiguous",
) -> Engine:
    """Create a normal Engine (no speculation) for comparison."""
    config = EngineConfig(
        model=_TARGET_MODEL,
        dtype="bfloat16",
        device="cuda",
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        batching_mode="continuous",
        kv_cache_backend=kv_cache_backend,
        batch_wait_timeout_s=0.0,
    )
    return Engine.from_components(config, target_model, tokenizer)


def _collect_output(queue: asyncio.Queue[StepOutput]) -> list[StepOutput]:
    """Drain all StepOutputs from a queue (non-async)."""
    outputs: list[StepOutput] = []
    while not queue.empty():
        outputs.append(queue.get_nowait())
    return outputs


def _run_engine_to_completion(
    engine: Engine,
    prompt_ids: list[int],
    sampling_params: SamplingParams,
    *,
    max_steps: int = 300,
) -> list[StepOutput]:
    """Add a single request and step the engine until done."""
    queue: asyncio.Queue[StepOutput] = asyncio.Queue()
    engine.add_request("req-0", prompt_ids, sampling_params, queue)

    all_outputs: list[StepOutput] = []
    for _ in range(max_steps):
        if not engine.has_work():
            break
        engine.step()
        all_outputs.extend(_collect_output(queue))
        if any(o.finished for o in all_outputs):
            break

    return all_outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_speculative_greedy_parity(device: str) -> None:
    """Greedy decode produces identical output with and without speculation.

    This is the key correctness test: speculative decoding with temperature=0
    must produce the exact same tokens as normal greedy decode.
    """
    _skip_if_no_cuda()
    dtype = torch.bfloat16
    target_model, draft_model, tokenizer, target_config = _load_models(device, dtype)

    for messages in _PROMPTS:
        prompt = render_chat_template(messages, model_type=target_config.model_type)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        params = SamplingParams(temperature=0.0, max_new_tokens=12)

        # Normal decode.
        normal_engine = _make_normal_engine(target_model, tokenizer)
        normal_outputs = _run_engine_to_completion(normal_engine, prompt_ids, params)
        normal_tokens = [o.token_id for o in normal_outputs if o.token_id is not None]
        del normal_engine
        torch.cuda.empty_cache()

        # Speculative decode.
        spec_engine = _make_speculative_engine(target_model, draft_model, tokenizer)
        spec_outputs = _run_engine_to_completion(spec_engine, prompt_ids, params)
        del spec_engine
        torch.cuda.empty_cache()
        spec_tokens = [o.token_id for o in spec_outputs if o.token_id is not None]

        # Must be identical.
        assert normal_tokens == spec_tokens, (
            f"Greedy parity failure for prompt '{messages[-1]['content'][:40]}...': "
            f"normal={normal_tokens[:10]}..., spec={spec_tokens[:10]}..."
        )


@pytest.mark.slow
def test_speculative_e2e_sampling(device: str) -> None:
    """Sampling mode produces coherent output and logs acceptance rate."""
    _skip_if_no_cuda()
    dtype = torch.bfloat16
    target_model, draft_model, tokenizer, target_config = _load_models(device, dtype)

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    prompt = render_chat_template(messages, model_type=target_config.model_type)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    params = SamplingParams(temperature=0.7, max_new_tokens=12, seed=42)

    engine = _make_speculative_engine(target_model, draft_model, tokenizer)
    outputs = _run_engine_to_completion(engine, prompt_ids, params)

    # Should have produced tokens.
    token_outputs = [o for o in outputs if o.token_id is not None]
    assert len(token_outputs) >= 1, "Expected at least one generated token"

    # Decode the text.
    token_ids = [o.token_id for o in token_outputs if o.token_id is not None]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert len(text) > 0, "Expected non-empty generated text"

    # The done event should have acceptance rate.
    done_outputs = [o for o in outputs if o.finished]
    assert len(done_outputs) == 1
    if done_outputs[0].acceptance_rate is not None:
        rate = done_outputs[0].acceptance_rate
        assert 0.0 <= rate <= 1.0, f"Acceptance rate out of range: {rate}"


@pytest.mark.slow
def test_speculative_with_continuous_batching(device: str) -> None:
    """Multiple concurrent requests with speculation all complete correctly."""
    _skip_if_no_cuda()
    dtype = torch.bfloat16
    target_model, draft_model, tokenizer, target_config = _load_models(device, dtype)

    engine = _make_speculative_engine(target_model, draft_model, tokenizer, max_batch_size=2)

    # Submit multiple requests.
    queues: list[asyncio.Queue[StepOutput]] = []
    prompts_text = [
        "What is 2+2?",
        "Name a color.",
    ]
    for i, prompt_text in enumerate(prompts_text):
        messages = [{"role": "user", "content": prompt_text}]
        prompt = render_chat_template(messages, model_type=target_config.model_type)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        params = SamplingParams(temperature=0.0, max_new_tokens=8)
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        queues.append(queue)
        engine.add_request(f"req-{i}", prompt_ids, params, queue)

    # Step until all complete.
    for _ in range(200):
        if not engine.has_work():
            break
        engine.step()

    # Verify all requests completed.
    for i, queue in enumerate(queues):
        outputs = _collect_output(queue)
        token_outputs = [o for o in outputs if o.token_id is not None]
        assert len(token_outputs) >= 1, f"Request {i} produced no tokens"

        done_outputs = [o for o in outputs if o.finished]
        assert len(done_outputs) == 1, (
            f"Request {i} did not finish (got {len(done_outputs)} done events)"
        )
