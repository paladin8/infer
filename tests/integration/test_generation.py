"""Integration tests for autoregressive generation with real models.

Tests load actual model weights and run generation end-to-end.
Marked ``@pytest.mark.slow`` — skip with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import pytest
import torch

from infer.engine.generate import generate
from infer.engine.sampler import SamplingParams
from infer.loader.chat_template import render_chat_template
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer

_DEV_MODELS = [
    pytest.param("meta-llama/Llama-3.2-1B-Instruct", id="llama"),
    pytest.param("Qwen/Qwen3-1.7B", id="qwen3-1.7b"),
    pytest.param("google/gemma-3-1b-it", id="gemma3"),
]

_PROMPTS = [
    [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    [{"role": "user", "content": "Write a haiku about programming."}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Name the four seasons."},
    ],
]


@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_functional_generation(model_id: str, device: str) -> None:
    """Generate responses to chat prompts and verify non-empty coherent output."""
    dtype = torch.bfloat16

    try:
        model, config = load_model(model_id, dtype=dtype, device=device)
    except Exception as exc:
        pytest.skip(f"Could not load model {model_id}: {exc}")

    try:
        tokenizer = Tokenizer(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {model_id}: {exc}")

    for messages in _PROMPTS:
        prompt = render_chat_template(messages, model_type=config.model_type)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        result = generate(
            model,
            tokenizer,
            prompt_ids,
            SamplingParams(temperature=0.0, max_new_tokens=64),
            device=device,
        )

        assert result.generated_tokens >= 1
        assert len(result.text) > 0
        assert result.finish_reason in ("eos", "stop", "length")
        assert result.timing.total_time_s > 0
        # Basic coherence: generated tokens are in valid range.
        for tid in result.token_ids:
            assert 0 <= tid < config.vocab_size


@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_deterministic_generation(model_id: str, device: str) -> None:
    """Greedy and seeded sampling produce identical results across runs."""
    dtype = torch.bfloat16

    try:
        model, config = load_model(model_id, dtype=dtype, device=device)
    except Exception as exc:
        pytest.skip(f"Could not load model {model_id}: {exc}")

    try:
        tokenizer = Tokenizer(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {model_id}: {exc}")

    prompt = render_chat_template(
        [{"role": "user", "content": "Count to five."}],
        model_type=config.model_type,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Greedy determinism.
    r1 = generate(
        model,
        tokenizer,
        prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=32),
        device=device,
    )
    r2 = generate(
        model,
        tokenizer,
        prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=32),
        device=device,
    )
    assert r1.token_ids == r2.token_ids

    # Seeded sampling determinism.
    r3 = generate(
        model,
        tokenizer,
        prompt_ids,
        SamplingParams(temperature=0.7, seed=42, max_new_tokens=32),
        device=device,
    )
    r4 = generate(
        model,
        tokenizer,
        prompt_ids,
        SamplingParams(temperature=0.7, seed=42, max_new_tokens=32),
        device=device,
    )
    assert r3.token_ids == r4.token_ids


@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_stop_string(model_id: str, device: str) -> None:
    """Stop string truncates output correctly."""
    dtype = torch.bfloat16

    try:
        model, config = load_model(model_id, dtype=dtype, device=device)
    except Exception as exc:
        pytest.skip(f"Could not load model {model_id}: {exc}")

    try:
        tokenizer = Tokenizer(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {model_id}: {exc}")

    prompt = render_chat_template(
        [{"role": "user", "content": "List three colors, one per line."}],
        model_type=config.model_type,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Generate WITHOUT stop string to confirm the model produces the pattern.
    baseline = generate(
        model,
        tokenizer,
        prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=128),
        device=device,
    )
    if "\n\n" not in baseline.text:
        pytest.skip("Model output does not contain stop pattern")

    # Generate WITH stop string and verify truncation.
    result = generate(
        model,
        tokenizer,
        prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=128, stop=["\n\n"]),
        device=device,
    )
    assert result.finish_reason == "stop"
    assert "\n\n" not in result.text
    assert result.generated_tokens < baseline.generated_tokens
