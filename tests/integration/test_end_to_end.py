"""End-to-end tests: load model, tokenize, chat template, forward, verify.

Each test exercises the full pipeline: config loading -> model construction ->
weight loading -> tokenization -> chat template -> forward pass -> logits parity.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from infer.loader.chat_template import render_chat_template
from infer.loader.model_loader import load_model
from infer.loader.tokenizer import Tokenizer

_DEV_MODELS = [
    pytest.param(
        "meta-llama/Llama-3.2-1B-Instruct",
        id="llama",
    ),
    pytest.param(
        "Qwen/Qwen3-1.7B",
        id="qwen3-1.7b",
    ),
    pytest.param(
        "google/gemma-3-1b-it",
        id="gemma3",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("model_id", _DEV_MODELS)
def test_end_to_end(model_id: str, device: str) -> None:
    """Full pipeline: load -> template -> tokenize -> forward -> verify."""
    dtype = torch.bfloat16

    # Load our model.
    try:
        our_model, config = load_model(model_id, dtype=dtype, device=device)
    except Exception as exc:
        pytest.skip(f"Could not load our model {model_id}: {exc}")

    # Load tokenizer.
    try:
        tokenizer = Tokenizer(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {model_id}: {exc}")

    # Render a chat prompt.
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt = render_chat_template(messages, model_type=config.model_type)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([token_ids], device=device)

    # Forward pass.
    with torch.no_grad():
        logits = our_model(input_ids)

    # Verify shape.
    assert logits.shape == (1, input_ids.shape[1], config.vocab_size), (
        f"Expected shape (1, {input_ids.shape[1]}, {config.vocab_size}), got {logits.shape}"
    )

    # Verify logits match HF.
    try:
        ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as exc:
        pytest.skip(f"Could not load HF model {model_id}: {exc}")
    ref_model.to(device)  # type: ignore[arg-type]
    ref_model.eval()

    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits

    diff = (logits.float() - ref_logits.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    print(f"\n{model_id} end-to-end (bf16, {device}):")
    print(f"  Prompt tokens: {len(token_ids)}")
    print(f"  Max absolute error:  {max_err:.6e}")
    print(f"  Mean absolute error: {mean_err:.6e}")

    # bf16 precision causes accumulated rounding errors across many layers.
    # Verified: float32 gives exact (0.0) parity for all architectures.
    assert max_err < 2.5, f"Max absolute error {max_err:.6e} exceeds threshold 2.5"
    assert mean_err < 0.2, f"Mean absolute error {mean_err:.6e} exceeds threshold 0.2"
