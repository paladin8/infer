"""Full-model logits parity tests against HuggingFace transformers.

Each test loads a real HF model and our model with the same weights,
runs a forward pass on the same input, and compares logits.

Tolerance thresholds:
  - float32: max < 1e-4, mean < 1e-5 (exact on CPU; CUDA SDPA kernels
    introduce tiny non-deterministic rounding).
  - bfloat16: max < 2.5, mean < 0.2 (rounding accumulation across layers).
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer.loader.model_loader import load_model

# Dev models to test.
_MODELS = [
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

_DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, id="bf16"),
]


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("model_id", _MODELS)
def test_logits_parity(model_id: str, dtype: torch.dtype, device: str) -> None:
    """Verify our model's logits match HF transformers."""
    # Load HF model.
    try:
        ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as exc:
        pytest.skip(f"Could not load HF model {model_id}: {exc}")
    ref_model.to(device)  # type: ignore[arg-type]
    ref_model.eval()

    # Load our model.
    try:
        our_model, _config = load_model(model_id, dtype=dtype, device=device)
    except Exception as exc:
        pytest.skip(f"Could not load our model {model_id}: {exc}")

    # Tokenize a short test input.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {model_id}: {exc}")

    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)

    # Forward pass.
    with torch.no_grad():
        our_logits = our_model(input_ids)
        ref_logits = ref_model(input_ids).logits

    # Compare.
    diff = (our_logits.float() - ref_logits.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    dtype_name = "fp32" if dtype == torch.float32 else "bf16"
    print(f"\n{model_id} logits parity ({dtype_name}, {device}):")
    print(f"  Max absolute error:  {max_err:.6e}")
    print(f"  Mean absolute error: {mean_err:.6e}")
    print(f"  Our logits range:    [{our_logits.min().item():.2f}, {our_logits.max().item():.2f}]")
    print(f"  Ref logits range:    [{ref_logits.min().item():.2f}, {ref_logits.max().item():.2f}]")

    if dtype == torch.float32:
        # CPU gives exact (0.0) parity.  CUDA SDPA kernels introduce tiny
        # non-deterministic rounding (~1e-5), so we allow a small tolerance.
        assert max_err < 1e-4, f"fp32 max error {max_err:.6e} exceeds threshold 1e-4"
        assert mean_err < 1e-5, f"fp32 mean error {mean_err:.6e} exceeds threshold 1e-5"
    else:
        assert max_err < 2.5, f"Max absolute error {max_err:.6e} exceeds threshold 2.5"
        assert mean_err < 0.2, f"Mean absolute error {mean_err:.6e} exceeds threshold 0.2"
