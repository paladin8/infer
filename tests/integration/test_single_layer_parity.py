"""Single-layer parity tests against HuggingFace reference models.

Each test loads a real HF model, extracts layer 0 weights, loads them
into our TransformerBlock, and compares outputs on random bf16 inputs.

Tests are marked ``@pytest.mark.slow`` and skip gracefully when models
are not accessible.

Tolerance thresholds (bf16):
- Max absolute error < 1e-2
- Mean absolute error < 1e-3
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from infer.models.common import causal_mask, sliding_window_causal_mask
from infer.models.gemma3 import Gemma3TransformerBlock
from infer.models.llama import LlamaTransformerBlock
from infer.models.qwen3 import Qwen3TransformerBlock

SEQ_LEN = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_rope(
    model: Any,
    seq_len: int,
    dtype: torch.dtype,
    layer_type: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Get RoPE (cos, sin) tables from an HF model's rotary_emb.

    Returns:
        ``(cos, sin)`` each of shape ``(1, seq_len, head_dim)``.
    """
    position_ids = torch.arange(seq_len).unsqueeze(0)
    hidden_size = model.config.hidden_size
    dummy = torch.zeros(1, seq_len, hidden_size, dtype=dtype)
    if layer_type is not None:
        cos, sin = model.model.rotary_emb(dummy, position_ids, layer_type)
    else:
        cos, sin = model.model.rotary_emb(dummy, position_ids)
    return cos, sin


def _load_hf_weights_into_block(
    hf_layer: Any,
    our_block: torch.nn.Module,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Copy matching weights from an HF decoder layer into our block.

    Converts our block to ``dtype`` after loading so dtypes match the HF layer.
    """
    our_keys = set(our_block.state_dict().keys())
    hf_state = hf_layer.state_dict()
    filtered = {k: v for k, v in hf_state.items() if k in our_keys}
    missing = our_keys - set(filtered.keys())
    assert not missing, f"Our block has weights not found in HF layer: {sorted(missing)}"
    our_block.load_state_dict(filtered, strict=True, assign=True)


def _assert_parity(
    hf_out: Tensor,
    our_out: Tensor,
    label: str,
    max_atol: float = 1e-2,
    mean_atol: float = 1e-3,
) -> None:
    """Compare outputs and assert within tolerance, printing actual errors."""
    diff = (hf_out - our_out).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(f"\n{label}: max_err={max_err:.6e}, mean_err={mean_err:.6e}")
    assert max_err < max_atol, f"{label}: max abs error {max_err:.6e} >= {max_atol}"
    assert mean_err < mean_atol, f"{label}: mean abs error {mean_err:.6e} >= {mean_atol}"


# ---------------------------------------------------------------------------
# Llama 3
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLlamaLayerParity:
    """Llama-3.2-1B-Instruct layer 0: pre-norm, no QK-norm, head_dim=64."""

    def test_layer0_bf16(self) -> None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                dtype=torch.bfloat16,
            )
        except Exception as exc:
            pytest.skip(f"Could not load Llama model: {exc}")

        cfg = model.config
        hf_layer = model.model.layers[0].eval()

        # RoPE from HF's rotary_emb.
        cos, sin = _extract_rope(model, SEQ_LEN, torch.bfloat16)

        # Build our block with matching config.
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        our_block = LlamaTransformerBlock(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        _load_hf_weights_into_block(hf_layer, our_block)
        our_block.eval()

        # Causal mask and random input.
        mask = causal_mask(SEQ_LEN).to(torch.bfloat16)
        torch.manual_seed(42)
        x = torch.randn(1, SEQ_LEN, cfg.hidden_size, dtype=torch.bfloat16)

        with torch.no_grad():
            hf_out = hf_layer(x, attention_mask=mask, position_embeddings=(cos, sin))
            if isinstance(hf_out, tuple):
                hf_out = hf_out[0]

            our_out = our_block(x, cos.squeeze(0), sin.squeeze(0), mask=mask)

        _assert_parity(hf_out, our_out, "Llama layer 0 (bf16)")


# ---------------------------------------------------------------------------
# Qwen 3
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestQwen3LayerParity:
    """Qwen3-1.7B layer 0: QK-norm, head_dim=128, vanilla RoPE with theta=1M."""

    def test_layer0_bf16(self) -> None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-1.7B",
                dtype=torch.bfloat16,
            )
        except Exception as exc:
            pytest.skip(f"Could not load Qwen3 model: {exc}")

        cfg = model.config
        hf_layer = model.model.layers[0].eval()

        # RoPE from HF's rotary_emb.
        cos, sin = _extract_rope(model, SEQ_LEN, torch.bfloat16)

        # Build our block with matching config.
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        our_block = Qwen3TransformerBlock(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        _load_hf_weights_into_block(hf_layer, our_block)
        our_block.eval()

        # Causal mask and random input.
        mask = causal_mask(SEQ_LEN).to(torch.bfloat16)
        torch.manual_seed(42)
        x = torch.randn(1, SEQ_LEN, cfg.hidden_size, dtype=torch.bfloat16)

        with torch.no_grad():
            hf_out = hf_layer(x, attention_mask=mask, position_embeddings=(cos, sin))
            if isinstance(hf_out, tuple):
                hf_out = hf_out[0]

            our_out = our_block(x, cos.squeeze(0), sin.squeeze(0), mask=mask)

        _assert_parity(hf_out, our_out, "Qwen3 layer 0 (bf16)")


# ---------------------------------------------------------------------------
# Gemma 3
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGemma3LayerParity:
    """gemma-3-1b-it layer 0: sandwich norm, QK-norm, GeGLU, sliding window."""

    def test_layer0_bf16(self) -> None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-3-1b-it",
                dtype=torch.bfloat16,
            )
        except Exception as exc:
            pytest.skip(f"Could not load Gemma3 model: {exc}")

        cfg = model.config
        hf_layer = model.model.layers[0].eval()

        # Layer 0 is sliding_attention (pattern=6 → layers 0-4 are local).
        layer_type = "sliding_attention"
        cos, sin = _extract_rope(model, SEQ_LEN, torch.bfloat16, layer_type=layer_type)

        # Build our block with matching config.
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        sliding_window = getattr(cfg, "sliding_window", 512)
        our_block = Gemma3TransformerBlock(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            query_pre_attn_scalar=cfg.query_pre_attn_scalar,
        )
        _load_hf_weights_into_block(hf_layer, our_block)
        our_block.eval()

        # Sliding window causal mask and random input.
        mask = sliding_window_causal_mask(SEQ_LEN, sliding_window).to(torch.bfloat16)
        torch.manual_seed(42)
        x = torch.randn(1, SEQ_LEN, cfg.hidden_size, dtype=torch.bfloat16)

        with torch.no_grad():
            hf_out = hf_layer(
                x,
                attention_mask=mask,
                position_embeddings=(cos, sin),
            )
            if isinstance(hf_out, tuple):
                hf_out = hf_out[0]

            our_out = our_block(x, cos.squeeze(0), sin.squeeze(0), mask=mask)

        _assert_parity(hf_out, our_out, "Gemma3 layer 0 (bf16)")
