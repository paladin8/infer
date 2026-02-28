"""Tests for FP8 block-wise quantized linear layer."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from infer.quant.fp8_linear import FP8Linear, fp8_block_dequant, replace_linear_with_fp8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_to_fp8(
    weight: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16/fp32 weight to FP8 + per-block scale (for testing).

    Mimics the quantization process used to create FP8 checkpoints:
    for each block_size x block_size block, compute scale = max(abs(block)) / 448.0,
    then quantize = (block / scale).clamp(-448, 448).to(float8_e4m3fn).

    Returns (weight_fp8, weight_scale_inv) where weight_scale_inv is the
    value you multiply by to dequantize.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    out_features, in_features = weight.shape
    pad_out = (block_size - out_features % block_size) % block_size
    pad_in = (block_size - in_features % block_size) % block_size

    w = weight.float()
    if pad_out > 0 or pad_in > 0:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))

    out_blocks = w.shape[0] // block_size
    in_blocks = w.shape[1] // block_size
    w = w.reshape(out_blocks, block_size, in_blocks, block_size)

    # Per-block max absolute value → scale.
    block_max = w.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = block_max / fp8_max  # [out_blocks, 1, in_blocks, 1]

    # Quantize: divide by scale, clamp to FP8 range, cast.
    w_scaled = (w / scale).clamp(-fp8_max, fp8_max)
    w_fp8 = w_scaled.reshape(out_blocks * block_size, in_blocks * block_size)[
        :out_features, :in_features
    ]
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)

    # Scale for dequantization (multiply to undo): weight_scale_inv = scale.
    # The name "inv" means inverse of quantization, i.e., the dequant multiplier.
    scale_inv = scale.squeeze(3).squeeze(1)  # [out_blocks, in_blocks]

    return w_fp8, scale_inv


# ---------------------------------------------------------------------------
# fp8_block_dequant tests
# ---------------------------------------------------------------------------


class TestFP8BlockDequant:
    """Tests for the fp8_block_dequant function."""

    def test_roundtrip_exact_multiple(self) -> None:
        """Quantize → dequantize roundtrip with dimensions that are exact multiples of 128."""
        torch.manual_seed(42)
        weight = torch.randn(256, 384, dtype=torch.bfloat16)
        w_fp8, scale = _quantize_to_fp8(weight, block_size=128)

        result = fp8_block_dequant(w_fp8, scale, block_size=128)

        assert result.dtype == torch.bfloat16
        assert result.shape == (256, 384)
        # FP8 has limited precision, so allow generous tolerance.
        torch.testing.assert_close(result, weight, atol=0.15, rtol=0.05)

    def test_roundtrip_non_multiple(self) -> None:
        """Quantize → dequantize roundtrip with dimensions NOT multiples of 128."""
        torch.manual_seed(42)
        weight = torch.randn(200, 300, dtype=torch.bfloat16)
        w_fp8, scale = _quantize_to_fp8(weight, block_size=128)

        result = fp8_block_dequant(w_fp8, scale, block_size=128)

        assert result.shape == (200, 300)
        torch.testing.assert_close(result, weight, atol=0.15, rtol=0.05)

    def test_small_block_size(self) -> None:
        """Test with a smaller block size for faster testing."""
        torch.manual_seed(42)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)
        w_fp8, scale = _quantize_to_fp8(weight, block_size=16)

        result = fp8_block_dequant(w_fp8, scale, block_size=16)

        assert result.shape == (32, 64)
        torch.testing.assert_close(result, weight, atol=0.15, rtol=0.05)

    def test_output_dtype(self) -> None:
        """Output is always bf16."""
        w_fp8 = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, 1, dtype=torch.float32)

        result = fp8_block_dequant(w_fp8, scale)
        assert result.dtype == torch.bfloat16

    def test_scale_applied_correctly(self) -> None:
        """Each block is multiplied by its scale."""
        # Create a simple 2-block case (block_size=4 for simplicity).
        w_fp8 = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[2.0, 3.0]], dtype=torch.float32)

        result = fp8_block_dequant(w_fp8, scale, block_size=4)

        # First 4 columns: 1.0 * 2.0 = 2.0
        torch.testing.assert_close(
            result[:, :4],
            torch.full((4, 4), 2.0, dtype=torch.bfloat16),
        )
        # Last 4 columns: 1.0 * 3.0 = 3.0
        torch.testing.assert_close(
            result[:, 4:],
            torch.full((4, 4), 3.0, dtype=torch.bfloat16),
        )


# ---------------------------------------------------------------------------
# FP8Linear tests
# ---------------------------------------------------------------------------


class TestFP8Linear:
    """Tests for the FP8Linear module."""

    def test_forward_shape(self) -> None:
        """Output shape matches standard linear."""
        layer = FP8Linear(256, 128, block_size=16)
        x = torch.randn(2, 10, 256, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (2, 10, 128)

    def test_forward_matches_dequantized_linear(self) -> None:
        """FP8Linear output should match nn.Linear with dequantized weights."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        block_size = 16

        # Create a reference weight, quantize it.
        ref_weight = torch.randn(out_f, in_f, dtype=torch.bfloat16)
        w_fp8, scale = _quantize_to_fp8(ref_weight, block_size=block_size)

        # Set up FP8Linear with the quantized weight.
        fp8_layer = FP8Linear(in_f, out_f, block_size=block_size)
        fp8_layer.weight = nn.Parameter(w_fp8, requires_grad=False)
        fp8_layer.weight_scale_inv = scale

        # Set up nn.Linear with the dequantized weight.
        dequant_weight = fp8_block_dequant(w_fp8, scale, block_size=block_size)
        ref_linear = nn.Linear(in_f, out_f, bias=False)
        ref_linear.weight = nn.Parameter(dequant_weight, requires_grad=False)

        x = torch.randn(2, 5, in_f, dtype=torch.bfloat16)
        torch.testing.assert_close(fp8_layer(x), ref_linear(x))

    def test_weight_dtype_preserved_after_to(self) -> None:
        """model.to(dtype=bf16) should not convert FP8 weights."""
        layer = FP8Linear(128, 64, block_size=16)
        assert layer.weight.dtype == torch.float8_e4m3fn

        layer.to(dtype=torch.bfloat16)
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale_inv.dtype == torch.float32

    def test_weight_dtype_preserved_after_to_device(self) -> None:
        """model.to(device=cpu) preserves FP8 weight dtype."""
        layer = FP8Linear(64, 32, block_size=16)
        assert layer.weight.dtype == torch.float8_e4m3fn

        layer.to(device="cpu")
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale_inv.dtype == torch.float32

    def test_extra_repr(self) -> None:
        """extra_repr shows in_features, out_features, block_size."""
        layer = FP8Linear(256, 128, block_size=64)
        r = layer.extra_repr()
        assert "in_features=256" in r
        assert "out_features=128" in r
        assert "block_size=64" in r
        assert "bias=False" in r


# ---------------------------------------------------------------------------
# replace_linear_with_fp8 tests
# ---------------------------------------------------------------------------


class TestReplaceLinearWithFP8:
    """Tests for the replace_linear_with_fp8 model surgery function."""

    def test_replaces_linear_layers(self) -> None:
        """Linear layers inside attention/mlp submodules are replaced."""

        class FakeAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 32, bias=False)

        class FakeBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = FakeAttn()
                self.input_layernorm = nn.Linear(64, 64, bias=False)  # norm — skip

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.layers = nn.ModuleList([FakeBlock()])
                self.lm_head = nn.Linear(64, 100, bias=False)

        model = FakeModel()
        replace_linear_with_fp8(model)

        block = model.layers[0]
        assert isinstance(block, FakeBlock)

        # Attention projections should be FP8Linear.
        assert isinstance(block.self_attn.q_proj, FP8Linear)
        assert isinstance(block.self_attn.k_proj, FP8Linear)

        # Norm layers should remain nn.Linear.
        assert isinstance(block.input_layernorm, nn.Linear)

        # lm_head should remain nn.Linear.
        assert isinstance(model.lm_head, nn.Linear)

    def test_raises_on_biased_linear(self) -> None:
        """Error if a linear layer to be replaced has bias."""

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64, bias=True)

        model = FakeModel()
        with pytest.raises(ValueError, match="does not support bias"):
            replace_linear_with_fp8(model)

    def test_preserves_embedding(self) -> None:
        """Embedding layers are not touched."""

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)

        model = FakeModel()
        replace_linear_with_fp8(model)
        assert isinstance(model.embed_tokens, nn.Embedding)

    def test_fp8_linear_dimensions(self) -> None:
        """Replaced FP8Linear has correct in_features/out_features."""

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(256, 128, bias=False)

        model = FakeModel()
        replace_linear_with_fp8(model)

        assert isinstance(model.proj, FP8Linear)
        assert model.proj.in_features == 256
        assert model.proj.out_features == 128


# ---------------------------------------------------------------------------
# _detect_quantization tests
# ---------------------------------------------------------------------------


class TestDetectQuantization:
    """Tests for auto-detection of quantization format from config."""

    def test_none_config(self) -> None:
        """No quantization_config → None."""
        from infer.loader.config import ModelConfig
        from infer.loader.model_loader import _detect_quantization

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=1024,
        )
        assert _detect_quantization(config) is None

    def test_fp8_config(self) -> None:
        """quant_method=fp8 → "fp8"."""
        from infer.loader.config import ModelConfig
        from infer.loader.model_loader import _detect_quantization

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=1024,
            quantization_config={"quant_method": "fp8", "fmt": "e4m3"},
        )
        assert _detect_quantization(config) == "fp8"

    def test_other_quant_method(self) -> None:
        """Unknown quant_method → None."""
        from infer.loader.config import ModelConfig
        from infer.loader.model_loader import _detect_quantization

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=1024,
            quantization_config={"quant_method": "gptq"},
        )
        assert _detect_quantization(config) is None
