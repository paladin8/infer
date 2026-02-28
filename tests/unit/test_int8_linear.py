"""Tests for INT8 per-channel quantized linear layer."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from infer.quant.int8_linear import INT8Linear, int8_channel_dequant, replace_linear_with_int8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_to_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16/fp32 weight to INT8 + per-channel scale (for testing).

    Per-channel symmetric quantization: for each row, compute
    scale = max(abs(row)) / 127, then quantize = round(row / scale).clamp(-128, 127).

    Returns (weight_int8, weight_scale) where weight_scale is the
    value you multiply by to dequantize.
    """
    w = weight.float()
    # Per-row max absolute value → scale.
    row_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [out, 1]
    scale = row_max / 127.0  # [out, 1]

    # Quantize: divide by scale, round, clamp to INT8 range, cast.
    w_scaled = (w / scale).round().clamp(-128, 127)
    w_int8 = w_scaled.to(torch.int8)

    return w_int8, scale


# ---------------------------------------------------------------------------
# int8_channel_dequant tests
# ---------------------------------------------------------------------------


class TestINT8ChannelDequant:
    """Tests for the int8_channel_dequant function."""

    def test_roundtrip(self) -> None:
        """Quantize → dequantize roundtrip."""
        torch.manual_seed(42)
        weight = torch.randn(256, 384, dtype=torch.bfloat16)
        w_int8, scale = _quantize_to_int8(weight)

        result = int8_channel_dequant(w_int8, scale)

        assert result.dtype == torch.bfloat16
        assert result.shape == (256, 384)
        # INT8 has limited precision, so allow generous tolerance.
        torch.testing.assert_close(result, weight, atol=0.05, rtol=0.02)

    def test_small_matrix(self) -> None:
        """Test with a small matrix."""
        torch.manual_seed(42)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)
        w_int8, scale = _quantize_to_int8(weight)

        result = int8_channel_dequant(w_int8, scale)

        assert result.shape == (32, 64)
        torch.testing.assert_close(result, weight, atol=0.05, rtol=0.02)

    def test_output_dtype(self) -> None:
        """Output is always bf16."""
        w_int8 = torch.zeros(128, 128, dtype=torch.int8)
        scale = torch.ones(128, 1, dtype=torch.float32)

        result = int8_channel_dequant(w_int8, scale)
        assert result.dtype == torch.bfloat16

    def test_scale_applied_correctly(self) -> None:
        """Each row is multiplied by its scale."""
        w_int8 = torch.ones(4, 8, dtype=torch.int8)
        scale = torch.tensor([[2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)

        result = int8_channel_dequant(w_int8, scale)

        for i, s in enumerate([2.0, 3.0, 4.0, 5.0]):
            torch.testing.assert_close(
                result[i],
                torch.full((8,), s, dtype=torch.bfloat16),
            )

    def test_negative_values(self) -> None:
        """Negative int8 values dequantize correctly."""
        w_int8 = torch.tensor([[-1, -2], [3, -4]], dtype=torch.int8)
        scale = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

        result = int8_channel_dequant(w_int8, scale)

        expected = torch.tensor([[-1.0, -2.0], [6.0, -8.0]], dtype=torch.bfloat16)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# INT8Linear tests
# ---------------------------------------------------------------------------


class TestINT8Linear:
    """Tests for the INT8Linear module."""

    def test_forward_shape(self) -> None:
        """Output shape matches standard linear."""
        layer = INT8Linear(256, 128)
        x = torch.randn(2, 10, 256, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (2, 10, 128)

    def test_forward_matches_dequantized_linear(self) -> None:
        """INT8Linear output should match nn.Linear with dequantized weights."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32

        # Create a reference weight, quantize it.
        ref_weight = torch.randn(out_f, in_f, dtype=torch.bfloat16)
        w_int8, scale = _quantize_to_int8(ref_weight)

        # Set up INT8Linear with the quantized weight.
        int8_layer = INT8Linear(in_f, out_f)
        int8_layer.weight = nn.Parameter(w_int8, requires_grad=False)
        int8_layer.weight_scale = scale

        # Set up nn.Linear with the dequantized weight.
        dequant_weight = int8_channel_dequant(w_int8, scale)
        ref_linear = nn.Linear(in_f, out_f, bias=False)
        ref_linear.weight = nn.Parameter(dequant_weight, requires_grad=False)

        x = torch.randn(2, 5, in_f, dtype=torch.bfloat16)
        torch.testing.assert_close(int8_layer(x), ref_linear(x))

    def test_weight_dtype_preserved_after_to(self) -> None:
        """model.to(dtype=bf16) should not convert INT8 weights."""
        layer = INT8Linear(128, 64)
        assert layer.weight.dtype == torch.int8

        layer.to(dtype=torch.bfloat16)
        assert layer.weight.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.float32

    def test_weight_dtype_preserved_after_to_device(self) -> None:
        """model.to(device=cpu) preserves INT8 weight dtype."""
        layer = INT8Linear(64, 32)
        assert layer.weight.dtype == torch.int8

        layer.to(device="cpu")
        assert layer.weight.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.float32

    def test_extra_repr(self) -> None:
        """extra_repr shows in_features, out_features."""
        layer = INT8Linear(256, 128)
        r = layer.extra_repr()
        assert "in_features=256" in r
        assert "out_features=128" in r
        assert "bias=False" in r


# ---------------------------------------------------------------------------
# replace_linear_with_int8 tests
# ---------------------------------------------------------------------------


class TestReplaceLinearWithINT8:
    """Tests for the replace_linear_with_int8 model surgery function."""

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
        replace_linear_with_int8(model)

        block = model.layers[0]
        assert isinstance(block, FakeBlock)

        # Attention projections should be INT8Linear.
        assert isinstance(block.self_attn.q_proj, INT8Linear)
        assert isinstance(block.self_attn.k_proj, INT8Linear)

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
            replace_linear_with_int8(model)

    def test_preserves_embedding(self) -> None:
        """Embedding layers are not touched."""

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)

        model = FakeModel()
        replace_linear_with_int8(model)
        assert isinstance(model.embed_tokens, nn.Embedding)

    def test_int8_linear_dimensions(self) -> None:
        """Replaced INT8Linear has correct in_features/out_features."""

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(256, 128, bias=False)

        model = FakeModel()
        replace_linear_with_int8(model)

        assert isinstance(model.proj, INT8Linear)
        assert model.proj.in_features == 256
        assert model.proj.out_features == 128


# ---------------------------------------------------------------------------
# _detect_quantization tests for INT8
# ---------------------------------------------------------------------------


class TestDetectQuantizationINT8:
    """Tests for auto-detection of INT8 quantization format from config."""

    def test_compressed_tensors_int_quantized(self) -> None:
        """compressed-tensors with int-quantized format → "int8"."""
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
            quantization_config={
                "quant_method": "compressed-tensors",
                "format": "int-quantized",
            },
        )
        assert _detect_quantization(config) == "int8"

    def test_compressed_tensors_other_format(self) -> None:
        """compressed-tensors with non-int format → None."""
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
            quantization_config={
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
            },
        )
        assert _detect_quantization(config) is None

    def test_fp8_still_detected(self) -> None:
        """FP8 detection still works alongside INT8."""
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


# ---------------------------------------------------------------------------
# _selective_to tests for INT8
# ---------------------------------------------------------------------------


class TestSelectiveToINT8:
    """Tests for _selective_to preserving INT8 dtypes."""

    def test_preserves_int8_weights_and_float32_scales(self) -> None:
        """_selective_to preserves int8 weight and float32 scale dtypes."""
        from infer.loader.model_loader import _selective_to

        model = nn.Module()
        layer = INT8Linear(64, 32)
        model.add_module("proj", layer)

        _selective_to(model, "cpu", torch.bfloat16)

        assert layer.weight.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.float32

    def test_converts_non_quantized_params(self) -> None:
        """_selective_to converts non-quantized params to target dtype."""
        from infer.loader.model_loader import _selective_to

        proj = INT8Linear(64, 32)
        norm = nn.Linear(64, 64, bias=False)
        norm.weight = nn.Parameter(torch.randn(64, 64, dtype=torch.float32))

        model = nn.Module()
        model.add_module("proj", proj)
        model.add_module("norm", norm)

        _selective_to(model, "cpu", torch.bfloat16)

        assert proj.weight.dtype == torch.int8
        assert norm.weight.dtype == torch.bfloat16
