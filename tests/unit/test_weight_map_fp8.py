"""Tests for weight map quantization scale tensor extensions (FP8 + INT8)."""

from __future__ import annotations

from infer.loader.config import ModelConfig
from infer.loader.weight_map import get_weight_map


def _make_config(model_type: str, num_layers: int = 2) -> ModelConfig:
    """Create a minimal ModelConfig for testing."""
    return ModelConfig(
        model_type=model_type,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=1024,
    )


class TestWeightMapFP8:
    """Tests for FP8 weight_scale_inv entries in weight maps."""

    def test_no_scales_without_quantization(self) -> None:
        """Without quantization, no weight_scale_inv entries."""
        config = _make_config("qwen3")
        wmap = get_weight_map(config, quantization=None)
        scale_keys = [k for k in wmap if "scale_inv" in k]
        assert scale_keys == []

    def test_scales_with_fp8_quantization(self) -> None:
        """With fp8 quantization, each linear layer gets weight_scale_inv."""
        config = _make_config("qwen3", num_layers=2)
        wmap = get_weight_map(config, quantization="fp8")
        scale_keys = sorted(k for k in wmap if "scale_inv" in k)

        # 7 linear layers per block (q/k/v/o + gate/up/down) x 2 layers = 14.
        assert len(scale_keys) == 14

        # Spot-check a few entries.
        assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in wmap
        assert wmap["model.layers.0.self_attn.q_proj.weight_scale_inv"] == (
            "layers.0.self_attn.q_proj.weight_scale_inv"
        )
        assert "model.layers.1.mlp.down_proj.weight_scale_inv" in wmap
        assert wmap["model.layers.1.mlp.down_proj.weight_scale_inv"] == (
            "layers.1.mlp.down_proj.weight_scale_inv"
        )

    def test_llama_fp8_scales(self) -> None:
        """Llama architecture also gets FP8 scale entries."""
        config = _make_config("llama", num_layers=1)
        wmap = get_weight_map(config, quantization="fp8")
        scale_keys = [k for k in wmap if "scale_inv" in k]
        assert len(scale_keys) == 7  # q/k/v/o + gate/up/down

    def test_gemma3_fp8_scales(self) -> None:
        """Gemma 3 architecture also gets FP8 scale entries."""
        config = _make_config("gemma3_text", num_layers=1)
        wmap = get_weight_map(config, quantization="fp8")
        scale_keys = [k for k in wmap if "scale_inv" in k]
        assert len(scale_keys) == 7

    def test_backward_compatible(self) -> None:
        """get_weight_map works without quantization parameter (default None)."""
        config = _make_config("qwen3")
        wmap = get_weight_map(config)
        assert "model.embed_tokens.weight" in wmap
        scale_keys = [k for k in wmap if "scale_inv" in k]
        assert scale_keys == []

    def test_non_quantized_layers_excluded(self) -> None:
        """embed_tokens, lm_head, and norms do NOT get scale entries."""
        config = _make_config("qwen3", num_layers=1)
        wmap = get_weight_map(config, quantization="fp8")

        # These should have weight entries but NOT scale entries.
        assert "model.embed_tokens.weight" in wmap
        assert "lm_head.weight" in wmap
        assert "model.layers.0.input_layernorm.weight" in wmap

        # None of the non-linear entries should have a scale counterpart.
        assert "model.embed_tokens.weight_scale_inv" not in wmap
        assert "lm_head.weight_scale_inv" not in wmap
        assert "model.layers.0.input_layernorm.weight_scale_inv" not in wmap


class TestWeightMapINT8:
    """Tests for INT8 weight_scale entries in weight maps."""

    def test_no_scales_without_quantization(self) -> None:
        """Without quantization, no weight_scale entries."""
        config = _make_config("qwen3")
        wmap = get_weight_map(config, quantization=None)
        scale_keys = [k for k in wmap if k.endswith("weight_scale")]
        assert scale_keys == []

    def test_scales_with_int8_quantization(self) -> None:
        """With int8 quantization, each linear layer gets weight_scale."""
        config = _make_config("qwen3", num_layers=2)
        wmap = get_weight_map(config, quantization="int8")
        scale_keys = sorted(k for k in wmap if k.endswith("weight_scale"))

        # 7 linear layers per block (q/k/v/o + gate/up/down) x 2 layers = 14.
        assert len(scale_keys) == 14

        # Spot-check a few entries.
        assert "model.layers.0.self_attn.q_proj.weight_scale" in wmap
        assert wmap["model.layers.0.self_attn.q_proj.weight_scale"] == (
            "layers.0.self_attn.q_proj.weight_scale"
        )
        assert "model.layers.1.mlp.down_proj.weight_scale" in wmap
        assert wmap["model.layers.1.mlp.down_proj.weight_scale"] == (
            "layers.1.mlp.down_proj.weight_scale"
        )

    def test_llama_int8_scales(self) -> None:
        """Llama architecture also gets INT8 scale entries."""
        config = _make_config("llama", num_layers=1)
        wmap = get_weight_map(config, quantization="int8")
        scale_keys = [k for k in wmap if k.endswith("weight_scale")]
        assert len(scale_keys) == 7

    def test_gemma3_int8_scales(self) -> None:
        """Gemma 3 architecture also gets INT8 scale entries."""
        config = _make_config("gemma3_text", num_layers=1)
        wmap = get_weight_map(config, quantization="int8")
        scale_keys = [k for k in wmap if k.endswith("weight_scale")]
        assert len(scale_keys) == 7

    def test_int8_no_fp8_scales(self) -> None:
        """INT8 quantization should NOT produce FP8 scale_inv entries."""
        config = _make_config("qwen3", num_layers=1)
        wmap = get_weight_map(config, quantization="int8")
        fp8_keys = [k for k in wmap if "scale_inv" in k]
        assert fp8_keys == []

    def test_non_quantized_layers_excluded(self) -> None:
        """embed_tokens, lm_head, and norms do NOT get INT8 scale entries."""
        config = _make_config("qwen3", num_layers=1)
        wmap = get_weight_map(config, quantization="int8")

        assert "model.embed_tokens.weight_scale" not in wmap
        assert "lm_head.weight_scale" not in wmap
        assert "model.layers.0.input_layernorm.weight_scale" not in wmap
