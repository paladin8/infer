"""Tests for the config reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from infer.loader.config import load_config

# ---------------------------------------------------------------------------
# Sample configs matching real HF config.json values
# ---------------------------------------------------------------------------

LLAMA_CONFIG: dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    },
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
}

QWEN3_CONFIG: dict[str, Any] = {
    "model_type": "qwen3",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
}

GEMMA3_CONFIG: dict[str, Any] = {
    "model_type": "gemma3_text",
    "hidden_size": 1152,
    "intermediate_size": 6912,
    "num_hidden_layers": 26,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "vocab_size": 262144,
    "max_position_embeddings": 32768,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_activation": "gelu_pytorch_tanh",  # Note: Gemma uses hidden_activation
    "query_pre_attn_scalar": 256,
    "sliding_window": 512,
    "sliding_window_pattern": 6,
    "tie_word_embeddings": True,
}


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    """Write a config dict to a config.json file and return the directory."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path


class TestLoadConfig:
    """Test loading configs for each architecture."""

    def test_llama(self, tmp_path: Path) -> None:
        model_dir = _write_config(tmp_path, LLAMA_CONFIG)
        cfg = load_config(model_dir)

        assert cfg.model_type == "llama"
        assert cfg.hidden_size == 2048
        assert cfg.intermediate_size == 8192
        assert cfg.num_hidden_layers == 16
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.vocab_size == 128256
        assert cfg.max_position_embeddings == 131072
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.rope_theta == 500000.0
        assert cfg.rope_scaling is not None
        assert cfg.rope_scaling["rope_type"] == "llama3"
        assert cfg.rope_scaling["factor"] == 32.0
        assert cfg.rope_scaling["high_freq_factor"] == 4.0
        assert cfg.rope_scaling["low_freq_factor"] == 1.0
        assert cfg.rope_scaling["original_max_position_embeddings"] == 8192
        assert cfg.attention_bias is False
        assert cfg.mlp_bias is False
        assert cfg.hidden_act == "silu"
        assert cfg.tie_word_embeddings is True
        assert cfg.head_dim is None
        assert cfg.layer_types is None  # no sliding window

    def test_qwen3(self, tmp_path: Path) -> None:
        model_dir = _write_config(tmp_path, QWEN3_CONFIG)
        cfg = load_config(model_dir)

        assert cfg.model_type == "qwen3"
        assert cfg.hidden_size == 2048
        assert cfg.intermediate_size == 6144
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.rope_theta == 1000000.0
        assert cfg.rope_scaling is None
        assert cfg.tie_word_embeddings is True
        assert cfg.layer_types is None  # no sliding window

    def test_gemma3(self, tmp_path: Path) -> None:
        model_dir = _write_config(tmp_path, GEMMA3_CONFIG)
        cfg = load_config(model_dir)

        assert cfg.model_type == "gemma3_text"
        assert cfg.hidden_size == 1152
        assert cfg.num_attention_heads == 4
        assert cfg.num_key_value_heads == 1
        assert cfg.head_dim == 256
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.rope_theta == 1000000.0
        assert cfg.rope_local_base_freq == 10000.0
        assert cfg.hidden_act == "gelu_pytorch_tanh"  # normalized from hidden_activation
        assert cfg.query_pre_attn_scalar == 256
        assert cfg.sliding_window == 512
        assert cfg.sliding_window_pattern == 6
        assert cfg.tie_word_embeddings is True

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        model_dir = _write_config(tmp_path, LLAMA_CONFIG)
        cfg = load_config(str(model_dir))
        assert cfg.model_type == "llama"


class TestUnsupportedModelType:
    """Test that unsupported model types fail fast."""

    def test_unknown_type_raises(self, tmp_path: Path) -> None:
        config = {**LLAMA_CONFIG, "model_type": "gpt2"}
        model_dir = _write_config(tmp_path, config)

        with pytest.raises(ValueError, match="Unsupported model_type: 'gpt2'"):
            load_config(model_dir)

    def test_missing_type_raises(self, tmp_path: Path) -> None:
        config = {k: v for k, v in LLAMA_CONFIG.items() if k != "model_type"}
        model_dir = _write_config(tmp_path, config)

        with pytest.raises(ValueError, match="Unsupported model_type"):
            load_config(model_dir)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        config = {"model_type": "llama", "hidden_size": 2048}
        model_dir = _write_config(tmp_path, config)

        with pytest.raises(TypeError):
            load_config(model_dir)


class TestConfigNormalization:
    """Test HF config normalization quirks."""

    def test_hidden_activation_mapped_to_hidden_act(self, tmp_path: Path) -> None:
        """Gemma 3 uses hidden_activation; reader should map it to hidden_act."""
        config = {**GEMMA3_CONFIG}
        assert "hidden_activation" in config
        assert "hidden_act" not in config

        model_dir = _write_config(tmp_path, config)
        cfg = load_config(model_dir)
        assert cfg.hidden_act == "gelu_pytorch_tanh"

    def test_nested_text_config(self, tmp_path: Path) -> None:
        """Gemma 3 multimodal configs nest text config under text_config."""
        nested = {
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                **{k: v for k, v in GEMMA3_CONFIG.items() if k != "model_type"},
            },
        }
        model_dir = _write_config(tmp_path, nested)
        cfg = load_config(model_dir)
        assert cfg.model_type == "gemma3_text"
        assert cfg.hidden_size == 1152

    def test_text_config_without_model_type_ignored(self, tmp_path: Path) -> None:
        """text_config without model_type is not a multimodal config — top level used."""
        config = {**LLAMA_CONFIG, "text_config": {"some_key": "some_value"}}
        model_dir = _write_config(tmp_path, config)
        cfg = load_config(model_dir)
        assert cfg.model_type == "llama"

    def test_extra_fields_ignored(self, tmp_path: Path) -> None:
        """Unknown fields in config.json should be silently ignored."""
        config = {**LLAMA_CONFIG, "torch_dtype": "bfloat16", "_name_or_path": "meta-llama/foo"}
        model_dir = _write_config(tmp_path, config)
        cfg = load_config(model_dir)
        assert cfg.model_type == "llama"

    def test_null_values_use_defaults(self, tmp_path: Path) -> None:
        """JSON null values for non-nullable fields should fall back to defaults."""
        config = {**QWEN3_CONFIG, "mlp_bias": None, "rope_scaling": None}
        model_dir = _write_config(tmp_path, config)
        cfg = load_config(model_dir)
        assert cfg.mlp_bias is False  # default, not None
        assert cfg.rope_scaling is None  # None is the default for this nullable field

    def test_does_not_mutate_input(self, tmp_path: Path) -> None:
        """Loading should not mutate the original config dict on disk (regression)."""
        config = {**GEMMA3_CONFIG}
        original_keys = set(config.keys())
        model_dir = _write_config(tmp_path, config)
        load_config(model_dir)
        # The dict in memory isn't passed directly, but verify the pattern:
        # re-reading should still have hidden_activation, not hidden_act
        with open(tmp_path / "config.json") as f:
            reloaded = json.load(f)
        assert set(reloaded.keys()) == original_keys

    def test_missing_config_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path)


class TestComputedHeadDim:
    """Test the computed_head_dim property."""

    def test_explicit_head_dim(self, tmp_path: Path) -> None:
        model_dir = _write_config(tmp_path, QWEN3_CONFIG)
        cfg = load_config(model_dir)
        assert cfg.head_dim == 128
        assert cfg.computed_head_dim == 128

    def test_inferred_head_dim(self, tmp_path: Path) -> None:
        model_dir = _write_config(tmp_path, LLAMA_CONFIG)
        cfg = load_config(model_dir)
        assert cfg.head_dim is None
        assert cfg.computed_head_dim == 2048 // 32  # 64

    def test_decoupled_head_dim(self, tmp_path: Path) -> None:
        """Gemma 3 1B: hidden_size=1152, num_heads=4, but head_dim=256 (not 288)."""
        model_dir = _write_config(tmp_path, GEMMA3_CONFIG)
        cfg = load_config(model_dir)
        assert cfg.computed_head_dim == 256
        assert cfg.hidden_size // cfg.num_attention_heads == 288  # would be wrong without head_dim


class TestLayerTypes:
    """Test layer_types resolution from sliding_window_pattern."""

    def test_resolved_from_sliding_window_pattern(self, tmp_path: Path) -> None:
        """Gemma 3: pattern=6 → every 6th layer (1-indexed) is full attention."""
        model_dir = _write_config(tmp_path, GEMMA3_CONFIG)
        cfg = load_config(model_dir)

        assert cfg.layer_types is not None
        assert len(cfg.layer_types) == 26  # num_hidden_layers

        # Layers 5, 11, 17, 23 (0-indexed) should be full_attention
        full_indices = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
        assert full_indices == [5, 11, 17, 23]

        # All others should be sliding_attention
        sliding_indices = [i for i, t in enumerate(cfg.layer_types) if t == "sliding_attention"]
        assert len(sliding_indices) == 22

    def test_no_resolution_without_pattern(self, tmp_path: Path) -> None:
        """Llama and Qwen don't use sliding window — layer_types stays None."""
        model_dir = _write_config(tmp_path, LLAMA_CONFIG)
        cfg = load_config(model_dir)
        assert cfg.layer_types is None

    def test_explicit_layer_types_preserved(self, tmp_path: Path) -> None:
        """If config already has layer_types, don't overwrite from pattern."""
        explicit_types = ["sliding_attention", "full_attention"] * 13
        config = {**GEMMA3_CONFIG, "layer_types": explicit_types}
        model_dir = _write_config(tmp_path, config)
        cfg = load_config(model_dir)
        assert cfg.layer_types == explicit_types
