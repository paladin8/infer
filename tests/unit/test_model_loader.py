"""Tests for the model loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from infer.loader.model_loader import _build_model, load_model
from infer.models.gemma3 import Gemma3Model
from infer.models.llama import LlamaModel
from infer.models.qwen3 import Qwen3Model

# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_llama(self) -> None:
        from infer.loader.config import ModelConfig

        config = ModelConfig(
            model_type="llama",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=128,
        )
        model = _build_model(config)
        assert isinstance(model, LlamaModel)

    def test_qwen3(self) -> None:
        from infer.loader.config import ModelConfig

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=128,
        )
        model = _build_model(config)
        assert isinstance(model, Qwen3Model)

    def test_gemma3(self) -> None:
        from infer.loader.config import ModelConfig

        config = ModelConfig(
            model_type="gemma3_text",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=128,
            head_dim=16,
            query_pre_attn_scalar=16.0,
        )
        model = _build_model(config)
        assert isinstance(model, Gemma3Model)

    def test_unsupported_type(self) -> None:
        from infer.loader.config import ModelConfig

        config = ModelConfig(
            model_type="unsupported",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=128,
        )
        with pytest.raises(ValueError, match="No model class"):
            _build_model(config)


# ---------------------------------------------------------------------------
# Tied embeddings test
# ---------------------------------------------------------------------------


class TestTiedEmbeddings:
    def test_missing_lm_head_filled_from_embed(self) -> None:
        """When tie_word_embeddings=True and lm_head.weight is missing from
        the checkpoint, load_model should copy embed_tokens.weight."""
        hidden = 32
        vocab = 50
        layers = 1
        heads = 2
        kv_heads = 2
        intermediate = 64

        # Build a config.json with tie_word_embeddings=True.
        config_data: dict[str, object] = {
            "model_type": "llama",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "vocab_size": vocab,
            "max_position_embeddings": 64,
            "tie_word_embeddings": True,
        }

        # Build a state dict with all expected weights EXCEPT lm_head.weight.
        from infer.loader.config import ModelConfig

        config = ModelConfig(
            model_type="llama",
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            vocab_size=vocab,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
        model = LlamaModel(config)
        full_sd = model.state_dict()

        # Remove lm_head.weight — this simulates a checkpoint that omits it.
        sd_without_lm_head = {k: v for k, v in full_sd.items() if k != "lm_head.weight"}

        # Add the "model." prefix back for HF naming convention.
        hf_sd: dict[str, torch.Tensor] = {}
        for k, v in sd_without_lm_head.items():
            if k == "embed_tokens.weight":
                hf_sd["model.embed_tokens.weight"] = v
            elif k == "norm.weight":
                hf_sd["model.norm.weight"] = v
            elif k.startswith("layers."):
                hf_sd[f"model.{k}"] = v

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write config.json.
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config_data))

            # Write safetensors without lm_head.weight.
            save_file(hf_sd, Path(tmpdir) / "model.safetensors")

            # load_model should succeed despite missing lm_head.weight.
            loaded_model, loaded_config = load_model(tmpdir, dtype=torch.float32, device="cpu")

            assert loaded_config.tie_word_embeddings is True
            # lm_head.weight should equal embed_tokens.weight.
            assert torch.equal(
                loaded_model.lm_head.weight,  # type: ignore[union-attr, arg-type]
                loaded_model.embed_tokens.weight,  # type: ignore[union-attr, arg-type]
            )
