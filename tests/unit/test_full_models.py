"""Unit tests for full model classes (LlamaModel, Qwen3Model, Gemma3Model).

Uses small configs with random weights to verify structure, shapes, and
architecture-specific behavior without downloading real model weights.
"""

from __future__ import annotations

import math

import torch

from infer.loader.config import ModelConfig
from infer.models.gemma3 import Gemma3Model
from infer.models.llama import LlamaModel
from infer.models.qwen3 import Qwen3Model

# ---------------------------------------------------------------------------
# Shared small configs
# ---------------------------------------------------------------------------

_LLAMA_CONFIG = ModelConfig(
    model_type="llama",
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    max_position_embeddings=64,
)

_QWEN3_CONFIG = ModelConfig(
    model_type="qwen3",
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    max_position_embeddings=64,
)

_GEMMA3_CONFIG = ModelConfig(
    model_type="gemma3_text",
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    max_position_embeddings=64,
    head_dim=16,
    query_pre_attn_scalar=16.0,
    sliding_window=4,
    sliding_window_pattern=4,
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
)


# ---------------------------------------------------------------------------
# LlamaModel
# ---------------------------------------------------------------------------


class TestLlamaModel:
    def test_output_shape(self) -> None:
        model = LlamaModel(_LLAMA_CONFIG)
        model.eval()
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_output_dtype_matches_model(self) -> None:
        model = LlamaModel(_LLAMA_CONFIG)
        model.eval()
        input_ids = torch.randint(0, 100, (2, 5))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.dtype == torch.float32

    def test_num_layers(self) -> None:
        model = LlamaModel(_LLAMA_CONFIG)
        assert len(model.layers) == 2

    def test_rope_buffers_registered(self) -> None:
        model = LlamaModel(_LLAMA_CONFIG)
        assert hasattr(model, "cos")
        assert hasattr(model, "sin")
        assert model.cos.shape[0] == 64  # max_position_embeddings
        assert model.sin.shape[0] == 64

    def test_batch_dimension(self) -> None:
        """Verify batch dim > 1 works correctly."""
        model = LlamaModel(_LLAMA_CONFIG)
        model.eval()
        input_ids = torch.randint(0, 100, (3, 6))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (3, 6, 100)


# ---------------------------------------------------------------------------
# Qwen3Model
# ---------------------------------------------------------------------------


class TestQwen3Model:
    def test_output_shape(self) -> None:
        model = Qwen3Model(_QWEN3_CONFIG)
        model.eval()
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_output_dtype_matches_model(self) -> None:
        model = Qwen3Model(_QWEN3_CONFIG)
        model.eval()
        input_ids = torch.randint(0, 100, (1, 5))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.dtype == torch.float32

    def test_num_layers(self) -> None:
        model = Qwen3Model(_QWEN3_CONFIG)
        assert len(model.layers) == 2

    def test_qk_norm_present(self) -> None:
        """Qwen3 blocks should have QK-norm."""
        model = Qwen3Model(_QWEN3_CONFIG)
        block = model.layers[0]
        assert block.self_attn.q_norm is not None  # type: ignore[union-attr]
        assert block.self_attn.k_norm is not None  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Gemma3Model
# ---------------------------------------------------------------------------


class TestGemma3Model:
    def test_output_shape(self) -> None:
        model = Gemma3Model(_GEMMA3_CONFIG)
        model.eval()
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (1, 8, 100)

    def test_output_dtype_matches_model(self) -> None:
        model = Gemma3Model(_GEMMA3_CONFIG)
        model.eval()
        input_ids = torch.randint(0, 100, (1, 5))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.dtype == torch.float32

    def test_num_layers(self) -> None:
        model = Gemma3Model(_GEMMA3_CONFIG)
        assert len(model.layers) == 4

    def test_embedding_scaling(self) -> None:
        """Gemma3 multiplies embeddings by sqrt(hidden_size)."""
        model = Gemma3Model(_GEMMA3_CONFIG)
        assert model.embedding_normalizer == math.sqrt(64)

    def test_dual_rope_buffers(self) -> None:
        """Gemma3 has 4 RoPE buffers: local_cos, local_sin, global_cos, global_sin."""
        model = Gemma3Model(_GEMMA3_CONFIG)
        for name in ("local_cos", "local_sin", "global_cos", "global_sin"):
            assert hasattr(model, name), f"Missing buffer: {name}"
            buf = getattr(model, name)
            assert buf.shape[0] == 64  # max_position_embeddings

    def test_layer_types_routing(self) -> None:
        """Verify layer_types are set from config."""
        model = Gemma3Model(_GEMMA3_CONFIG)
        assert model.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]

    def test_sliding_window_from_config(self) -> None:
        model = Gemma3Model(_GEMMA3_CONFIG)
        assert model.sliding_window == 4

    def test_gemma3_rms_norm_for_final_norm(self) -> None:
        """Gemma3 uses Gemma3RMSNorm (1+weight convention) for the final norm."""
        from infer.models.gemma3 import Gemma3RMSNorm

        model = Gemma3Model(_GEMMA3_CONFIG)
        assert isinstance(model.norm, Gemma3RMSNorm)

    def test_layer_types_default_when_none(self) -> None:
        """When layer_types is None, defaults to all full_attention."""
        config = ModelConfig(
            model_type="gemma3_text",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=64,
            head_dim=16,
            query_pre_attn_scalar=16.0,
        )
        model = Gemma3Model(config)
        assert model.layer_types == ["full_attention", "full_attention"]
