"""Tests for the layer-by-layer activation diff tooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from infer.debug import ModelDiff, compare_models, format_diff
from infer.loader.config import ModelConfig
from infer.models.llama import LlamaModel


def _make_small_config() -> ModelConfig:
    """Create a small config for testing."""
    return ModelConfig(
        model_type="llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=50,
        max_position_embeddings=64,
    )


class _FakeHFOutput:
    """Minimal stand-in for HF model output with .logits attribute."""

    def __init__(self, logits: Tensor) -> None:
        self.logits = logits


class _FakeHFModel(nn.Module):
    """Minimal HF-like model with model.layers, model.norm, and model.embed_tokens.

    Wraps a LlamaModel to mimic the HF structure where the backbone
    is under ``model.model``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model = LlamaModel(config)

    def forward(self, input_ids: Tensor) -> _FakeHFOutput:
        logits = self.model(input_ids)
        return _FakeHFOutput(logits)


class TestCompareModels:
    def test_smoke_returns_model_diff(self) -> None:
        """compare_models returns a ModelDiff with the right number of layer diffs."""
        config = _make_small_config()
        our_model = LlamaModel(config)
        ref_model = _FakeHFModel(config)
        our_model.eval()
        ref_model.eval()

        input_ids = torch.randint(0, 50, (1, 8))
        diff = compare_models(our_model, ref_model, input_ids)

        assert isinstance(diff, ModelDiff)
        assert len(diff.layer_diffs) == 2  # 2 layers
        assert diff.embedding_diff is not None
        assert diff.final_norm_diff is not None
        assert diff.logits_diff is not None

    def test_identical_models_zero_error(self) -> None:
        """Two models with identical weights should have zero error."""
        config = _make_small_config()
        our_model = LlamaModel(config)
        ref_model = _FakeHFModel(config)

        # Copy our weights into the ref model so they are identical.
        ref_model.model.load_state_dict(our_model.state_dict())
        our_model.eval()
        ref_model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, 50, (1, 8))
        diff = compare_models(our_model, ref_model, input_ids)

        assert diff.embedding_diff.max_abs_error == 0.0
        for ld in diff.layer_diffs:
            assert ld.max_abs_error == 0.0
        assert diff.final_norm_diff.max_abs_error == 0.0
        assert diff.logits_diff.max_abs_error == 0.0


class TestFormatDiff:
    def test_produces_table(self) -> None:
        """format_diff returns a string with the expected table structure."""
        config = _make_small_config()
        our_model = LlamaModel(config)
        ref_model = _FakeHFModel(config)
        ref_model.model.load_state_dict(our_model.state_dict())
        our_model.eval()
        ref_model.eval()

        input_ids = torch.randint(0, 50, (1, 8))
        diff = compare_models(our_model, ref_model, input_ids)
        table = format_diff(diff)

        assert isinstance(table, str)
        lines = table.strip().split("\n")
        # Header + separator + embed + 2 layers + final_norm + logits = 7 lines.
        assert len(lines) == 7
        assert "embed" in lines[2]
        assert "layer_0" in lines[3]
        assert "layer_1" in lines[4]
        assert "final_norm" in lines[5]
        assert "logits" in lines[6]
