"""Tests for the weight name mapping."""

from __future__ import annotations

import json

import pytest

from infer.loader.config import ModelConfig
from infer.loader.weight_map import (
    gemma3_weight_map,
    get_weight_map,
    llama_weight_map,
    qwen3_weight_map,
)

# Use small layer counts to keep tests readable.
NUM_LAYERS = 2


def _layer_keys(mapping: dict[str, str], layer: int) -> set[str]:
    """Extract internal names for a specific layer from a weight map."""
    prefix = f"layers.{layer}."
    return {v for v in mapping.values() if v.startswith(prefix)}


class TestLlamaWeightMap:
    """Test Llama 3 weight mapping."""

    def test_global_weights(self) -> None:
        m = llama_weight_map(NUM_LAYERS)
        internal_values = set(m.values())
        assert "embed_tokens.weight" in internal_values
        assert "norm.weight" in internal_values
        assert "lm_head.weight" in internal_values

    def test_per_layer_weights(self) -> None:
        m = llama_weight_map(NUM_LAYERS)
        layer_keys = _layer_keys(m, 0)

        # 4 attn projections + 3 MLP projections + 2 norms = 9 per layer
        assert len(layer_keys) == 9

        # Attention projections
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert f"layers.0.self_attn.{proj}.weight" in layer_keys

        # MLP projections
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"layers.0.mlp.{proj}.weight" in layer_keys

        # Layer norms
        assert "layers.0.input_layernorm.weight" in layer_keys
        assert "layers.0.post_attention_layernorm.weight" in layer_keys

    def test_no_qk_norm(self) -> None:
        m = llama_weight_map(NUM_LAYERS)
        internal_values = set(m.values())
        assert not any("q_norm" in v for v in internal_values)
        assert not any("k_norm" in v for v in internal_values)

    def test_total_weight_count(self) -> None:
        m = llama_weight_map(NUM_LAYERS)
        # embed + norm + lm_head + (9 per layer * 2 layers) = 3 + 18 = 21
        assert len(m) == 21

    def test_hf_names_have_model_prefix(self) -> None:
        """HF names should start with 'model.' (except lm_head)."""
        m = llama_weight_map(NUM_LAYERS)
        for hf_name in m:
            assert hf_name.startswith("model.") or hf_name == "lm_head.weight"

    def test_all_layers_present(self) -> None:
        m = llama_weight_map(NUM_LAYERS)
        for i in range(NUM_LAYERS):
            assert len(_layer_keys(m, i)) == 9


class TestQwen3WeightMap:
    """Test Qwen 3 weight mapping — same as Llama plus QK-norm."""

    def test_has_qk_norm(self) -> None:
        m = qwen3_weight_map(NUM_LAYERS)
        layer_keys = _layer_keys(m, 0)
        assert "layers.0.self_attn.q_norm.weight" in layer_keys
        assert "layers.0.self_attn.k_norm.weight" in layer_keys

    def test_per_layer_weights(self) -> None:
        m = qwen3_weight_map(NUM_LAYERS)
        layer_keys = _layer_keys(m, 0)
        # 4 attn + 2 QK-norm + 3 MLP + 2 norms = 11 per layer
        assert len(layer_keys) == 11

    def test_total_weight_count(self) -> None:
        m = qwen3_weight_map(NUM_LAYERS)
        # embed + norm + lm_head + (11 per layer * 2) = 3 + 22 = 25
        assert len(m) == 25


class TestGemma3WeightMap:
    """Test Gemma 3 weight mapping — QK-norm and 4 norms per layer."""

    def test_has_qk_norm(self) -> None:
        m = gemma3_weight_map(NUM_LAYERS)
        layer_keys = _layer_keys(m, 0)
        assert "layers.0.self_attn.q_norm.weight" in layer_keys
        assert "layers.0.self_attn.k_norm.weight" in layer_keys

    def test_has_four_norms(self) -> None:
        m = gemma3_weight_map(NUM_LAYERS)
        layer_keys = _layer_keys(m, 0)
        assert "layers.0.input_layernorm.weight" in layer_keys
        assert "layers.0.post_attention_layernorm.weight" in layer_keys
        assert "layers.0.pre_feedforward_layernorm.weight" in layer_keys
        assert "layers.0.post_feedforward_layernorm.weight" in layer_keys

    def test_per_layer_weights(self) -> None:
        m = gemma3_weight_map(NUM_LAYERS)
        layer_keys = _layer_keys(m, 0)
        # 4 attn + 2 QK-norm + 3 MLP + 4 norms = 13 per layer
        assert len(layer_keys) == 13

    def test_total_weight_count(self) -> None:
        m = gemma3_weight_map(NUM_LAYERS)
        # embed + norm + lm_head + (13 per layer * 2) = 3 + 26 = 29
        assert len(m) == 29


class TestGetWeightMap:
    """Test the dispatcher function."""

    def _make_config(self, model_type: str, num_layers: int = 2) -> ModelConfig:
        """Create a minimal ModelConfig for testing."""
        return ModelConfig(
            model_type=model_type,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=2048,
        )

    def test_dispatches_llama(self) -> None:
        cfg = self._make_config("llama")
        m = get_weight_map(cfg)
        # Llama has no QK-norm
        assert not any("q_norm" in k for k in m.values())
        assert len(m) == 21  # 3 global + 9*2 layers

    def test_dispatches_qwen3(self) -> None:
        cfg = self._make_config("qwen3")
        m = get_weight_map(cfg)
        assert any("q_norm" in k for k in m.values())
        assert len(m) == 25  # 3 global + 11*2 layers

    def test_dispatches_gemma3(self) -> None:
        cfg = self._make_config("gemma3_text")
        m = get_weight_map(cfg)
        assert any("pre_feedforward_layernorm" in k for k in m.values())
        assert len(m) == 29  # 3 global + 13*2 layers

    def test_uses_num_layers_from_config(self) -> None:
        cfg = self._make_config("llama", num_layers=4)
        m = get_weight_map(cfg)
        # 3 global + 9*4 layers = 39
        assert len(m) == 39

    def test_unsupported_model_type_raises(self) -> None:
        cfg = self._make_config("llama")
        cfg.model_type = "gpt2"
        with pytest.raises(ValueError, match="No weight map"):
            get_weight_map(cfg)


class TestDevModelLayerCounts:
    """Verify weight counts at real dev model layer counts."""

    @pytest.mark.parametrize(
        ("fn", "num_layers", "per_layer", "expected"),
        [
            (llama_weight_map, 16, 9, 16 * 9 + 3),  # Llama-3.2-1B-Instruct
            (qwen3_weight_map, 28, 11, 28 * 11 + 3),  # Qwen3-1.7B
            (gemma3_weight_map, 26, 13, 26 * 13 + 3),  # gemma-3-1b-it
        ],
        ids=["llama-1b", "qwen3-1.7b", "gemma3-1b"],
    )
    def test_real_layer_counts(
        self,
        fn: type,
        num_layers: int,
        per_layer: int,
        expected: int,
    ) -> None:
        m = fn(num_layers)
        assert len(m) == expected


class TestMappingConsistency:
    """Cross-architecture consistency checks."""

    @pytest.mark.parametrize(
        "fn",
        [llama_weight_map, qwen3_weight_map, gemma3_weight_map],
        ids=["llama", "qwen3", "gemma3"],
    )
    def test_internal_names_strip_model_prefix(self, fn: type) -> None:
        """Internal names should be the HF name with 'model.' stripped."""
        m = fn(NUM_LAYERS)
        for hf_name, internal_name in m.items():
            if hf_name.startswith("model."):
                assert internal_name == hf_name.removeprefix("model.")
            else:
                # lm_head.weight has no model. prefix
                assert internal_name == hf_name

    @pytest.mark.parametrize(
        "fn",
        [llama_weight_map, qwen3_weight_map, gemma3_weight_map],
        ids=["llama", "qwen3", "gemma3"],
    )
    def test_bijective_mapping(self, fn: type) -> None:
        """Each HF name maps to a unique internal name (no collisions)."""
        m = fn(NUM_LAYERS)
        assert len(set(m.values())) == len(m), f"Duplicate internal names in {fn.__name__}"

    def test_qwen3_superset_of_llama(self) -> None:
        """Qwen 3's internal names should be a superset of Llama's."""
        llama = set(llama_weight_map(NUM_LAYERS).values())
        qwen3 = set(qwen3_weight_map(NUM_LAYERS).values())
        assert llama.issubset(qwen3)
        # The extra keys should be q_norm and k_norm
        extra = qwen3 - llama
        assert all("q_norm" in k or "k_norm" in k for k in extra)

    def test_lm_head_always_included(self) -> None:
        """All architectures always include lm_head.weight."""
        for fn in (llama_weight_map, qwen3_weight_map, gemma3_weight_map):
            m = fn(NUM_LAYERS)
            assert "lm_head.weight" in m


def _real_tensor_names(model_id: str) -> set[str]:
    """Download just enough metadata to get the real tensor names from HF Hub."""
    from huggingface_hub import hf_hub_download

    # Try sharded layout first (index JSON is a few KB).
    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        return set(index["weight_map"].keys())
    except Exception:
        pass

    # Fall back to single-file layout — read header only, no full download needed
    # (safetensors stores tensor names in a small header at the start of the file).
    from safetensors import safe_open

    path = hf_hub_download(model_id, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        return set(f.keys())


@pytest.mark.slow
class TestRealModelWeightNames:
    """Validate our weight maps against real HF checkpoint tensor names.

    Our maps always include ``lm_head.weight`` by design.  Some checkpoints
    omit it when ``tie_word_embeddings=True`` (e.g. Gemma 3 1B), so we allow
    that as the sole permitted difference.
    """

    def _assert_covers_real_tensors(self, weight_map: dict[str, str], real_names: set[str]) -> None:
        our_hf_names = set(weight_map.keys())
        # Every real tensor must be in our map.
        missing = real_names - our_hf_names
        assert not missing, f"Tensors in checkpoint but not in our map: {sorted(missing)[:10]}"
        # Any extra keys in our map should only be lm_head.weight.
        extra = our_hf_names - real_names
        assert extra <= {"lm_head.weight"}, (
            f"Unexpected extra keys in our map: {sorted(extra)[:10]}"
        )

    def test_qwen3(self) -> None:
        try:
            real = _real_tensor_names("Qwen/Qwen3-1.7B")
        except Exception as exc:
            pytest.skip(f"Qwen3 not available: {exc}")
        m = qwen3_weight_map(28)
        self._assert_covers_real_tensors(m, real)

    def test_gemma3(self) -> None:
        try:
            real = _real_tensor_names("google/gemma-3-1b-it")
        except Exception as exc:
            pytest.skip(f"Gemma3 not available: {exc}")
        m = gemma3_weight_map(26)
        self._assert_covers_real_tensors(m, real)

    def test_llama(self) -> None:
        try:
            real = _real_tensor_names("meta-llama/Llama-3.2-1B-Instruct")
        except Exception as exc:
            pytest.skip(f"Llama not available: {exc}")
        m = llama_weight_map(16)
        self._assert_covers_real_tensors(m, real)
