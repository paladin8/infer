"""Tests for the safetensors weight loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from infer.loader.weights import load_weights


class TestSingleFileLoading:
    """Test loading from a single model.safetensors file."""

    def test_loads_tensors(self, tmp_path: Path) -> None:
        tensors = {
            "weight_a": torch.randn(4, 8),
            "weight_b": torch.randn(16),
        }
        save_file(tensors, tmp_path / "model.safetensors")

        loaded = load_weights(tmp_path)
        assert set(loaded.keys()) == {"weight_a", "weight_b"}
        assert loaded["weight_a"].shape == (4, 8)
        assert loaded["weight_b"].shape == (16,)
        torch.testing.assert_close(loaded["weight_a"], tensors["weight_a"])
        torch.testing.assert_close(loaded["weight_b"], tensors["weight_b"])

    def test_dtype_conversion(self, tmp_path: Path) -> None:
        original = torch.randn(4, 4, dtype=torch.float32)
        save_file({"w": original}, tmp_path / "model.safetensors")

        loaded = load_weights(tmp_path, dtype=torch.bfloat16)
        assert loaded["w"].dtype == torch.bfloat16
        torch.testing.assert_close(loaded["w"], original.to(torch.bfloat16))

    def test_device_propagation(self, tmp_path: Path) -> None:
        save_file({"w": torch.zeros(2)}, tmp_path / "model.safetensors")
        loaded = load_weights(tmp_path, device="cpu")
        assert loaded["w"].device == torch.device("cpu")

    def test_accepts_torch_device(self, tmp_path: Path) -> None:
        save_file({"w": torch.zeros(2)}, tmp_path / "model.safetensors")
        loaded = load_weights(tmp_path, device=torch.device("cpu"))
        assert loaded["w"].device == torch.device("cpu")

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        save_file({"w": torch.zeros(2)}, tmp_path / "model.safetensors")
        loaded = load_weights(str(tmp_path))
        assert "w" in loaded

    def test_mixed_dtypes_preserved(self, tmp_path: Path) -> None:
        """Without dtype arg, original dtypes from the file are preserved."""
        tensors = {
            "fp32": torch.randn(4, dtype=torch.float32),
            "fp16": torch.randn(4, dtype=torch.float16),
        }
        save_file(tensors, tmp_path / "model.safetensors")

        loaded = load_weights(tmp_path)
        assert loaded["fp32"].dtype == torch.float32
        assert loaded["fp16"].dtype == torch.float16

    def test_mixed_dtypes_converted(self, tmp_path: Path) -> None:
        """With dtype arg, all tensors are converted regardless of original dtype."""
        tensors = {
            "fp32": torch.randn(4, dtype=torch.float32),
            "fp16": torch.randn(4, dtype=torch.float16),
        }
        save_file(tensors, tmp_path / "model.safetensors")

        loaded = load_weights(tmp_path, dtype=torch.bfloat16)
        assert loaded["fp32"].dtype == torch.bfloat16
        assert loaded["fp16"].dtype == torch.bfloat16


class TestShardedLoading:
    """Test loading from sharded safetensors with an index JSON."""

    def _write_sharded(self, tmp_path: Path) -> dict[str, torch.Tensor]:
        """Create a two-shard layout and return the expected tensors."""
        shard0_tensors = {
            "layer.0.weight": torch.randn(4, 8),
            "layer.0.bias": torch.randn(4),
        }
        shard1_tensors = {
            "layer.1.weight": torch.randn(4, 8),
            "layer.1.bias": torch.randn(4),
        }

        save_file(shard0_tensors, tmp_path / "model-00001-of-00002.safetensors")
        save_file(shard1_tensors, tmp_path / "model-00002-of-00002.safetensors")

        weight_map = {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.0.bias": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors",
            "layer.1.bias": "model-00002-of-00002.safetensors",
        }
        index = {"metadata": {}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        return {**shard0_tensors, **shard1_tensors}

    def test_loads_all_shards(self, tmp_path: Path) -> None:
        expected = self._write_sharded(tmp_path)
        loaded = load_weights(tmp_path)

        assert set(loaded.keys()) == set(expected.keys())
        for name, tensor in expected.items():
            torch.testing.assert_close(loaded[name], tensor)

    def test_dtype_conversion(self, tmp_path: Path) -> None:
        expected = self._write_sharded(tmp_path)
        loaded = load_weights(tmp_path, dtype=torch.float16)
        for name, tensor in loaded.items():
            assert tensor.dtype == torch.float16
            torch.testing.assert_close(tensor, expected[name].to(torch.float16))

    def test_missing_tensor_raises(self, tmp_path: Path) -> None:
        """Index references a tensor that doesn't exist in any shard."""
        save_file({"a": torch.zeros(2)}, tmp_path / "model-00001-of-00001.safetensors")

        weight_map = {
            "a": "model-00001-of-00001.safetensors",
            "b": "model-00001-of-00001.safetensors",  # not in the actual file
        }
        index = {"metadata": {}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        with pytest.raises(ValueError, match="Missing"):
            load_weights(tmp_path)

    def test_unexpected_tensor_raises(self, tmp_path: Path) -> None:
        """Shard contains a tensor not listed in the index."""
        save_file(
            {"a": torch.zeros(2), "extra": torch.zeros(3)},
            tmp_path / "model-00001-of-00001.safetensors",
        )

        weight_map = {"a": "model-00001-of-00001.safetensors"}
        index = {"metadata": {}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        with pytest.raises(ValueError, match="Unexpected"):
            load_weights(tmp_path)

    def test_prefers_index_over_single_file(self, tmp_path: Path) -> None:
        """When both exist, sharded index takes precedence."""
        # Write a single file with different content
        save_file({"single": torch.zeros(2)}, tmp_path / "model.safetensors")

        # Write sharded layout
        expected = self._write_sharded(tmp_path)

        loaded = load_weights(tmp_path)
        # Should load from shards, not the single file
        assert set(loaded.keys()) == set(expected.keys())


class TestMissingFiles:
    """Test error handling for missing files."""

    def test_no_safetensors_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"No model\.safetensors"):
            load_weights(tmp_path)

    def test_missing_shard_file_raises(self, tmp_path: Path) -> None:
        """Index references a shard file that doesn't exist on disk."""
        weight_map = {"w": "model-00001-of-00001.safetensors"}
        index = {"metadata": {}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        with pytest.raises(FileNotFoundError, match="Shard file"):
            load_weights(tmp_path)
