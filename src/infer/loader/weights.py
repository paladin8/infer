"""Safetensors weight loader: load model weights into a flat tensor dict."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_weights(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Load model weights from safetensors files.

    Supports two layouts:
    - Single-file: a ``model.safetensors`` file.
    - Sharded: a ``model.safetensors.index.json`` pointing to multiple shard files.

    Args:
        model_path: Directory containing the safetensors file(s).
        device: Device to load tensors onto (passed to ``safetensors.torch.load_file``).
        dtype: If provided, each tensor is converted to this dtype after loading.

    Returns:
        A flat ``dict[str, torch.Tensor]`` with HF-namespaced tensor names as keys.

    Raises:
        FileNotFoundError: If neither single-file nor sharded layout is found.
        ValueError: If sharded loading produces tensors that don't match the index.
    """
    model_dir = Path(model_path)
    single_file = model_dir / "model.safetensors"
    index_file = model_dir / "model.safetensors.index.json"

    if index_file.exists():
        return _load_sharded(index_file, device=device, dtype=dtype)
    elif single_file.exists():
        return _load_single(single_file, device=device, dtype=dtype)
    else:
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json found in {model_dir}"
        )


def _apply_dtype(tensors: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    """Convert all tensors to the given dtype in place."""
    for name in tensors:
        tensors[name] = tensors[name].to(dtype)


def _load_single(
    path: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Load from a single model.safetensors file."""
    tensors = load_file(str(path), device=str(device))
    if dtype is not None:
        _apply_dtype(tensors, dtype)
    return tensors


def _load_sharded(
    index_path: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Load from sharded safetensors files using an index JSON."""
    with open(index_path) as f:
        index = json.load(f)

    weight_map: dict[str, str] = index["weight_map"]
    expected_names = set(weight_map.keys())

    # Determine unique shard filenames (preserving load order isn't critical,
    # but using dict.fromkeys gives deterministic insertion-order iteration).
    shard_files = list(dict.fromkeys(weight_map.values()))

    tensors: dict[str, torch.Tensor] = {}
    model_dir = index_path.parent

    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Shard file {shard_file} referenced in {index_path.name} not found at {shard_path}"
            )
        shard_tensors = load_file(str(shard_path), device=str(device))
        tensors.update(shard_tensors)

    # Validate: loaded tensors must exactly match the index's weight_map.
    loaded_names = set(tensors.keys())

    missing = expected_names - loaded_names
    if missing:
        raise ValueError(
            f"Missing {len(missing)} tensor(s) after loading shards: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    unexpected = loaded_names - expected_names
    if unexpected:
        raise ValueError(
            f"Unexpected {len(unexpected)} tensor(s) not in index: "
            f"{sorted(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}"
        )

    if dtype is not None:
        _apply_dtype(tensors, dtype)

    return tensors
