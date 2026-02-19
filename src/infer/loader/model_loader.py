"""Model loader: build a model from config and load HuggingFace weights."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import nn

from infer.loader.config import ModelConfig, load_config
from infer.loader.weight_map import get_weight_map
from infer.loader.weights import load_weights
from infer.models.gemma3 import Gemma3Model
from infer.models.llama import LlamaModel
from infer.models.qwen3 import Qwen3Model

_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "llama": LlamaModel,
    "qwen3": Qwen3Model,
    "gemma3_text": Gemma3Model,
}


def _build_model(config: ModelConfig) -> nn.Module:
    """Construct the right model class for the given config.

    Args:
        config: A loaded ModelConfig instance.

    Returns:
        An uninitialized model (random weights).

    Raises:
        ValueError: If model_type is not supported.
    """
    cls = _MODEL_CLASSES.get(config.model_type)
    if cls is None:
        raise ValueError(
            f"No model class for model_type: {config.model_type!r}. "
            f"Supported types: {sorted(_MODEL_CLASSES.keys())}"
        )
    return cls(config)


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace Hub model ID to a local cache path.

    If ``model_path`` is already a local directory (contains config.json),
    it is returned as-is.  Otherwise, ``huggingface_hub.snapshot_download``
    is used to download the model to the HF cache.
    """
    if Path(model_path).is_dir():
        return model_path
    return snapshot_download(model_path)


def load_model(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[nn.Module, ModelConfig]:
    """Load a model from a local directory or HuggingFace Hub model ID.

    Args:
        model_path: Path to a local directory containing config.json and
            safetensors weights, or a HuggingFace Hub model ID
            (e.g. ``"meta-llama/Llama-3.2-1B-Instruct"``).
        dtype: Target dtype for model parameters.
        device: Target device.

    Returns:
        A tuple of ``(model, config)``.  The model has weights loaded
        and is in eval mode.
    """
    local_path = _resolve_model_path(model_path)
    config = load_config(local_path)
    model = _build_model(config)

    # Load weights to CPU first to avoid doubling GPU memory.
    raw_weights = load_weights(local_path, device="cpu", dtype=dtype)
    weight_map = get_weight_map(config)

    # Rename HF checkpoint names to internal module names.
    renamed: dict[str, torch.Tensor] = {}
    for hf_name, internal_name in weight_map.items():
        if hf_name in raw_weights:
            renamed[internal_name] = raw_weights[hf_name]

    # Handle tied embeddings: if lm_head.weight is missing from checkpoint,
    # copy embed_tokens.weight to fill the gap.  Some HF configs omit the
    # tie_word_embeddings flag entirely (e.g. Gemma 3), so we also handle
    # the case where the flag isn't set but lm_head.weight is absent.
    if "lm_head.weight" not in renamed and "embed_tokens.weight" in renamed:
        renamed["lm_head.weight"] = renamed["embed_tokens.weight"]

    model.load_state_dict(renamed, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config
