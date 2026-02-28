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
from infer.quant import replace_linear_with_fp8

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


def _detect_quantization(config: ModelConfig) -> str | None:
    """Auto-detect quantization format from checkpoint metadata.

    Returns ``"fp8"`` if the config indicates block-wise FP8 quantization,
    ``None`` otherwise.
    """
    qc = config.quantization_config
    if qc is None:
        return None
    if qc.get("quant_method") == "fp8":
        return "fp8"
    return None


def _selective_to(
    model: nn.Module,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Move model to device and dtype, preserving FP8 parameter dtypes.

    ``model.to(dtype=...)`` would convert FP8 weights to bf16, destroying
    the quantized values.  This function moves all tensors to the target
    device and converts non-FP8 parameters to the target dtype.

    The buffer heuristic preserves float32 buffers (FP8 scale tensors) and
    converts everything else to the target dtype.  This works because at this
    point in the pipeline, the only float32 buffers are ``weight_scale_inv``
    from ``FP8Linear`` modules — RoPE cos/sin buffers arrive as bf16 from
    the checkpoint.  ``FP8Linear._apply`` provides a second layer of defense
    if ``model.to()`` is called later.
    """
    for param in model.parameters():
        if param.dtype == torch.float8_e4m3fn:
            param.data = param.data.to(device=device)
        else:
            param.data = param.data.to(device=device, dtype=dtype)
    for buf in model.buffers():
        if buf.dtype == torch.float32:
            buf.data = buf.data.to(device=device)
        else:
            buf.data = buf.data.to(device=device, dtype=dtype)


def load_model(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    quantization: str | None = None,
) -> tuple[nn.Module, ModelConfig]:
    """Load a model from a local directory or HuggingFace Hub model ID.

    Args:
        model_path: Path to a local directory containing config.json and
            safetensors weights, or a HuggingFace Hub model ID
            (e.g. ``"meta-llama/Llama-3.2-1B-Instruct"``).
        dtype: Target dtype for model parameters (non-FP8).
        device: Target device.
        quantization: Quantization format (``None`` or ``"fp8"``).
            When ``None``, auto-detected from checkpoint metadata.

    Returns:
        A tuple of ``(model, config)``.  The model has weights loaded
        and is in eval mode.
    """
    local_path = _resolve_model_path(model_path)
    config = load_config(local_path)

    # Auto-detect quantization from checkpoint if not explicitly set.
    if quantization is None:
        quantization = _detect_quantization(config)

    model = _build_model(config)

    # Apply FP8 model surgery before loading weights.
    if quantization == "fp8":
        replace_linear_with_fp8(model)

    # Load weights.  For FP8 checkpoints, pass dtype=None to preserve
    # float8_e4m3fn tensors; for standard checkpoints, convert to target dtype.
    load_dtype = None if quantization == "fp8" else dtype
    raw_weights = load_weights(local_path, device="cpu", dtype=load_dtype)
    weight_map = get_weight_map(config, quantization=quantization)

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

    if quantization == "fp8":
        # Selective move: preserve FP8 weight dtypes.
        _selective_to(model, device, dtype)
    else:
        model.to(device=device, dtype=dtype)

    model.eval()
    return model, config
