"""Weight name mapping: map HF checkpoint tensor names to internal module names.

The weight map always includes ``lm_head.weight`` regardless of
``tie_word_embeddings``.  Real HF checkpoints are inconsistent: some include
``lm_head.weight`` even when embeddings are tied (e.g. Qwen3-1.7B).  Whether
to reuse ``embed_tokens.weight`` as the LM head is a model construction
concern, not a weight loading concern.
"""

from __future__ import annotations

from infer.loader.config import ModelConfig

# Linear projections that get weight_scale_inv entries for FP8 quantization.
_ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJS = ("gate_proj", "up_proj", "down_proj")


def _add_fp8_scales(
    mapping: dict[str, str],
    num_layers: int,
) -> None:
    """Add ``weight_scale_inv`` entries for all quantized linear layers."""
    for i in range(num_layers):
        hf = f"model.layers.{i}"
        internal = f"layers.{i}"
        for proj in _ATTN_PROJS:
            mapping[f"{hf}.self_attn.{proj}.weight_scale_inv"] = (
                f"{internal}.self_attn.{proj}.weight_scale_inv"
            )
        for proj in _MLP_PROJS:
            mapping[f"{hf}.mlp.{proj}.weight_scale_inv"] = f"{internal}.mlp.{proj}.weight_scale_inv"


def _add_int8_scales(
    mapping: dict[str, str],
    num_layers: int,
) -> None:
    """Add ``weight_scale`` entries for all quantized linear layers."""
    for i in range(num_layers):
        hf = f"model.layers.{i}"
        internal = f"layers.{i}"
        for proj in _ATTN_PROJS:
            mapping[f"{hf}.self_attn.{proj}.weight_scale"] = (
                f"{internal}.self_attn.{proj}.weight_scale"
            )
        for proj in _MLP_PROJS:
            mapping[f"{hf}.mlp.{proj}.weight_scale"] = f"{internal}.mlp.{proj}.weight_scale"


def llama_weight_map(num_layers: int) -> dict[str, str]:
    """Map HF tensor names to internal names for Llama 3.

    Llama 3 has 2 norms per layer (input_layernorm, post_attention_layernorm)
    and no QK-norm.

    Args:
        num_layers: Number of transformer layers.

    Returns:
        Mapping from HF checkpoint names to internal parameter names.
    """
    mapping: dict[str, str] = {}

    mapping["model.embed_tokens.weight"] = "embed_tokens.weight"

    for i in range(num_layers):
        hf = f"model.layers.{i}"
        internal = f"layers.{i}"

        # Attention projections
        for proj in _ATTN_PROJS:
            mapping[f"{hf}.self_attn.{proj}.weight"] = f"{internal}.self_attn.{proj}.weight"

        # MLP projections
        for proj in _MLP_PROJS:
            mapping[f"{hf}.mlp.{proj}.weight"] = f"{internal}.mlp.{proj}.weight"

        # Layer norms (2 per layer)
        mapping[f"{hf}.input_layernorm.weight"] = f"{internal}.input_layernorm.weight"
        mapping[f"{hf}.post_attention_layernorm.weight"] = (
            f"{internal}.post_attention_layernorm.weight"
        )

    mapping["model.norm.weight"] = "norm.weight"
    mapping["lm_head.weight"] = "lm_head.weight"

    return mapping


def qwen3_weight_map(num_layers: int) -> dict[str, str]:
    """Map HF tensor names to internal names for Qwen 3.

    Same structure as Llama 3, plus q_norm/k_norm per layer.

    Args:
        num_layers: Number of transformer layers.

    Returns:
        Mapping from HF checkpoint names to internal parameter names.
    """
    mapping = llama_weight_map(num_layers)

    # Add QK-norm weights
    for i in range(num_layers):
        hf = f"model.layers.{i}"
        internal = f"layers.{i}"
        for norm in ("q_norm", "k_norm"):
            mapping[f"{hf}.self_attn.{norm}.weight"] = f"{internal}.self_attn.{norm}.weight"

    return mapping


def gemma3_weight_map(num_layers: int) -> dict[str, str]:
    """Map HF tensor names to internal names for Gemma 3.

    Gemma 3 has QK-norm and 4 norms per layer (sandwich norm pattern):
    input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
    post_feedforward_layernorm.

    Args:
        num_layers: Number of transformer layers.

    Returns:
        Mapping from HF checkpoint names to internal parameter names.
    """
    mapping: dict[str, str] = {}

    mapping["model.embed_tokens.weight"] = "embed_tokens.weight"

    for i in range(num_layers):
        hf = f"model.layers.{i}"
        internal = f"layers.{i}"

        # Attention projections
        for proj in _ATTN_PROJS:
            mapping[f"{hf}.self_attn.{proj}.weight"] = f"{internal}.self_attn.{proj}.weight"

        # QK-norm
        for norm in ("q_norm", "k_norm"):
            mapping[f"{hf}.self_attn.{norm}.weight"] = f"{internal}.self_attn.{norm}.weight"

        # MLP projections
        for proj in _MLP_PROJS:
            mapping[f"{hf}.mlp.{proj}.weight"] = f"{internal}.mlp.{proj}.weight"

        # Layer norms (4 per layer — sandwich norm)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            mapping[f"{hf}.{norm}.weight"] = f"{internal}.{norm}.weight"

    mapping["model.norm.weight"] = "norm.weight"
    mapping["lm_head.weight"] = "lm_head.weight"

    return mapping


def get_weight_map(
    config: ModelConfig,
    quantization: str | None = None,
) -> dict[str, str]:
    """Select the right weight map based on model_type.

    Args:
        config: A loaded ModelConfig instance.
        quantization: Quantization format (``None``, ``"fp8"``, or ``"int8"``).
            When ``"fp8"``, adds ``weight_scale_inv`` entries for each
            quantized linear layer.  When ``"int8"``, adds ``weight_scale``
            entries instead.

    Returns:
        Mapping from HF checkpoint names to internal parameter names.

    Raises:
        ValueError: If model_type is not supported.
    """
    dispatchers = {
        "llama": llama_weight_map,
        "qwen3": qwen3_weight_map,
        "gemma3_text": gemma3_weight_map,
    }
    fn = dispatchers.get(config.model_type)
    if fn is None:
        raise ValueError(
            f"No weight map for model_type: {config.model_type!r}. "
            f"Supported types: {sorted(dispatchers.keys())}"
        )
    mapping = fn(config.num_hidden_layers)

    if quantization == "fp8":
        _add_fp8_scales(mapping, config.num_hidden_layers)
    elif quantization == "int8":
        _add_int8_scales(mapping, config.num_hidden_layers)

    return mapping
