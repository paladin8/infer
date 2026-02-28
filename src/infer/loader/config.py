"""Config reader: parse HuggingFace config.json into a typed dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen3", "gemma3_text"})


@dataclass
class ModelConfig:
    """Typed representation of a HuggingFace model config."""

    # Identity
    model_type: str  # "llama", "qwen3", "gemma3_text"

    # Dimensions
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    head_dim: int | None = None

    # Normalization
    rms_norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rope_local_base_freq: float | None = None

    # Projection biases
    attention_bias: bool = False
    mlp_bias: bool = False

    # Activation
    hidden_act: str = "silu"

    # Attention scaling
    query_pre_attn_scalar: float | None = None

    # Sliding window attention
    sliding_window: int | None = None
    sliding_window_pattern: int | None = None

    # Layer types (resolved from sliding_window_pattern or loaded directly from config)
    layer_types: list[str] | None = None

    # Embeddings
    tie_word_embeddings: bool = False

    # Quantization (from checkpoint metadata)
    quantization_config: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def computed_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize HF config quirks before constructing the dataclass."""
    raw = dict(raw)  # shallow copy to avoid mutating caller's dict

    # Gemma 3 multimodal configs (4B+) nest text config under "text_config".
    # The nested text_config contains its own model_type (e.g. "gemma3_text").
    if "text_config" in raw:
        text_cfg = raw["text_config"]
        if isinstance(text_cfg, dict) and "model_type" in text_cfg:
            raw = dict(text_cfg)

    # Gemma 3 uses "hidden_activation" instead of "hidden_act".
    if "hidden_activation" in raw and "hidden_act" not in raw:
        raw["hidden_act"] = raw.pop("hidden_activation")

    return raw


def _resolve_layer_types(config: ModelConfig) -> None:
    """Resolve layer_types from sliding_window_pattern if not already set."""
    if config.layer_types is not None:
        return
    if config.sliding_window_pattern is None:
        return
    # Every sliding_window_pattern-th layer (1-indexed) is full attention,
    # the rest are sliding window. E.g. pattern=6 → layers 5,11,17,23 are global.
    pattern = config.sliding_window_pattern
    config.layer_types = [
        "full_attention" if (i + 1) % pattern == 0 else "sliding_attention"
        for i in range(config.num_hidden_layers)
    ]


def load_config(model_path: str | Path) -> ModelConfig:
    """Load and parse a HuggingFace config.json into a ModelConfig.

    Args:
        model_path: Path to the model directory containing config.json.

    Returns:
        A populated ModelConfig instance.

    Raises:
        FileNotFoundError: If config.json does not exist.
        ValueError: If model_type is not supported.
    """
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        raw = json.load(f)

    raw = _normalize_raw_config(raw)

    model_type = raw.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. "
            f"Supported types: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    # Extract only the fields that ModelConfig knows about.
    # Drop None values so non-nullable fields fall back to their defaults
    # (real HF configs sometimes have null for fields like mlp_bias).
    known_fields = {f.name for f in fields(ModelConfig)}
    filtered = {k: v for k, v in raw.items() if k in known_fields and v is not None}

    config = ModelConfig(**filtered)
    _resolve_layer_types(config)
    return config
