"""INT8 quantized linear layer with per-channel dequantization.

Supports the compressed-tensors INT8 checkpoint format used by
nytopop/Qwen3-8B.w8a8: each linear layer stores a signed int8 weight
tensor and a per-channel float32 scale tensor (``weight_scale``).

Dequantization multiplies each row of the int8 weight by its
corresponding scale factor, producing a bf16 weight for the matmul.
The quantization is symmetric (zero point = 0), so no zero-point
tensor is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def int8_channel_dequant(
    weight: Tensor,
    scale: Tensor,
) -> Tensor:
    """Dequantize per-channel symmetric INT8 weight to bf16.

    Each row of the int8 weight is multiplied by its corresponding
    scalar from the scale tensor.  The multiply is done in float32
    for precision before the final bf16 cast.

    Args:
        weight: INT8 weight tensor, shape ``[out_features, in_features]``,
            dtype ``int8``.
        scale: Per-channel scale tensor, shape ``[out_features, 1]``,
            dtype ``float32``.

    Returns:
        Dequantized weight in bf16, shape ``[out_features, in_features]``.
    """
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


class INT8Linear(nn.Module):
    """Linear layer with INT8 per-channel quantized weights.

    Replaces ``nn.Linear`` for layers loaded from INT8 checkpoints.  Stores the
    weight as ``int8`` and the per-channel scale as ``float32``.  On each
    forward pass, the weight is dequantized to bf16 before the matmul.

    Does not support bias (all target architectures use bias-free linears).

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight placeholder — populated by load_state_dict.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )

        # Scale placeholder — persistent buffer so load_state_dict can populate it.
        # Stored as float32 for precision; checkpoint bf16 scales are coerced
        # to float32 by load_state_dict.
        self.weight_scale: Tensor
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, 1, dtype=torch.float32),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Dequantize INT8 weight to bf16 and compute linear transform."""
        w = int8_channel_dequant(self.weight, self.weight_scale)
        return F.linear(x, w)

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> INT8Linear:
        """Override to prevent dtype conversion of INT8 weights and scale buffers.

        When ``model.to(dtype=bf16)`` is called, PyTorch's default ``_apply``
        would convert the int8 weight to bf16, destroying the quantized values,
        and convert the float32 scale buffer to bf16, losing precision.
        This override only applies device moves to INT8 parameters and scale
        buffers, preserving their original dtypes.
        """
        # Apply fn to buffers, but only accept device changes (not dtype changes)
        # for the weight_scale buffer which must stay float32.
        for key, buf in self._buffers.items():
            if buf is not None:
                new_buf = fn(buf)
                if isinstance(new_buf, Tensor):
                    if key == "weight_scale" and new_buf.dtype != buf.dtype:
                        # Dtype conversion requested — only apply device move.
                        if new_buf.device != buf.device:
                            self._buffers[key] = buf.to(device=new_buf.device)
                    else:
                        self._buffers[key] = new_buf

        # Apply to parameters, but skip dtype changes for int8 parameters.
        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    new_data = fn(param.data)
                    if isinstance(new_data, Tensor):
                        if param.dtype == torch.int8 and new_data.dtype != torch.int8:
                            # Dtype conversion requested — only apply device move.
                            if new_data.device != param.data.device:
                                self._parameters[key] = nn.Parameter(
                                    param.data.to(device=new_data.device),
                                    requires_grad=param.requires_grad,
                                )
                        else:
                            self._parameters[key] = nn.Parameter(
                                new_data, requires_grad=param.requires_grad
                            )

        # Recurse into child modules.
        if recurse:
            for module in self._modules.values():
                if module is not None:
                    module._apply(fn, recurse)

        return self

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"


def replace_linear_with_int8(model: nn.Module) -> None:
    """Replace ``nn.Linear`` modules with ``INT8Linear`` for INT8 quantization.

    Walks the module tree and replaces every ``nn.Linear`` that should be
    quantized.  Layers that should remain in full precision are skipped:

    - ``embed_tokens`` (embedding, not a linear)
    - ``lm_head`` (final vocab projection)
    - Any module whose name contains ``norm`` (RMSNorm, QK norms)

    Args:
        model: The model to modify in-place.

    Raises:
        ValueError: If a linear layer to be replaced has a bias.
    """
    _SKIP_NAMES = {"embed_tokens", "lm_head"}

    for parent_name, parent_module in list(model.named_modules()):
        for name, child in list(parent_module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            if not isinstance(child, nn.Linear):
                continue

            # Skip layers that should stay in full precision.
            if name in _SKIP_NAMES or "norm" in name:
                continue

            if child.bias is not None:
                raise ValueError(f"INT8Linear does not support bias, but {full_name} has bias=True")

            int8 = INT8Linear(child.in_features, child.out_features)
            setattr(parent_module, name, int8)
