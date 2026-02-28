"""FP8 quantized linear layer with block-wise dequantization.

Supports the block-wise FP8 checkpoint format used by Qwen/Qwen3-8B-FP8
(DeepSeek-V3 style): each linear layer stores an FP8 weight tensor and a
per-block float32 scale tensor (``weight_scale_inv``).

Dequantization multiplies each 128x128 block of the FP8 weight by its
corresponding scale factor, producing a bf16 weight for the matmul.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def fp8_block_dequant(
    weight: Tensor,
    scale: Tensor,
    block_size: int = 128,
) -> Tensor:
    """Dequantize block-wise FP8 weight to bf16.

    Each ``block_size x block_size`` block of the FP8 weight is multiplied by
    its corresponding scalar from the scale tensor.  The intermediate multiply
    is done in float32 for precision before the final bf16 cast.

    Args:
        weight: FP8 weight tensor, shape ``[out_features, in_features]``,
            dtype ``float8_e4m3fn``.
        scale: Per-block scale tensor, shape
            ``[ceil(out/block_size), ceil(in/block_size)]``, dtype ``float32``.
        block_size: Block size (default 128, matching the checkpoint format).

    Returns:
        Dequantized weight in bf16, shape ``[out_features, in_features]``.
    """
    out_features, in_features = weight.shape

    # Pad to multiple of block_size if needed (F.pad doesn't support fp8).
    pad_out = (block_size - out_features % block_size) % block_size
    pad_in = (block_size - in_features % block_size) % block_size
    if pad_out > 0 or pad_in > 0:
        w = F.pad(weight.to(torch.float32), (0, pad_in, 0, pad_out))
    else:
        w = weight.to(torch.float32)

    # Reshape to [out/B, B, in/B, B], broadcast-multiply by scale.
    out_blocks = w.shape[0] // block_size
    in_blocks = w.shape[1] // block_size
    w = w.reshape(out_blocks, block_size, in_blocks, block_size)
    w = w * scale[:, None, :, None]

    # Reshape back and trim padding.
    w = w.reshape(out_blocks * block_size, in_blocks * block_size)
    w = w[:out_features, :in_features]
    return w.to(torch.bfloat16)


class FP8Linear(nn.Module):
    """Linear layer with FP8 block-quantized weights.

    Replaces ``nn.Linear`` for layers loaded from FP8 checkpoints.  Stores the
    weight as ``float8_e4m3fn`` and the per-block scale as ``float32``.  On each
    forward pass, the weight is dequantized to bf16 before the matmul.

    Does not support bias (all target architectures use bias-free linears).

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        block_size: Block size for dequantization (default 128).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 128,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Weight placeholder — populated by load_state_dict.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )

        # Scale placeholder — persistent buffer so load_state_dict can populate it.
        scale_out = -(-out_features // block_size)  # ceil division
        scale_in = -(-in_features // block_size)
        self.weight_scale_inv: Tensor
        self.register_buffer(
            "weight_scale_inv",
            torch.ones(scale_out, scale_in, dtype=torch.float32),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Dequantize FP8 weight to bf16 and compute linear transform."""
        w = fp8_block_dequant(self.weight, self.weight_scale_inv, self.block_size)
        return F.linear(x, w)

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> FP8Linear:
        """Override to prevent dtype conversion of FP8 weights and scale buffers.

        When ``model.to(dtype=bf16)`` is called, PyTorch's default ``_apply``
        would convert the FP8 weight to bf16, destroying the quantized values,
        and convert the float32 scale buffer to bf16, losing precision.
        This override only applies device moves to FP8 parameters and scale
        buffers, preserving their original dtypes.
        """
        # Apply fn to buffers, but only accept device changes (not dtype changes)
        # for the weight_scale_inv buffer which must stay float32.
        for key, buf in self._buffers.items():
            if buf is not None:
                new_buf = fn(buf)
                if isinstance(new_buf, Tensor):
                    if key == "weight_scale_inv" and new_buf.dtype != buf.dtype:
                        # Dtype conversion requested — only apply device move.
                        if new_buf.device != buf.device:
                            self._buffers[key] = buf.to(device=new_buf.device)
                    else:
                        self._buffers[key] = new_buf

        # Apply to parameters, but skip dtype changes for FP8 parameters.
        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    new_data = fn(param.data)
                    if isinstance(new_data, Tensor):
                        if (
                            param.dtype == torch.float8_e4m3fn
                            and new_data.dtype != torch.float8_e4m3fn
                        ):
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
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, bias=False"
        )


def replace_linear_with_fp8(model: nn.Module) -> None:
    """Replace ``nn.Linear`` modules with ``FP8Linear`` for FP8 quantization.

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
                raise ValueError(f"FP8Linear does not support bias, but {full_name} has bias=True")

            fp8 = FP8Linear(child.in_features, child.out_features)
            setattr(parent_module, name, fp8)
