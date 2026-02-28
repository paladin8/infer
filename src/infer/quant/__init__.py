"""Weight quantization support (FP8 and INT8)."""

from infer.quant.fp8_linear import FP8Linear, fp8_block_dequant, replace_linear_with_fp8
from infer.quant.int8_linear import INT8Linear, int8_channel_dequant, replace_linear_with_int8

__all__ = [
    "FP8Linear",
    "INT8Linear",
    "fp8_block_dequant",
    "int8_channel_dequant",
    "replace_linear_with_fp8",
    "replace_linear_with_int8",
]
