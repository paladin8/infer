"""FP8 weight quantization support."""

from infer.quant.fp8_linear import FP8Linear, fp8_block_dequant, replace_linear_with_fp8

__all__ = ["FP8Linear", "fp8_block_dequant", "replace_linear_with_fp8"]
