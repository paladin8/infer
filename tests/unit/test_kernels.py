"""Tests for fused Triton kernels.

Each kernel is compared against the PyTorch reference implementation.
Tests are skipped when CUDA is not available.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from infer.models.common import build_rope_cos_sin

# Skip entire module if CUDA is unavailable.  The triton kernel modules import
# triton unconditionally, which is only available in CUDA environments.
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Triton kernel tests", allow_module_level=True)

from infer.kernels.activation import triton_fused_gated_activation
from infer.kernels.fused_norm_residual import triton_fused_residual_rms_norm
from infer.kernels.rms_norm import triton_rms_norm
from infer.kernels.rope import triton_apply_rope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_DTYPE_IDS = ["bf16", "fp16", "fp32"]


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return (atol, rtol) for a given dtype."""
    if dtype == torch.float32:
        return (1e-5, 1e-5)
    return (1e-2, 1e-2)


def _assert_close(actual: Tensor, expected: Tensor, dtype: torch.dtype) -> None:
    """Assert tensors are close within dtype tolerance."""
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _pytorch_rms_norm(x: Tensor, weight: Tensor, eps: float, gemma_style: bool) -> Tensor:
    """PyTorch reference RMSNorm."""
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    if gemma_style:
        return ((1.0 + weight.float()) * normed).to(input_dtype)
    return weight.to(input_dtype) * normed.to(input_dtype)


def _pytorch_fused_residual_rms_norm(
    residual: Tensor, x: Tensor, weight: Tensor, eps: float, gemma_style: bool
) -> tuple[Tensor, Tensor]:
    """PyTorch reference: residual add + RMSNorm."""
    combined = residual + x
    normed = _pytorch_rms_norm(combined, weight, eps, gemma_style)
    return combined, normed


def _rotate_half(x: Tensor) -> Tensor:
    """Swap and negate halves: [a, b] -> [-b, a]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _pytorch_apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """PyTorch reference RoPE."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rotated = q * cos + _rotate_half(q) * sin
    k_rotated = k * cos + _rotate_half(k) * sin
    return q_rotated.to(q.dtype), k_rotated.to(k.dtype)


def _pytorch_fused_gated_activation(gate: Tensor, up: Tensor, use_gelu: bool) -> Tensor:
    """PyTorch reference: act_fn(gate) * up."""
    if use_gelu:
        activated = torch.nn.functional.gelu(gate, approximate="tanh")
    else:
        activated = torch.nn.functional.silu(gate)
    return activated * up


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------


class TestTritonRMSNorm:
    """Test fused RMSNorm kernel against PyTorch reference."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize(
        "shape",
        [(1, 128), (4, 1024), (1, 4096), (2, 8, 256)],
        ids=["1x128", "4x1024", "1x4096", "2x8x256"],
    )
    def test_standard_rms_norm(self, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
        N = shape[-1]
        x = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(N, dtype=dtype, device="cuda")
        eps = 1e-6

        result = triton_rms_norm(x, weight, eps=eps, gemma_style=False)
        expected = _pytorch_rms_norm(x, weight, eps, gemma_style=False)

        _assert_close(result, expected, dtype)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize(
        "shape",
        [(1, 128), (4, 1024), (2, 8, 256)],
        ids=["1x128", "4x1024", "2x8x256"],
    )
    def test_gemma3_rms_norm(self, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
        N = shape[-1]
        x = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(N, dtype=dtype, device="cuda")  # offset-from-1 convention
        eps = 1e-6

        result = triton_rms_norm(x, weight, eps=eps, gemma_style=True)
        expected = _pytorch_rms_norm(x, weight, eps, gemma_style=True)

        _assert_close(result, expected, dtype)

    def test_output_dtype_preserved(self) -> None:
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
        result = triton_rms_norm(x, weight, eps=1e-6)
        assert result.dtype == torch.bfloat16

    def test_output_shape_preserved(self) -> None:
        x = torch.randn(2, 4, 8, 64, dtype=torch.float32, device="cuda")
        weight = torch.randn(64, dtype=torch.float32, device="cuda")
        result = triton_rms_norm(x, weight, eps=1e-6)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# RoPE tests
# ---------------------------------------------------------------------------


class TestTritonRoPE:
    """Test fused RoPE kernel against PyTorch reference."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize(
        "batch,heads,kv_heads,seq_len,head_dim",
        [
            (1, 4, 2, 1, 64),  # decode step
            (1, 4, 2, 8, 64),  # short seq
            (2, 8, 4, 16, 128),  # medium
            (1, 4, 1, 32, 256),  # large head_dim (Gemma3)
        ],
        ids=["decode", "short", "medium", "large_hd"],
    )
    def test_rope_matches_pytorch(
        self,
        dtype: torch.dtype,
        batch: int,
        heads: int,
        kv_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> None:
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device="cuda")
        cos, sin = build_rope_cos_sin(head_dim, seq_len)
        cos = cos.to(dtype=dtype, device="cuda")
        sin = sin.to(dtype=dtype, device="cuda")

        q_tri, k_tri = triton_apply_rope(q, k, cos, sin)
        q_ref, k_ref = _pytorch_apply_rope(q, k, cos, sin)

        _assert_close(q_tri, q_ref, dtype)
        _assert_close(k_tri, k_ref, dtype)

    def test_rope_preserves_norm(self) -> None:
        head_dim = 64
        q = torch.randn(1, 4, 8, head_dim, dtype=torch.float32, device="cuda")
        k = torch.randn(1, 2, 8, head_dim, dtype=torch.float32, device="cuda")
        cos, sin = build_rope_cos_sin(head_dim, 8)
        cos = cos.to(device="cuda")
        sin = sin.to(device="cuda")

        q_rot, k_rot = triton_apply_rope(q, k, cos, sin)
        torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_rope_non_contiguous_inputs(self) -> None:
        """RoPE must handle non-contiguous Q/K (from .transpose(1,2) in Attention)."""
        batch, seq_len, num_heads, kv_heads, head_dim = 1, 6, 32, 8, 64
        # Simulate Attention: view then transpose → non-contiguous
        q = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda").transpose(1, 2)
        k = torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda").transpose(1, 2)
        assert not q.is_contiguous()

        cos, sin = build_rope_cos_sin(head_dim, seq_len)
        cos = cos.to(device="cuda")
        sin = sin.to(device="cuda")

        q_tri, k_tri = triton_apply_rope(q, k, cos, sin)
        q_ref, k_ref = _pytorch_apply_rope(q, k, cos, sin)
        _assert_close(q_tri, q_ref, torch.float32)
        _assert_close(k_tri, k_ref, torch.float32)

    def test_rope_output_shapes(self) -> None:
        q = torch.randn(2, 8, 4, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(2, 2, 4, 64, dtype=torch.bfloat16, device="cuda")
        cos, sin = build_rope_cos_sin(64, 4)
        cos = cos.to(dtype=torch.bfloat16, device="cuda")
        sin = sin.to(dtype=torch.bfloat16, device="cuda")

        q_rot, k_rot = triton_apply_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert q_rot.dtype == q.dtype
        assert k_rot.dtype == k.dtype

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_rope_3d_cos_sin(self, dtype: torch.dtype) -> None:
        """3D cos/sin [batch, seq_len, head_dim] for continuous batching decode."""
        batch, heads, kv_heads, seq_len, head_dim = 4, 8, 4, 1, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device="cuda")

        # Build full RoPE table, then index with different positions per batch element.
        max_pos = 128
        cos_full, sin_full = build_rope_cos_sin(head_dim, max_pos)
        cos_full = cos_full.to(dtype=dtype, device="cuda")
        sin_full = sin_full.to(dtype=dtype, device="cuda")

        # Simulate continuous batching: each request at a different position.
        positions = torch.tensor([[10], [25], [50], [100]], device="cuda")
        cos_3d = cos_full[positions]  # [4, 1, head_dim]
        sin_3d = sin_full[positions]  # [4, 1, head_dim]

        q_tri, k_tri = triton_apply_rope(q, k, cos_3d, sin_3d)

        # Compare element-wise against 2D RoPE at each position.
        for i in range(batch):
            pos = positions[i, 0].item()
            cos_2d = cos_full[pos : pos + 1]  # [1, head_dim]
            sin_2d = sin_full[pos : pos + 1]
            q_ref_i, k_ref_i = _pytorch_apply_rope(q[i : i + 1], k[i : i + 1], cos_2d, sin_2d)
            _assert_close(q_tri[i : i + 1], q_ref_i, dtype)
            _assert_close(k_tri[i : i + 1], k_ref_i, dtype)

    def test_rope_3d_matches_2d_when_same_position(self) -> None:
        """3D cos/sin with identical positions per batch element matches 2D result."""
        batch, heads, kv_heads, seq_len, head_dim = 2, 4, 2, 1, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")

        cos_full, sin_full = build_rope_cos_sin(head_dim, 32)
        cos_full = cos_full.to(device="cuda")
        sin_full = sin_full.to(device="cuda")

        pos = 5
        cos_2d = cos_full[pos : pos + 1]  # [1, head_dim]
        sin_2d = sin_full[pos : pos + 1]

        # 3D: same position for all batch elements
        positions = torch.tensor([[pos], [pos]], device="cuda")
        cos_3d = cos_full[positions]  # [2, 1, head_dim]
        sin_3d = sin_full[positions]

        q_2d, k_2d = triton_apply_rope(q, k, cos_2d, sin_2d)
        q_3d, k_3d = triton_apply_rope(q, k, cos_3d, sin_3d)

        torch.testing.assert_close(q_2d, q_3d)
        torch.testing.assert_close(k_2d, k_3d)

    def test_rope_3d_multi_token(self) -> None:
        """3D cos/sin with seq_len > 1 exercises stride_cos_batch * stride_cos_seq interaction."""
        batch, heads, kv_heads, seq_len, head_dim = 2, 4, 2, 4, 64
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")

        max_pos = 64
        cos_full, sin_full = build_rope_cos_sin(head_dim, max_pos)
        cos_full = cos_full.to(device="cuda")
        sin_full = sin_full.to(device="cuda")

        # Batch 0 at positions [5,6,7,8], batch 1 at positions [20,21,22,23]
        positions = torch.tensor([[5, 6, 7, 8], [20, 21, 22, 23]], device="cuda")
        cos_3d = cos_full[positions]  # [2, 4, head_dim]
        sin_3d = sin_full[positions]

        q_tri, k_tri = triton_apply_rope(q, k, cos_3d, sin_3d)

        # Compare per batch element against 2D reference
        for i in range(batch):
            cos_2d = cos_full[positions[i]]  # [4, head_dim]
            sin_2d = sin_full[positions[i]]
            q_ref_i, k_ref_i = _pytorch_apply_rope(q[i : i + 1], k[i : i + 1], cos_2d, sin_2d)
            _assert_close(q_tri[i : i + 1], q_ref_i, torch.float32)
            _assert_close(k_tri[i : i + 1], k_ref_i, torch.float32)

    def test_rope_3d_rejects_invalid_shapes(self) -> None:
        """Wrapper rejects 1D and 4D cos/sin."""
        q = torch.randn(1, 4, 1, 64, device="cuda")
        k = torch.randn(1, 2, 1, 64, device="cuda")
        cos_1d = torch.randn(64, device="cuda")
        sin_1d = torch.randn(64, device="cuda")
        with pytest.raises(ValueError, match=r"2-D.*or 3-D"):
            triton_apply_rope(q, k, cos_1d, sin_1d)

    def test_rope_mismatched_cos_sin_ndim(self) -> None:
        """Wrapper rejects cos and sin with different ndim."""
        q = torch.randn(1, 4, 1, 64, device="cuda")
        k = torch.randn(1, 2, 1, 64, device="cuda")
        cos_2d = torch.randn(1, 64, device="cuda")
        sin_3d = torch.randn(1, 1, 64, device="cuda")
        with pytest.raises(ValueError, match="same number of dimensions"):
            triton_apply_rope(q, k, cos_2d, sin_3d)


# ---------------------------------------------------------------------------
# Fused residual + RMSNorm tests
# ---------------------------------------------------------------------------


class TestTritonFusedResidualRMSNorm:
    """Test fused residual + RMSNorm kernel against PyTorch reference."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize(
        "shape",
        [(1, 128), (4, 1024), (2, 8, 256)],
        ids=["1x128", "4x1024", "2x8x256"],
    )
    def test_standard_fused_norm(self, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
        N = shape[-1]
        residual = torch.randn(shape, dtype=dtype, device="cuda")
        x = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(N, dtype=dtype, device="cuda")
        eps = 1e-6

        combined, normed = triton_fused_residual_rms_norm(
            residual, x, weight, eps=eps, gemma_style=False
        )
        expected_comb, expected_norm = _pytorch_fused_residual_rms_norm(
            residual, x, weight, eps, gemma_style=False
        )

        _assert_close(combined, expected_comb, dtype)
        _assert_close(normed, expected_norm, dtype)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    def test_gemma3_fused_norm(self, dtype: torch.dtype) -> None:
        shape = (2, 4, 256)
        N = shape[-1]
        residual = torch.randn(shape, dtype=dtype, device="cuda")
        x = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(N, dtype=dtype, device="cuda")
        eps = 1e-6

        combined, normed = triton_fused_residual_rms_norm(
            residual, x, weight, eps=eps, gemma_style=True
        )
        expected_comb, expected_norm = _pytorch_fused_residual_rms_norm(
            residual, x, weight, eps, gemma_style=True
        )

        _assert_close(combined, expected_comb, dtype)
        _assert_close(normed, expected_norm, dtype)


# ---------------------------------------------------------------------------
# Fused gated activation tests
# ---------------------------------------------------------------------------


class TestTritonFusedGatedActivation:
    """Test fused gated activation kernel against PyTorch reference."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
    @pytest.mark.parametrize(
        "shape",
        [(1, 128), (4, 1024), (1, 4096), (2, 8, 256)],
        ids=["1x128", "4x1024", "1x4096", "2x8x256"],
    )
    @pytest.mark.parametrize("use_gelu", [False, True], ids=["silu", "gelu"])
    def test_fused_activation(
        self, dtype: torch.dtype, shape: tuple[int, ...], use_gelu: bool
    ) -> None:
        gate = torch.randn(shape, dtype=dtype, device="cuda")
        up = torch.randn(shape, dtype=dtype, device="cuda")

        result = triton_fused_gated_activation(gate, up, use_gelu=use_gelu)
        expected = _pytorch_fused_gated_activation(gate, up, use_gelu=use_gelu)

        _assert_close(result, expected, dtype)

    def test_output_dtype_preserved(self) -> None:
        gate = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        result = triton_fused_gated_activation(gate, up, use_gelu=False)
        assert result.dtype == torch.bfloat16

    def test_output_shape_preserved(self) -> None:
        gate = torch.randn(2, 4, 256, dtype=torch.float32, device="cuda")
        up = torch.randn(2, 4, 256, dtype=torch.float32, device="cuda")
        result = triton_fused_gated_activation(gate, up, use_gelu=True)
        assert result.shape == gate.shape
