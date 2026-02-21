"""Tests for shared model components."""

from __future__ import annotations

import math

import pytest
import torch

from infer.cache.simple import KVCache
from infer.models.common import (
    Attention,
    GatedMLP,
    RMSNorm,
    apply_rope,
    build_rope_cos_sin,
    causal_mask,
    sliding_window_causal_mask,
)

_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

_requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required for Triton kernels")


@_requires_cuda
class TestRMSNorm:
    """Test RMSNorm matches the expected formula."""

    def test_output_matches_formula(self) -> None:
        dim = 16
        norm = RMSNorm(dim, eps=1e-6).to(_DEVICE)
        # Use a non-trivial weight so we're not just multiplying by 1.
        norm.weight = torch.nn.Parameter(torch.randn(dim, device=_DEVICE))
        x = torch.randn(2, 4, dim, device=_DEVICE)

        result = norm(x)
        # Reference computation in float32 (matches the upcast inside RMSNorm).
        x_f32 = x.to(torch.float32)
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = (norm.weight * normed).to(x.dtype)
        torch.testing.assert_close(result, expected)

    def test_output_shape(self) -> None:
        dim = 32
        norm = RMSNorm(dim).to(_DEVICE)
        x = torch.randn(2, 8, dim, device=_DEVICE)
        assert norm(x).shape == x.shape

    def test_output_dtype(self) -> None:
        dim = 16
        norm = RMSNorm(dim).to(_DEVICE)
        x = torch.randn(2, 4, dim, device=_DEVICE)
        assert norm(x).dtype == x.dtype

    def test_unit_weight_is_identity_scale(self) -> None:
        """With weight=ones, output should equal normalized x."""
        dim = 8
        norm = RMSNorm(dim, eps=1e-6).to(_DEVICE)
        x = torch.randn(3, dim, device=_DEVICE)

        result = norm(x)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = x * rms
        torch.testing.assert_close(result, expected)

    def test_different_eps(self) -> None:
        """Different eps values produce different outputs for near-zero input."""
        dim = 4
        norm_small = RMSNorm(dim, eps=1e-10).to(_DEVICE)
        norm_large = RMSNorm(dim, eps=1.0).to(_DEVICE)
        x = torch.full((1, dim), 1e-6, device=_DEVICE)
        out_small = norm_small(x)
        out_large = norm_large(x)
        assert not torch.allclose(out_small, out_large)

    def test_bfloat16_matches_float32(self) -> None:
        """bfloat16 input should produce results close to a float32 reference."""
        dim = 64
        norm = RMSNorm(dim, eps=1e-6).to(_DEVICE)
        x_f32 = torch.randn(2, 4, dim, device=_DEVICE)
        x_bf16 = x_f32.to(torch.bfloat16)
        ref = norm(x_f32)
        out = norm(x_bf16)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


class TestBuildRopeCosSin:
    """Test the RoPE cos/sin table factory."""

    def test_vanilla_shape(self) -> None:
        head_dim, seq_len = 64, 128
        cos, sin = build_rope_cos_sin(head_dim, seq_len)
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

    def test_llama3_scaled_shape(self) -> None:
        head_dim, seq_len = 64, 128
        scaling = {
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        }
        cos, sin = build_rope_cos_sin(head_dim, seq_len, theta=500000.0, rope_scaling=scaling)
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

    def test_linear_scaled_shape(self) -> None:
        head_dim, seq_len = 64, 128
        scaling = {"rope_type": "linear", "factor": 4.0}
        cos, sin = build_rope_cos_sin(head_dim, seq_len, rope_scaling=scaling)
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

    def test_different_theta_produces_different_tables(self) -> None:
        cos_a, sin_a = build_rope_cos_sin(64, 128, theta=10000.0)
        cos_b, sin_b = build_rope_cos_sin(64, 128, theta=1000000.0)
        assert not torch.allclose(cos_a, cos_b)
        assert not torch.allclose(sin_a, sin_b)

    def test_position_zero_cos_is_one(self) -> None:
        """At position 0, all angles are 0 so cos should be 1."""
        cos, sin = build_rope_cos_sin(32, 16)
        torch.testing.assert_close(cos[0], torch.ones(32))
        torch.testing.assert_close(sin[0], torch.zeros(32))

    def test_vanilla_matches_manual_computation(self) -> None:
        """Verify vanilla RoPE against hand-computed values for head_dim=4."""
        head_dim, seq_len, theta = 4, 3, 10000.0
        cos, sin = build_rope_cos_sin(head_dim, seq_len, theta=theta)

        # inv_freq: [theta^0, theta^(-2/4)] = [1.0, 1/100]
        inv_freq = torch.tensor([1.0, 1.0 / 100.0])
        for pos in range(seq_len):
            angles = inv_freq * pos  # [head_dim/2]
            expected_cos = torch.cat([angles.cos(), angles.cos()])
            expected_sin = torch.cat([angles.sin(), angles.sin()])
            torch.testing.assert_close(cos[pos], expected_cos)
            torch.testing.assert_close(sin[pos], expected_sin)

    def test_linear_scaling_halves_frequencies(self) -> None:
        """Linear scaling by factor=2 should halve inv_freq, so tables at
        position 2*p with factor=2 should match position p without scaling."""
        head_dim, theta = 16, 10000.0
        cos_vanilla, sin_vanilla = build_rope_cos_sin(head_dim, 64, theta=theta)
        cos_scaled, sin_scaled = build_rope_cos_sin(
            head_dim, 128, theta=theta, rope_scaling={"rope_type": "linear", "factor": 2.0}
        )
        # Position p with factor=2 should equal position p/2 without scaling,
        # i.e. scaled[2*p] == vanilla[p].
        for p in range(64):
            torch.testing.assert_close(cos_scaled[2 * p], cos_vanilla[p])
            torch.testing.assert_close(sin_scaled[2 * p], sin_vanilla[p])

    def test_llama3_high_freq_unchanged(self) -> None:
        """High-frequency components should be identical to vanilla RoPE."""
        head_dim, theta = 128, 500000.0
        scaling = {
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        }
        cos_vanilla, _ = build_rope_cos_sin(head_dim, 64, theta=theta)
        cos_scaled, _ = build_rope_cos_sin(head_dim, 64, theta=theta, rope_scaling=scaling)

        # Find which frequency components are in the high-freq band.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        wavelen = 2 * math.pi / inv_freq
        high_freq_wavelen = 8192 / 4.0
        high_freq_mask = wavelen < high_freq_wavelen

        # The cos table repeats frequencies: [f0,f1,...,f_{d/2-1}, f0,f1,...].
        full_mask = torch.cat([high_freq_mask, high_freq_mask])
        if full_mask.any():
            torch.testing.assert_close(cos_scaled[:, full_mask], cos_vanilla[:, full_mask])

    def test_llama3_low_freq_fully_scaled(self) -> None:
        """Low-frequency components should be scaled by 1/factor."""
        head_dim, theta = 128, 500000.0
        factor = 32.0
        scaling = {
            "rope_type": "llama3",
            "factor": factor,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        }
        cos_scaled, _ = build_rope_cos_sin(head_dim, 64, theta=theta, rope_scaling=scaling)

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        wavelen = 2 * math.pi / inv_freq
        low_freq_wavelen = 8192 / 1.0
        low_freq_mask = wavelen > low_freq_wavelen

        if low_freq_mask.any():
            # For low-freq band: scaled_inv_freq = inv_freq / factor,
            # so scaled angles = vanilla angles / factor.
            # Build expected by linear-scaling those components.
            cos_linear, _ = build_rope_cos_sin(
                head_dim, 64, theta=theta, rope_scaling={"rope_type": "linear", "factor": factor}
            )
            full_mask = torch.cat([low_freq_mask, low_freq_mask])
            torch.testing.assert_close(cos_scaled[:, full_mask], cos_linear[:, full_mask])

    def test_rope_type_default_is_vanilla(self) -> None:
        """rope_type='default' should produce the same tables as None."""
        cos_none, sin_none = build_rope_cos_sin(32, 16)
        cos_default, sin_default = build_rope_cos_sin(32, 16, rope_scaling={"rope_type": "default"})
        torch.testing.assert_close(cos_none, cos_default)
        torch.testing.assert_close(sin_none, sin_default)

    def test_unknown_rope_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown rope_type"):
            build_rope_cos_sin(64, 128, rope_scaling={"rope_type": "unknown"})


@_requires_cuda
class TestApplyRope:
    """Test RoPE rotation application."""

    def test_output_shapes(self) -> None:
        batch, heads, kv_heads, seq_len, head_dim = 2, 4, 2, 8, 16
        q = torch.randn(batch, heads, seq_len, head_dim, device=_DEVICE)
        k = torch.randn(batch, kv_heads, seq_len, head_dim, device=_DEVICE)
        cos, sin = build_rope_cos_sin(head_dim, seq_len)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_zero_is_identity(self) -> None:
        """At position 0, angles are all 0, so rotation should be identity."""
        head_dim = 8
        q = torch.randn(1, 1, 1, head_dim, device=_DEVICE)
        k = torch.randn(1, 1, 1, head_dim, device=_DEVICE)
        cos, sin = build_rope_cos_sin(head_dim, 1)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        torch.testing.assert_close(q_rot, q)
        torch.testing.assert_close(k_rot, k)

    def test_hand_computed_rotation(self) -> None:
        """Verify rotation against manual calculation with head_dim=4."""
        head_dim, theta = 4, 10000.0
        pos = 1

        # Build tables for 2 positions, use position 1.
        cos, sin = build_rope_cos_sin(head_dim, 2, theta=theta)
        cos_p = cos[pos]  # [head_dim]
        sin_p = sin[pos]

        # Input: q = [a, b, c, d] at position 1.
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], device=_DEVICE)  # [1, 1, 1, 4]
        k = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]], device=_DEVICE)

        q_rot, _k_rot = apply_rope(
            q, k, cos_p.unsqueeze(0).to(_DEVICE), sin_p.unsqueeze(0).to(_DEVICE)
        )

        # Manual: rotate_half([a,b,c,d]) = [-c,-d,a,b]
        # rotated = [a,b,c,d]*cos + [-c,-d,a,b]*sin
        inv_freq = torch.tensor([1.0, 1.0 / 100.0])
        angles = inv_freq * pos
        c = torch.cat([angles.cos(), angles.cos()])
        s = torch.cat([angles.sin(), angles.sin()])

        q_flat = q.squeeze()
        q_half = torch.cat([-q_flat[2:], q_flat[:2]])
        expected_q = q_flat * c.to(_DEVICE) + q_half * s.to(_DEVICE)

        torch.testing.assert_close(q_rot.squeeze(), expected_q)

    def test_different_positions_differ(self) -> None:
        """Different positions should produce different rotations."""
        head_dim = 16
        q = torch.randn(1, 1, 4, head_dim, device=_DEVICE)
        k = torch.randn(1, 1, 4, head_dim, device=_DEVICE)
        cos, sin = build_rope_cos_sin(head_dim, 4)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        q_rot, _ = apply_rope(q, k, cos, sin)
        # Positions 0 and 1 should differ (unless input happens to be zero).
        assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 1])

    def test_rotation_preserves_norm(self) -> None:
        """RoPE rotation should preserve vector norms."""
        head_dim = 32
        q = torch.randn(2, 4, 8, head_dim, device=_DEVICE)
        k = torch.randn(2, 2, 8, head_dim, device=_DEVICE)
        cos, sin = build_rope_cos_sin(head_dim, 8)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=1e-5)

    def test_preserves_dtype(self) -> None:
        """Output dtype should match input dtype even when cos/sin are float32."""
        head_dim = 16
        q = torch.randn(1, 1, 4, head_dim, dtype=torch.bfloat16, device=_DEVICE)
        k = torch.randn(1, 1, 4, head_dim, dtype=torch.bfloat16, device=_DEVICE)
        cos, sin = build_rope_cos_sin(head_dim, 4)  # float32
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.dtype == torch.bfloat16
        assert k_rot.dtype == torch.bfloat16


@_requires_cuda
class TestAttention:
    """Test the multi-head attention module."""

    # Shared small dimensions for most tests.
    HIDDEN = 32
    NUM_HEADS = 4
    NUM_KV_HEADS = 2
    HEAD_DIM = 8

    def _make_attn(self, **overrides: object) -> Attention:
        kwargs: dict[str, object] = {
            "hidden_size": self.HIDDEN,
            "num_heads": self.NUM_HEADS,
            "num_kv_heads": self.NUM_KV_HEADS,
            "head_dim": self.HEAD_DIM,
        }
        kwargs.update(overrides)
        return Attention(**kwargs).to(_DEVICE)  # type: ignore[arg-type]

    def _forward(
        self, attn: Attention, batch: int = 1, seq_len: int = 4, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = torch.randn(batch, seq_len, self.HIDDEN, device=_DEVICE)
        cos, sin = build_rope_cos_sin(attn.head_dim, seq_len)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        return attn(x, cos, sin, mask=mask)

    def test_output_shape(self) -> None:
        attn = self._make_attn()
        out = self._forward(attn, batch=2, seq_len=6)
        assert out.shape == (2, 6, self.HIDDEN)

    def test_output_shape_no_gqa(self) -> None:
        """When num_kv_heads == num_heads, no GQA expansion needed."""
        attn = self._make_attn(num_kv_heads=self.NUM_HEADS)
        out = self._forward(attn, batch=1, seq_len=4)
        assert out.shape == (1, 4, self.HIDDEN)

    def test_with_qk_norm(self) -> None:
        attn = self._make_attn(qk_norm=True)
        assert attn.q_norm is not None
        assert attn.k_norm is not None
        out = self._forward(attn)
        assert out.shape == (1, 4, self.HIDDEN)

    def test_without_qk_norm(self) -> None:
        attn = self._make_attn(qk_norm=False)
        assert attn.q_norm is None
        assert attn.k_norm is None
        out = self._forward(attn)
        assert out.shape == (1, 4, self.HIDDEN)

    def test_with_bias(self) -> None:
        attn = self._make_attn(bias=True)
        assert attn.q_proj.bias is not None
        assert attn.o_proj.bias is not None
        out = self._forward(attn)
        assert out.shape == (1, 4, self.HIDDEN)

    def test_without_bias(self) -> None:
        attn = self._make_attn(bias=False)
        assert attn.q_proj.bias is None
        assert attn.o_proj.bias is None

    def test_custom_scale(self) -> None:
        """Custom query_pre_attn_scalar should override default scaling."""
        scalar = 256.0
        attn = self._make_attn(scale=scalar**-0.5)
        assert attn.scale == scalar**-0.5
        out = self._forward(attn)
        assert out.shape == (1, 4, self.HIDDEN)

    def test_default_scale(self) -> None:
        attn = self._make_attn()
        assert attn.scale == self.HEAD_DIM**-0.5

    def test_with_causal_mask(self) -> None:
        seq_len = 6
        mask = causal_mask(seq_len, device=_DEVICE)
        attn = self._make_attn()
        out = self._forward(attn, seq_len=seq_len, mask=mask)
        assert out.shape == (1, seq_len, self.HIDDEN)

    def test_decoupled_head_dim(self) -> None:
        """head_dim can differ from hidden_size // num_heads (like Gemma 3 1B)."""
        # Gemma 3 1B style: hidden=1152, heads=4, head_dim=256
        attn = Attention(
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,  # != 64 // 4 = 16
        ).to(_DEVICE)
        x = torch.randn(1, 4, 64, device=_DEVICE)
        cos, sin = build_rope_cos_sin(32, 4)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        out = attn(x, cos, sin)
        assert out.shape == (1, 4, 64)

    def test_projection_dimensions(self) -> None:
        """Verify projection sizes use num_heads * head_dim, not hidden_size."""
        attn = self._make_attn()
        assert attn.q_proj.out_features == self.NUM_HEADS * self.HEAD_DIM
        assert attn.k_proj.out_features == self.NUM_KV_HEADS * self.HEAD_DIM
        assert attn.v_proj.out_features == self.NUM_KV_HEADS * self.HEAD_DIM
        assert attn.o_proj.in_features == self.NUM_HEADS * self.HEAD_DIM
        assert attn.o_proj.out_features == self.HIDDEN

    def test_indivisible_heads_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            Attention(
                hidden_size=self.HIDDEN,
                num_heads=4,
                num_kv_heads=3,
                head_dim=self.HEAD_DIM,
            )

    # --- KV cache tests ---

    def test_kv_cache_prefill_output_shape(self) -> None:
        """With KV cache, prefill output shape matches no-cache output."""
        attn = self._make_attn()
        seq_len = 6
        cache = KVCache.allocate(
            num_layers=1,
            num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=seq_len + 10,
            dtype=torch.float32,
            device=_DEVICE,
        )
        x = torch.randn(1, seq_len, self.HIDDEN, device=_DEVICE)
        cos, sin = build_rope_cos_sin(self.HEAD_DIM, seq_len)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        mask = causal_mask(seq_len, device=_DEVICE)
        out = attn(x, cos, sin, mask=mask, kv_cache=cache, layer_idx=0)
        assert out.shape == (1, seq_len, self.HIDDEN)

    def test_kv_cache_decode_output_shape(self) -> None:
        """Single-token decode with cache produces [batch, 1, hidden] output."""
        attn = self._make_attn()
        prompt_len = 4
        cache = KVCache.allocate(
            num_layers=1,
            num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=prompt_len + 5,
            dtype=torch.float32,
            device=_DEVICE,
        )
        # Prefill
        x = torch.randn(1, prompt_len, self.HIDDEN, device=_DEVICE)
        cos, sin = build_rope_cos_sin(self.HEAD_DIM, prompt_len)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        mask = causal_mask(prompt_len, device=_DEVICE)
        attn(x, cos, sin, mask=mask, kv_cache=cache, layer_idx=0)
        cache.advance(prompt_len)

        # Decode one token
        x_decode = torch.randn(1, 1, self.HIDDEN, device=_DEVICE)
        cos_d, sin_d = build_rope_cos_sin(self.HEAD_DIM, prompt_len + 1)
        cos_d = cos_d[prompt_len : prompt_len + 1].to(_DEVICE)
        sin_d = sin_d[prompt_len : prompt_len + 1].to(_DEVICE)
        out = attn(x_decode, cos_d, sin_d, mask=None, kv_cache=cache, layer_idx=0)
        assert out.shape == (1, 1, self.HIDDEN)

    def test_kv_cache_equivalence(self) -> None:
        """Cached prefill + decode produces same output as full no-cache forward."""
        attn = self._make_attn()
        torch.manual_seed(42)
        seq_len = 5
        x_all = torch.randn(1, seq_len, self.HIDDEN, device=_DEVICE)
        cos_all, sin_all = build_rope_cos_sin(self.HEAD_DIM, seq_len)
        cos_all, sin_all = cos_all.to(_DEVICE), sin_all.to(_DEVICE)
        mask_all = causal_mask(seq_len, device=_DEVICE)

        # Full forward without cache.
        out_no_cache = attn(x_all, cos_all, sin_all, mask=mask_all)

        # Prefill first 4, then decode 1.
        cache = KVCache.allocate(
            num_layers=1,
            num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=seq_len,
            dtype=torch.float32,
            device=_DEVICE,
        )
        prefill_len = seq_len - 1
        mask_prefill = causal_mask(prefill_len, device=_DEVICE)
        attn(
            x_all[:, :prefill_len, :],
            cos_all[:prefill_len],
            sin_all[:prefill_len],
            mask=mask_prefill,
            kv_cache=cache,
            layer_idx=0,
        )
        cache.advance(prefill_len)

        # Decode last token — no mask needed for full attention.
        out_decode = attn(
            x_all[:, prefill_len : prefill_len + 1, :],
            cos_all[prefill_len : prefill_len + 1],
            sin_all[prefill_len : prefill_len + 1],
            mask=None,
            kv_cache=cache,
            layer_idx=0,
        )

        # The last position's output should match.
        torch.testing.assert_close(
            out_decode[:, 0, :], out_no_cache[:, -1, :], atol=1e-5, rtol=1e-5
        )

    def test_kv_cache_none_is_noop(self) -> None:
        """Passing kv_cache=None behaves identically to the original forward."""
        attn = self._make_attn()
        torch.manual_seed(0)
        x = torch.randn(1, 4, self.HIDDEN, device=_DEVICE)
        cos, sin = build_rope_cos_sin(self.HEAD_DIM, 4)
        cos, sin = cos.to(_DEVICE), sin.to(_DEVICE)
        mask = causal_mask(4, device=_DEVICE)

        out1 = attn(x, cos, sin, mask=mask)
        out2 = attn(x, cos, sin, mask=mask, kv_cache=None, layer_idx=0)
        torch.testing.assert_close(out1, out2)

    def test_kv_cache_equivalence_with_qk_norm(self) -> None:
        """Cached path matches no-cache path when QK-norm is enabled."""
        attn = self._make_attn(qk_norm=True)
        torch.manual_seed(99)
        seq_len = 5
        x_all = torch.randn(1, seq_len, self.HIDDEN, device=_DEVICE)
        cos_all, sin_all = build_rope_cos_sin(self.HEAD_DIM, seq_len)
        cos_all, sin_all = cos_all.to(_DEVICE), sin_all.to(_DEVICE)
        mask_all = causal_mask(seq_len, device=_DEVICE)

        # Full forward without cache.
        out_no_cache = attn(x_all, cos_all, sin_all, mask=mask_all)

        # Prefill first 4, then decode 1.
        cache = KVCache.allocate(
            num_layers=1,
            num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=seq_len,
            dtype=torch.float32,
            device=_DEVICE,
        )
        prefill_len = seq_len - 1
        attn(
            x_all[:, :prefill_len, :],
            cos_all[:prefill_len],
            sin_all[:prefill_len],
            mask=causal_mask(prefill_len, device=_DEVICE),
            kv_cache=cache,
            layer_idx=0,
        )
        cache.advance(prefill_len)

        out_decode = attn(
            x_all[:, prefill_len : prefill_len + 1, :],
            cos_all[prefill_len : prefill_len + 1],
            sin_all[prefill_len : prefill_len + 1],
            mask=None,
            kv_cache=cache,
            layer_idx=0,
        )

        torch.testing.assert_close(
            out_decode[:, 0, :], out_no_cache[:, -1, :], atol=1e-5, rtol=1e-5
        )

    def test_kv_cache_multi_step_decode(self) -> None:
        """Prefill(3) + decode 3 tokens one-at-a-time matches a full 6-token forward."""
        attn = self._make_attn()
        torch.manual_seed(7)
        total_len = 6
        prefill_len = 3
        x_all = torch.randn(1, total_len, self.HIDDEN, device=_DEVICE)
        cos_all, sin_all = build_rope_cos_sin(self.HEAD_DIM, total_len)
        cos_all, sin_all = cos_all.to(_DEVICE), sin_all.to(_DEVICE)
        mask_all = causal_mask(total_len, device=_DEVICE)

        # Full forward without cache.
        out_no_cache = attn(x_all, cos_all, sin_all, mask=mask_all)

        # Cached: prefill first 3.
        cache = KVCache.allocate(
            num_layers=1,
            num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=total_len,
            dtype=torch.float32,
            device=_DEVICE,
        )
        attn(
            x_all[:, :prefill_len, :],
            cos_all[:prefill_len],
            sin_all[:prefill_len],
            mask=causal_mask(prefill_len, device=_DEVICE),
            kv_cache=cache,
            layer_idx=0,
        )
        cache.advance(prefill_len)

        # Decode tokens 3, 4, 5 one at a time.
        for i in range(prefill_len, total_len):
            out_decode = attn(
                x_all[:, i : i + 1, :],
                cos_all[i : i + 1],
                sin_all[i : i + 1],
                mask=None,
                kv_cache=cache,
                layer_idx=0,
            )
            cache.advance(1)

            torch.testing.assert_close(
                out_decode[:, 0, :], out_no_cache[:, i, :], atol=1e-5, rtol=1e-5
            )

    def test_kv_cache_equivalence_no_gqa(self) -> None:
        """Cached path works when num_kv_heads == num_heads (MHA, no GQA expansion)."""
        attn = self._make_attn(num_kv_heads=self.NUM_HEADS)
        torch.manual_seed(13)
        seq_len = 5
        x_all = torch.randn(1, seq_len, self.HIDDEN, device=_DEVICE)
        cos_all, sin_all = build_rope_cos_sin(self.HEAD_DIM, seq_len)
        cos_all, sin_all = cos_all.to(_DEVICE), sin_all.to(_DEVICE)
        mask_all = causal_mask(seq_len, device=_DEVICE)

        out_no_cache = attn(x_all, cos_all, sin_all, mask=mask_all)

        cache = KVCache.allocate(
            num_layers=1,
            num_kv_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=seq_len,
            dtype=torch.float32,
            device=_DEVICE,
        )
        prefill_len = seq_len - 1
        attn(
            x_all[:, :prefill_len, :],
            cos_all[:prefill_len],
            sin_all[:prefill_len],
            mask=causal_mask(prefill_len, device=_DEVICE),
            kv_cache=cache,
            layer_idx=0,
        )
        cache.advance(prefill_len)

        out_decode = attn(
            x_all[:, prefill_len : prefill_len + 1, :],
            cos_all[prefill_len : prefill_len + 1],
            sin_all[prefill_len : prefill_len + 1],
            mask=None,
            kv_cache=cache,
            layer_idx=0,
        )

        torch.testing.assert_close(
            out_decode[:, 0, :], out_no_cache[:, -1, :], atol=1e-5, rtol=1e-5
        )


@_requires_cuda
class TestGatedMLP:
    """Test the gated MLP module."""

    HIDDEN = 32
    INTERMEDIATE = 64

    def test_output_shape(self) -> None:
        mlp = GatedMLP(self.HIDDEN, self.INTERMEDIATE).to(_DEVICE)
        x = torch.randn(2, 8, self.HIDDEN, device=_DEVICE)
        assert mlp(x).shape == x.shape

    def test_silu_matches_manual(self) -> None:
        """Verify output matches down(silu(gate(x)) * up(x))."""
        mlp = GatedMLP(self.HIDDEN, self.INTERMEDIATE, act_fn="silu").to(_DEVICE)
        x = torch.randn(1, 4, self.HIDDEN, device=_DEVICE)

        expected = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        torch.testing.assert_close(mlp(x), expected)

    def test_gelu_tanh_matches_manual(self) -> None:
        """Verify output matches down(gelu_tanh(gate(x)) * up(x))."""
        mlp = GatedMLP(self.HIDDEN, self.INTERMEDIATE, act_fn="gelu_pytorch_tanh").to(_DEVICE)
        x = torch.randn(1, 4, self.HIDDEN, device=_DEVICE)

        gelu = torch.nn.GELU(approximate="tanh")
        expected = mlp.down_proj(gelu(mlp.gate_proj(x)) * mlp.up_proj(x))
        torch.testing.assert_close(mlp(x), expected)

    def test_with_bias(self) -> None:
        mlp = GatedMLP(self.HIDDEN, self.INTERMEDIATE, bias=True).to(_DEVICE)
        assert mlp.gate_proj.bias is not None
        assert mlp.up_proj.bias is not None
        assert mlp.down_proj.bias is not None
        x = torch.randn(1, 4, self.HIDDEN, device=_DEVICE)
        assert mlp(x).shape == (1, 4, self.HIDDEN)

    def test_without_bias(self) -> None:
        mlp = GatedMLP(self.HIDDEN, self.INTERMEDIATE, bias=False)
        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_projection_dimensions(self) -> None:
        mlp = GatedMLP(self.HIDDEN, self.INTERMEDIATE)
        assert mlp.gate_proj.in_features == self.HIDDEN
        assert mlp.gate_proj.out_features == self.INTERMEDIATE
        assert mlp.up_proj.in_features == self.HIDDEN
        assert mlp.up_proj.out_features == self.INTERMEDIATE
        assert mlp.down_proj.in_features == self.INTERMEDIATE
        assert mlp.down_proj.out_features == self.HIDDEN

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            GatedMLP(self.HIDDEN, self.INTERMEDIATE, act_fn="relu")


class TestCausalMask:
    """Test the causal mask helper."""

    def test_shape(self) -> None:
        mask = causal_mask(8)
        assert mask.shape == (1, 1, 8, 8)

    def test_diagonal_is_zero(self) -> None:
        """Positions can attend to themselves."""
        mask = causal_mask(4)
        for i in range(4):
            assert mask[0, 0, i, i].item() == 0.0

    def test_lower_triangle_is_zero(self) -> None:
        """Positions can attend to all earlier positions."""
        mask = causal_mask(4)
        for i in range(4):
            for j in range(i + 1):
                assert mask[0, 0, i, j].item() == 0.0

    def test_upper_triangle_is_neginf(self) -> None:
        """Positions cannot attend to future positions."""
        mask = causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[0, 0, i, j].item() == float("-inf")

    def test_seq_len_one(self) -> None:
        mask = causal_mask(1)
        assert mask.shape == (1, 1, 1, 1)
        assert mask[0, 0, 0, 0].item() == 0.0


class TestSlidingWindowCausalMask:
    """Test the sliding window causal mask helper."""

    def test_shape(self) -> None:
        mask = sliding_window_causal_mask(8, window_size=3)
        assert mask.shape == (1, 1, 8, 8)

    def test_future_is_masked(self) -> None:
        """Positions cannot attend to future positions."""
        mask = sliding_window_causal_mask(6, window_size=3)
        for i in range(6):
            for j in range(i + 1, 6):
                assert mask[0, 0, i, j].item() == float("-inf")

    def test_within_window_is_zero(self) -> None:
        """Positions within the window can attend."""
        mask = sliding_window_causal_mask(6, window_size=3)
        # Position 3 should attend to positions 1, 2, 3 (window_size=3).
        assert mask[0, 0, 3, 1].item() == 0.0
        assert mask[0, 0, 3, 2].item() == 0.0
        assert mask[0, 0, 3, 3].item() == 0.0

    def test_beyond_window_is_masked(self) -> None:
        """Positions beyond the window are masked."""
        mask = sliding_window_causal_mask(6, window_size=3)
        # Position 3 should NOT attend to position 0 (distance=3 >= window_size=3).
        assert mask[0, 0, 3, 0].item() == float("-inf")
        # Position 5 should NOT attend to positions 0, 1, 2.
        assert mask[0, 0, 5, 0].item() == float("-inf")
        assert mask[0, 0, 5, 1].item() == float("-inf")
        assert mask[0, 0, 5, 2].item() == float("-inf")

    def test_large_window_equals_causal(self) -> None:
        """A window >= seq_len should produce the same mask as causal_mask."""
        seq_len = 8
        cm = causal_mask(seq_len)
        sw = sliding_window_causal_mask(seq_len, window_size=seq_len)
        torch.testing.assert_close(cm, sw)

    def test_window_size_one(self) -> None:
        """Window of 1 means each position only attends to itself."""
        mask = sliding_window_causal_mask(4, window_size=1)
        for i in range(4):
            for j in range(4):
                if i == j:
                    assert mask[0, 0, i, j].item() == 0.0
                else:
                    assert mask[0, 0, i, j].item() == float("-inf")


class TestMaskDtypeDevice:
    """Tests for dtype and device arguments on mask helpers."""

    def test_causal_mask_default_dtype(self) -> None:
        mask = causal_mask(4)
        assert mask.dtype == torch.float32
        assert mask.device == torch.device("cpu")

    def test_causal_mask_bfloat16(self) -> None:
        mask = causal_mask(4, dtype=torch.bfloat16)
        assert mask.dtype == torch.bfloat16

    def test_causal_mask_float16(self) -> None:
        mask = causal_mask(4, dtype=torch.float16)
        assert mask.dtype == torch.float16

    def test_sliding_window_default_dtype(self) -> None:
        mask = sliding_window_causal_mask(4, window_size=2)
        assert mask.dtype == torch.float32
        assert mask.device == torch.device("cpu")

    def test_sliding_window_bfloat16(self) -> None:
        mask = sliding_window_causal_mask(4, window_size=2, dtype=torch.bfloat16)
        assert mask.dtype == torch.bfloat16

    def test_causal_mask_values_preserved_in_bf16(self) -> None:
        """Verify 0.0 and -inf values are correct in bfloat16."""
        mask = causal_mask(3, dtype=torch.bfloat16)
        # Diagonal should be 0.0 (attend to self).
        assert mask[0, 0, 0, 0].item() == 0.0
        # Upper triangle should be -inf.
        assert mask[0, 0, 0, 1].item() == float("-inf")
