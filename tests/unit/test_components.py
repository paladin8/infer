"""Tests for shared model components."""

from __future__ import annotations

import math

import torch

from infer.models.common import RMSNorm, apply_rope, build_rope_cos_sin


class TestRMSNorm:
    """Test RMSNorm matches the expected formula."""

    def test_output_matches_formula(self) -> None:
        dim = 16
        norm = RMSNorm(dim, eps=1e-6)
        # Use a non-trivial weight so we're not just multiplying by 1.
        norm.weight = torch.nn.Parameter(torch.randn(dim))
        x = torch.randn(2, 4, dim)

        result = norm(x)
        # Reference computation in float32 (matches the upcast inside RMSNorm).
        x_f32 = x.to(torch.float32)
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = (norm.weight * normed).to(x.dtype)
        torch.testing.assert_close(result, expected)

    def test_output_shape(self) -> None:
        dim = 32
        norm = RMSNorm(dim)
        x = torch.randn(2, 8, dim)
        assert norm(x).shape == x.shape

    def test_output_dtype(self) -> None:
        dim = 16
        norm = RMSNorm(dim)
        x = torch.randn(2, 4, dim)
        assert norm(x).dtype == x.dtype

    def test_unit_weight_is_identity_scale(self) -> None:
        """With weight=ones, output should equal normalized x."""
        dim = 8
        norm = RMSNorm(dim, eps=1e-6)
        x = torch.randn(3, dim)

        result = norm(x)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = x * rms
        torch.testing.assert_close(result, expected)

    def test_different_eps(self) -> None:
        """Different eps values produce different outputs for near-zero input."""
        dim = 4
        norm_small = RMSNorm(dim, eps=1e-10)
        norm_large = RMSNorm(dim, eps=1.0)
        x = torch.full((1, dim), 1e-6)
        out_small = norm_small(x)
        out_large = norm_large(x)
        assert not torch.allclose(out_small, out_large)

    def test_bfloat16_matches_float32(self) -> None:
        """bfloat16 input should produce results close to a float32 reference."""
        dim = 64
        norm = RMSNorm(dim, eps=1e-6)
        x_f32 = torch.randn(2, 4, dim)
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
        import pytest

        with pytest.raises(ValueError, match="Unknown rope_type"):
            build_rope_cos_sin(64, 128, rope_scaling={"rope_type": "unknown"})


class TestApplyRope:
    """Test RoPE rotation application."""

    def test_output_shapes(self) -> None:
        batch, heads, kv_heads, seq_len, head_dim = 2, 4, 2, 8, 16
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, kv_heads, seq_len, head_dim)
        cos, sin = build_rope_cos_sin(head_dim, seq_len)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_zero_is_identity(self) -> None:
        """At position 0, angles are all 0, so rotation should be identity."""
        head_dim = 8
        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)
        cos, sin = build_rope_cos_sin(head_dim, 1)
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
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # [1, 1, 1, 4]
        k = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])

        q_rot, _k_rot = apply_rope(q, k, cos_p.unsqueeze(0), sin_p.unsqueeze(0))

        # Manual: rotate_half([a,b,c,d]) = [-c,-d,a,b]
        # rotated = [a,b,c,d]*cos + [-c,-d,a,b]*sin
        inv_freq = torch.tensor([1.0, 1.0 / 100.0])
        angles = inv_freq * pos
        c = torch.cat([angles.cos(), angles.cos()])
        s = torch.cat([angles.sin(), angles.sin()])

        q_flat = q.squeeze()
        q_half = torch.cat([-q_flat[2:], q_flat[:2]])
        expected_q = q_flat * c + q_half * s

        torch.testing.assert_close(q_rot.squeeze(), expected_q)

    def test_different_positions_differ(self) -> None:
        """Different positions should produce different rotations."""
        head_dim = 16
        q = torch.randn(1, 1, 4, head_dim)
        k = torch.randn(1, 1, 4, head_dim)
        cos, sin = build_rope_cos_sin(head_dim, 4)
        q_rot, _ = apply_rope(q, k, cos, sin)
        # Positions 0 and 1 should differ (unless input happens to be zero).
        assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 1])

    def test_rotation_preserves_norm(self) -> None:
        """RoPE rotation should preserve vector norms."""
        head_dim = 32
        q = torch.randn(2, 4, 8, head_dim)
        k = torch.randn(2, 2, 8, head_dim)
        cos, sin = build_rope_cos_sin(head_dim, 8)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=1e-5)

    def test_preserves_dtype(self) -> None:
        """Output dtype should match input dtype even when cos/sin are float32."""
        head_dim = 16
        q = torch.randn(1, 1, 4, head_dim, dtype=torch.bfloat16)
        k = torch.randn(1, 1, 4, head_dim, dtype=torch.bfloat16)
        cos, sin = build_rope_cos_sin(head_dim, 4)  # float32
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert q_rot.dtype == torch.bfloat16
        assert k_rot.dtype == torch.bfloat16
