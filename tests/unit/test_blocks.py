"""Tests for per-architecture transformer blocks."""

from __future__ import annotations

import torch

from infer.models.common import RMSNorm, build_rope_cos_sin, causal_mask, sliding_window_causal_mask
from infer.models.gemma3 import Gemma3RMSNorm, Gemma3TransformerBlock
from infer.models.llama import LlamaTransformerBlock
from infer.models.qwen3 import Qwen3TransformerBlock

# Small dimensions for fast unit tests.
HIDDEN = 64
INTER = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
SEQ_LEN = 8
BATCH = 2
EPS = 1e-5


def _make_rope(
    seq_len: int = SEQ_LEN, head_dim: int = HEAD_DIM
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build vanilla RoPE cos/sin tables."""
    cos, sin = build_rope_cos_sin(head_dim, seq_len)
    return cos, sin


class TestLlamaTransformerBlock:
    """Test the Llama 3 transformer block."""

    def test_output_shape(self) -> None:
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_output_dtype_matches_input(self) -> None:
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        assert out.dtype == x.dtype

    def test_with_causal_mask(self) -> None:
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        mask = causal_mask(SEQ_LEN)
        out = block(x, cos, sin, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_residual_connection(self) -> None:
        """With zeroed-out sub-layers, output should equal input (residual passthrough)."""
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        # Zero all parameters in attention and MLP to make them output zero.
        with torch.no_grad():
            for name, param in block.named_parameters():
                if "layernorm" not in name:
                    param.zero_()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_no_qk_norm(self) -> None:
        """Llama block should not have QK-norm."""
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        assert block.self_attn.q_norm is None
        assert block.self_attn.k_norm is None

    def test_no_bias(self) -> None:
        """Llama block should not use bias in projections."""
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        assert block.self_attn.q_proj.bias is None
        assert block.self_attn.k_proj.bias is None
        assert block.self_attn.v_proj.bias is None
        assert block.self_attn.o_proj.bias is None
        assert block.mlp.gate_proj.bias is None
        assert block.mlp.up_proj.bias is None
        assert block.mlp.down_proj.bias is None

    def test_silu_activation(self) -> None:
        """Llama block should use SiLU (SwiGLU)."""
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        assert isinstance(block.mlp.act_fn, torch.nn.SiLU)

    def test_state_dict_keys(self) -> None:
        """State dict keys should match our internal weight naming convention."""
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        keys = set(block.state_dict().keys())
        expected = {
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }
        assert keys == expected

    def test_different_inputs_give_different_outputs(self) -> None:
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        cos, sin = _make_rope()
        x1 = torch.randn(1, SEQ_LEN, HIDDEN)
        x2 = torch.randn(1, SEQ_LEN, HIDDEN)
        out1 = block(x1, cos, sin)
        out2 = block(x2, cos, sin)
        assert not torch.allclose(out1, out2)

    def test_deterministic(self) -> None:
        """Same input should produce the same output."""
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        cos, sin = _make_rope()
        x = torch.randn(1, SEQ_LEN, HIDDEN)
        out1 = block(x, cos, sin)
        out2 = block(x, cos, sin)
        torch.testing.assert_close(out1, out2)

    def test_bfloat16(self) -> None:
        """Block should produce valid bf16 output without NaN."""
        block = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        block = block.to(torch.bfloat16)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.bfloat16)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()


class TestQwen3TransformerBlock:
    """Test the Qwen 3 transformer block."""

    def test_output_shape(self) -> None:
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_with_causal_mask(self) -> None:
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        mask = causal_mask(SEQ_LEN)
        out = block(x, cos, sin, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_has_qk_norm(self) -> None:
        """Qwen 3 block should have QK-norm enabled."""
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        assert block.self_attn.q_norm is not None
        assert block.self_attn.k_norm is not None

    def test_qk_norm_dim_is_head_dim(self) -> None:
        """QK-norm should normalize over head_dim, not hidden_size."""
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        assert block.self_attn.q_norm is not None
        assert block.self_attn.q_norm.weight.shape == (HEAD_DIM,)

    def test_no_bias(self) -> None:
        """Qwen 3 block should not use bias in projections."""
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        assert block.self_attn.q_proj.bias is None
        assert block.mlp.gate_proj.bias is None

    def test_silu_activation(self) -> None:
        """Qwen 3 block should use SiLU (SwiGLU)."""
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        assert isinstance(block.mlp.act_fn, torch.nn.SiLU)

    def test_state_dict_keys(self) -> None:
        """State dict should be Llama keys + q_norm/k_norm."""
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        keys = set(block.state_dict().keys())
        expected = {
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }
        assert keys == expected

    def test_qwen3_superset_of_llama_keys(self) -> None:
        """Qwen 3 state dict keys should be a superset of Llama's."""
        llama = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        qwen3 = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        llama_keys = set(llama.state_dict().keys())
        qwen3_keys = set(qwen3.state_dict().keys())
        assert llama_keys.issubset(qwen3_keys)
        extra = qwen3_keys - llama_keys
        assert extra == {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}

    def test_residual_connection(self) -> None:
        """With zeroed-out sub-layers, output should equal input."""
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        with torch.no_grad():
            for name, param in block.named_parameters():
                if "layernorm" not in name:
                    param.zero_()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_bfloat16(self) -> None:
        """Block should produce valid bf16 output without NaN."""
        block = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        block = block.to(torch.bfloat16)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.bfloat16)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()

    def test_different_from_llama_with_same_weights(self) -> None:
        """QK-norm should cause different outputs even with identical base weights."""
        torch.manual_seed(42)
        llama = LlamaTransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        torch.manual_seed(42)
        qwen3 = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        x = torch.randn(1, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out_llama = llama(x, cos, sin)
        out_qwen3 = qwen3(x, cos, sin)
        # QK-norm changes the computation, so outputs should differ.
        assert not torch.allclose(out_llama, out_qwen3)


class TestGemma3RMSNorm:
    """Test the Gemma 3 (1 + weight) RMSNorm variant."""

    def test_zero_weight_is_identity_scaling(self) -> None:
        """With default zero weights, (1 + 0) = 1, so it's just normalization."""
        dim = 16
        norm = Gemma3RMSNorm(dim, eps=1e-6)
        x = torch.randn(2, 4, dim)
        out = norm(x)
        # Compare to standard normalization (no weight scaling).
        x_f32 = x.to(torch.float32)
        expected = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(out, expected.to(x.dtype), atol=1e-5, rtol=1e-5)

    def test_weight_initialized_to_zeros(self) -> None:
        norm = Gemma3RMSNorm(16)
        torch.testing.assert_close(norm.weight.data, torch.zeros(16))

    def test_standard_rmsnorm_weight_initialized_to_ones(self) -> None:
        """Contrast: standard RMSNorm uses ones, Gemma3 uses zeros."""
        standard = RMSNorm(16)
        gemma3 = Gemma3RMSNorm(16)
        torch.testing.assert_close(standard.weight.data, torch.ones(16))
        torch.testing.assert_close(gemma3.weight.data, torch.zeros(16))

    def test_nonzero_weight_applies_offset(self) -> None:
        """With weight=0.5, should scale by (1 + 0.5) = 1.5."""
        dim = 8
        norm = Gemma3RMSNorm(dim, eps=1e-6)
        with torch.no_grad():
            norm.weight.fill_(0.5)
        x = torch.randn(1, 4, dim)
        out = norm(x)
        # Manual computation.
        x_f32 = x.to(torch.float32)
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = 1.5 * normed
        torch.testing.assert_close(out, expected.to(x.dtype), atol=1e-5, rtol=1e-5)

    def test_output_shape(self) -> None:
        norm = Gemma3RMSNorm(16)
        x = torch.randn(2, 4, 16)
        assert norm(x).shape == x.shape

    def test_bfloat16_stability(self) -> None:
        """Should produce stable results with bfloat16 input."""
        dim = 32
        norm = Gemma3RMSNorm(dim)
        x_f32 = torch.randn(2, 4, dim)
        x_bf16 = x_f32.to(torch.bfloat16)
        out_f32 = norm(x_f32)
        out_bf16 = norm(x_bf16)
        torch.testing.assert_close(out_f32, out_bf16.float(), atol=5e-3, rtol=5e-3)

    def test_equivalent_to_standard_at_init(self) -> None:
        """At init, Gemma3 (zeros, 1+w) should match standard (ones, w) RMSNorm."""
        dim = 16
        standard = RMSNorm(dim, eps=1e-6)
        gemma3 = Gemma3RMSNorm(dim, eps=1e-6)
        x = torch.randn(2, 4, dim)
        # Both should produce the same output at init time.
        torch.testing.assert_close(standard(x), gemma3(x), atol=1e-5, rtol=1e-5)


class TestGemma3TransformerBlock:
    """Test the Gemma 3 transformer block."""

    def _make_block(self) -> Gemma3TransformerBlock:
        return Gemma3TransformerBlock(
            HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS, query_pre_attn_scalar=256.0
        )

    def test_output_shape(self) -> None:
        block = self._make_block()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_with_causal_mask(self) -> None:
        block = self._make_block()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        mask = causal_mask(SEQ_LEN)
        out = block(x, cos, sin, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_with_sliding_window_mask(self) -> None:
        block = self._make_block()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        mask = sliding_window_causal_mask(SEQ_LEN, window_size=4)
        out = block(x, cos, sin, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_has_qk_norm(self) -> None:
        """Gemma 3 block should have QK-norm."""
        block = self._make_block()
        assert block.self_attn.q_norm is not None
        assert block.self_attn.k_norm is not None

    def test_qk_norm_is_gemma3_variant(self) -> None:
        """QK-norm should use the Gemma 3 (1+weight) variant."""
        block = self._make_block()
        assert isinstance(block.self_attn.q_norm, Gemma3RMSNorm)
        assert isinstance(block.self_attn.k_norm, Gemma3RMSNorm)

    def test_layer_norms_are_gemma3_variant(self) -> None:
        """All 4 layer norms should use the Gemma 3 (1+weight) variant."""
        block = self._make_block()
        assert isinstance(block.input_layernorm, Gemma3RMSNorm)
        assert isinstance(block.post_attention_layernorm, Gemma3RMSNorm)
        assert isinstance(block.pre_feedforward_layernorm, Gemma3RMSNorm)
        assert isinstance(block.post_feedforward_layernorm, Gemma3RMSNorm)

    def test_has_four_norms(self) -> None:
        """Gemma 3 should have 4 block norms (sandwich pattern)."""
        block = self._make_block()
        norm_names = [name for name, _ in block.named_modules() if isinstance(_, Gemma3RMSNorm)]
        # 4 block norms + 2 QK-norms = 6 total Gemma3RMSNorm instances.
        assert len(norm_names) == 6

    def test_gelu_activation(self) -> None:
        """Gemma 3 block should use GELU-tanh (GeGLU)."""
        block = self._make_block()
        assert isinstance(block.mlp.act_fn, torch.nn.GELU)

    def test_attention_scale(self) -> None:
        """Attention scale should use query_pre_attn_scalar, not head_dim."""
        block = self._make_block()
        expected_scale = 256.0**-0.5  # 1/16
        assert block.self_attn.scale == expected_scale

    def test_no_bias(self) -> None:
        block = self._make_block()
        assert block.self_attn.q_proj.bias is None
        assert block.mlp.gate_proj.bias is None

    def test_state_dict_keys(self) -> None:
        """State dict should have QK-norm + 4 sandwich norms."""
        block = self._make_block()
        keys = set(block.state_dict().keys())
        expected = {
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }
        assert keys == expected

    def test_residual_connection(self) -> None:
        """With zeroed-out sub-layers, output should equal input."""
        block = self._make_block()
        with torch.no_grad():
            for name, param in block.named_parameters():
                if "layernorm" not in name:
                    param.zero_()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_bfloat16(self) -> None:
        """Block should produce valid bf16 output without NaN."""
        block = self._make_block().to(torch.bfloat16)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.bfloat16)
        cos, sin = _make_rope()
        out = block(x, cos, sin)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()

    def test_sandwich_norm_differs_from_pre_norm(self) -> None:
        """Sandwich norm should produce different results from pre-norm (Qwen 3)."""
        torch.manual_seed(42)
        qwen3 = Qwen3TransformerBlock(HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)
        torch.manual_seed(42)
        gemma3 = self._make_block()
        x = torch.randn(1, SEQ_LEN, HIDDEN)
        cos, sin = _make_rope()
        out_qwen3 = qwen3(x, cos, sin)
        out_gemma3 = gemma3(x, cos, sin)
        assert not torch.allclose(out_qwen3, out_gemma3)
