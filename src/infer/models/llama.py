"""Llama 3 transformer block.

Pre-norm architecture with 2 block norms, no QK-norm, SwiGLU MLP.

Block structure::

    residual = x
    x = input_layernorm(x)
    x = attention(x, cos, sin, mask) + residual
    residual = x
    x = post_attention_layernorm(x)
    x = mlp(x) + residual
"""

from __future__ import annotations

from torch import Tensor, nn

from infer.models.common import Attention, GatedMLP, RMSNorm


class LlamaTransformerBlock(nn.Module):
    """Single Llama 3 transformer block.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: MLP inner dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Dimension of each attention head.
        rms_norm_eps: Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=False,
            qk_norm=False,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            act_fn="silu",
        )

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[batch, seq_len, hidden_size]``.
            cos: RoPE cosine table ``[seq_len, head_dim]``.
            sin: RoPE sine table ``[seq_len, head_dim]``.
            mask: Attention mask ``[1, 1, seq_len, seq_len]`` (additive, float).

        Returns:
            Output tensor ``[batch, seq_len, hidden_size]``.
        """
        # Attention sub-layer with pre-norm.
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask)
        x = residual + x

        # MLP sub-layer with pre-norm.
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x
