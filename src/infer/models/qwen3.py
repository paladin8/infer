"""Qwen 3 model and transformer block.

Pre-norm architecture with 2 block norms, QK-norm, SwiGLU MLP.
Same block structure as Llama 3 — the only difference is QK-norm
(RMSNorm on Q and K per-head, after projection, before RoPE).

Block structure::

    residual = x
    x = input_layernorm(x)
    x = attention(x, cos, sin, mask) + residual   # QK-norm on Q,K before RoPE
    residual = x
    x = post_attention_layernorm(x)
    x = mlp(x) + residual
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from infer.loader.config import ModelConfig

if TYPE_CHECKING:
    from infer.cache.simple import KVCache
from infer.models.common import (
    Attention,
    GatedMLP,
    RMSNorm,
    build_rope_cos_sin,
    causal_mask,
)


class Qwen3TransformerBlock(nn.Module):
    """Single Qwen 3 transformer block.

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
            qk_norm=True,
            rms_norm_eps=rms_norm_eps,
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
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[batch, seq_len, hidden_size]``.
            cos: RoPE cosine table ``[seq_len, head_dim]``.
            sin: RoPE sine table ``[seq_len, head_dim]``.
            mask: Attention mask (additive, float).
            kv_cache: Optional KV cache (passed through to attention).
            layer_idx: Layer index for cache indexing.

        Returns:
            Output tensor ``[batch, seq_len, hidden_size]``.
        """
        # Attention sub-layer with pre-norm.
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask, kv_cache=kv_cache, layer_idx=layer_idx)
        x = residual + x

        # MLP sub-layer with pre-norm.
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen3Model(nn.Module):
    """Full Qwen 3 model: embedding, transformer blocks, final norm, LM head.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3TransformerBlock(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.computed_head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE tables as non-persistent buffers.
        cos, sin = build_rope_cos_sin(
            config.computed_head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.cos: Tensor
        self.sin: Tensor
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs, shape ``[batch, seq_len]``.
            kv_cache: Optional KV cache for incremental decoding.

        Returns:
            Logits, shape ``[batch, seq_len, vocab_size]``
            (``[batch, 1, vocab_size]`` when using cache).
        """
        x = self.embed_tokens(input_ids)
        seq_len = x.shape[1]

        if kv_cache is not None:
            pos = kv_cache.seq_len
            if seq_len > 1:
                assert pos == 0, "Chunked prefill not supported in Phase 3"
            cos = self.cos[pos : pos + seq_len]
            sin = self.sin[pos : pos + seq_len]
            mask = None if seq_len == 1 else causal_mask(seq_len, dtype=x.dtype, device=x.device)
        else:
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]
            mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, kv_cache=kv_cache, layer_idx=i)

        if kv_cache is not None:
            kv_cache.advance(seq_len)
            x = x[:, -1:, :]

        x = self.norm(x)
        return self.lm_head(x)
