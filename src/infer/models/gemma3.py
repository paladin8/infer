"""Gemma 3 model and transformer block.

Sandwich-norm architecture with 4 block norms, QK-norm, GeGLU MLP.

Post-sub-layer norms are applied *before* the residual add, unlike the
pre-norm pattern used by Llama and Qwen.

Uses the ``(1 + weight)`` RMSNorm convention: weights are stored as offsets
from 1.0 (initialized to zeros in HF checkpoints).

Block structure::

    residual = x
    x = input_layernorm(x)
    x = attention(x, cos, sin, mask)     # QK-norm, scale from query_pre_attn_scalar
    x = post_attention_layernorm(x)       # post-norm BEFORE residual add
    x = residual + x
    residual = x
    x = pre_feedforward_layernorm(x)
    x = mlp(x)                            # GeGLU (gelu_pytorch_tanh)
    x = post_feedforward_layernorm(x)     # post-norm BEFORE residual add
    x = residual + x
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
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
    sliding_window_causal_mask,
)


class Gemma3RMSNorm(RMSNorm):
    """Gemma 3 RMSNorm with ``(1 + weight)`` convention.

    Weight is initialized to zeros.  Forward computes:
    ``(1 + weight) * x * rsqrt(mean(x^2) + eps)``

    This is mathematically equivalent to standard RMSNorm at init time
    (since ``1 + 0 = 1``), but HF checkpoints store the offset weight.

    Args:
        dim: The dimension to normalize over (last axis).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim, eps)
        # Override: zeros instead of ones (HF stores the offset from 1).
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Unlike standard RMSNorm which casts back to input_dtype before the
        # weight multiply, HF's Gemma3RMSNorm multiplies in float32 then casts:
        #   HF Llama/Qwen3: self.weight * normed.to(input_dtype)  (bf16 * bf16)
        #   HF Gemma3:      (1 + self.weight.float()) * normed    (f32 * f32)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * normed).to(input_dtype)


class Gemma3TransformerBlock(nn.Module):
    """Single Gemma 3 transformer block.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: MLP inner dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Dimension of each attention head.
        rms_norm_eps: Epsilon for RMSNorm layers.
        query_pre_attn_scalar: Scaling denominator for attention.
            The attention scale is ``query_pre_attn_scalar ** -0.5``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-5,
        query_pre_attn_scalar: float = 256.0,
    ) -> None:
        super().__init__()
        # Attention with QK-norm disabled at init. We assign Gemma3RMSNorm
        # instances manually because Attention creates standard RMSNorm by
        # default, and Gemma 3 needs the (1 + weight) variant. This is safe:
        # nn.Module.__setattr__ registers the new modules correctly, and
        # Attention.forward checks `if self.q_norm is not None`.
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=False,
            qk_norm=False,
            scale=query_pre_attn_scalar**-0.5,
        )
        self.self_attn.q_norm = Gemma3RMSNorm(head_dim, eps=rms_norm_eps)
        self.self_attn.k_norm = Gemma3RMSNorm(head_dim, eps=rms_norm_eps)

        # MLP with GeGLU activation.
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            act_fn="gelu_pytorch_tanh",
        )

        # Sandwich norms — all use Gemma 3's (1 + weight) convention.
        self.input_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=rms_norm_eps)

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
        # Attention sub-layer with sandwich norm.
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, mask, kv_cache=kv_cache, layer_idx=layer_idx)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP sub-layer with sandwich norm.
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x


class Gemma3Model(nn.Module):
    """Full Gemma 3 model: embedding (with scaling), transformer blocks, final norm, LM head.

    Gemma 3 differences from Llama/Qwen:

    - Embedding output is scaled by ``sqrt(hidden_size)``.
    - Dual RoPE tables: local (``rope_local_base_freq``) for sliding-window
      layers, global (``rope_theta``) for full-attention layers.
    - Per-layer mask selection: sliding-window layers use a windowed causal
      mask, full-attention layers use a standard causal mask.
    - Final norm uses ``Gemma3RMSNorm`` (``1 + weight`` convention).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Gemma3TransformerBlock(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.computed_head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    query_pre_attn_scalar=config.query_pre_attn_scalar or config.computed_head_dim,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Model-specific attributes.
        self.embedding_normalizer = math.sqrt(config.hidden_size)
        self.layer_types: list[str] = (
            config.layer_types or ["full_attention"] * config.num_hidden_layers
        )
        self.sliding_window = config.sliding_window or 512

        # Precompute local RoPE tables (rope_local_base_freq, used for sliding-window layers).
        local_theta = config.rope_local_base_freq or config.rope_theta
        local_cos, local_sin = build_rope_cos_sin(
            config.computed_head_dim,
            config.max_position_embeddings,
            theta=local_theta,
        )
        self.local_cos: Tensor
        self.local_sin: Tensor
        self.register_buffer("local_cos", local_cos, persistent=False)
        self.register_buffer("local_sin", local_sin, persistent=False)

        # Precompute global RoPE tables (rope_theta, with optional rope_scaling).
        global_cos, global_sin = build_rope_cos_sin(
            config.computed_head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.global_cos: Tensor
        self.global_sin: Tensor
        self.register_buffer("global_cos", global_cos, persistent=False)
        self.register_buffer("global_sin", global_sin, persistent=False)

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
        x = x * self.embedding_normalizer
        seq_len = x.shape[1]

        if kv_cache is not None:
            pos = kv_cache.seq_len
            if seq_len > 1:
                assert pos == 0, "Chunked prefill not supported in Phase 3"

            # RoPE tables: offset by current cache position.
            local_cos = self.local_cos[pos : pos + seq_len]
            local_sin = self.local_sin[pos : pos + seq_len]
            global_cos = self.global_cos[pos : pos + seq_len]
            global_sin = self.global_sin[pos : pos + seq_len]

            if seq_len == 1:
                # Single-token decode.
                cached_len = pos + 1
                global_mask = None
                cutoff = max(0, cached_len - self.sliding_window)
                if cutoff > 0:
                    local_mask = torch.zeros(1, 1, 1, cached_len, dtype=x.dtype, device=x.device)
                    local_mask[:, :, :, :cutoff] = float("-inf")
                else:
                    local_mask = None
            else:
                # Prefill: standard masks.
                local_mask = sliding_window_causal_mask(
                    seq_len, self.sliding_window, dtype=x.dtype, device=x.device
                )
                global_mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)
        else:
            local_cos = self.local_cos[:seq_len]
            local_sin = self.local_sin[:seq_len]
            global_cos = self.global_cos[:seq_len]
            global_sin = self.global_sin[:seq_len]
            local_mask = sliding_window_causal_mask(
                seq_len, self.sliding_window, dtype=x.dtype, device=x.device
            )
            global_mask = causal_mask(seq_len, dtype=x.dtype, device=x.device)

        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == "sliding_attention":
                x = layer(x, local_cos, local_sin, local_mask, kv_cache=kv_cache, layer_idx=i)
            else:
                x = layer(x, global_cos, global_sin, global_mask, kv_cache=kv_cache, layer_idx=i)

        if kv_cache is not None:
            kv_cache.advance(seq_len)
            x = x[:, -1:, :]

        x = self.norm(x)
        return self.lm_head(x)
