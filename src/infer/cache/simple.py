"""Pre-allocated contiguous KV cache."""

from __future__ import annotations

import torch
from torch import Tensor

from infer.loader.config import ModelConfig


class KVCache:
    """Pre-allocated contiguous KV cache.

    Stores key and value tensors for every layer in two contiguous
    allocations.  A position counter (``seq_len``) tracks how many
    positions have been filled.

    The cache stores K/V *after* QK-norm and RoPE, so cached entries
    are position-encoded and ready for attention.

    Attributes:
        k: Key cache, shape ``[num_layers, batch, num_kv_heads, max_seq_len, head_dim]``.
        v: Value cache, same shape as ``k``.
        seq_len: Number of positions currently filled (same for all layers).
    """

    def __init__(self, k: Tensor, v: Tensor, seq_len: int = 0) -> None:
        self.k = k
        self.v = v
        self.seq_len = seq_len

    @staticmethod
    def allocate(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> KVCache:
        """Pre-allocate cache tensors filled with zeros."""
        shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        k = torch.zeros(shape, dtype=dtype, device=device)
        v = torch.zeros(shape, dtype=dtype, device=device)
        return KVCache(k, v)

    @staticmethod
    def from_model_config(
        config: ModelConfig,
        max_seq_len: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> KVCache:
        """Allocate a cache sized for the given model config.

        Extracts ``num_hidden_layers``, ``num_key_value_heads``, and
        ``computed_head_dim`` from ``config``.
        """
        return KVCache.allocate(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.computed_head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store new K/V entries and return the full valid K/V for this layer.

        Writes ``k`` and ``v`` at positions ``[seq_len : seq_len + new_len]``
        in the cache for the given layer.  Returns views of the cache covering
        positions ``[0 : seq_len + new_len]``.

        Args:
            layer_idx: Which layer's cache to update.
            k: New key tensor ``[batch, num_kv_heads, new_len, head_dim]``.
            v: New value tensor ``[batch, num_kv_heads, new_len, head_dim]``.

        Returns:
            ``(cached_k, cached_v)`` covering all valid positions for this layer.
                Both have shape ``[batch, num_kv_heads, seq_len + new_len, head_dim]``.
        """
        new_len = k.shape[2]
        start = self.seq_len
        end = start + new_len
        assert end <= self.k.shape[3], (
            f"KV cache overflow: writing to position {end} but max_seq_len is {self.k.shape[3]}"
        )
        self.k[layer_idx, :, :, start:end, :] = k
        self.v[layer_idx, :, :, start:end, :] = v
        return self.k[layer_idx, :, :, :end, :], self.v[layer_idx, :, :, :end, :]

    def advance(self, n: int) -> None:
        """Advance the position counter by ``n`` tokens.

        Called once per forward pass, after all layers have written their entries.
        """
        self.seq_len += n

    @property
    def memory_bytes(self) -> int:
        """Total GPU memory used by cache tensors in bytes."""
        return self.k.nbytes + self.v.nbytes
