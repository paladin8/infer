"""Slotted KV cache pool for continuous batching."""

from __future__ import annotations

import torch
from torch import Tensor

from infer.loader.config import ModelConfig


class SlottedKVCache:
    """Pre-allocated KV cache pool for continuous batching.

    Allocates cache for ``max_batch_size`` sequences at engine startup.
    Each slot has an independent position counter, enabling sequences at
    different generation stages to coexist in the same cache tensor.

    Attributes:
        k: Key cache, shape ``[num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim]``.
        v: Value cache, same shape as ``k``.
        seq_lens: Per-slot position counters.
    """

    def __init__(
        self,
        k: Tensor,
        v: Tensor,
        max_batch_size: int,
    ) -> None:
        self.k = k
        self.v = v
        self.max_batch_size = max_batch_size
        self.seq_lens: list[int] = [0] * max_batch_size
        self._free_slots: set[int] = set(range(max_batch_size))

    @staticmethod
    def from_model_config(
        config: ModelConfig,
        max_seq_len: int,
        max_batch_size: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> SlottedKVCache:
        """Allocate a pool sized for the given model config."""
        shape = (
            config.num_hidden_layers,
            max_batch_size,
            config.num_key_value_heads,
            max_seq_len,
            config.computed_head_dim,
        )
        k = torch.zeros(shape, dtype=dtype, device=device)
        v = torch.zeros(shape, dtype=dtype, device=device)
        return SlottedKVCache(k, v, max_batch_size)

    def allocate_slot(self) -> int:
        """Claim a free slot. Raises RuntimeError if none available."""
        if not self._free_slots:
            raise RuntimeError("No free cache slots available")
        return self._free_slots.pop()

    def free_slot(self, slot: int) -> None:
        """Release a slot and reset its position counter."""
        self.seq_lens[slot] = 0
        self._free_slots.add(slot)

    def free_slot_count(self) -> int:
        """Number of available slots."""
        return len(self._free_slots)

    def prefill_view(self, slot: int) -> PrefillCacheView:
        """Return a single-slot view for prefill (KVCacheProtocol-compatible)."""
        return PrefillCacheView(self, slot)

    def batched_prefill_view(
        self, slots: list[int], prompt_lens: list[int]
    ) -> BatchedPrefillCacheView:
        """Return a multi-slot view for batched prefill."""
        return BatchedPrefillCacheView(self, slots, prompt_lens)

    def decode_view(self, active_slots: list[int]) -> DecodeCacheView:
        """Return a multi-slot view for batched decode."""
        return DecodeCacheView(self, active_slots)


class PrefillCacheView:
    """KVCacheProtocol-compatible view for single-slot prefill.

    Writes K/V directly into the pool at the assigned slot.
    The model forward code calls ``update()`` and ``advance()``
    exactly as it does with a regular KVCache.
    """

    def __init__(self, pool: SlottedKVCache, slot: int) -> None:
        self.pool = pool
        self.slot = slot
        self.seq_len: int = 0  # starts at 0 for new request

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store K/V at this slot's position and return the valid cache.

        k, v shape: ``[1, num_kv_heads, new_len, head_dim]``.
        Returns cache covering ``[0 : seq_len + new_len]`` for this slot.
        """
        new_len = k.shape[2]
        start = self.seq_len
        end = start + new_len
        max_seq_len = self.pool.k.shape[3]
        assert end <= max_seq_len, (
            f"KV cache overflow: writing to position {end} but max_seq_len is {max_seq_len}"
        )
        self.pool.k[layer_idx, self.slot : self.slot + 1, :, start:end, :] = k
        self.pool.v[layer_idx, self.slot : self.slot + 1, :, start:end, :] = v
        return (
            self.pool.k[layer_idx, self.slot : self.slot + 1, :, :end, :],
            self.pool.v[layer_idx, self.slot : self.slot + 1, :, :end, :],
        )

    def advance(self, n: int) -> None:
        """Advance this slot's position counter."""
        self.seq_len += n
        self.pool.seq_lens[self.slot] = self.seq_len


class BatchedPrefillCacheView:
    """KVCacheProtocol-compatible view for batching multiple prefill requests.

    Scatter-writes each batch element's K/V to its assigned pool slot.
    On ``advance()``, per-slot ``seq_lens`` are set to actual prompt lengths
    (not the padded length), so subsequent decode steps use correct positions.
    """

    def __init__(self, pool: SlottedKVCache, slots: list[int], prompt_lens: list[int]) -> None:
        self.pool = pool
        self.slots = slots
        self.prompt_lens = prompt_lens
        self.seq_len: int = 0  # starts at 0 for new requests

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Scatter-write K/V to pool slots and return for attention.

        k, v shape: ``[batch, num_kv_heads, padded_len, head_dim]``.
        Writes each batch element to its assigned slot in the pool.
        Returns ``(k, v)`` directly — during prefill the input IS the full
        cache, and the padding mask handles different actual lengths.
        """
        padded_len = k.shape[2]
        start = self.seq_len
        end = start + padded_len
        max_seq_len = self.pool.k.shape[3]
        assert end <= max_seq_len, (
            f"KV cache overflow: writing to position {end} but max_seq_len is {max_seq_len}"
        )
        for i, slot in enumerate(self.slots):
            self.pool.k[layer_idx, slot, :, start:end, :] = k[i]
            self.pool.v[layer_idx, slot, :, start:end, :] = v[i]
        return k, v

    def advance(self, n: int) -> None:
        """Advance position counters.

        Sets per-slot ``seq_lens`` to actual prompt lengths (not the padded
        length ``n``), so decode uses the correct positions.
        """
        self.seq_len += n
        for i, slot in enumerate(self.slots):
            self.pool.seq_lens[slot] = self.prompt_lens[i]


class DecodeCacheView:
    """Multi-slot cache view for batched decode.

    Provides KVCacheProtocol-compatible interface over a subset of pool slots.
    Each slot's K/V is written at its own position.  The full cache is
    gathered from the pool for attention.
    """

    def __init__(self, pool: SlottedKVCache, active_slots: list[int]) -> None:
        self.pool = pool
        self.slots = active_slots
        self.slot_seq_lens = [pool.seq_lens[s] for s in active_slots]
        self._seq_len = max(self.slot_seq_lens) if active_slots else 0
        # Pre-compute slot index tensor to avoid per-layer allocation.
        self._slot_idx: Tensor | None = None

    @property
    def seq_len(self) -> int:
        """Max of active slot seq_lens (used for mask width calculation)."""
        return self._seq_len

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write per-slot K/V and return gathered cache for attention.

        k, v shape: ``[active_batch, num_kv_heads, 1, head_dim]``.
        Writes each batch element's K/V to its slot's current position.
        Returns gathered cache ``[active_batch, num_kv_heads, max_len, head_dim]``.
        """
        for i, slot in enumerate(self.slots):
            pos = self.slot_seq_lens[i]
            self.pool.k[layer_idx, slot, :, pos, :] = k[i, :, 0, :]
            self.pool.v[layer_idx, slot, :, pos, :] = v[i, :, 0, :]

        max_len = self._seq_len + 1
        if self._slot_idx is None:
            self._slot_idx = torch.tensor(self.slots, device=k.device)
        cached_k = self.pool.k[layer_idx, self._slot_idx, :, :max_len, :]
        cached_v = self.pool.v[layer_idx, self._slot_idx, :, :max_len, :]
        return cached_k, cached_v

    def advance(self, n: int) -> None:
        """Advance all active slots by n positions."""
        for i, slot in enumerate(self.slots):
            self.pool.seq_lens[slot] += n
            self.slot_seq_lens[i] += n
        self._seq_len += n
