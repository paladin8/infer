"""Cache protocols: structural interfaces for all cache implementations."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class KVCacheProtocol(Protocol):
    """Interface that all KV cache implementations must satisfy.

    Models call ``update()`` per layer to store and retrieve K/V,
    ``advance()`` once per forward pass, and read ``seq_len`` for
    mask width calculation.  Phase 5 views, Phase 6 paged views,
    and the original ``KVCache`` all implement this protocol.
    """

    @property
    def seq_len(self) -> int:
        """Current sequence length (read by model for mask width)."""
        ...

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]: ...

    def advance(self, n: int) -> None: ...

    def is_paged(self) -> bool: ...


class CachePoolProtocol(Protocol):
    """Interface for KV cache pools (slotted and paged).

    Abstracts allocation, deallocation, view creation, and capacity
    queries so the runner and engine operate identically across backends.
    """

    def allocate_slot(self, initial_tokens: int = 0) -> int:
        """Allocate a cache slot for a new sequence.

        Args:
            initial_tokens: Number of tokens to pre-allocate capacity for
                (prompt length). Paged backends use this to eagerly allocate
                blocks; contiguous backends ignore it (slots have fixed
                pre-allocated capacity).

        Returns:
            Integer slot/sequence ID, stored in ``request.slot_idx``.

        Raises:
            RuntimeError: If insufficient capacity is available.
        """
        ...

    def free_slot(self, slot: int) -> None:
        """Release a slot and all associated resources (blocks, seq_lens, page tables)."""
        ...

    def get_seq_len(self, slot: int) -> int:
        """Return the current sequence length for a slot."""
        ...

    def free_slot_count(self) -> int:
        """Number of available slots (contiguous) or a bound on allocatable sequences (paged)."""
        ...

    def free_token_capacity(self) -> int | None:
        """Number of tokens that can be stored in free capacity.

        Returns ``None`` for backends where token budget is not applicable
        (contiguous — capacity is per-slot, not token-granular).
        """
        ...

    def prefill_view(self, slot: int) -> KVCacheProtocol:
        """Return a single-slot view for prefill."""
        ...

    def decode_view(self, active_slots: list[int]) -> KVCacheProtocol:
        """Return a multi-slot view for batched decode."""
        ...

    def batched_prefill_view(
        self,
        slots: list[int],
        prompt_lens: list[int],
    ) -> KVCacheProtocol:
        """Return a multi-slot view for batched prefill."""
        ...

    def batched_chunked_prefill_view(
        self,
        slots: list[int],
        start_positions: list[int],
        chunk_lens: list[int],
    ) -> KVCacheProtocol:
        """Return a multi-slot view for batched chunked prefill.

        Each slot writes K/V at ``[start_pos, start_pos + chunk_len)`` and
        the view returns gathered KV padded to ``max(start_pos + chunk_len)``
        for batched attention.
        """
        ...

    def is_paged(self) -> bool:
        """Whether this pool uses paged block allocation."""
        ...
