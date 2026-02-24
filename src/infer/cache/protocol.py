"""KV cache protocol: structural interface for all cache implementations."""

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

    seq_len: int

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]: ...

    def advance(self, n: int) -> None: ...
