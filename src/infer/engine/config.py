"""Engine configuration."""

from __future__ import annotations

from dataclasses import dataclass

_VALID_BATCHING_MODES = {"static", "continuous"}
_VALID_SCHEDULER_POLICIES = {"fcfs"}
_VALID_KV_CACHE_BACKENDS = {"contiguous"}  # Phase 6 adds "paged"
_VALID_DTYPES = {"bfloat16", "float16"}


@dataclass
class EngineConfig:
    """Configuration for the serving engine.

    Attributes:
        model: Full HuggingFace model ID (e.g. ``meta-llama/Llama-3.2-1B-Instruct``).
        dtype: Model weight dtype (``"bfloat16"`` or ``"float16"``).
        device: Torch device string.
        max_seq_len: Maximum total sequence length (prompt + generation).
        max_batch_size: Maximum requests per static batch.
        max_waiting_requests: Admission limit for the waiting queue (503 when full).
        seed: Global random seed (per-request seeds override this).
        batching_mode: ``"static"`` (Phase 4) or ``"continuous"`` (Phase 5).
        scheduler_policy: ``"fcfs"`` (Phase 5 adds more).
        batch_wait_timeout_s: Max seconds to wait for a batch to fill before dispatching.
        kv_cache_backend: ``"contiguous"`` (Phase 3) or ``"paged"`` (Phase 6).
    """

    model: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_seq_len: int = 4096
    max_batch_size: int = 8
    max_waiting_requests: int = 64
    seed: int | None = None

    # Batching and scheduling — fixed for Phase 4, extended in Phase 5.
    batching_mode: str = "static"
    scheduler_policy: str = "fcfs"
    batch_wait_timeout_s: float = 0.05

    # KV cache backend — Phase 6 adds "paged".
    kv_cache_backend: str = "contiguous"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration values, raising ``ValueError`` on invalid settings."""
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Unsupported dtype: {self.dtype!r}. Choose from {sorted(_VALID_DTYPES)}"
            )
        if self.batching_mode not in _VALID_BATCHING_MODES:
            raise ValueError(
                f"Unsupported batching_mode: {self.batching_mode!r}. "
                f"Choose from {sorted(_VALID_BATCHING_MODES)}"
            )
        if self.scheduler_policy not in _VALID_SCHEDULER_POLICIES:
            raise ValueError(
                f"Unsupported scheduler_policy: {self.scheduler_policy!r}. "
                f"Choose from {sorted(_VALID_SCHEDULER_POLICIES)}"
            )
        if self.kv_cache_backend not in _VALID_KV_CACHE_BACKENDS:
            raise ValueError(
                f"Unsupported kv_cache_backend: {self.kv_cache_backend!r}. "
                f"Choose from {sorted(_VALID_KV_CACHE_BACKENDS)}"
            )
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.max_waiting_requests < 1:
            raise ValueError(f"max_waiting_requests must be >= 1, got {self.max_waiting_requests}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.batch_wait_timeout_s < 0:
            raise ValueError(f"batch_wait_timeout_s must be >= 0, got {self.batch_wait_timeout_s}")
