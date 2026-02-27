"""Engine configuration."""

from __future__ import annotations

from dataclasses import dataclass

_VALID_BATCHING_MODES = {"static", "continuous"}
_VALID_SCHEDULER_POLICIES = {"fcfs"}
_VALID_KV_CACHE_BACKENDS = {"contiguous", "paged"}
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
        use_chunked_prefill: Split long prefills into chunks (Phase 7).
        prefill_chunk_size: Tokens per prefill chunk.
        max_prefill_chunks_per_step: Cap on chunks per step (``None`` = no cap).
        use_prefix_caching: Cache KV blocks for shared prefixes (Phase 8).
            Requires ``kv_cache_backend="paged"`` and ``use_chunked_prefill=True``.
        use_cuda_graphs: Capture decode forward pass as CUDA graphs (Phase 9).
            Requires ``kv_cache_backend="paged"`` and ``batching_mode="continuous"``.
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

    # KV cache backend.
    kv_cache_backend: str = "contiguous"

    # Chunked prefill.
    use_chunked_prefill: bool = False
    prefill_chunk_size: int = 512
    max_prefill_chunks_per_step: int | None = None  # None = no cap (batch all)

    # Prefix caching.
    use_prefix_caching: bool = False

    # CUDA graphs (Phase 9).
    use_cuda_graphs: bool = False

    # Paged backend configuration.
    block_size: int = 16  # tokens per KV cache block (paged backend only)
    num_gpu_blocks: int | None = None  # total blocks; None = auto-compute

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
        if self.use_chunked_prefill:
            if self.batching_mode != "continuous":
                raise ValueError("Chunked prefill requires batching_mode='continuous'")
            if self.prefill_chunk_size < 1:
                raise ValueError(f"prefill_chunk_size must be >= 1, got {self.prefill_chunk_size}")
            if (
                self.max_prefill_chunks_per_step is not None
                and self.max_prefill_chunks_per_step < 1
            ):
                raise ValueError(
                    f"max_prefill_chunks_per_step must be >= 1 or None, "
                    f"got {self.max_prefill_chunks_per_step}"
                )
        if self.kv_cache_backend == "paged":
            if self.batching_mode != "continuous":
                raise ValueError("Paged KV cache requires batching_mode='continuous'")
            if self.block_size < 1:
                raise ValueError(f"block_size must be >= 1, got {self.block_size}")
            if self.num_gpu_blocks is not None and self.num_gpu_blocks < 1:
                raise ValueError(f"num_gpu_blocks must be >= 1, got {self.num_gpu_blocks}")
        if self.use_prefix_caching:
            if self.kv_cache_backend != "paged":
                raise ValueError("Prefix caching requires kv_cache_backend='paged'")
            if not self.use_chunked_prefill:
                raise ValueError("Prefix caching requires use_chunked_prefill=True")
        if self.use_cuda_graphs:
            if self.kv_cache_backend != "paged":
                raise ValueError("CUDA graphs require kv_cache_backend='paged'")
            if self.batching_mode != "continuous":
                raise ValueError("CUDA graphs require batching_mode='continuous'")
