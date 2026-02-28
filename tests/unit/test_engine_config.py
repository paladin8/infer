"""Unit tests for EngineConfig."""

from __future__ import annotations

import pytest

from infer.engine.config import EngineConfig


class TestEngineConfigDefaults:
    def test_defaults(self) -> None:
        cfg = EngineConfig(model="meta-llama/Llama-3.2-1B-Instruct")
        assert cfg.model == "meta-llama/Llama-3.2-1B-Instruct"
        assert cfg.dtype == "bfloat16"
        assert cfg.device == "cuda"
        assert cfg.max_seq_len == 4096
        assert cfg.max_batch_size == 8
        assert cfg.max_waiting_requests == 64
        assert cfg.seed is None
        assert cfg.batching_mode == "static"
        assert cfg.scheduler_policy == "fcfs"
        assert cfg.batch_wait_timeout_s == 0.05
        assert cfg.kv_cache_backend == "contiguous"

    def test_custom_values(self) -> None:
        cfg = EngineConfig(
            model="Qwen/Qwen3-1.7B",
            dtype="float16",
            max_seq_len=2048,
            max_batch_size=4,
            max_waiting_requests=32,
            seed=42,
            batch_wait_timeout_s=0.1,
        )
        assert cfg.model == "Qwen/Qwen3-1.7B"
        assert cfg.dtype == "float16"
        assert cfg.max_seq_len == 2048
        assert cfg.max_batch_size == 4
        assert cfg.seed == 42


class TestEngineConfigValidation:
    def test_invalid_batching_mode(self) -> None:
        with pytest.raises(ValueError, match="batching_mode"):
            EngineConfig(model="m", batching_mode="unknown")

    def test_invalid_scheduler_policy(self) -> None:
        with pytest.raises(ValueError, match="scheduler_policy"):
            EngineConfig(model="m", scheduler_policy="priority")

    def test_invalid_kv_cache_backend(self) -> None:
        with pytest.raises(ValueError, match="kv_cache_backend"):
            EngineConfig(model="m", kv_cache_backend="block")

    def test_max_batch_size_zero(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            EngineConfig(model="m", max_batch_size=0)

    def test_max_waiting_requests_zero(self) -> None:
        with pytest.raises(ValueError, match="max_waiting_requests"):
            EngineConfig(model="m", max_waiting_requests=0)

    def test_max_seq_len_zero(self) -> None:
        with pytest.raises(ValueError, match="max_seq_len"):
            EngineConfig(model="m", max_seq_len=0)

    def test_negative_batch_wait_timeout(self) -> None:
        with pytest.raises(ValueError, match="batch_wait_timeout_s"):
            EngineConfig(model="m", batch_wait_timeout_s=-1.0)

    def test_zero_batch_wait_timeout_is_valid(self) -> None:
        cfg = EngineConfig(model="m", batch_wait_timeout_s=0.0)
        assert cfg.batch_wait_timeout_s == 0.0

    def test_continuous_batching_mode_accepted(self) -> None:
        cfg = EngineConfig(model="m", batching_mode="continuous")
        assert cfg.batching_mode == "continuous"

    def test_paged_kv_cache_backend_accepted(self) -> None:
        cfg = EngineConfig(model="m", kv_cache_backend="paged", batching_mode="continuous")
        assert cfg.kv_cache_backend == "paged"

    def test_paged_requires_continuous_batching(self) -> None:
        with pytest.raises(ValueError, match="batching_mode='continuous'"):
            EngineConfig(model="m", kv_cache_backend="paged", batching_mode="static")

    def test_paged_invalid_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            EngineConfig(
                model="m",
                kv_cache_backend="paged",
                batching_mode="continuous",
                block_size=0,
            )

    def test_paged_invalid_num_gpu_blocks(self) -> None:
        with pytest.raises(ValueError, match="num_gpu_blocks must be >= 1"):
            EngineConfig(
                model="m",
                kv_cache_backend="paged",
                batching_mode="continuous",
                num_gpu_blocks=0,
            )

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            EngineConfig(model="m", dtype="float32")

    def test_float16_dtype_accepted(self) -> None:
        cfg = EngineConfig(model="m", dtype="float16")
        assert cfg.dtype == "float16"


class TestChunkedPrefillConfig:
    def test_defaults(self) -> None:
        cfg = EngineConfig(model="m", batching_mode="continuous")
        assert cfg.use_chunked_prefill is False
        assert cfg.prefill_chunk_size == 512
        assert cfg.max_prefill_chunks_per_step is None

    def test_chunked_prefill_accepted(self) -> None:
        cfg = EngineConfig(
            model="m",
            batching_mode="continuous",
            use_chunked_prefill=True,
            prefill_chunk_size=256,
        )
        assert cfg.use_chunked_prefill is True
        assert cfg.prefill_chunk_size == 256

    def test_chunked_requires_continuous(self) -> None:
        with pytest.raises(ValueError, match="batching_mode='continuous'"):
            EngineConfig(model="m", batching_mode="static", use_chunked_prefill=True)

    def test_chunk_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="prefill_chunk_size must be >= 1"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                use_chunked_prefill=True,
                prefill_chunk_size=0,
            )

    def test_max_chunks_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_prefill_chunks_per_step must be >= 1 or None"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                use_chunked_prefill=True,
                max_prefill_chunks_per_step=0,
            )

    def test_max_chunks_none_accepted(self) -> None:
        cfg = EngineConfig(
            model="m",
            batching_mode="continuous",
            use_chunked_prefill=True,
            max_prefill_chunks_per_step=None,
        )
        assert cfg.max_prefill_chunks_per_step is None

    def test_max_chunks_positive_accepted(self) -> None:
        cfg = EngineConfig(
            model="m",
            batching_mode="continuous",
            use_chunked_prefill=True,
            max_prefill_chunks_per_step=4,
        )
        assert cfg.max_prefill_chunks_per_step == 4

    def test_disabled_skips_validation(self) -> None:
        """When use_chunked_prefill=False, invalid chunk params don't raise."""
        cfg = EngineConfig(
            model="m",
            batching_mode="static",
            use_chunked_prefill=False,
            prefill_chunk_size=0,
        )
        assert cfg.prefill_chunk_size == 0


class TestPrefixCachingConfig:
    def test_default_disabled(self) -> None:
        cfg = EngineConfig(model="m")
        assert cfg.use_prefix_caching is False

    def test_prefix_caching_accepted(self) -> None:
        cfg = EngineConfig(
            model="m",
            batching_mode="continuous",
            kv_cache_backend="paged",
            use_chunked_prefill=True,
            use_prefix_caching=True,
        )
        assert cfg.use_prefix_caching is True

    def test_prefix_requires_paged(self) -> None:
        with pytest.raises(ValueError, match="kv_cache_backend='paged'"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                kv_cache_backend="contiguous",
                use_chunked_prefill=True,
                use_prefix_caching=True,
            )

    def test_prefix_requires_chunked_prefill(self) -> None:
        with pytest.raises(ValueError, match="use_chunked_prefill=True"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                kv_cache_backend="paged",
                use_chunked_prefill=False,
                use_prefix_caching=True,
            )

    def test_disabled_skips_validation(self) -> None:
        """When use_prefix_caching=False, backend/chunked requirements don't apply."""
        cfg = EngineConfig(
            model="m",
            batching_mode="static",
            use_prefix_caching=False,
        )
        assert cfg.use_prefix_caching is False


class TestCUDAGraphsConfig:
    def test_default_disabled(self) -> None:
        cfg = EngineConfig(model="m")
        assert cfg.use_cuda_graphs is False

    def test_cuda_graphs_accepted(self) -> None:
        cfg = EngineConfig(
            model="m",
            batching_mode="continuous",
            kv_cache_backend="paged",
            use_cuda_graphs=True,
        )
        assert cfg.use_cuda_graphs is True

    def test_cuda_graphs_requires_paged(self) -> None:
        with pytest.raises(ValueError, match="kv_cache_backend='paged'"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                kv_cache_backend="contiguous",
                use_cuda_graphs=True,
            )

    def test_cuda_graphs_rejects_contiguous_and_static(self) -> None:
        """CUDA graphs with contiguous + static rejects (paged check fires first)."""
        with pytest.raises(ValueError, match="kv_cache_backend='paged'"):
            EngineConfig(
                model="m",
                batching_mode="static",
                kv_cache_backend="contiguous",
                use_cuda_graphs=True,
            )

    def test_disabled_skips_validation(self) -> None:
        """When use_cuda_graphs=False, backend/batching requirements don't apply."""
        cfg = EngineConfig(
            model="m",
            batching_mode="static",
            use_cuda_graphs=False,
        )
        assert cfg.use_cuda_graphs is False


class TestSpeculativeDecodingConfig:
    def test_default_disabled(self) -> None:
        cfg = EngineConfig(model="m")
        assert cfg.use_speculative_decoding is False
        assert cfg.draft_model is None
        assert cfg.spec_length == 5

    def test_speculative_decoding_valid_config(self) -> None:
        cfg = EngineConfig(
            model="m",
            batching_mode="continuous",
            use_speculative_decoding=True,
            draft_model="d",
            spec_length=3,
        )
        assert cfg.use_speculative_decoding is True
        assert cfg.draft_model == "d"
        assert cfg.spec_length == 3

    def test_speculative_decoding_requires_draft_model(self) -> None:
        with pytest.raises(ValueError, match="draft_model"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                use_speculative_decoding=True,
                draft_model=None,
            )

    def test_speculative_decoding_requires_continuous(self) -> None:
        with pytest.raises(ValueError, match="batching_mode='continuous'"):
            EngineConfig(
                model="m",
                batching_mode="static",
                use_speculative_decoding=True,
                draft_model="d",
            )

    def test_speculative_decoding_rejects_cuda_graphs(self) -> None:
        with pytest.raises(ValueError, match="incompatible with use_cuda_graphs"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                kv_cache_backend="paged",
                use_speculative_decoding=True,
                draft_model="d",
                use_cuda_graphs=True,
            )

    def test_spec_length_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="spec_length must be >= 1"):
            EngineConfig(
                model="m",
                batching_mode="continuous",
                use_speculative_decoding=True,
                draft_model="d",
                spec_length=0,
            )

    def test_disabled_skips_validation(self) -> None:
        """When use_speculative_decoding=False, draft_model/spec_length are not validated."""
        cfg = EngineConfig(
            model="m",
            batching_mode="static",
            use_speculative_decoding=False,
            draft_model=None,
            spec_length=0,
        )
        assert cfg.use_speculative_decoding is False
