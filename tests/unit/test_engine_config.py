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

    def test_continuous_batching_mode_rejected(self) -> None:
        """Phase 5 value is not yet supported."""
        with pytest.raises(ValueError, match="batching_mode"):
            EngineConfig(model="m", batching_mode="continuous")

    def test_paged_kv_cache_backend_rejected(self) -> None:
        """Phase 6 value is not yet supported."""
        with pytest.raises(ValueError, match="kv_cache_backend"):
            EngineConfig(model="m", kv_cache_backend="paged")

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            EngineConfig(model="m", dtype="float32")

    def test_float16_dtype_accepted(self) -> None:
        cfg = EngineConfig(model="m", dtype="float16")
        assert cfg.dtype == "float16"
