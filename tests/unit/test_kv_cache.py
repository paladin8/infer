"""Unit tests for the KVCache data structure."""

from __future__ import annotations

import pytest
import torch

from infer.cache.simple import KVCache
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
MAX_SEQ_LEN = 16
BATCH = 1
DTYPE = torch.float32
DEVICE = "cpu"


@pytest.fixture()
def cache() -> KVCache:
    """A small KVCache for testing."""
    return KVCache.allocate(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH,
        dtype=DTYPE,
        device=DEVICE,
    )


@pytest.fixture()
def config() -> ModelConfig:
    """A minimal ModelConfig for testing from_model_config."""
    return ModelConfig(
        model_type="llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=NUM_KV_HEADS,
        vocab_size=100,
        max_position_embeddings=128,
        head_dim=HEAD_DIM,
    )


# ---------------------------------------------------------------------------
# allocate
# ---------------------------------------------------------------------------


class TestAllocate:
    def test_shape(self, cache: KVCache) -> None:
        expected = (NUM_LAYERS, BATCH, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        assert cache.k.shape == expected
        assert cache.v.shape == expected

    def test_dtype_and_device(self, cache: KVCache) -> None:
        assert cache.k.dtype == DTYPE
        assert cache.v.dtype == DTYPE
        assert cache.k.device == torch.device(DEVICE)

    def test_initial_seq_len(self, cache: KVCache) -> None:
        assert cache.seq_len == 0

    def test_initialized_to_zeros(self, cache: KVCache) -> None:
        assert torch.all(cache.k == 0)
        assert torch.all(cache.v == 0)


# ---------------------------------------------------------------------------
# from_model_config
# ---------------------------------------------------------------------------


class TestFromModelConfig:
    def test_matches_allocate(self, config: ModelConfig) -> None:
        cache = KVCache.from_model_config(
            config, max_seq_len=MAX_SEQ_LEN, dtype=DTYPE, device=DEVICE
        )
        expected = (NUM_LAYERS, 1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        assert cache.k.shape == expected
        assert cache.v.shape == expected
        assert cache.seq_len == 0

    def test_batch_size(self, config: ModelConfig) -> None:
        cache = KVCache.from_model_config(
            config, max_seq_len=MAX_SEQ_LEN, batch_size=4, dtype=DTYPE, device=DEVICE
        )
        expected = (NUM_LAYERS, 4, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        assert cache.k.shape == expected
        assert cache.v.shape == expected

    def test_uses_computed_head_dim(self) -> None:
        """When head_dim is None, computed_head_dim = hidden_size // num_attention_heads."""
        config = ModelConfig(
            model_type="llama",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=100,
            max_position_embeddings=128,
            head_dim=None,
        )
        cache = KVCache.from_model_config(config, max_seq_len=32, dtype=DTYPE, device=DEVICE)
        assert cache.k.shape == (2, 1, 4, 32, 8)  # head_dim = 64 // 8 = 8


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_prefill(self, cache: KVCache) -> None:
        """Prefill writes multiple positions at once."""
        prompt_len = 5
        k_new = torch.randn(BATCH, NUM_KV_HEADS, prompt_len, HEAD_DIM)
        v_new = torch.randn(BATCH, NUM_KV_HEADS, prompt_len, HEAD_DIM)

        cached_k, cached_v = cache.update(layer_idx=0, k=k_new, v=v_new)

        # Returned tensors cover [0 : seq_len + new_len] = [0 : 5]
        assert cached_k.shape == (BATCH, NUM_KV_HEADS, prompt_len, HEAD_DIM)
        assert cached_v.shape == (BATCH, NUM_KV_HEADS, prompt_len, HEAD_DIM)

        # Content matches what we wrote
        torch.testing.assert_close(cached_k, k_new)
        torch.testing.assert_close(cached_v, v_new)

    def test_decode_step(self, cache: KVCache) -> None:
        """After advance, update appends a single token and returns the full range."""
        # Simulate prefill of 3 tokens
        k_prefill = torch.randn(BATCH, NUM_KV_HEADS, 3, HEAD_DIM)
        v_prefill = torch.randn(BATCH, NUM_KV_HEADS, 3, HEAD_DIM)
        cache.update(layer_idx=0, k=k_prefill, v=v_prefill)
        cache.advance(3)

        # Decode step: single new token
        k_decode = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM)
        v_decode = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM)
        cached_k, cached_v = cache.update(layer_idx=0, k=k_decode, v=v_decode)

        # Should return positions [0:4]
        assert cached_k.shape == (BATCH, NUM_KV_HEADS, 4, HEAD_DIM)
        assert cached_v.shape == (BATCH, NUM_KV_HEADS, 4, HEAD_DIM)

        # First 3 positions match prefill, position 3 matches decode
        torch.testing.assert_close(cached_k[:, :, :3, :], k_prefill)
        torch.testing.assert_close(cached_k[:, :, 3:4, :], k_decode)

    def test_multiple_layers(self, cache: KVCache) -> None:
        """Each layer has independent storage."""
        k0 = torch.ones(BATCH, NUM_KV_HEADS, 2, HEAD_DIM)
        v0 = torch.ones(BATCH, NUM_KV_HEADS, 2, HEAD_DIM)
        k1 = torch.ones(BATCH, NUM_KV_HEADS, 2, HEAD_DIM) * 2
        v1 = torch.ones(BATCH, NUM_KV_HEADS, 2, HEAD_DIM) * 2

        cache.update(layer_idx=0, k=k0, v=v0)
        cache.update(layer_idx=1, k=k1, v=v1)

        # Layer 0 should have 1s, layer 1 should have 2s
        torch.testing.assert_close(cache.k[0, :, :, :2, :], k0)
        torch.testing.assert_close(cache.k[1, :, :, :2, :], k1)

    def test_overflow_asserts(self, cache: KVCache) -> None:
        """Writing past max_seq_len raises AssertionError."""
        k_too_big = torch.randn(BATCH, NUM_KV_HEADS, MAX_SEQ_LEN + 1, HEAD_DIM)
        v_too_big = torch.randn(BATCH, NUM_KV_HEADS, MAX_SEQ_LEN + 1, HEAD_DIM)
        with pytest.raises(AssertionError, match="KV cache overflow"):
            cache.update(layer_idx=0, k=k_too_big, v=v_too_big)

    def test_overflow_after_advance(self, cache: KVCache) -> None:
        """Overflow detected after advancing past available space."""
        # Fill to capacity
        k_full = torch.randn(BATCH, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        v_full = torch.randn(BATCH, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        cache.update(layer_idx=0, k=k_full, v=v_full)
        cache.advance(MAX_SEQ_LEN)

        # One more token should overflow
        k_one = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM)
        v_one = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM)
        with pytest.raises(AssertionError, match="KV cache overflow"):
            cache.update(layer_idx=0, k=k_one, v=v_one)

    def test_update_returns_views(self, cache: KVCache) -> None:
        """Returned tensors are views into the cache (not copies)."""
        k_new = torch.randn(BATCH, NUM_KV_HEADS, 3, HEAD_DIM)
        v_new = torch.randn(BATCH, NUM_KV_HEADS, 3, HEAD_DIM)
        cached_k, _cached_v = cache.update(layer_idx=0, k=k_new, v=v_new)

        # Modifying the returned view should modify the cache
        assert cached_k.data_ptr() == cache.k[0, :, :, :3, :].data_ptr()


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------


class TestAdvance:
    def test_advance_increments(self, cache: KVCache) -> None:
        assert cache.seq_len == 0
        cache.advance(5)
        assert cache.seq_len == 5
        cache.advance(3)
        assert cache.seq_len == 8

    def test_advance_by_zero(self, cache: KVCache) -> None:
        cache.advance(0)
        assert cache.seq_len == 0


# ---------------------------------------------------------------------------
# memory_bytes
# ---------------------------------------------------------------------------


class TestMemoryBytes:
    def test_memory_bytes(self, cache: KVCache) -> None:
        element_size = cache.k.element_size()  # 4 for float32
        elements_per_tensor = NUM_LAYERS * BATCH * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM
        expected = 2 * elements_per_tensor * element_size
        assert cache.memory_bytes == expected

    def test_memory_bytes_bfloat16(self) -> None:
        cache = KVCache.allocate(
            num_layers=2,
            num_kv_heads=4,
            head_dim=16,
            max_seq_len=32,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        element_size = 2  # bfloat16
        elements_per_tensor = 2 * 1 * 4 * 32 * 16
        expected = 2 * elements_per_tensor * element_size
        assert cache.memory_bytes == expected


# ---------------------------------------------------------------------------
# Re-export from __init__
# ---------------------------------------------------------------------------


class TestReExport:
    def test_import_from_package(self) -> None:
        from infer.cache import KVCache as KVCacheFromInit

        assert KVCacheFromInit is KVCache
