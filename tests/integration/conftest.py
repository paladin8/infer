"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def device() -> str:
    """Return 'cuda' — Triton kernels require CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton kernel tests")
    return "cuda"
