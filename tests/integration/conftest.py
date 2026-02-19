"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def device() -> str:
    """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
