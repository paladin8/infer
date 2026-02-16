import torch


def test_torch_available() -> None:
    assert torch.__version__


def test_cuda_available() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
