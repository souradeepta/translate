"""CUDA and GPU availability utilities."""

from __future__ import annotations


def is_cuda_available() -> bool:
    """Return True if PyTorch can use a CUDA device."""
    try:
        import torch  # type: ignore[import-untyped]

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_best_device() -> str:
    """Return 'cuda' if available, otherwise 'cpu'."""
    return "cuda" if is_cuda_available() else "cpu"


def get_free_vram_mib() -> int:
    """
    Return free VRAM in MiB for the default CUDA device.
    Returns 0 if CUDA is unavailable.
    """
    try:
        import torch  # type: ignore[import-untyped]

        if not torch.cuda.is_available():
            return 0
        free, _ = torch.cuda.mem_get_info(0)
        return int(free / 1024 / 1024)
    except (ImportError, RuntimeError):
        return 0


def get_total_vram_mib() -> int:
    """Return total VRAM in MiB for the default CUDA device. Returns 0 if unavailable."""
    try:
        import torch  # type: ignore[import-untyped]

        if not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(0)
        return int(props.total_memory / 1024 / 1024)
    except (ImportError, RuntimeError):
        return 0
