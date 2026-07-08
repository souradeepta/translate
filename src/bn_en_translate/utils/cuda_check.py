"""CUDA and GPU availability utilities."""

from __future__ import annotations


def is_cuda_available() -> bool:
    """Return True if PyTorch can use a CUDA device."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_best_device() -> str:
    """Return 'cuda' if available, otherwise 'cpu'."""
    return "cuda" if is_cuda_available() else "cpu"


def require_cuda(model_name: str) -> None:
    """Raise RuntimeError if CUDA is not available.

    Call at the end of model load() when device='cuda' is configured.
    GPU inference is required — CPU fallback is explicitly prohibited.
    """
    if not is_cuda_available():
        raise RuntimeError(
            f"{model_name}: device='cuda' configured but CUDA is not available. "
            "GPU inference is required — do not fall back to CPU."
        )


def get_free_vram_mib() -> int:
    """
    Return free VRAM in MiB for the default CUDA device.
    Returns 0 if CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free, _ = torch.cuda.mem_get_info(0)
        return int(free / 1024 / 1024)
    except (ImportError, RuntimeError):
        return 0


def reset_cuda_state() -> None:
    """Release cached CUDA allocations and reset peak-memory statistics.

    Call between models in multi-model runs so each model's VRAM peak reading
    starts from a clean slate (residual allocations from a previous model
    otherwise inflate the next model's reported peak — see
    monitor/observations.md 2026-04-10).
    Safe no-op when torch or CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def get_total_vram_mib() -> int:
    """Return total VRAM in MiB for the default CUDA device. Returns 0 if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(0)
        return int(props.total_memory / 1024 / 1024)
    except (ImportError, RuntimeError):
        return 0
