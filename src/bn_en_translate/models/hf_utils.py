"""Shared helpers for HuggingFace-native translator implementations.

Used by milmmt.py, madlad.py, seamless.py, indicTrans2.py — keep these
free of model-specific logic.
"""

from __future__ import annotations

import importlib.util


def flash_attn_available() -> bool:
    """True if the flash-attn package is importable."""
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attn_implementation(use_flash: bool, fallback: str = "sdpa") -> str:
    """flash_attention_2 if installed and requested; else the given fallback.

    flash-attn is not installable on sm_120/WSL2 as of 2026-07, so the fallback
    is the effective default on this machine. Not every architecture supports
    every fallback: SDPA is fast and broadly supported (Gemma3/MiLMMT), but T5
    (MADLAD) rejects sdpa in transformers 5.4.0 (T5PreTrainedModel._supports_sdpa
    is False, ValueError at load) — T5-based callers must pass fallback="eager".
    """
    if use_flash and flash_attn_available():
        return "flash_attention_2"
    return fallback


def resolve_device(config_device: str) -> str:
    """Resolve 'auto' to the best available device; pass through otherwise."""
    from bn_en_translate.utils.cuda_check import get_best_device

    return get_best_device() if config_device == "auto" else config_device


def free_cuda_memory() -> None:
    """Release cached CUDA allocations. Safe no-op without torch/CUDA."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
