"""Shared CTranslate2 utilities used across multiple translator backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def probe_compute_type(
    model_path: str,
    device: str,
    probe_fn: Callable[[Any], None],
) -> str:
    """Select the best working CTranslate2 compute type for the given device.

    Tries candidates in preference order (highest speed/lowest memory first).
    INT8 variants fail on Blackwell sm_120 + CUDA 12.x with CUBLAS_STATUS_NOT_SUPPORTED;
    the probe catches this at load time rather than during first translation.

    Args:
        model_path: Path to the CTranslate2 model directory.
        device:     "cuda" or "cpu".
        probe_fn:   Callable that runs a real translation on the probe translator.
                    Must raise on failure (any exception is caught and retried).

    Returns:
        The first compute type that works, or "float32" as an ultimate fallback.
    """
    import ctranslate2

    if device == "cpu":
        return "int8"

    supported = ctranslate2.get_supported_compute_types(device)
    for ct in ("int8_float16", "int8", "float16", "bfloat16", "float32"):
        if ct not in supported:
            continue
        try:
            translator = ctranslate2.Translator(
                model_path, device=device, compute_type=ct
            )
            probe_fn(translator)
            del translator
            return ct
        except Exception:
            continue
    return "float32"
