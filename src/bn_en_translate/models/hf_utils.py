"""Shared helpers for HuggingFace-native translator implementations.

Used by milmmt.py, madlad.py, seamless.py, indicTrans2.py — keep these
free of model-specific logic.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types
from collections.abc import Callable
from typing import Any


def stub_transformers_onnx() -> None:
    """Stub the removed `transformers.onnx` module for trust_remote_code models.

    transformers 5.x deleted the onnx export submodule entirely. Some HF
    remote-code repos (e.g. ai4bharat/indictrans2-indic-en-1B) still
    unconditionally `from transformers.onnx import OnnxConfig,
    OnnxSeq2SeqConfigWithPast` at module import time for an ONNX-export
    config class we never instantiate. Without this stub, AutoConfig/
    AutoModel.from_pretrained(trust_remote_code=True) raises ModuleNotFoundError
    before the real (needed) code ever runs.
    """
    if "transformers.onnx" in sys.modules:
        return

    class _OnnxConfig:
        default_fixed_batch = 2
        default_fixed_sequence = 8

    class _OnnxSeq2SeqConfigWithPast(_OnnxConfig):
        pass

    def _compute_effective_axis_dimension(
        dimension: int, fixed_dimension: int, num_token_to_add: int = 0
    ) -> int:
        if dimension <= 0:
            dimension = fixed_dimension
        dimension -= num_token_to_add
        return dimension

    stub = types.ModuleType("transformers.onnx")
    stub.__path__ = []  # mark as a package
    stub.OnnxConfig = _OnnxConfig  # type: ignore[attr-defined]
    stub.OnnxSeq2SeqConfigWithPast = _OnnxSeq2SeqConfigWithPast  # type: ignore[attr-defined]

    utils_stub = types.ModuleType("transformers.onnx.utils")
    utils_stub.compute_effective_axis_dimension = (  # type: ignore[attr-defined]
        _compute_effective_axis_dimension
    )
    stub.utils = utils_stub  # type: ignore[attr-defined]

    sys.modules["transformers.onnx"] = stub
    sys.modules["transformers.onnx.utils"] = utils_stub


def load_indictrans_tokenizer(model_id: str) -> Any:
    """Load an IndicTrans2 tokenizer with a narrow Transformers 5.x shim.

    AI4Bharat's remote tokenizer assigns special-token attributes before calling
    ``PreTrainedTokenizerBase.__init__``. Transformers 5.x validates those
    assignments against ``_special_tokens_map``, which does not exist until the
    base initializer runs. The shim initializes only that missing map during the
    remote ``AutoTokenizer`` call and restores the public class immediately after
    loading. This keeps the compatibility workaround isolated from other models.
    """
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    original_setattr: Callable[[Any, str, Any], None] = PreTrainedTokenizerBase.__setattr__

    def _compat_setattr(instance: Any, name: str, value: Any) -> None:
        if name != "_special_tokens_map" and not hasattr(instance, "_special_tokens_map"):
            object.__setattr__(instance, "_special_tokens_map", {})
        original_setattr(instance, name, value)

    # The dynamic remote class delegates attribute handling to this base class.
    # Restore it in finally so a failed model load cannot alter unrelated backends.
    PreTrainedTokenizerBase.__setattr__ = _compat_setattr  # type: ignore[method-assign]
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    finally:
        PreTrainedTokenizerBase.__setattr__ = original_setattr  # type: ignore[method-assign]


def compatibility_report() -> str:
    """Return an actionable startup report for model-backend dependencies."""
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "not installed"

    try:
        toolkit_version = importlib.metadata.version("IndicTransToolkit")
    except importlib.metadata.PackageNotFoundError:
        toolkit_version = "not installed (IndicTrans2 uses its tokenizer fallback)"

    if transformers_version == "not installed":
        status = "ERROR: install transformers>=5.4,<5.5"
    elif not transformers_version.startswith("5.4."):
        status = (
            "WARNING: tested range is transformers>=5.4,<5.5; "
            "install that range before loading IndicTrans2 remote code"
        )
    else:
        status = "OK: NLLB uses explicit seq2seq API; IndicTrans2 tokenizer shim enabled"
    return (
        f"Backend compatibility: transformers={transformers_version}; "
        f"IndicTransToolkit={toolkit_version}; {status}"
    )


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
    # Call-time import: no cycle exists, but this matches the repo convention
    # and keeps the cuda_check.get_best_device monkeypatch seam working.
    from bn_en_translate.utils.cuda_check import get_best_device

    return get_best_device() if config_device == "auto" else config_device


def free_cuda_memory() -> None:
    """Release cached CUDA allocations. Safe no-op without torch/CUDA.

    Does NOT reset peak-memory statistics, so it is safe to call from
    unload() without disturbing VRAM measurements — for between-model
    measurement resets use cuda_check.reset_cuda_state instead.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
