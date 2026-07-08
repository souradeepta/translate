"""Tests for shared HF model helpers."""

from __future__ import annotations

from bn_en_translate.models import hf_utils


def test_resolve_attn_sdpa_default_fallback(monkeypatch) -> None:
    monkeypatch.setattr(hf_utils, "flash_attn_available", lambda: False)
    assert hf_utils.resolve_attn_implementation(use_flash=True) == "sdpa"
    assert hf_utils.resolve_attn_implementation(use_flash=False) == "sdpa"


def test_resolve_attn_respects_fallback_param(monkeypatch) -> None:
    """T5 (MADLAD) rejects sdpa in transformers 5.4.0 — callers pass fallback='eager'."""
    monkeypatch.setattr(hf_utils, "flash_attn_available", lambda: False)
    assert hf_utils.resolve_attn_implementation(use_flash=True, fallback="eager") == "eager"
    assert hf_utils.resolve_attn_implementation(use_flash=False, fallback="eager") == "eager"


def test_resolve_attn_flash_when_available(monkeypatch) -> None:
    monkeypatch.setattr(hf_utils, "flash_attn_available", lambda: True)
    assert hf_utils.resolve_attn_implementation(use_flash=True) == "flash_attention_2"
    assert (
        hf_utils.resolve_attn_implementation(use_flash=True, fallback="eager")
        == "flash_attention_2"
    )


def test_resolve_device_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(
        "bn_en_translate.utils.cuda_check.get_best_device", lambda: "cuda"
    )
    assert hf_utils.resolve_device("auto") == "cuda"
    assert hf_utils.resolve_device("cuda") == "cuda"
    assert hf_utils.resolve_device("cpu") == "cpu"


def test_free_cuda_memory_never_raises() -> None:
    hf_utils.free_cuda_memory()
