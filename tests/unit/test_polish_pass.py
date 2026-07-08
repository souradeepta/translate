"""Tests for the Ollama polish pass orchestration (mocked Ollama)."""

from __future__ import annotations

import pytest

from bn_en_translate.config import PipelineConfig
from bn_en_translate.pipeline.pipeline import polish_with_ollama


class FakeOllama:
    """Stands in for OllamaTranslator: records lifecycle and inputs."""

    def __init__(self) -> None:
        self.loaded = False
        self.polished: list[str] = []

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self.loaded, "polish called before load()"
        self.polished.extend(texts)
        return [f"POLISHED: {t}" for t in texts]


def test_polish_preserves_paragraph_count(monkeypatch) -> None:
    fake = FakeOllama()
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline._make_ollama", lambda config: fake
    )
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline.ensure_vram_available",
        lambda required_mib, context: None,
    )
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    result = polish_with_ollama(text, PipelineConfig())
    assert result.count("\n\n") == 2
    assert result.startswith("POLISHED: ")
    assert not fake.loaded  # unloaded afterwards


def test_polish_unloads_even_when_ollama_raises(monkeypatch) -> None:
    fake = FakeOllama()

    def _boom(texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        raise ConnectionError("ollama died mid-request")

    fake.translate = _boom  # type: ignore[method-assign]
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline._make_ollama", lambda config: fake
    )
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline.ensure_vram_available",
        lambda required_mib, context: None,
    )
    with pytest.raises(ConnectionError):
        polish_with_ollama("text", PipelineConfig())
    assert not fake.loaded


def test_polish_raises_on_low_vram(monkeypatch) -> None:
    def _raise(required_mib: int, context: str) -> None:
        raise RuntimeError(f"{context}: needs {required_mib} MiB")

    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline.ensure_vram_available", _raise
    )
    with pytest.raises(RuntimeError, match="polish"):
        polish_with_ollama("text", PipelineConfig())


def test_polish_vram_requirement_lookup(monkeypatch) -> None:
    """Known Ollama tags use their table entry; unknown tags use the max (4800)."""
    seen: list[int] = []
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline.ensure_vram_available",
        lambda required_mib, context: seen.append(required_mib),
    )
    fake = FakeOllama()
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline._make_ollama", lambda config: fake
    )
    polish_with_ollama("t", PipelineConfig(ollama_model="gemma3:12b"))
    polish_with_ollama("t", PipelineConfig(ollama_model="some-unknown-model:1b"))
    assert seen == [4700, 4800]
