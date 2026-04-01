"""E2E tests for Ollama integration. Requires Ollama running locally."""

from __future__ import annotations

import pytest

from bn_en_translate.config import PipelineConfig
from bn_en_translate.models.ollama_translator import OllamaTranslator


@pytest.mark.e2e
def test_ollama_connects_and_returns_english() -> None:
    """Requires: ollama serve + ollama pull qwen2.5:7b-instruct-q4_K_M"""
    config = PipelineConfig(ollama_model="qwen2.5:7b-instruct-q4_K_M")
    translator = OllamaTranslator(config)

    try:
        translator.load()
    except RuntimeError as e:
        pytest.skip(f"Ollama not running: {e}")

    result = translator.translate(
        ["আমি ভালো আছি।"], src_lang="ben_Beng", tgt_lang="eng_Latn"
    )
    translator.unload()

    assert len(result) == 1
    assert len(result[0]) > 0
    # Should contain English text (no Bengali characters)
    assert not any("\u0980" <= c <= "\u09ff" for c in result[0])


@pytest.mark.e2e
def test_ollama_fails_gracefully_when_not_running() -> None:
    config = PipelineConfig(ollama_base_url="http://localhost:19999")  # wrong port
    translator = OllamaTranslator(config)
    with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
        translator.load()
