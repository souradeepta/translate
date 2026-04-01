"""Integration tests with a real NLLB-600M model. Marked slow — downloads ~1.2 GB."""

from __future__ import annotations

import pytest

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.nllb import NLLBTranslator
from bn_en_translate.pipeline.pipeline import TranslationPipeline


@pytest.mark.slow
def test_nllb_model_loads_without_error() -> None:
    config = ModelConfig(model_name="nllb-600M", device="cpu")
    translator = NLLBTranslator(config)
    translator.load()
    translator.unload()


@pytest.mark.slow
def test_single_bengali_sentence_produces_english() -> None:
    config = ModelConfig(model_name="nllb-600M", device="cpu")
    translator = NLLBTranslator(config)
    with translator:
        result = translator.translate(
            ["আমি ভালো আছি।"], src_lang="ben_Beng", tgt_lang="eng_Latn"
        )
    assert len(result) == 1
    # Should produce something in English (no Bengali chars in output)
    assert not any("\u0980" <= c <= "\u09ff" for c in result[0])


@pytest.mark.slow
def test_nllb_output_is_valid_utf8(short_bengali_text: str) -> None:
    config = ModelConfig(model_name="nllb-600M", device="cpu")
    translator = NLLBTranslator(config)
    pipeline = TranslationPipeline(translator, PipelineConfig(model=config))
    with translator:
        result = pipeline.translate(short_bengali_text)
    # Should be encodable as UTF-8 (it's str in Python, so always true, but check chars)
    result.encode("utf-8")  # raises if invalid
    assert len(result) > 0


@pytest.mark.slow
def test_nllb_output_has_no_bengali_characters(short_bengali_text: str) -> None:
    config = ModelConfig(model_name="nllb-600M", device="cpu")
    translator = NLLBTranslator(config)
    pipeline = TranslationPipeline(translator, PipelineConfig(model=config))
    with translator:
        result = pipeline.translate(short_bengali_text)
    # Translation should not contain Bengali script characters
    bengali_chars = [c for c in result if "\u0980" <= c <= "\u09ff"]
    assert len(bengali_chars) == 0, f"Found Bengali chars in output: {bengali_chars[:10]}"
