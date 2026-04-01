"""E2E quality tests for IndicTrans2. Marked e2e — requires model download + GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.indicTrans2 import IndicTrans2Translator
from bn_en_translate.pipeline.pipeline import TranslationPipeline


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MIN_BLEU_SCORE = 25.0  # Conservative floor — IndicTrans2 typically scores 30+


@pytest.mark.e2e
@pytest.mark.gpu
def test_indictrans2_model_loads() -> None:
    config = ModelConfig(model_name="indicTrans2-1B", device="cuda")
    translator = IndicTrans2Translator(config)
    translator.load()
    translator.unload()


@pytest.mark.e2e
@pytest.mark.gpu
def test_bleu_score_above_threshold() -> None:
    """BLEU score on the short fixture must exceed MIN_BLEU_SCORE."""
    sacrebleu = pytest.importorskip("sacrebleu")

    config = ModelConfig(model_name="indicTrans2-1B", device="cuda")
    translator = IndicTrans2Translator(config)
    pipeline = TranslationPipeline(translator, PipelineConfig(model=config))

    bengali_text = (FIXTURES_DIR / "sample_short.bn.txt").read_text(encoding="utf-8")
    reference = (FIXTURES_DIR / "expected_short.en.txt").read_text(encoding="utf-8")

    with translator:
        hypothesis = pipeline.translate(bengali_text)

    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
    assert bleu.score >= MIN_BLEU_SCORE, (
        f"BLEU score {bleu.score:.1f} < threshold {MIN_BLEU_SCORE}. "
        f"Hypothesis: {hypothesis!r}"
    )


@pytest.mark.e2e
@pytest.mark.gpu
def test_named_entities_preserved() -> None:
    """Proper nouns like 'Rabindranath' should appear in the output."""
    config = ModelConfig(model_name="indicTrans2-1B", device="cuda")
    translator = IndicTrans2Translator(config)
    pipeline = TranslationPipeline(translator, PipelineConfig(model=config))

    bengali_text = (FIXTURES_DIR / "sample_short.bn.txt").read_text(encoding="utf-8")

    with translator:
        result = pipeline.translate(bengali_text)

    # 'রবীন্দ্রনাথ' should be transliterated to 'Rabindranath' or similar
    result_lower = result.lower()
    assert "rabindranath" in result_lower or "tagore" in result_lower, (
        f"Expected named entities in output, got: {result!r}"
    )


@pytest.mark.e2e
@pytest.mark.gpu
def test_full_medium_story_translated(fixtures_dir: Path) -> None:
    config = ModelConfig(model_name="indicTrans2-1B", device="cuda")
    translator = IndicTrans2Translator(config)
    pipeline = TranslationPipeline(translator, PipelineConfig(model=config))

    bengali_text = (fixtures_dir / "sample_medium.bn.txt").read_text(encoding="utf-8")

    with translator:
        result = pipeline.translate(bengali_text)

    assert len(result) > 50
    # No Bengali characters in output
    assert not any("\u0980" <= c <= "\u09ff" for c in result)
    # Should have paragraph structure preserved
    assert "\n\n" in result
