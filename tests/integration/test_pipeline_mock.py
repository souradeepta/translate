"""Integration tests for the full pipeline using MockTranslator (no GPU needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bn_en_translate.config import PipelineConfig
from bn_en_translate.pipeline.pipeline import TranslationPipeline
from tests.conftest import MockTranslator


def _make_pipeline(config: PipelineConfig | None = None) -> TranslationPipeline:
    translator = MockTranslator()
    translator.load()
    return TranslationPipeline(translator, config)


def test_short_story_translated_end_to_end(short_bengali_text: str) -> None:
    pipeline = _make_pipeline()
    result = pipeline.translate(short_bengali_text)
    assert isinstance(result, str)
    assert len(result) > 0
    # MockTranslator prefixes with [MOCK]
    assert "[MOCK]" in result


def test_medium_story_translated_end_to_end(medium_bengali_text: str) -> None:
    pipeline = _make_pipeline()
    result = pipeline.translate(medium_bengali_text)
    assert isinstance(result, str)
    assert len(result) > 0


def test_paragraph_structure_preserved(medium_bengali_text: str) -> None:
    """A 3-paragraph story should produce output with paragraph breaks."""
    pipeline = _make_pipeline()
    result = pipeline.translate(medium_bengali_text)
    # medium_bengali_text has 3 paragraphs separated by \n\n
    assert "\n\n" in result


def test_paragraph_count_preserved(medium_bengali_text: str) -> None:
    pipeline = _make_pipeline()
    result = pipeline.translate(medium_bengali_text)
    input_paragraphs = [p for p in medium_bengali_text.split("\n\n") if p.strip()]
    output_paragraphs = [p for p in result.split("\n\n") if p.strip()]
    assert len(output_paragraphs) == len(input_paragraphs)


def test_pipeline_handles_empty_story() -> None:
    pipeline = _make_pipeline()
    result = pipeline.translate("")
    assert result == ""


def test_pipeline_handles_single_sentence() -> None:
    pipeline = _make_pipeline()
    result = pipeline.translate("আমি ভালো আছি।")
    assert isinstance(result, str)
    assert len(result) > 0


def test_pipeline_handles_whitespace_only() -> None:
    pipeline = _make_pipeline()
    result = pipeline.translate("   \n\n   ")
    assert result == ""


def test_file_input_output_roundtrip(tmp_path: Path, fixtures_dir: Path) -> None:
    pipeline = _make_pipeline()
    input_file = str(fixtures_dir / "sample_short.bn.txt")
    output_file = str(tmp_path / "output.txt")

    result = pipeline.translate_file(input_file, output_file)

    assert Path(output_file).exists()
    written = Path(output_file).read_text(encoding="utf-8")
    assert written == result
    assert "[MOCK]" in written


def test_batch_size_respected() -> None:
    """Verify that we can set a small batch size without errors."""
    from bn_en_translate.config import ChunkConfig

    config = PipelineConfig(chunk=ChunkConfig(batch_size=2))
    pipeline = _make_pipeline(config)
    # 3 paragraphs → will be chunked and processed in batches of 2
    text = "এক।\n\nদুই।\n\nতিন।"
    result = pipeline.translate(text)
    assert isinstance(result, str)
