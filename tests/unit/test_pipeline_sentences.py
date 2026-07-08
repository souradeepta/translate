"""Tests for TranslationPipeline.translate_sentences (batched, 1:1, order-preserving)."""

from __future__ import annotations

from bn_en_translate.config import ChunkConfig, PipelineConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.pipeline.pipeline import TranslationPipeline


class RecordingTranslator(TranslatorBase):
    """Mock that records every batch it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        self.batches.append(list(texts))
        return [f"[MOCK] {t}" for t in texts]


def _make_pipeline(batch_size: int = 3) -> tuple[TranslationPipeline, RecordingTranslator]:
    translator = RecordingTranslator()
    translator.load()
    config = PipelineConfig(chunk=ChunkConfig(batch_size=batch_size))
    return TranslationPipeline(translator, config), translator


def test_translate_sentences_one_to_one_and_ordered() -> None:
    pipeline, _ = _make_pipeline()
    sentences = [f"বাক্য {i}।" for i in range(7)]
    out = pipeline.translate_sentences(sentences)
    assert len(out) == 7
    for i, o in enumerate(out):
        assert f"বাক্য {i}" in o


def test_translate_sentences_batches_by_batch_size() -> None:
    pipeline, translator = _make_pipeline(batch_size=3)
    pipeline.translate_sentences([f"বাক্য {i}।" for i in range(7)])
    assert [len(b) for b in translator.batches] == [3, 3, 1]


def test_translate_sentences_normalizes_input() -> None:
    pipeline, translator = _make_pipeline()
    pipeline.translate_sentences(["  বাক্য\t\tএক।  "])
    # normalize() collapses runs of spaces/tabs and strips
    assert translator.batches[0][0] == "বাক্য এক।"


def test_translate_sentences_empty_list() -> None:
    pipeline, translator = _make_pipeline()
    assert pipeline.translate_sentences([]) == []
    assert translator.batches == []


def test_translate_sentences_blank_input_maps_to_blank_output() -> None:
    pipeline, translator = _make_pipeline(batch_size=2)
    out = pipeline.translate_sentences(["বাক্য এক।", "   ", "বাক্য দুই।"])
    assert len(out) == 3
    assert out[1] == ""
    assert "বাক্য এক" in out[0] and "বাক্য দুই" in out[2]
    # blank never reached the backend
    for batch in translator.batches:
        assert "" not in batch


def test_batches_are_length_sorted_but_output_order_restored() -> None:
    pipeline, translator = _make_pipeline(batch_size=2)
    # Mixed lengths, deliberately unsorted (long first) so that a naive
    # in-input-order batching produces a non-ascending first batch — this
    # is what makes the assertion below actually exercise the sort fix
    # instead of passing by coincidence.
    sentences = ["এটি একটি অনেক অনেক অনেক লম্বা বাংলা বাক্য যা চলতেই থাকে।", "ছোট।", "মাঝারি বাক্য।"]
    out = pipeline.translate_sentences(sentences)
    # Output order must match input order exactly
    assert [o.replace("[MOCK] ", "") for o in out] == [
        "এটি একটি অনেক অনেক অনেক লম্বা বাংলা বাক্য যা চলতেই থাকে।",
        "ছোট।",
        "মাঝারি বাক্য।",
    ]
    # Each batch must be internally ordered shortest-to-longest input
    for batch in translator.batches:
        lengths = [len(t) for t in batch]
        assert lengths == sorted(lengths)


def test_document_translate_still_preserves_paragraphs(mock_translator) -> None:
    """Regression guard: sorting inside _translate_in_batches must not break reassembly."""
    from bn_en_translate.pipeline.pipeline import TranslationPipeline

    mock_translator.load()
    pipeline = TranslationPipeline(mock_translator)
    text = "প্রথম অনুচ্ছেদ।\n\nদ্বিতীয় অনুচ্ছেদ যা একটু লম্বা।\n\nতৃতীয়।"
    result = pipeline.translate(text)
    assert result.count("\n\n") == 2
