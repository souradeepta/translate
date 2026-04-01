"""TDD tests for the postprocessor / reassembler."""

from __future__ import annotations

from bn_en_translate.pipeline.chunker import ChunkResult
from bn_en_translate.pipeline.postprocessor import reassemble, _clean_english


# ---------------------------------------------------------------------------
# reassemble()
# ---------------------------------------------------------------------------


def test_empty_chunks_returns_empty_string() -> None:
    assert reassemble([], []) == ""


def test_single_chunk_single_translation() -> None:
    chunks = [ChunkResult(chunk_id=0, para_id=0, text="hello", sentence_start=0, sentence_end=1)]
    translations = ["Hello world."]
    result = reassemble(chunks, translations)
    assert result == "Hello world."


def test_two_chunks_same_paragraph_joined_with_space() -> None:
    chunks = [
        ChunkResult(chunk_id=0, para_id=0, text="A.", sentence_start=0, sentence_end=1),
        ChunkResult(chunk_id=1, para_id=0, text="B.", sentence_start=1, sentence_end=2),
    ]
    translations = ["First sentence.", "Second sentence."]
    result = reassemble(chunks, translations)
    assert "First sentence." in result
    assert "Second sentence." in result


def test_two_paragraphs_separated_by_double_newline() -> None:
    chunks = [
        ChunkResult(chunk_id=0, para_id=0, text="para1", sentence_start=0, sentence_end=1),
        ChunkResult(chunk_id=1, para_id=1, text="para2", sentence_start=0, sentence_end=1),
    ]
    translations = ["First paragraph.", "Second paragraph."]
    result = reassemble(chunks, translations)
    assert "\n\n" in result
    paragraphs = result.split("\n\n")
    assert len(paragraphs) == 2


def test_paragraphs_reassembled_in_original_order() -> None:
    chunks = [
        ChunkResult(chunk_id=0, para_id=0, text="A", sentence_start=0, sentence_end=1),
        ChunkResult(chunk_id=1, para_id=1, text="B", sentence_start=0, sentence_end=1),
        ChunkResult(chunk_id=2, para_id=2, text="C", sentence_start=0, sentence_end=1),
    ]
    translations = ["Alpha", "Beta", "Gamma"]
    result = reassemble(chunks, translations)
    idx_alpha = result.index("Alpha")
    idx_beta = result.index("Beta")
    idx_gamma = result.index("Gamma")
    assert idx_alpha < idx_beta < idx_gamma


# ---------------------------------------------------------------------------
# _clean_english()
# ---------------------------------------------------------------------------


def test_removes_space_before_period() -> None:
    assert _clean_english("Hello .") == "Hello."


def test_removes_space_before_comma() -> None:
    assert _clean_english("Hello , world.") == "Hello, world."


def test_removes_double_spaces() -> None:
    assert _clean_english("Hello  world.") == "Hello world."


def test_removes_repeated_article_the() -> None:
    result = _clean_english("She went to the the store.")
    assert "the the" not in result


def test_removes_repeated_article_a() -> None:
    result = _clean_english("He found a a key.")
    assert "a a" not in result


def test_clean_english_empty_string() -> None:
    assert _clean_english("") == ""


def test_clean_english_no_changes_needed() -> None:
    text = "She walked to the store and bought bread."
    assert _clean_english(text) == text
