"""TDD tests for the Chunker — pure logic, no model or GPU needed."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bn_en_translate.config import ChunkConfig
from bn_en_translate.pipeline.chunker import Chunker, ChunkResult


# ---------------------------------------------------------------------------
# Basic edge cases
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty_list() -> None:
    chunker = Chunker()
    assert chunker.chunk("") == []


def test_whitespace_only_returns_empty_list() -> None:
    chunker = Chunker()
    assert chunker.chunk("   \n\n   ") == []


def test_single_sentence_is_one_chunk() -> None:
    chunker = Chunker()
    text = "আমি বাংলায় গান গাই।"
    result = chunker.chunk(text)
    assert len(result) == 1
    assert isinstance(result[0], ChunkResult)


def test_single_sentence_chunk_contains_text() -> None:
    chunker = Chunker()
    text = "আমি বাংলায় গান গাই।"
    result = chunker.chunk(text)
    assert result[0].text.strip() != ""


# ---------------------------------------------------------------------------
# Token budget / splitting
# ---------------------------------------------------------------------------


def test_sentence_below_max_tokens_is_not_split() -> None:
    config = ChunkConfig(max_tokens_per_chunk=400)
    chunker = Chunker(config)
    # A single short sentence
    text = "আমি ভালো আছি।"
    result = chunker.chunk(text)
    assert len(result) == 1


def test_two_short_sentences_same_paragraph_merge_into_one_chunk() -> None:
    config = ChunkConfig(max_tokens_per_chunk=400)
    chunker = Chunker(config)
    text = "আমি ভালো আছি। তুমি কেমন আছ?"
    result = chunker.chunk(text)
    # Both are tiny — should fit in one chunk
    assert len(result) == 1


def test_chunk_never_exceeds_max_tokens() -> None:
    config = ChunkConfig(max_tokens_per_chunk=20)
    chunker = Chunker(config)
    # Each sentence is ~10 chars → ~2-3 tokens; max=20 tokens → expect splits
    text = "এক। দুই। তিন। চার। পাঁচ। ছয়। সাত। আট। নয়। দশ।"
    result = chunker.chunk(text)
    for chunk in result:
        assert chunk.estimated_tokens <= 20 + 10, (
            f"Chunk '{chunk.text[:30]}' has {chunk.estimated_tokens} tokens, "
            f"exceeding soft limit of 20"
        )


# ---------------------------------------------------------------------------
# Chunk IDs and metadata
# ---------------------------------------------------------------------------


def test_chunk_ids_are_sequential() -> None:
    chunker = Chunker(ChunkConfig(max_tokens_per_chunk=10))
    text = "এক। দুই। তিন। চার। পাঁচ।"
    result = chunker.chunk(text)
    ids = [c.chunk_id for c in result]
    assert ids == list(range(len(result)))


def test_single_paragraph_chunks_all_have_para_id_zero() -> None:
    chunker = Chunker(ChunkConfig(max_tokens_per_chunk=10))
    text = "এক। দুই। তিন।"
    result = chunker.chunk(text)
    for chunk in result:
        assert chunk.para_id == 0


def test_two_paragraphs_have_different_para_ids() -> None:
    chunker = Chunker()
    text = "প্রথম অনুচ্ছেদ।\n\nদ্বিতীয় অনুচ্ছেদ।"
    result = chunker.chunk(text)
    para_ids = {c.para_id for c in result}
    assert len(para_ids) == 2, f"Expected 2 unique para_ids, got: {para_ids}"


def test_para_ids_preserved_in_order() -> None:
    chunker = Chunker()
    text = "প্রথম।\n\nদ্বিতীয়।\n\nতৃতীয়।"
    result = chunker.chunk(text)
    para_ids = [c.para_id for c in result]
    assert para_ids == sorted(para_ids), "para_ids should be non-decreasing"


# ---------------------------------------------------------------------------
# Bengali sentence boundary detection
# ---------------------------------------------------------------------------


def test_danda_is_sentence_boundary() -> None:
    chunker = Chunker()
    # Two sentences separated by danda + space
    text = "প্রথম বাক্য। দ্বিতীয় বাক্য।"
    result = chunker.chunk(text)
    # They might be in one chunk (short) but the total text should be preserved
    all_text = " ".join(c.text for c in result)
    assert "প্রথম বাক্য" in all_text
    assert "দ্বিতীয় বাক্য" in all_text


def test_double_danda_is_sentence_boundary() -> None:
    chunker = Chunker()
    text = "প্রথম বাক্য॥ দ্বিতীয় বাক্য।"
    result = chunker.chunk(text)
    all_text = " ".join(c.text for c in result)
    assert "প্রথম বাক্য" in all_text


# ---------------------------------------------------------------------------
# Property-based test (Hypothesis)
# ---------------------------------------------------------------------------


@given(st.text(min_size=0, max_size=200))
@settings(max_examples=100)
def test_chunker_never_raises_on_arbitrary_text(text: str) -> None:
    """Chunker should never raise on any input string."""
    chunker = Chunker()
    # Should not raise
    result = chunker.chunk(text)
    assert isinstance(result, list)


@given(st.integers(min_value=10, max_value=500))
def test_all_chunk_ids_are_unique(max_tokens: int) -> None:
    config = ChunkConfig(max_tokens_per_chunk=max_tokens)
    chunker = Chunker(config)
    text = "এক। দুই। তিন। চার। পাঁচ।\n\nছয়। সাত। আট।"
    result = chunker.chunk(text)
    ids = [c.chunk_id for c in result]
    assert len(ids) == len(set(ids)), "All chunk_ids must be unique"
