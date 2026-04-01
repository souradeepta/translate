"""TDD tests for text utility functions."""

from __future__ import annotations

from bn_en_translate.utils.text_utils import (
    count_bengali_sentences,
    estimate_tokens,
    normalize_unicode,
    split_paragraphs,
    split_sentences_bengali,
)


# ---------------------------------------------------------------------------
# split_paragraphs()
# ---------------------------------------------------------------------------


def test_split_paragraphs_single_paragraph() -> None:
    result = split_paragraphs("Hello world.")
    assert result == ["Hello world."]


def test_split_paragraphs_double_newline() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    result = split_paragraphs(text)
    assert len(result) == 2


def test_split_paragraphs_strips_each_paragraph() -> None:
    text = "  First.  \n\n  Second.  "
    result = split_paragraphs(text)
    assert result[0] == "First."
    assert result[1] == "Second."


def test_split_paragraphs_ignores_empty_paragraphs() -> None:
    text = "First.\n\n\n\nSecond."
    result = split_paragraphs(text)
    assert len(result) == 2


def test_split_paragraphs_empty_string() -> None:
    assert split_paragraphs("") == []


# ---------------------------------------------------------------------------
# split_sentences_bengali()
# ---------------------------------------------------------------------------


def test_split_sentences_single_danda() -> None:
    text = "আমি ভালো আছি।"
    result = split_sentences_bengali(text)
    assert len(result) == 1


def test_split_sentences_two_dandas() -> None:
    text = "প্রথম বাক্য। দ্বিতীয় বাক্য।"
    result = split_sentences_bengali(text)
    assert len(result) == 2


def test_split_sentences_double_danda() -> None:
    text = "প্রথম বাক্য॥ দ্বিতীয় বাক্য।"
    result = split_sentences_bengali(text)
    assert len(result) == 2


def test_split_sentences_empty_string() -> None:
    result = split_sentences_bengali("")
    assert result == []


def test_split_sentences_no_punctuation() -> None:
    # A sentence fragment without ending punctuation
    text = "আমি ভালো আছি"
    result = split_sentences_bengali(text)
    assert len(result) >= 1
    assert text in result[0]


# ---------------------------------------------------------------------------
# estimate_tokens()
# ---------------------------------------------------------------------------


def test_estimate_tokens_positive() -> None:
    assert estimate_tokens("Hello world") > 0


def test_estimate_tokens_empty_is_one() -> None:
    # Even empty string shouldn't return 0 (to avoid division/logic issues)
    assert estimate_tokens("") == 1


def test_estimate_tokens_longer_text_more_tokens() -> None:
    short = "Hello."
    long = "Hello world this is a much longer sentence than the short one."
    assert estimate_tokens(long) > estimate_tokens(short)


# ---------------------------------------------------------------------------
# count_bengali_sentences()
# ---------------------------------------------------------------------------


def test_count_bengali_sentences_one() -> None:
    assert count_bengali_sentences("আমি ভালো আছি।") == 1


def test_count_bengali_sentences_two() -> None:
    text = "প্রথম বাক্য। দ্বিতীয় বাক্য।"
    assert count_bengali_sentences(text) == 2


# ---------------------------------------------------------------------------
# normalize_unicode()
# ---------------------------------------------------------------------------


def test_normalize_unicode_returns_nfc() -> None:
    import unicodedata

    nfd = unicodedata.normalize("NFD", "আমার")
    result = normalize_unicode(nfd)
    assert unicodedata.is_normalized("NFC", result)


def test_normalize_unicode_idempotent() -> None:
    text = "ভালো"
    assert normalize_unicode(normalize_unicode(text)) == normalize_unicode(text)
