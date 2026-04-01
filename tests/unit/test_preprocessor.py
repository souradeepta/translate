"""TDD tests for Bengali text preprocessor."""

from __future__ import annotations

import unicodedata

from bn_en_translate.pipeline.preprocessor import normalize, _collapse_whitespace


def test_nfc_normalization_applied() -> None:
    # Create a string that is NFD (decomposed) - ক + combining vowel
    nfd_text = unicodedata.normalize("NFD", "আমার")
    result = normalize(nfd_text)
    assert unicodedata.is_normalized("NFC", result)


def test_empty_string_returns_empty() -> None:
    assert normalize("") == ""


def test_whitespace_only_returns_empty() -> None:
    assert normalize("   \n\n  ") == ""


def test_extra_spaces_collapsed() -> None:
    text = "আমি   ভালো   আছি।"
    result = normalize(text)
    assert "  " not in result


def test_paragraph_breaks_preserved() -> None:
    text = "প্রথম অনুচ্ছেদ।\n\nদ্বিতীয় অনুচ্ছেদ।"
    result = normalize(text)
    assert "\n\n" in result


def test_excessive_newlines_collapsed_to_double() -> None:
    text = "প্রথম।\n\n\n\n\nদ্বিতীয়।"
    result = normalize(text)
    assert "\n\n\n" not in result
    assert "\n\n" in result


def test_tabs_normalized_to_space() -> None:
    text = "আমি\tভালো\tআছি।"
    result = normalize(text)
    assert "\t" not in result


def test_text_is_stripped() -> None:
    text = "   আমি ভালো আছি।   "
    result = normalize(text)
    assert result == result.strip()


def test_normal_bengali_text_unchanged_structurally() -> None:
    text = "রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যের কবি।"
    result = normalize(text)
    # The Bengali content should be intact
    assert "রবীন্দ্রনাথ" in result
    assert "ঠাকুর" in result
