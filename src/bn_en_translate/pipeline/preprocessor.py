"""Bengali text preprocessor: normalize unicode and fix common artifacts."""

from __future__ import annotations

import re
import unicodedata


# Zero-width joiner / non-joiner
ZWJ = "\u200d"
ZWNJ = "\u200c"

# Bengali Hasanta (virama)
HASANTA = "\u09cd"


def normalize(text: str) -> str:
    """
    Full normalization pipeline for Bengali input text.

    Steps:
    1. NFC Unicode normalization (canonical composition)
    2. Collapse excessive whitespace (preserve paragraph breaks)
    3. Normalize Bengali punctuation variants
    """
    text = unicodedata.normalize("NFC", text)
    text = _collapse_whitespace(text)
    text = _normalize_punctuation(text)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to one; preserve newlines."""
    # Replace runs of spaces/tabs (not newlines) with single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ newlines to exactly 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_punctuation(text: str) -> str:
    """Normalize Unicode punctuation variants to standard Bengali forms."""
    # Some texts use U+0964 variants or ASCII period for Bengali danda
    # Normalize fullwidth period to ASCII
    text = text.replace("\uff0e", ".")
    # Normalize double danda variants
    text = text.replace("\u0965", "\u0965")  # identity (already canonical)
    return text
