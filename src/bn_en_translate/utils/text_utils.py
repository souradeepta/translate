"""Utilities for Bengali text processing."""

from __future__ import annotations

import re
import unicodedata


# Bengali sentence-ending punctuation
BENGALI_DANDA = "\u0964"       # ।
BENGALI_DOUBLE_DANDA = "\u0965"  # ॥
SENTENCE_ENDERS = {BENGALI_DANDA, BENGALI_DOUBLE_DANDA, ".", "!", "?"}

# Rough token estimate: Bengali words average ~6 Unicode chars
# English words average ~5 chars. We use a simple character-based estimate.
CHARS_PER_TOKEN = 4.5


def split_paragraphs(text: str) -> list[str]:
    """Split text on double newlines; strip each paragraph."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences_bengali(text: str) -> list[str]:
    """
    Split Bengali text into sentences.

    Uses danda (।) and double-danda (॥) as primary sentence boundaries,
    falling back to Western punctuation (.!?) for mixed-script texts.
    Does NOT require indic-nlp-library — pure regex approach for reliability.
    """
    # Split on sentence-ending punctuation, keeping the delimiter
    pattern = rf"(?<=[{re.escape(BENGALI_DANDA + BENGALI_DOUBLE_DANDA)}!?.])\s+"
    raw = re.split(pattern, text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def estimate_tokens(text: str) -> int:
    """Rough token count estimate for chunking decisions."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def count_bengali_sentences(text: str) -> int:
    """Count approximate sentence count in Bengali text."""
    return len(split_sentences_bengali(text))


def normalize_unicode(text: str) -> str:
    """Normalize to NFC Unicode form (canonical composition)."""
    return unicodedata.normalize("NFC", text)
