"""Reassembles translated chunks back into a coherent English story."""

from __future__ import annotations

import re

from bn_en_translate.pipeline.chunker import ChunkResult


def reassemble(chunks: list[ChunkResult], translations: list[str]) -> str:
    """
    Reassemble translated chunks into the original paragraph structure.

    Args:
        chunks: Original ChunkResult list (provides para_id ordering).
        translations: Translated text for each chunk, same order as chunks.

    Returns:
        Final English story with paragraphs separated by double newlines.
    """
    if not chunks:
        return ""

    # Group translations by paragraph
    para_map: dict[int, list[str]] = {}
    for chunk, translated_text in zip(chunks, translations):
        para_map.setdefault(chunk.para_id, []).append(translated_text)

    # Join sentences within paragraphs, then join paragraphs
    paragraphs = []
    for para_id in sorted(para_map.keys()):
        para_text = " ".join(para_map[para_id])
        para_text = _clean_english(para_text)
        paragraphs.append(para_text)

    return "\n\n".join(paragraphs)


def _clean_english(text: str) -> str:
    """Fix common machine translation artifacts in English output."""
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Collapse multiple spaces to one
    text = re.sub(r" {2,}", " ", text)
    # Fix repeated articles (the the, a a)
    text = re.sub(r"\b(the|a|an) \1\b", r"\1", text, flags=re.IGNORECASE)
    return text.strip()
