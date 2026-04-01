"""Splits long Bengali stories into translation-ready chunks."""

from __future__ import annotations

from dataclasses import dataclass

from bn_en_translate.config import ChunkConfig
from bn_en_translate.utils.text_utils import (
    estimate_tokens,
    split_paragraphs,
    split_sentences_bengali,
)


@dataclass(frozen=True)
class ChunkResult:
    """A single chunk ready for translation."""

    chunk_id: int
    para_id: int
    text: str
    sentence_start: int  # index of first sentence within the paragraph
    sentence_end: int    # index after last sentence (exclusive)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.text)


class Chunker:
    """
    Splits a Bengali story into chunks that fit within a model's context window.

    - Splits on paragraph boundaries first, then sentences within paragraphs.
    - Never splits mid-sentence.
    - Produces ChunkResult objects preserving para_id for reassembly.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    def chunk(self, text: str) -> list[ChunkResult]:
        """Chunk the full story text into a flat list of ChunkResult objects."""
        if not text or not text.strip():
            return []

        paragraphs = split_paragraphs(text)
        results: list[ChunkResult] = []
        chunk_id = 0

        for para_id, paragraph in enumerate(paragraphs):
            sentences = split_sentences_bengali(paragraph)
            if not sentences:
                continue

            para_chunks = self._chunk_paragraph(sentences, para_id, chunk_id)
            results.extend(para_chunks)
            chunk_id += len(para_chunks)

        return results

    def _chunk_paragraph(
        self, sentences: list[str], para_id: int, start_chunk_id: int
    ) -> list[ChunkResult]:
        chunks: list[ChunkResult] = []
        chunk_id = start_chunk_id
        current_sentences: list[str] = []
        current_tokens = 0
        sentence_start = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = estimate_tokens(sentence)

            # If adding this sentence would exceed the limit AND we already have content,
            # flush the current chunk first.
            if (
                current_tokens + sentence_tokens > self.config.max_tokens_per_chunk
                and current_sentences
            ):
                chunks.append(
                    ChunkResult(
                        chunk_id=chunk_id,
                        para_id=para_id,
                        text=" ".join(current_sentences),
                        sentence_start=sentence_start,
                        sentence_end=i,
                    )
                )
                chunk_id += 1
                sentence_start = i
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Flush remaining sentences
        if current_sentences:
            chunks.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    para_id=para_id,
                    text=" ".join(current_sentences),
                    sentence_start=sentence_start,
                    sentence_end=sentence_start + len(current_sentences),
                )
            )

        return chunks
