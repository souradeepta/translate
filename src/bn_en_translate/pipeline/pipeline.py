"""Main translation pipeline: chunk → translate → reassemble."""

from __future__ import annotations

from bn_en_translate.config import PipelineConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.pipeline.chunker import Chunker
from bn_en_translate.pipeline.postprocessor import reassemble
from bn_en_translate.pipeline.preprocessor import normalize


class TranslationPipeline:
    """
    Orchestrates the full Bengali → English translation workflow.

    Usage:
        config = PipelineConfig()
        translator = get_translator(config)
        pipeline = TranslationPipeline(translator, config)

        with translator:
            result = pipeline.translate("Bengali story text...")
    """

    def __init__(self, translator: TranslatorBase, config: PipelineConfig | None = None) -> None:
        self.translator = translator
        self.config = config or PipelineConfig()
        self.chunker = Chunker(self.config.chunk)

    def translate(self, text: str) -> str:
        """
        Translate a Bengali story string to English.

        The translator must already be loaded (via load() or context manager).
        """
        if not text or not text.strip():
            return ""

        # 1. Normalize input
        normalized = normalize(text)

        # 2. Chunk into translation-sized pieces
        chunks = self.chunker.chunk(normalized)
        if not chunks:
            return ""

        # 3. Translate in batches
        chunk_texts = [c.text for c in chunks]
        translations = self._translate_in_batches(chunk_texts)

        # 4. Reassemble into the original paragraph structure
        return reassemble(chunks, translations)

    def translate_sentences(self, sentences: list[str]) -> list[str]:
        """
        Translate pre-split sentences with true batching, 1:1 in/out.

        Unlike translate(), this does no chunking or reassembly — each input
        string maps to exactly one output string, in order. Intended for
        benchmark corpora where hypothesis/reference alignment must be exact.
        Inputs are normalized the same way translate() normalizes documents.

        Empty or whitespace-only inputs are mapped to "" locally and never
        sent to the backend, while still occupying their original position
        in the returned list (1:1 ordering is preserved).
        """
        if not sentences:
            return []
        normalized = [normalize(s) for s in sentences]
        non_blank_indices = [i for i, s in enumerate(normalized) if s]
        non_blank_texts = [normalized[i] for i in non_blank_indices]
        translated = self._translate_in_batches(non_blank_texts) if non_blank_texts else []

        results = [""] * len(normalized)
        for idx, text in zip(non_blank_indices, translated, strict=True):
            results[idx] = text
        return results

    def _translate_in_batches(self, texts: list[str]) -> list[str]:
        batch_size = self.config.chunk.batch_size
        results: list[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            translated = self.translator.translate(
                batch,
                src_lang=self.config.model.src_lang,
                tgt_lang=self.config.model.tgt_lang,
            )
            results.extend(translated)

        return results

    def translate_file(self, input_path: str, output_path: str) -> str:
        """Read a Bengali story file, translate it, write English output."""
        from bn_en_translate.utils.file_io import read_story, write_translation

        text = read_story(input_path)
        result = self.translate(text)
        write_translation(result, output_path)
        return result
