"""Main translation pipeline: chunk → translate → reassemble."""

from __future__ import annotations

from bn_en_translate.config import PipelineConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.pipeline.chunker import Chunker
from bn_en_translate.pipeline.postprocessor import reassemble
from bn_en_translate.pipeline.preprocessor import normalize
from bn_en_translate.utils.cuda_check import ensure_vram_available


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
        """Translate texts in batches, length-sorted to minimize padding waste.

        HF models pad each batch to its longest member; grouping similar
        lengths cuts wasted decode steps. Original order is restored before
        returning, so callers (and reassemble()) see 1:1 positional mapping.
        CT2 backends sort internally — this is harmless there.
        """
        batch_size = self.config.chunk.batch_size
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: list[str] = [""] * len(texts)

        for start in range(0, len(order), batch_size):
            index_batch = order[start : start + batch_size]
            batch = [texts[i] for i in index_batch]
            translated = self.translator.translate(
                batch,
                src_lang=self.config.model.src_lang,
                tgt_lang=self.config.model.tgt_lang,
            )
            for idx, out in zip(index_batch, translated, strict=True):
                results[idx] = out

        return results

    def translate_file(self, input_path: str, output_path: str) -> str:
        """Read a Bengali story file, translate it, write English output."""
        from bn_en_translate.utils.file_io import read_story, write_translation

        text = read_story(input_path)
        result = self.translate(text)
        write_translation(result, output_path)
        return result


def _make_ollama(config: PipelineConfig) -> TranslatorBase:
    """Seam for tests — constructs the real OllamaTranslator."""
    from bn_en_translate.models.ollama_translator import OllamaTranslator

    return OllamaTranslator(config)


def _ollama_vram_requirement_mib(ollama_model: str) -> int:
    """Look up the polish model's VRAM need; unknown tags assume the largest."""
    from bn_en_translate.config import MODEL_VRAM_MIB

    exact = MODEL_VRAM_MIB.get(f"ollama-{ollama_model}")
    if exact is not None:
        return exact
    # Tags often carry quant suffixes (qwen2.5:7b-instruct-q4_K_M) — prefix match
    for key, mib in MODEL_VRAM_MIB.items():
        if key.startswith("ollama-") and ollama_model.startswith(key.removeprefix("ollama-")):
            return mib
    return max(v for k, v in MODEL_VRAM_MIB.items() if k.startswith("ollama-"))


def polish_with_ollama(english_text: str, config: PipelineConfig) -> str:
    """Run the Ollama literary polish pass over translated English text.

    Per-paragraph so paragraph structure survives (key invariant #3).
    Caller must have unloaded the translation model first — this checks the
    Ollama model's VRAM requirement and raises rather than OOM-ing.
    """
    from bn_en_translate.utils.text_utils import split_paragraphs

    ensure_vram_available(
        _ollama_vram_requirement_mib(config.ollama_model), context="Ollama polish pass"
    )

    paragraphs = split_paragraphs(english_text)
    ollama = _make_ollama(config)
    ollama.load()
    try:
        polished = ollama.translate(paragraphs, src_lang="eng_Latn", tgt_lang="eng_Latn")
    finally:
        ollama.unload()
    return "\n\n".join(polished)
