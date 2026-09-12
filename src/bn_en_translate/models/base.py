"""Abstract base class for all translation model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from bn_en_translate.models.capabilities import (
    CONSERVATIVE_CAPABILITIES,
    TranslatorCapabilities,
    capabilities_for_adapter,
)


class TranslatorBase(ABC):
    """
    Contract that all translator implementations must satisfy.

    Lifecycle:
        1. Instantiate with config.
        2. Call load() — downloads/loads model into memory.
        3. Call translate() one or more times.
        4. Call unload() to free GPU/CPU memory.
    """

    DEFAULT_BEAM_SIZE: int = 4
    """Per-model default beam size. Subclasses override this."""

    CAPABILITIES: TranslatorCapabilities = CONSERVATIVE_CAPABILITIES

    def __init__(self) -> None:
        self._loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model into memory (GPU or CPU)."""

    @abstractmethod
    def unload(self) -> None:
        """Free model from memory."""

    @abstractmethod
    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a list of texts. Called only when loaded."""

    def _effective_beam_size(self) -> int:
        """Return beam_size from config if explicitly set, else this model's DEFAULT_BEAM_SIZE."""
        config = getattr(self, "config", None)
        if config is not None and getattr(config, "beam_size", None) is not None:
            return int(config.beam_size)
        return self.DEFAULT_BEAM_SIZE

    @property
    def capabilities(self) -> TranslatorCapabilities:
        """Return backend limits and feature support without loading a model.

        Tokenizer-backed adapters expose exact counting after ``load()``.  Before
        then, their declared limit remains useful but callers must use the
        conservative estimate returned by :meth:`count_input_tokens`.
        """
        declared = capabilities_for_adapter(type(self).__name__)
        if self.CAPABILITIES is not CONSERVATIVE_CAPABILITIES:
            declared = self.CAPABILITIES
        if self._token_counter() is not None and not declared.token_count_is_exact:
            return replace(declared, token_count_is_exact=True)
        return declared

    def count_input_tokens(self, text: str) -> int:
        """Count request input tokens, conservatively when no tokenizer is loaded."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        counter = self._token_counter()
        if counter is not None:
            return counter(text)
        # A tokenizer with byte fallback can emit more than one token for a
        # Unicode code point.  UTF-8 byte length is deliberately pessimistic but
        # guarantees a planning caller will not treat the old character/4 guess
        # as an exact limit.
        return max(1, len(text.encode("utf-8")))

    def _token_counter(self) -> Callable[[str], int] | None:
        """Return an exact loaded-tokenizer counter for common HF/CT2 adapters."""
        tokenizer: Any = getattr(self, "_tokenizer", None)
        if tokenizer is not None:
            def count_with_hf(value: str) -> int:
                encoded = tokenizer(value, add_special_tokens=True, truncation=False)
                token_ids = encoded["input_ids"]
                return len(token_ids)
            return count_with_hf
        sentencepiece: Any = getattr(self, "_sp", None)
        if sentencepiece is not None:
            # CT2 NLLB/IndicTrans2 requests append EOS and the source language.
            return lambda value: len(sentencepiece.encode(value, out_type=str)) + 2
        sentencepiece = getattr(self, "_sp_src", None)
        if sentencepiece is not None:
            return lambda value: len(sentencepiece.encode(value, out_type=str))
        processor: Any = getattr(self, "_processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            def count_with_processor(value: str) -> int:
                encoded = tokenizer(value, add_special_tokens=True, truncation=False)
                return len(encoded["input_ids"])
            return count_with_processor
        return None

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """
        Translate texts from src_lang to tgt_lang.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        if not self._loaded:
            raise RuntimeError(
                "Model is not loaded. Call load() before translate()."
            )
        if not texts:
            return []
        return self._translate_batch(texts, src_lang, tgt_lang)

    def __enter__(self) -> TranslatorBase:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()
