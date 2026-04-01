"""Abstract base class for all translation model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TranslatorBase(ABC):
    """
    Contract that all translator implementations must satisfy.

    Lifecycle:
        1. Instantiate with config.
        2. Call load() — downloads/loads model into memory.
        3. Call translate() one or more times.
        4. Call unload() to free GPU/CPU memory.
    """

    _loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model into memory (GPU or CPU)."""

    @abstractmethod
    def unload(self) -> None:
        """Free model from memory."""

    @abstractmethod
    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a list of texts. Called only when loaded."""

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

    def __enter__(self) -> "TranslatorBase":
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()
