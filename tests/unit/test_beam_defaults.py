"""Tests for per-model beam size defaults."""
from __future__ import annotations
from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


class _ConcreteTranslator(TranslatorBase):
    """Minimal concrete subclass for testing TranslatorBase directly."""
    DEFAULT_BEAM_SIZE = 7

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        return texts


def test_effective_beam_size_uses_config_when_set() -> None:
    cfg = ModelConfig(beam_size=3)
    t = _ConcreteTranslator(cfg)
    assert t._effective_beam_size() == 3


def test_effective_beam_size_uses_class_default_when_config_is_none() -> None:
    cfg = ModelConfig(beam_size=None)
    t = _ConcreteTranslator(cfg)
    assert t._effective_beam_size() == 7


def test_base_default_beam_size_is_4() -> None:
    assert TranslatorBase.DEFAULT_BEAM_SIZE == 4
