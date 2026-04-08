"""Unit tests for MADLAD-400-3B translator."""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from bn_en_translate.config import ModelConfig


def test_madlad_import() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    assert MADLADTranslator is not None


def test_madlad_default_config() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    assert t.config.model_name == "madlad-3b"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"


def test_madlad_default_beam_size() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    assert t.DEFAULT_BEAM_SIZE == 4


def test_madlad_translate_raises_before_load() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_madlad_prepends_target_language_tag() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    result = t._build_input_texts(["আমি ভাত খাই।"], "eng_Latn")
    assert result == ["<2en> আমি ভাত খাই।"]


def test_madlad_empty_input_returns_empty() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._tokenizer = MagicMock()
    result = t.translate([], "ben_Beng", "eng_Latn")
    assert result == []
