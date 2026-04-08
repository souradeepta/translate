"""Unit tests for SeamlessM4T-v2 translator."""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from bn_en_translate.config import ModelConfig


def test_seamless_import() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    assert SeamlessTranslator is not None


def test_seamless_default_config() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    assert t.config.model_name == "seamless-medium"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"


def test_seamless_default_beam_size() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    assert t.DEFAULT_BEAM_SIZE == 5


def test_seamless_translate_raises_before_load() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_seamless_flores_to_seamless_lang_code() -> None:
    from bn_en_translate.models.seamless import _to_seamless_lang
    assert _to_seamless_lang("ben_Beng") == "ben"
    assert _to_seamless_lang("eng_Latn") == "eng"
    assert _to_seamless_lang("hin_Deva") == "hin"


def test_seamless_empty_input_returns_empty() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._processor = MagicMock()
    result = t.translate([], "ben_Beng", "eng_Latn")
    assert result == []
