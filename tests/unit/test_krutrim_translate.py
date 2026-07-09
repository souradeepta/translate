"""Unit tests for Krutrim-Translate (Ola Krutrim, distilled IndicTrans2, CT2-native)."""
from __future__ import annotations

from unittest.mock import MagicMock

from bn_en_translate.config import ModelConfig


def test_krutrim_translate_default_config() -> None:
    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator
    t = KrutrimTranslateTranslator()
    assert t.config.model_name == "krutrim-translate"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"


def test_krutrim_translate_default_beam_size() -> None:
    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator
    t = KrutrimTranslateTranslator()
    assert t.DEFAULT_BEAM_SIZE == 3


def test_krutrim_translate_raises_before_load() -> None:
    import pytest

    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator
    t = KrutrimTranslateTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_krutrim_translate_load_raises_on_missing_model_dir() -> None:
    import pytest

    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator
    cfg = ModelConfig(
        model_name="krutrim-translate",
        model_path="/nonexistent/path/ct_model_indic_english",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
    )
    t = KrutrimTranslateTranslator(cfg)
    with pytest.raises(FileNotFoundError):
        t.load()


def test_krutrim_translate_unload_clears_state() -> None:
    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator
    t = KrutrimTranslateTranslator()
    t._loaded = True
    t._translator = MagicMock()
    t._sp_src = MagicMock()
    t._sp_tgt = MagicMock()
    t.unload()
    assert t._translator is None
    assert t._sp_src is None
    assert t._sp_tgt is None
    assert not t._loaded


def test_krutrim_translate_batch_uses_separate_src_tgt_spm() -> None:
    """Krutrim ships distinct SRC/TGT SentencePiece models (unlike shared-vocab IndicTrans2).

    No manual </s>+src_lang token appending and no target_prefix: verified
    empirically that Krutrim's distilled export needs neither — CT2 handles
    EOS/decoder-start via config.json, and IndicProcessor.preprocess_batch()
    embeds the language pair as plain text before SPM tokenization.
    """
    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator

    t = KrutrimTranslateTranslator()
    t._loaded = True

    mock_sp_src = MagicMock()
    mock_sp_src.encode = MagicMock(return_value=["▁test"])
    mock_sp_tgt = MagicMock()
    mock_sp_tgt.decode = MagicMock(return_value="I eat rice.")

    mock_result = MagicMock()
    mock_result.hypotheses = [["▁I", "▁eat", "▁rice", "."]]
    mock_translator = MagicMock()
    mock_translator.translate_batch = MagicMock(return_value=[mock_result])

    mock_processor = MagicMock()
    mock_processor.preprocess_batch = MagicMock(return_value=["ben_Beng eng_Latn আমি ভাত খাই।"])
    mock_processor.postprocess_batch = MagicMock(return_value=["I eat rice."])

    t._sp_src = mock_sp_src
    t._sp_tgt = mock_sp_tgt
    t._translator = mock_translator
    t._processor = mock_processor

    result = t._translate_batch(["আমি ভাত খাই।"], "ben_Beng", "eng_Latn")

    assert result == ["I eat rice."]
    mock_sp_src.encode.assert_called_once()
    mock_processor.preprocess_batch.assert_called_once_with(
        ["আমি ভাত খাই।"], src_lang="ben_Beng", tgt_lang="eng_Latn"
    )
    # no target_prefix — Krutrim's config.json drives decoder start, not CT2 kwargs
    _, kwargs = mock_translator.translate_batch.call_args
    assert "target_prefix" not in kwargs


def test_krutrim_translate_custom_config() -> None:
    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator
    cfg = ModelConfig(
        model_name="krutrim-translate", model_path="", src_lang="ben_Beng", tgt_lang="eng_Latn"
    )
    t = KrutrimTranslateTranslator(cfg)
    assert t.config.src_lang == "ben_Beng"
