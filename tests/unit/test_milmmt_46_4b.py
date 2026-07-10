"""Unit tests for MiLMMT-46-4B translator (Gemma3-4B, 4-bit quantized)."""
from __future__ import annotations

from bn_en_translate.config import ModelConfig


def test_milmmt_4b_default_config() -> None:
    from bn_en_translate.models.milmmt import MiLMMT4BTranslator
    t = MiLMMT4BTranslator()
    assert t.config.model_name == "milmmt-46-4b"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"
    assert t.config.load_in_4bit is True


def test_milmmt_4b_uses_own_hf_id_and_local_path() -> None:
    from bn_en_translate.models.milmmt import MiLMMT4BTranslator, MiLMMTTranslator
    t = MiLMMT4BTranslator()
    assert t.HF_MODEL_ID == "xiaomi-research/MiLMMT-46-4B-v0.1"
    assert "milmmt-46-4B" in t._LOCAL_PATH or "milmmt-46-4b" in t._LOCAL_PATH.lower()
    # different checkpoint from the 1B variant
    assert t._LOCAL_PATH != MiLMMTTranslator._LOCAL_PATH


def test_milmmt_4b_default_beam_size() -> None:
    from bn_en_translate.models.milmmt import MiLMMT4BTranslator
    t = MiLMMT4BTranslator()
    assert t.DEFAULT_BEAM_SIZE == 1


def test_milmmt_4b_custom_config_can_override_quantization() -> None:
    from bn_en_translate.models.milmmt import MiLMMT4BTranslator
    cfg = ModelConfig(
        model_name="milmmt-46-4b",
        model_path="",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        load_in_4bit=False,
    )
    t = MiLMMT4BTranslator(cfg)
    assert t.config.load_in_4bit is False
