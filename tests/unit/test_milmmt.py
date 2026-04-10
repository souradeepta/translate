"""Unit tests for MiLMMT-46-1B translator."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import torch

from bn_en_translate.config import ModelConfig


def test_milmmt_import() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    assert MiLMMTTranslator is not None


def test_milmmt_default_config() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    assert t.config.model_name == "milmmt-46-1b"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"


def test_milmmt_default_beam_size() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    assert t.DEFAULT_BEAM_SIZE == 1


def test_milmmt_translate_raises_before_load() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_milmmt_build_prompts_bengali_to_english() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    prompts = t._build_prompts(["আমি ভাত খাই।"], "ben_Beng", "eng_Latn")
    assert len(prompts) == 1
    assert "Translate this from Bengali to English:" in prompts[0]
    assert "Bengali: আমি ভাত খাই।" in prompts[0]
    assert prompts[0].endswith("English:")


def test_milmmt_build_prompts_unknown_lang_falls_back_gracefully() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    prompts = t._build_prompts(["test"], "xyz_Latn", "abc_Cyrl")
    # Should not raise; should produce a reasonable prompt with capitalized prefix
    assert "Translate this from" in prompts[0]
    assert "Xyz" in prompts[0]
    assert "Abc" in prompts[0]


def test_milmmt_build_prompts_batch() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    texts = ["আমি ভাত খাই।", "সে স্কুলে যায়।"]
    prompts = t._build_prompts(texts, "ben_Beng", "eng_Latn")
    assert len(prompts) == 2
    for prompt in prompts:
        assert "Bengali:" in prompt
        assert "English:" in prompt


def test_milmmt_empty_input_returns_empty() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._tokenizer = MagicMock()
    result = t.translate([], "ben_Beng", "eng_Latn")
    assert result == []


def test_milmmt_custom_config() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    cfg = ModelConfig(model_name="milmmt-46-1b", model_path="", src_lang="ben_Beng", tgt_lang="eng_Latn")
    t = MiLMMTTranslator(cfg)
    assert t.config.src_lang == "ben_Beng"
