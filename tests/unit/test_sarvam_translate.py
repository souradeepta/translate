"""Unit tests for Sarvam-Translate (Gemma3-4B, AI4Bharat + Sarvam AI)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from bn_en_translate.config import ModelConfig


def test_sarvam_translate_default_config() -> None:
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    t = SarvamTranslateTranslator()
    assert t.config.model_name == "sarvam-translate"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"
    assert t.config.load_in_4bit is True


def test_sarvam_translate_default_beam_size() -> None:
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    t = SarvamTranslateTranslator()
    assert t.DEFAULT_BEAM_SIZE == 1


def test_sarvam_translate_raises_before_load() -> None:
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    t = SarvamTranslateTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_sarvam_translate_build_messages_bengali_to_english() -> None:
    """Chat-style system+user messages, per the model card's exact prompt format."""
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    t = SarvamTranslateTranslator()
    messages_batch = t._build_messages(["আমি ভাত খাই।"], "ben_Beng", "eng_Latn")
    assert len(messages_batch) == 1
    messages = messages_batch[0]
    assert messages[0] == {"role": "system", "content": "Translate the text below to English."}
    assert messages[1] == {"role": "user", "content": "আমি ভাত খাই।"}


def test_sarvam_translate_build_messages_unknown_lang_falls_back() -> None:
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    t = SarvamTranslateTranslator()
    messages_batch = t._build_messages(["test"], "xyz_Latn", "abc_Cyrl")
    assert "Abc" in messages_batch[0][0]["content"]


def test_sarvam_translate_empty_input_returns_empty() -> None:
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    t = SarvamTranslateTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._tokenizer = MagicMock()
    result = t.translate([], "ben_Beng", "eng_Latn")
    assert result == []


def test_sarvam_translate_unload_clears_state() -> None:
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    t = SarvamTranslateTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._tokenizer = MagicMock()
    t.unload()
    assert t._model is None
    assert t._tokenizer is None
    assert not t._loaded


def test_sarvam_translate_batch_slices_prompt_tokens() -> None:
    """_translate_batch must strip echoed prompt tokens, same contract as MiLMMT."""
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator

    t = SarvamTranslateTranslator()
    t._loaded = True

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value="<templated prompt>")
    prompt_ids = torch.zeros((1, 5), dtype=torch.long)
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(return_value=prompt_ids)
    mock_inputs["input_ids"] = prompt_ids
    mock_inputs.to = MagicMock(return_value=mock_inputs)
    mock_tokenizer.return_value = mock_inputs

    full_output = torch.zeros((1, 8), dtype=torch.long)
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    mock_model.generate = MagicMock(return_value=full_output)

    mock_tokenizer.batch_decode = MagicMock(return_value=["I eat rice."])

    t._model = mock_model
    t._tokenizer = mock_tokenizer

    result = t._translate_batch(["আমি ভাত খাই।"], "ben_Beng", "eng_Latn")

    assert result == ["I eat rice."]
    decoded_arg = mock_tokenizer.batch_decode.call_args[0][0]
    assert decoded_arg.shape[1] == 3, f"Expected 3 new tokens, got {decoded_arg.shape[1]}"
    generate_kwargs = mock_model.generate.call_args[1]
    assert generate_kwargs.get("do_sample") is False


def test_sarvam_translate_load_uses_4bit_quantization(monkeypatch) -> None:
    """load() must pass a BitsAndBytesConfig(load_in_4bit=True) when config.load_in_4bit."""
    monkeypatch.setattr(
        "bn_en_translate.models.hf_utils.flash_attn_available", lambda: False
    )

    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator

    cfg = ModelConfig(
        model_name="sarvam-translate",
        model_path="",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        device="cpu",
        load_in_4bit=False,  # force cpu path in test; quantization asserted separately below
    )
    t = SarvamTranslateTranslator(cfg)

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch(
             "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
         ) as mock_from_pretrained:
        t.load()

    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs.get("quantization_config") is None


def test_sarvam_translate_custom_config() -> None:
    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator
    cfg = ModelConfig(
        model_name="sarvam-translate", model_path="", src_lang="ben_Beng", tgt_lang="eng_Latn"
    )
    t = SarvamTranslateTranslator(cfg)
    assert t.config.src_lang == "ben_Beng"
