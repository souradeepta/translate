"""Unit tests for NiuTrans LMT-60-1.7B translator (Qwen3-based causal LM)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from bn_en_translate.config import ModelConfig


def test_lmt60_default_config() -> None:
    from bn_en_translate.models.lmt60 import LMT60Translator
    t = LMT60Translator()
    assert t.config.model_name == "lmt-60-1.7b"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"


def test_lmt60_default_beam_size_is_five() -> None:
    """Model card quickstart uses num_beams=5, do_sample=False."""
    from bn_en_translate.models.lmt60 import LMT60Translator
    t = LMT60Translator()
    assert t.DEFAULT_BEAM_SIZE == 5


def test_lmt60_translate_raises_before_load() -> None:
    from bn_en_translate.models.lmt60 import LMT60Translator
    t = LMT60Translator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_lmt60_build_prompts_bengali_to_english() -> None:
    """Prompt format verbatim from the model card (verified 2026-07-08)."""
    from bn_en_translate.models.lmt60 import LMT60Translator
    t = LMT60Translator()
    prompts = t._build_prompts(["আমি ভাত খাই।"], "ben_Beng", "eng_Latn")
    assert len(prompts) == 1
    assert "Translate the following text from Bengali into English:" in prompts[0]
    assert "Bengali: আমি ভাত খাই।" in prompts[0]
    assert prompts[0].endswith("English:")


def test_lmt60_build_prompts_unknown_lang_falls_back_gracefully() -> None:
    from bn_en_translate.models.lmt60 import LMT60Translator
    t = LMT60Translator()
    prompts = t._build_prompts(["test"], "xyz_Latn", "abc_Cyrl")
    assert "Translate the following text from" in prompts[0]
    assert "Xyz" in prompts[0]
    assert "Abc" in prompts[0]


def test_lmt60_factory_routing() -> None:
    from bn_en_translate.config import PipelineConfig
    from bn_en_translate.models.factory import get_translator
    from bn_en_translate.models.lmt60 import LMT60Translator

    config = PipelineConfig(model=ModelConfig(model_name="lmt-60-1.7b"))
    assert isinstance(get_translator(config), LMT60Translator)


def test_lmt60_unload_clears_state() -> None:
    from bn_en_translate.models.lmt60 import LMT60Translator
    t = LMT60Translator()
    t._loaded = True
    t._model = MagicMock()
    t._tokenizer = MagicMock()
    t.unload()
    assert t._model is None
    assert t._tokenizer is None
    assert not t._loaded


def test_lmt60_translate_batch_applies_chat_template_and_slices_prompt() -> None:
    """The model card wraps the prompt in a user chat message; output must be
    decoded from generated tokens only (prompt echo stripped at input_len)."""
    from bn_en_translate.models.lmt60 import LMT60Translator

    t = LMT60Translator()
    t._loaded = True

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(
        side_effect=lambda messages, **kw: f"<chat>{messages[0]['content']}</chat>"
    )
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
    # Chat template was applied with a single user message
    tmpl_args = mock_tokenizer.apply_chat_template.call_args
    assert tmpl_args[0][0][0]["role"] == "user"
    assert tmpl_args[1].get("add_generation_prompt") is True
    assert tmpl_args[1].get("tokenize") is False
    # The tokenizer received the chat-templated string
    tokenized_texts = mock_tokenizer.call_args[0][0]
    assert tokenized_texts[0].startswith("<chat>")
    # Only the 3 generated tokens were decoded
    decoded_arg = mock_tokenizer.batch_decode.call_args[0][0]
    assert decoded_arg.shape[1] == 3
    # Deterministic beam search per model card
    generate_kwargs = mock_model.generate.call_args[1]
    assert generate_kwargs.get("do_sample") is False
    assert generate_kwargs.get("num_beams") == 5


def test_lmt60_load_sets_left_padding_and_bf16(monkeypatch) -> None:
    """Causal LM batching requires left padding; Qwen3 native dtype is bf16."""
    monkeypatch.setattr(
        "bn_en_translate.models.hf_utils.flash_attn_available", lambda: False
    )

    from bn_en_translate.models.lmt60 import LMT60Translator

    cfg = ModelConfig(
        model_name="lmt-60-1.7b",
        model_path="",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        device="cpu",
    )
    t = LMT60Translator(cfg)

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch(
             "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
         ) as mock_from_pretrained:
        t.load()

    assert mock_tokenizer.padding_side == "left"
    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["dtype"] == torch.bfloat16
    assert kwargs["attn_implementation"] == "sdpa"
