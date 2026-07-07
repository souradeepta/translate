"""Unit tests for MiLMMT-46-1B translator."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import torch

from bn_en_translate.config import ModelConfig


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


def test_milmmt_unload_clears_state() -> None:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    t = MiLMMTTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._tokenizer = MagicMock()
    t.unload()
    assert t._model is None
    assert t._tokenizer is None
    assert not t._loaded


def test_milmmt_translate_batch_slices_prompt_tokens() -> None:
    """_translate_batch must strip echoed prompt tokens from generated output."""
    from bn_en_translate.models.milmmt import MiLMMTTranslator

    t = MiLMMTTranslator()
    t._loaded = True

    # Simulate tokenizer: encodes prompts to 5-token tensors (input_len=5),
    # generates 8 tokens total (first 5 are prompt, last 3 are translation).
    mock_tokenizer = MagicMock()
    mock_tokenizer.padding_side = "right"  # will be overridden in load(); test batch only
    prompt_ids = torch.zeros((1, 5), dtype=torch.long)
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(return_value=prompt_ids)
    mock_inputs["input_ids"] = prompt_ids
    mock_inputs.to = MagicMock(return_value=mock_inputs)
    mock_tokenizer.return_value = mock_inputs

    # generate() returns 8 tokens: prompt (5) + translation (3)
    full_output = torch.zeros((1, 8), dtype=torch.long)
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    mock_model.generate = MagicMock(return_value=full_output)

    mock_tokenizer.batch_decode = MagicMock(return_value=["I eat rice."])

    t._model = mock_model
    t._tokenizer = mock_tokenizer

    result = t._translate_batch(["আমি ভাত খাই।"], "ben_Beng", "eng_Latn")

    assert result == ["I eat rice."]
    # Verify only the 3 generated tokens (not all 8) were decoded
    decoded_arg = mock_tokenizer.batch_decode.call_args[0][0]
    assert decoded_arg.shape[1] == 3, f"Expected 3 new tokens, got {decoded_arg.shape[1]}"
    # Verify do_sample=False was passed
    generate_kwargs = mock_model.generate.call_args[1]
    assert generate_kwargs.get("do_sample") is False


def test_attn_fallback_is_sdpa(monkeypatch) -> None:
    """Without flash-attn installed, the fallback must be sdpa, not eager."""
    import bn_en_translate.models.milmmt as milmmt_mod

    monkeypatch.setattr(milmmt_mod, "_flash_attn_available", lambda: False)
    assert milmmt_mod._resolve_attn_implementation(use_flash=True) == "sdpa"
    assert milmmt_mod._resolve_attn_implementation(use_flash=False) == "sdpa"


def test_attn_uses_flash_when_available(monkeypatch) -> None:
    import bn_en_translate.models.milmmt as milmmt_mod

    monkeypatch.setattr(milmmt_mod, "_flash_attn_available", lambda: True)
    assert milmmt_mod._resolve_attn_implementation(use_flash=True) == "flash_attention_2"


def test_milmmt_load_passes_resolved_attn_impl_to_from_pretrained(monkeypatch) -> None:
    """load() must pass the resolver's output (not a hardcoded string) as attn_implementation.

    Patches the AutoModelForCausalLM/AutoTokenizer from_pretrained classmethods directly
    on the real `transformers` module (load()'s local `from transformers import ...`
    resolves to the same class objects), and forces device="cpu" so no CUDA/download occurs.
    """
    import bn_en_translate.models.milmmt as milmmt_mod

    monkeypatch.setattr(milmmt_mod, "_flash_attn_available", lambda: False)

    from bn_en_translate.models.milmmt import MiLMMTTranslator

    cfg = ModelConfig(
        model_name="milmmt-46-1b",
        model_path="",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        device="cpu",
    )
    t = MiLMMTTranslator(cfg)

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch(
             "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
         ) as mock_from_pretrained:
        t.load()

    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["attn_implementation"] == "sdpa"
