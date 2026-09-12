"""Capabilities and token-counter contract tests."""

from __future__ import annotations

import pytest

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.models.capabilities import TranslatorCapabilities
from bn_en_translate.models.factory import get_translator, supported_model_names


class _Translator(TranslatorBase):
    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        return texts


class _Tokenizer:
    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split()) + 2))}


def test_default_capabilities_are_conservative_and_immutable() -> None:
    translator = _Translator()
    assert translator.capabilities.max_input_tokens == 384
    assert not translator.capabilities.token_count_is_exact
    assert translator.count_input_tokens("বাংলা") == len("বাংলা".encode())
    with pytest.raises(AttributeError):
        translator.capabilities.max_input_tokens = 1  # type: ignore[misc]


def test_loaded_huggingface_tokenizer_enables_exact_counting() -> None:
    translator = _Translator()
    translator._tokenizer = _Tokenizer()  # type: ignore[attr-defined]
    assert translator.capabilities.token_count_is_exact
    assert translator.count_input_tokens("one two three") == 5


def test_capabilities_validate_limits_and_language_pairs() -> None:
    capabilities = TranslatorCapabilities(
        max_input_tokens=10,
        max_output_tokens=5,
        supports_batching=True,
        supports_context_prompt=False,
        supports_glossary_constraints=False,
        supports_json_output=False,
        supports_seed=True,
        token_count_is_exact=True,
        supported_language_pairs=frozenset({("ben_Beng", "eng_Latn")}),
    )
    assert capabilities.supports_language_pair("ben_Beng", "eng_Latn")
    assert not capabilities.supports_language_pair("eng_Latn", "ben_Beng")
    with pytest.raises(ValueError, match="limits"):
        TranslatorCapabilities(0, 1, False, False, False, False, False, False)


def test_every_registered_model_reports_a_non_fallback_profile() -> None:
    for name in supported_model_names():
        translator = get_translator(PipelineConfig(model=ModelConfig(model_name=name)))
        capabilities = translator.capabilities
        assert capabilities.max_input_tokens >= 512, name
        assert capabilities.max_output_tokens >= 256, name
