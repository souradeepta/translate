"""TDD tests for the TranslatorBase abstract contract."""

from __future__ import annotations

import pytest

from bn_en_translate.models.base import TranslatorBase
from tests.conftest import MockTranslator


def test_translator_base_is_abstract() -> None:
    """Cannot instantiate TranslatorBase directly."""
    with pytest.raises(TypeError):
        TranslatorBase()  # type: ignore[abstract]


def test_translate_raises_if_not_loaded() -> None:
    translator = MockTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        translator.translate(["test"], "ben_Beng", "eng_Latn")


def test_translate_succeeds_after_load() -> None:
    translator = MockTranslator()
    translator.load()
    result = translator.translate(["hello"], "ben_Beng", "eng_Latn")
    assert len(result) == 1


def test_translate_returns_same_count_as_input() -> None:
    translator = MockTranslator()
    translator.load()
    inputs = ["one", "two", "three"]
    result = translator.translate(inputs, "ben_Beng", "eng_Latn")
    assert len(result) == len(inputs)


def test_translate_accepts_empty_list() -> None:
    translator = MockTranslator()
    translator.load()
    result = translator.translate([], "ben_Beng", "eng_Latn")
    assert result == []


def test_unload_prevents_further_translation() -> None:
    translator = MockTranslator()
    translator.load()
    translator.unload()
    with pytest.raises(RuntimeError, match="not loaded"):
        translator.translate(["test"], "ben_Beng", "eng_Latn")


def test_context_manager_loads_and_unloads() -> None:
    translator = MockTranslator()
    assert not translator._loaded
    with translator:
        assert translator._loaded
        result = translator.translate(["hello"], "ben_Beng", "eng_Latn")
        assert len(result) == 1
    assert not translator._loaded


def test_mock_translator_prefixes_output() -> None:
    translator = MockTranslator()
    translator.load()
    result = translator.translate(["আমি"], "ben_Beng", "eng_Latn")
    assert result[0].startswith("[MOCK]")
