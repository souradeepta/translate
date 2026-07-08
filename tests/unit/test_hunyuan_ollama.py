"""Hunyuan-MT via Ollama: factory routing and prompt format."""

from __future__ import annotations

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.factory import get_translator
from bn_en_translate.models.ollama_translator import OllamaTranslator


def test_factory_routes_hunyuan_to_ollama_translator() -> None:
    config = PipelineConfig(model=ModelConfig(model_name="hunyuan-mt-7b"))
    translator = get_translator(config)
    assert isinstance(translator, OllamaTranslator)
    # Hunyuan-MT prompt format (model card): plain instruction, no chat template
    assert "Translate the following segment into English" in translator.prompt_template


def test_hunyuan_uses_dedicated_model_tag() -> None:
    config = PipelineConfig(model=ModelConfig(model_name="hunyuan-mt-7b"))
    translator = get_translator(config)
    assert isinstance(translator, OllamaTranslator)
    assert "HY-MT" in translator.model_tag


def test_default_ollama_keeps_literary_prompt_and_config_tag() -> None:
    """The plain 'ollama' route must be unaffected by the Hunyuan additions."""
    config = PipelineConfig(model=ModelConfig(model_name="ollama"))
    translator = get_translator(config)
    assert isinstance(translator, OllamaTranslator)
    assert "professional literary translator" in translator.prompt_template
    assert translator.model_tag == config.ollama_model
