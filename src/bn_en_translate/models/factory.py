"""Factory for creating translator instances by name."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.base import TranslatorBase

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry maps a lower-case model name to a callable that accepts a
# PipelineConfig and returns a TranslatorBase.  Adding a new model is a
# one-line registration — no if/elif chain to touch.

_REGISTRY: dict[str, Callable[[PipelineConfig], TranslatorBase]] = {}


def register_model(name: str) -> Callable:
    """Decorator that registers a factory function under the given model name."""
    def _decorator(fn: Callable[[PipelineConfig], TranslatorBase]) -> Callable:
        _REGISTRY[name.lower()] = fn
        return fn
    return _decorator


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

@register_model("nllb-600m")
@register_model("nllb-1.3b")
def _make_nllb(config: PipelineConfig) -> TranslatorBase:
    ct2_path = _ct2_path(config.model)
    if ct2_path.exists():
        from bn_en_translate.models.nllb_ct2 import NLLBCt2Translator
        return NLLBCt2Translator(config.model)
    from bn_en_translate.models.nllb import NLLBTranslator
    return NLLBTranslator(config.model)


@register_model("indictrans2-1b")
@register_model("indictrans2")
def _make_indictrans2(config: PipelineConfig) -> TranslatorBase:
    ct2_path = _ct2_path(config.model)
    if ct2_path.exists():
        from bn_en_translate.models.indicTrans2_ct2 import IndicTrans2Ct2Translator
        return IndicTrans2Ct2Translator(config.model)
    from bn_en_translate.models.indicTrans2 import IndicTrans2Translator
    return IndicTrans2Translator(config.model)


@register_model("ollama")
def _make_ollama(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.ollama_translator import OllamaTranslator
    return OllamaTranslator(config)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_translator(config: PipelineConfig) -> TranslatorBase:
    """
    Return the appropriate translator based on PipelineConfig.model.model_name.

    Supported model names:
      - "nllb-600M"       -> NLLBCt2Translator if CT2 model exists, else NLLBTranslator
      - "nllb-1.3B"       -> NLLBCt2Translator if CT2 model exists, else NLLBTranslator
      - "indicTrans2-1B"  -> IndicTrans2Ct2Translator if CT2 exists, else IndicTrans2Translator
      - "ollama"          -> OllamaTranslator (local Ollama LLM)

    Extend by calling @register_model("new-name") on a new factory function.
    """
    name = config.model.model_name.lower()
    factory = _REGISTRY.get(name)
    if factory is None:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model name: '{config.model.model_name}'. "
            f"Supported: {supported}"
        )
    return factory(config)


def _ct2_path(model_config: ModelConfig) -> Path:
    """Return the expected CTranslate2 model path for a given ModelConfig."""
    if model_config.model_path and model_config.model_path != "models/nllb-600M-ct2":
        return Path(model_config.model_path)
    name = model_config.model_name.lower()
    mapping = {
        "nllb-600m": "models/nllb-600M-ct2",
        "nllb-1.3b": "models/nllb-1.3B-ct2",
        "indictrans2-1b": "models/indicTrans2-1B-ct2",
        "indictrans2": "models/indicTrans2-1B-ct2",
    }
    return Path(mapping.get(name, f"models/{model_config.model_name}-ct2"))
