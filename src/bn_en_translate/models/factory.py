"""Factory for creating translator instances by name."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bn_en_translate.config import CT2_MODEL_PATHS, ModelConfig, PipelineConfig
from bn_en_translate.models.base import TranslatorBase

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry maps a lower-case model name to a callable that accepts a
# PipelineConfig and returns a TranslatorBase.  Adding a new model is a
# one-line registration — no if/elif chain to touch.

_Factory = Callable[[PipelineConfig], TranslatorBase]

_REGISTRY: dict[str, _Factory] = {}


def register_model(name: str) -> Callable[[_Factory], _Factory]:
    """Decorator that registers a factory function under the given model name."""
    def _decorator(fn: _Factory) -> _Factory:
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


@register_model("hunyuan-mt-7b")
@register_model("hunyuan")
def _make_hunyuan(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.ollama_translator import (
        HUNYUAN_MT_PROMPT,
        OllamaTranslator,
    )
    return OllamaTranslator(
        config,
        prompt_template=HUNYUAN_MT_PROMPT,
        model_tag="demonbyron/HY-MT1.5-7B:Q4_K_M",
    )


@register_model("madlad-3b")
@register_model("madlad")
def _make_madlad(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.madlad import MADLADTranslator
    return MADLADTranslator(config.model)


@register_model("seamless-medium")
@register_model("seamless")
def _make_seamless(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.seamless import SeamlessTranslator
    return SeamlessTranslator(config.model)


@register_model("milmmt-46-1b")
@register_model("milmmt")
def _make_milmmt(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.milmmt import MiLMMTTranslator
    return MiLMMTTranslator(config.model)


@register_model("milmmt-46-4b")
def _make_milmmt_4b(config: PipelineConfig) -> TranslatorBase:
    from dataclasses import replace

    from bn_en_translate.models.milmmt import MiLMMT4BTranslator

    # Same two ModelConfig-shared-default pitfalls as sarvam-translate:
    # model_path defaults to nllb-600M's CT2 dir, and load_in_4bit defaults
    # to False. 4B params in bf16 (~8 GB) doesn't fit this 8 GB card, so
    # 4-bit is required, not optional, for this specific translator.
    model_config = replace(
        config.model, model_path="", load_in_4bit=True
    )
    return MiLMMT4BTranslator(model_config)


@register_model("lmt-60-1.7b")
@register_model("lmt-60")
def _make_lmt60(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.lmt60 import LMT60Translator
    return LMT60Translator(config.model)


@register_model("sarvam-translate")
@register_model("sarvam")
def _make_sarvam_translate(config: PipelineConfig) -> TranslatorBase:
    from dataclasses import replace

    from bn_en_translate.models.sarvam_translate import SarvamTranslateTranslator

    # 4B params in bf16 (~8 GB) is too tight on this 8 GB card; 4-bit bnb
    # quantization (measured ~1.4 GB VRAM) is the only viable path here, not
    # an optional preference — force it regardless of ModelConfig's shared
    # dataclass default (False), mirroring how other translators hard-code
    # their own known-good settings (e.g. MiLMMT's bf16 dtype).
    model_config = replace(config.model, load_in_4bit=True)
    return SarvamTranslateTranslator(model_config)


@register_model("krutrim-translate")
@register_model("krutrim")
def _make_krutrim_translate(config: PipelineConfig) -> TranslatorBase:
    from dataclasses import replace

    from bn_en_translate.models.krutrim_translate import KrutrimTranslateTranslator

    # ModelConfig.model_path's shared dataclass default points at nllb-600M's
    # CT2 dir (the system's global default model) — callers that build a
    # plain ModelConfig(model_name="krutrim-translate", ...) without setting
    # model_path (CLI, benchmark.py) would otherwise silently try to load
    # this translator from the wrong directory.
    model_config = replace(config.model, model_path=KrutrimTranslateTranslator._LOCAL_PATH)
    return KrutrimTranslateTranslator(model_config)


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
      - "madlad-3b"       -> MADLADTranslator (Google MADLAD-400-3B)
      - "seamless-medium" -> SeamlessTranslator (Meta SeamlessM4T-v2)
      - "milmmt-46-1b"    -> MiLMMTTranslator (Xiaomi MiLMMT-46-1B, Gemma3-based)

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
    """Return the CTranslate2 model directory for a given ModelConfig.

    Known models resolve to their canonical absolute path from CT2_MODEL_PATHS.
    Unknown models fall back to model_config.model_path.
    """
    name = model_config.model_name.lower()
    if name in CT2_MODEL_PATHS:
        return Path(CT2_MODEL_PATHS[name])
    return Path(model_config.model_path)
