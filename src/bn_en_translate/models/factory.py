"""Factory for creating translator instances by name."""

from __future__ import annotations

from pathlib import Path

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.base import TranslatorBase


def get_translator(config: PipelineConfig) -> TranslatorBase:
    """
    Return the appropriate translator based on PipelineConfig.model.model_name.

    Supported model names:
      - "nllb-600M"       -> NLLBCt2Translator if CT2 model exists, else NLLBTranslator
      - "nllb-1.3B"       -> NLLBCt2Translator if CT2 model exists, else NLLBTranslator
      - "indicTrans2-1B"  -> IndicTrans2Ct2Translator if CT2 model exists, else IndicTrans2Translator
      - "ollama"          -> OllamaTranslator (local Ollama LLM)

    CTranslate2 INT8 models are preferred when available (faster, less VRAM).
    """
    name = config.model.model_name.lower()

    if name in ("nllb-600m", "nllb-1.3b"):
        ct2_path = _ct2_path(config.model)
        if ct2_path.exists():
            from bn_en_translate.models.nllb_ct2 import NLLBCt2Translator
            return NLLBCt2Translator(config.model)
        from bn_en_translate.models.nllb import NLLBTranslator
        return NLLBTranslator(config.model)

    if name in ("indictrans2-1b", "indictrans2"):
        ct2_path = _ct2_path(config.model)
        if ct2_path.exists():
            from bn_en_translate.models.indicTrans2_ct2 import IndicTrans2Ct2Translator
            return IndicTrans2Ct2Translator(config.model)
        from bn_en_translate.models.indicTrans2 import IndicTrans2Translator
        return IndicTrans2Translator(config.model)

    if name == "ollama":
        from bn_en_translate.models.ollama_translator import OllamaTranslator
        return OllamaTranslator(config)

    raise ValueError(
        f"Unknown model name: '{config.model.model_name}'. "
        "Supported: nllb-600M, nllb-1.3B, indicTrans2-1B, ollama"
    )


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
