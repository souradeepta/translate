"""NLLB-200 translator using HuggingFace Transformers (CPU/GPU)."""

from __future__ import annotations

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


class NLLBTranslator(TranslatorBase):
    """
    Facebook NLLB-200 translation model via HuggingFace Transformers.

    Supports:
      - facebook/nllb-200-distilled-600M  (fits easily in 8 GB VRAM)
      - facebook/nllb-200-distilled-1.3B  (better quality, ~2.6 GB fp16)

    Language codes use the FLORES-200 format: ben_Beng, eng_Latn, etc.
    """

    DEFAULT_BEAM_SIZE: int = 4

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(model_name="nllb-600M")
        self._pipeline: object | None = None

    def load(self) -> None:
        from transformers import pipeline  # type: ignore[import-untyped]

        model_id = self._resolve_model_id()
        device = 0 if self.config.device == "cuda" else -1  # HF pipeline convention

        # Do not pass max_length here: for encoder-decoder models it caps total tokens
        # (input + output), not just new tokens. Pass max_new_tokens per-call instead.
        self._pipeline = pipeline(
            "translation",
            model=model_id,
            device=device,
            src_lang=self.config.src_lang,
            tgt_lang=self.config.tgt_lang,
            num_beams=self._effective_beam_size(),
        )
        self._loaded = True

    def unload(self) -> None:
        self._pipeline = None
        self._loaded = False
        # Release GPU memory if torch is available
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self._pipeline is not None
        results = self._pipeline(  # type: ignore[operator]
            texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=self.config.inference_batch_size,
            max_new_tokens=self.config.max_decoding_length,
        )
        return [r["translation_text"] for r in results]  # type: ignore[index]

    def _resolve_model_id(self) -> str:
        """Map short model names to HuggingFace model IDs."""
        aliases = {
            "nllb-600M": "facebook/nllb-200-distilled-600M",
            "nllb-1.3B": "facebook/nllb-200-distilled-1.3B",
        }
        return aliases.get(self.config.model_name, self.config.model_name)
