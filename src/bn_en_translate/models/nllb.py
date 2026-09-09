"""NLLB-200 translator using HuggingFace Transformers (CPU/GPU)."""

from __future__ import annotations

from typing import Any

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.models.hf_utils import free_cuda_memory, resolve_device


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
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from bn_en_translate.utils.cuda_check import require_cuda

        model_id = self._resolve_model_id()
        device = resolve_device(self.config.device)

        # Use the explicit seq2seq API rather than pipeline("translation"). The
        # generic translation pipeline was removed/changed across Transformers
        # releases and hid language-token and output-length configuration.
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        if device == "cuda":
            require_cuda(type(self).__name__)
            self._model = self._model.to("cuda")
        self._model.eval()
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        free_cuda_memory()

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch

        assert self._model is not None
        assert self._tokenizer is not None
        model_device = next(self._model.parameters()).device

        # NLLB requires the source language on the tokenizer and the target
        # language as the forced first decoder token for every request.
        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model_device)
        target_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                num_beams=self._effective_beam_size(),
                max_new_tokens=self.config.max_decoding_length,
            )

        return list(self._tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ))

    def _resolve_model_id(self) -> str:
        """Map short model names to HuggingFace model IDs."""
        aliases = {
            "nllb-600m": "facebook/nllb-200-distilled-600M",
            "nllb-1.3b": "facebook/nllb-200-distilled-1.3B",
        }
        return aliases.get(self.config.model_name.lower(), self.config.model_name)
