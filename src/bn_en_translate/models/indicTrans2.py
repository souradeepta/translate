"""IndicTrans2 translator — best quality for Bengali → English."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bn_en_translate.config import REPO_ROOT, ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.models.hf_utils import (
    free_cuda_memory,
    load_indictrans_tokenizer,
    resolve_attn_implementation,
    stub_transformers_onnx,
)

LOGGER = logging.getLogger(__name__)


class IndicTrans2Translator(TranslatorBase):
    """
    AI4Bharat IndicTrans2 translation model.

    Model: ai4bharat/indictrans2-indic-en-1B
    This is the recommended primary model for Bengali → English translation.

    Language codes: ben_Beng → eng_Latn (same FLORES-200 format as NLLB)

    Setup (one-time):
        pip install IndicTransToolkit
        # Then download the model:
        python scripts/download_models.py --model indicTrans2-1B

    Backend: HF-native (AutoModelForSeq2SeqLM), not CTranslate2 — CT2's
    converter registry has no entry for IndicTransConfig, so conversion is
    architecturally unsupported.

    VRAM usage (float16, HF-native): ~3 GB
    """

    HF_MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"
    _LOCAL_PATH: str = str(REPO_ROOT / "models/indicTrans2-1B-hf")
    DEFAULT_BEAM_SIZE: int = 5

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="indicTrans2-1B",
            model_path="models/indicTrans2-1B-ct2",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """
        Load IndicTrans2 via the IndicTrans2 HuggingFace interface.
        Falls back to standard HuggingFace transformers if the custom
        interface is not installed.
        """
        try:
            self._load_via_indictrans2_interface()
        except ImportError as exc:
            LOGGER.warning(
                "IndicTransToolkit unavailable; using the lower-level IndicTrans2 "
                "fallback tokenizer (%s). Install IndicTransToolkit for the supported "
                "pre/post-processing path.",
                exc,
            )
            self._load_via_transformers_fallback()
        self._loaded = True

    def _load_via_indictrans2_interface(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM

        stub_transformers_onnx()

        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            # IndicTransToolkit (PyPI) imports PreTrainedTokenizerBase from the
            # pre-5.x location; transformers 5.4 moved it to tokenization_utils_base.
            # Patch the old path back before retrying rather than losing the
            # IndicProcessor preprocessing (script/numeral normalization) to
            # the generic transformers_fallback path.
            import transformers.tokenization_utils as _ttu
            from transformers.tokenization_utils_base import (
                PreTrainedTokenizerBase as _PTB,
            )

            _ttu.PreTrainedTokenizerBase = _PTB
            from IndicTransToolkit import IndicProcessor

        from bn_en_translate.utils.cuda_check import require_cuda

        model_id = self._LOCAL_PATH if Path(self._LOCAL_PATH).exists() else self.HF_MODEL_ID
        self._tokenizer = load_indictrans_tokenizer(model_id)
        # IndicTrans2's sdpa support is unverified — keep eager as the fallback
        # (behavior-preserving; do not switch to sdpa without dedicated testing).
        attn_impl = resolve_attn_implementation(self.config.use_flash_attention, fallback="eager")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            dtype=torch.float16,
        )
        self._processor = IndicProcessor(inference=True)

        if self.config.device == "cuda":
            require_cuda(type(self).__name__)
            self._model.to("cuda")

    def _load_via_transformers_fallback(self) -> None:
        """Fallback: load as a standard seq2seq model (lower quality tokenization)."""
        import torch
        from transformers import AutoModelForSeq2SeqLM

        from bn_en_translate.utils.cuda_check import require_cuda

        stub_transformers_onnx()

        model_id = self._LOCAL_PATH if Path(self._LOCAL_PATH).exists() else self.HF_MODEL_ID
        self._tokenizer = load_indictrans_tokenizer(model_id)
        attn_impl = resolve_attn_implementation(self.config.use_flash_attention, fallback="eager")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            dtype=torch.float16,
        )

        if self.config.device == "cuda":
            require_cuda(type(self).__name__)
            self._model.to("cuda")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        free_cuda_memory()

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch

        model_device = next(self._model.parameters()).device

        if self._processor is not None:
            # Use IndicTransToolkit preprocessing
            batch = self._processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = self._tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(model_device)
        else:
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model_device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                num_beams=self._effective_beam_size(),
                max_new_tokens=self.config.max_decoding_length,
            )

        decoded = self._tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if self._processor is not None:
            decoded = self._processor.postprocess_batch(decoded, lang=tgt_lang)

        return list(decoded)
