"""IndicTrans2 translator — best quality for Bengali → English."""

from __future__ import annotations

import importlib.util

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


class IndicTrans2Translator(TranslatorBase):
    """
    AI4Bharat IndicTrans2 translation model.

    Model: ai4bharat/indictrans2-indic-en-1B
    This is the recommended primary model for Bengali → English translation.

    Language codes: ben_Beng → eng_Latn (same FLORES-200 format as NLLB)

    Setup (one-time):
        pip install git+https://github.com/AI4Bharat/IndicTrans2.git#subdirectory=huggingface_interface
        # Then download the model:
        python scripts/download_models.py --model indicTrans2-1B

    VRAM usage (INT8 via CTranslate2): ~1.0–1.5 GB
    """

    HF_MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"
    DEFAULT_BEAM_SIZE: int = 5

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="indicTrans2-1B",
            model_path="models/indicTrans2-1B-ct2",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: object | None = None
        self._tokenizer: object | None = None

    def load(self) -> None:
        """
        Load IndicTrans2 via the IndicTrans2 HuggingFace interface.
        Falls back to standard HuggingFace transformers if the custom
        interface is not installed.
        """
        try:
            self._load_via_indictrans2_interface()
        except ImportError:
            self._load_via_transformers_fallback()
        self._loaded = True

    def _load_via_indictrans2_interface(self) -> None:
        from IndicTransToolkit import IndicProcessor  # type: ignore[import-untyped]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-untyped]

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True
        )
        attn_impl = (
            "flash_attention_2"
            if self.config.use_flash_attention and _flash_attn_available()
            else "eager"
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True, attn_implementation=attn_impl
        )
        self._processor = IndicProcessor(inference=True)

        import torch  # type: ignore[import-untyped]

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model.to("cuda")  # type: ignore[union-attr]

    def _load_via_transformers_fallback(self) -> None:
        """Fallback: load as a standard seq2seq model (lower quality tokenization)."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-untyped]

        self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
        attn_impl = (
            "flash_attention_2"
            if self.config.use_flash_attention and _flash_attn_available()
            else "eager"
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.HF_MODEL_ID, attn_implementation=attn_impl
        )
        self._processor = None

        import torch  # type: ignore[import-untyped]

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model.to("cuda")  # type: ignore[union-attr]

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch  # type: ignore[import-untyped]

        device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"

        if self._processor is not None:
            # Use IndicTransToolkit preprocessing
            batch = self._processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = self._tokenizer(  # type: ignore[operator]
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
        else:
            inputs = self._tokenizer(  # type: ignore[operator]
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

        with torch.no_grad():
            generated = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                num_beams=self._effective_beam_size(),
                num_return_sequences=1,
                max_length=self.config.max_decoding_length,
            )

        decoded = self._tokenizer.batch_decode(  # type: ignore[union-attr]
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if self._processor is not None:
            decoded = self._processor.postprocess_batch(decoded, lang=tgt_lang)

        return decoded
