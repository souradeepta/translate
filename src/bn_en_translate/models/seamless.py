"""SeamlessM4T-v2 translator — Meta's multilingual text translation model."""

from __future__ import annotations

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase

# Map FLORES-200 codes to SeamlessM4T short codes
_SEAMLESS_LANG_MAP: dict[str, str] = {
    "ben_Beng": "ben",
    "eng_Latn": "eng",
    "hin_Deva": "hin",
    "urd_Arab": "urd",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
}


def _to_seamless_lang(flores_code: str) -> str:
    """Convert FLORES-200 language code to SeamlessM4T short code."""
    return _SEAMLESS_LANG_MAP.get(flores_code, flores_code.split("_")[0])


class SeamlessTranslator(TranslatorBase):
    """
    Meta SeamlessM4T-v2 text translation model.

    Architecture: Custom encoder-decoder (SeamlessM4Tv2ForTextToText)
    HF ID: facebook/seamless-m4t-v2-large
    VRAM (float16): 3.9 GB measured (RTX 5050)
    FLORES-200 bn→en: BLEU 67.0 / chrF 80.2 @ 32 ch/s (measured)

    Language codes: callers use FLORES-200 format (e.g. 'ben_Beng'); this class
    maps them to SeamlessM4T short codes ('ben') automatically.

    Critical constraints:
      - device_map="auto" is NOT supported; load float16 then .to("cuda") explicitly
      - generate_speech=False must NOT be passed to generate() (text-only model)
      - padding=True, truncation=True required in processor call for batches

    Setup:
        python scripts/download_models.py --model seamless-medium
    """

    HF_MODEL_ID: str = "facebook/seamless-m4t-v2-large"
    _LOCAL_PATH: str = "models/seamless-medium-hf"
    DEFAULT_BEAM_SIZE: int = 5

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="seamless-medium",
            model_path="",  # HF native — no CT2 conversion
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: object | None = None
        self._processor: object | None = None

    def load(self) -> None:
        import torch  # type: ignore[import-untyped]
        from pathlib import Path
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText  # type: ignore[import-untyped]

        from bn_en_translate.utils.cuda_check import get_best_device

        # Prefer local download; fall back to HF Hub
        model_id = self._LOCAL_PATH if Path(self._LOCAL_PATH).exists() else self.HF_MODEL_ID

        device = (
            get_best_device() if self.config.device == "auto" else self.config.device
        )

        # SeamlessM4Tv2ForTextToText does not support device_map="auto" — load in float16
        # then move to GPU explicitly. Text-only portion is ~3.5 GB, fits in 8 GB VRAM.
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = SeamlessM4Tv2ForTextToText.from_pretrained(
            model_id,
            dtype=torch.float16,
        )
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda")  # type: ignore[union-attr]

        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch  # type: ignore[import-untyped]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch  # type: ignore[import-untyped]

        model_device = next(self._model.parameters()).device  # type: ignore[union-attr]

        seamless_src = _to_seamless_lang(src_lang)
        seamless_tgt = _to_seamless_lang(tgt_lang)

        inputs = self._processor(  # type: ignore[operator]
            text=texts,
            src_lang=seamless_src,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model_device)

        with torch.no_grad():
            output_tokens = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                tgt_lang=seamless_tgt,
                num_beams=self._effective_beam_size(),
                max_new_tokens=self.config.max_decoding_length,
            )

        return self._processor.batch_decode(  # type: ignore[union-attr]
            output_tokens, skip_special_tokens=True
        )
