"""MADLAD-400-3B translator — Google's dedicated multilingual MT model.

WARNING: The local checkpoint at models/madlad-3b-hf/ has a known weight mismatch
(shared.weight != decoder.embed_tokens.weight) that produces degenerate output (BLEU 0).
Additionally, 3B float16 parameters require ~6 GB weights + KV cache, exceeding the 8 GB
VRAM budget and forcing CPU offload (~30 s/sentence). This model is EXCLUDED from
benchmarks. Re-download cleanly from HF Hub before using.
"""

from __future__ import annotations
import importlib.util
import warnings

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


# Map FLORES-200 language codes to MADLAD-400 target tags
_MADLAD_LANG_TAG: dict[str, str] = {
    "eng_Latn": "<2en>",
    "ben_Beng": "<2bn>",
    "hin_Deva": "<2hi>",
}


class MADLADTranslator(TranslatorBase):
    """
    Google MADLAD-400-3B translation model via HuggingFace Transformers.

    Architecture: T5-based encoder-decoder
    HF ID: google/madlad400-3b-mt
    VRAM (float16): 8.1 GB measured — exceeds 8 GB budget, triggers CPU offload
    FLORES-200 bn→en: EXCLUDED (checkpoint weight mismatch → degenerate output)

    Source text is prefixed with the target language tag, e.g. '<2en> <bengali text>'.
    No source language tag is required. Always use max_new_tokens in generate(), not
    max_length — T5's max_length caps input+output tokens combined.

    Setup:
        python scripts/download_models.py --model madlad-3b
    """

    HF_MODEL_ID = "google/madlad400-3b-mt"
    DEFAULT_BEAM_SIZE: int = 4

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="madlad-3b",
            model_path="models/madlad-3b-hf",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: object | None = None
        self._tokenizer: object | None = None

    _LOCAL_PATH = "models/madlad-3b-hf"

    def load(self) -> None:
        import torch  # type: ignore[import-untyped]
        from pathlib import Path
        from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore[import-untyped]

        from bn_en_translate.utils.cuda_check import get_best_device

        warnings.warn(
            "MADLADTranslator: the local checkpoint at models/madlad-3b-hf/ has a known "
            "shared.weight != decoder.embed_tokens.weight mismatch that produces garbage output. "
            "3B float16 also exceeds the 8 GB VRAM budget, forcing slow CPU offload. "
            "Re-download cleanly from HF Hub before benchmarking.",
            UserWarning,
            stacklevel=2,
        )

        # Prefer local download; fall back to HF Hub (auto-downloads on first use)
        model_id = self._LOCAL_PATH if Path(self._LOCAL_PATH).exists() else self.HF_MODEL_ID

        attn_impl = (
            "flash_attention_2"
            if self.config.use_flash_attention and _flash_attn_available()
            else "eager"
        )

        device = (
            get_best_device() if self.config.device == "auto" else self.config.device
        )
        # device_map="auto" is necessary because 3B float16 weights (~6 GB) + KV cache
        # exceed 8 GB VRAM. This forces CPU offload and reduces throughput to ~2 ch/s.
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else None

        self._tokenizer = T5Tokenizer.from_pretrained(model_id)
        self._model = T5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=torch.float16,
            device_map=device_map,
        )

        self._loaded = True

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

    def _build_input_texts(self, texts: list[str], tgt_lang: str) -> list[str]:
        """Prefix each text with the MADLAD-400 target language tag."""
        tag = _MADLAD_LANG_TAG.get(tgt_lang, "<2en>")
        return [f"{tag} {t}" for t in texts]

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch  # type: ignore[import-untyped]

        # When using device_map="auto", the model manages its own device placement.
        # Move inputs to the same device as the model's first parameter.
        model_device = next(self._model.parameters()).device  # type: ignore[union-attr]

        input_texts = self._build_input_texts(texts, tgt_lang)
        inputs = self._tokenizer(  # type: ignore[operator]
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model_device)

        with torch.no_grad():
            generated = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                num_beams=self._effective_beam_size(),
                max_new_tokens=256,
            )

        return self._tokenizer.batch_decode(  # type: ignore[union-attr]
            generated, skip_special_tokens=True
        )
