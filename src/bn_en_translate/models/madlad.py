"""MADLAD-400-3B translator — Google's dedicated multilingual MT model.

WARNING: The local checkpoint at models/madlad-3b-hf/ has a known weight mismatch
(shared.weight != decoder.embed_tokens.weight) that produces degenerate output (BLEU 0).
Additionally, 3B float16 parameters require ~6 GB weights + KV cache, exceeding the 8 GB
VRAM budget and forcing CPU offload (~30 s/sentence). This model is EXCLUDED from
benchmarks. Re-download cleanly from HF Hub before using.
"""

from __future__ import annotations

import warnings

from bn_en_translate.config import REPO_ROOT, ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.models.hf_utils import (
    flash_attn_available as _flash_attn_available,  # noqa: F401
)
from bn_en_translate.models.hf_utils import free_cuda_memory, resolve_device


def _resolve_attn_implementation(use_flash: bool, fallback: str = "sdpa") -> str:
    """flash_attention_2 if installed and requested; else the given fallback.

    Thin wrapper around hf_utils.resolve_attn_implementation that calls the
    module-level `_flash_attn_available` name (not hf_utils directly) so tests
    can monkeypatch it by module attribute.
    """
    if use_flash and _flash_attn_available():
        return "flash_attention_2"
    return fallback


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

    _LOCAL_PATH: str = str(REPO_ROOT / "models/madlad-3b-hf")

    def load(self) -> None:
        from pathlib import Path

        import torch  # type: ignore[import-untyped]
        from transformers import (  # type: ignore[import-untyped]
            T5ForConditionalGeneration,
            T5Tokenizer,
        )

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

        attn_impl = _resolve_attn_implementation(self.config.use_flash_attention, fallback="eager")

        device = resolve_device(self.config.device)
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

        self._verify_tied_embeddings(self._model)

        self._loaded = True

    @staticmethod
    def _verify_tied_embeddings(model: object) -> None:
        """Detect the known corrupt-checkpoint failure mode at load time.

        A healthy T5 MT checkpoint has shared.weight tied to
        decoder.embed_tokens.weight. The local madlad-3b-hf checkpoint was
        observed with untied (randomised) weights, which produces degenerate
        output (BLEU 0) with no error. Compare a slice — the embedding matrix
        is ~512M params, too large to compare in full, and the documented
        corruption randomises the whole matrix. Corruption confined to rows
        >= 64 would pass this check; that mode has never been observed.
        """
        import torch  # type: ignore[import-untyped]

        shared = model.shared.weight  # type: ignore[attr-defined]
        decoder = model.decoder.embed_tokens.weight  # type: ignore[attr-defined]
        if getattr(shared, "is_meta", False) or getattr(decoder, "is_meta", False):
            raise RuntimeError(
                "MADLAD embeddings are disk-offloaded (meta tensors); cannot verify "
                "checkpoint integrity — insufficient RAM for device_map='auto'."
            )
        if not torch.equal(shared[:64].float().cpu(), decoder[:64].float().cpu()):
            raise RuntimeError(
                "MADLAD checkpoint tied-embedding mismatch: shared.weight != "
                "decoder.embed_tokens.weight. This checkpoint produces garbage "
                "output. Re-download cleanly: rm -rf models/madlad-3b-hf && "
                "python scripts/download_models.py --model madlad-3b"
            )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        free_cuda_memory()

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
