"""Krutrim-Translate — Ola Krutrim's distilled IndicTrans2, CT2-native.

krutrim-ai-labs/Krutrim-Translate extends IndicTrans2 with a longer context
(256 -> 4096 tokens) and distills it to 6 encoder / 3 decoder layers for ~4x
lower latency at "minimal decline" in quality per the model card. It ships
pre-converted as two directional CTranslate2 exports:
    ct_model_english_indic/   (English -> Indic)
    ct_model_indic_english/   (Indic -> English)  <- the one we need

Unlike this repo's own indicTrans2_ct2.py (which uses ONE shared SPM model
for both source and target, M2M-100 style, with manual `</s>+src_lang`
token appending and a `target_prefix`), Krutrim's distilled export uses a
DIFFERENT, simpler convention — verified empirically 2026-07-09 (raw SPM
input produced 100% `<unk>` garbage until this was worked out):
  - Separate SRC/TGT SentencePiece models (`vocab/model.SRC`, `vocab/model.TGT`)
  - `config.json` sets `add_source_eos: true` / `decoder_start_token: "</s>"` —
    CT2 handles EOS/BOS itself; no manual token appending, no `target_prefix`
  - Language tags are NOT SPM tokens at all. `IndicProcessor.preprocess_batch()`
    prepends them as plain text ("ben_Beng eng_Latn <script-normalized text>")
    before SPM tokenization — this is not optional. Skipping it (raw SPM only)
    produces near-100% `<unk>` tokens; IndicProcessor's script normalization
    step is load-bearing, not a quality nicety.

Setup (gated repo — must accept license first):
    accept terms at huggingface.co/krutrim-ai-labs/Krutrim-Translate
    python scripts/download_models.py --model krutrim-translate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bn_en_translate.config import REPO_ROOT, ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.utils.ct2_utils import probe_compute_type


def _import_indic_processor() -> Any:
    """Import IndicProcessor, patching IndicTransToolkit's transformers-5.x
    incompatibility (same fix as indicTrans2.py — the PyPI package imports
    PreTrainedTokenizerBase from the pre-5.x location)."""
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        import transformers.tokenization_utils as _ttu
        from transformers.tokenization_utils_base import (
            PreTrainedTokenizerBase as _PTB,
        )

        _ttu.PreTrainedTokenizerBase = _PTB
        from IndicTransToolkit import IndicProcessor
    return IndicProcessor


class KrutrimTranslateTranslator(TranslatorBase):
    """
    Ola Krutrim's Krutrim-Translate (Indic -> English direction), CT2-native.

    Architecture: distilled IndicTrans2, 6 encoder / 3 decoder layers, 4096 ctx
    HF ID: krutrim-ai-labs/Krutrim-Translate (gated — requires license acceptance)

    Setup:
        python scripts/download_models.py --model krutrim-translate
    """

    HF_MODEL_ID = "krutrim-ai-labs/Krutrim-Translate"
    _LOCAL_PATH: str = str(
        REPO_ROOT / "models/krutrim-translate-hf/ct_model_indic_english"
    )
    DEFAULT_BEAM_SIZE: int = 3  # per the model's own example.ipynb (beam_len=3)

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="krutrim-translate",
            model_path=self._LOCAL_PATH,
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._translator: Any = None
        self._sp_src: Any = None
        self._sp_tgt: Any = None
        self._processor: Any = None

    def load(self) -> None:
        import ctranslate2
        import sentencepiece as spm

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"CTranslate2 model not found at '{model_path}'. "
                "Run: python scripts/download_models.py --model krutrim-translate"
            )

        device = self.config.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

        self._sp_src = spm.SentencePieceProcessor()
        self._sp_src.load(str(model_path / "vocab" / "model.SRC"))
        self._sp_tgt = spm.SentencePieceProcessor()
        self._sp_tgt.load(str(model_path / "vocab" / "model.TGT"))

        IndicProcessor = _import_indic_processor()
        self._processor = IndicProcessor(inference=True)

        probe_text = self._processor.preprocess_batch(
            ["Rabindranath Tagore is an unforgettable poet of Bengali literature."],
            src_lang="eng_Latn",
            tgt_lang="ben_Beng",
        )[0]
        probe_src = self._sp_src.encode(probe_text, out_type=str)
        compute_type = probe_compute_type(
            str(model_path),
            device,
            lambda t: t.translate_batch([probe_src], beam_size=1, max_decoding_length=20),
        )

        self._translator = ctranslate2.Translator(
            str(model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=4,
        )

        self._loaded = True

    def unload(self) -> None:
        self._translator = None
        self._sp_src = None
        self._sp_tgt = None
        self._processor = None
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self._translator is not None
        assert self._sp_src is not None
        assert self._sp_tgt is not None
        assert self._processor is not None

        # IndicProcessor prepends "src_lang tgt_lang " as plain text and
        # applies script normalization — required, not optional (see module
        # docstring: raw SPM without this produces near-100% <unk> tokens).
        preprocessed = self._processor.preprocess_batch(
            texts, src_lang=src_lang, tgt_lang=tgt_lang
        )

        tokenized = [self._sp_src.encode(t, out_type=str) for t in preprocessed]

        results = self._translator.translate_batch(
            tokenized,
            beam_size=self._effective_beam_size(),
            max_decoding_length=self.config.max_decoding_length,
            max_batch_size=self.config.max_ct2_batch_size,
        )

        decoded = [
            self._sp_tgt.decode(result.hypotheses[0]) for result in results
        ]
        return list(self._processor.postprocess_batch(decoded, lang=tgt_lang))
