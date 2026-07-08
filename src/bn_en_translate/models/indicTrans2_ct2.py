"""IndicTrans2-1B via CTranslate2 float16 — optimal quality + GPU efficiency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.utils.ct2_utils import probe_compute_type


class IndicTrans2Ct2Translator(TranslatorBase):
    """
    AI4Bharat IndicTrans2-1B via CTranslate2 float16.

    This is the recommended path for production use:
      - Compute type probed at load time: float16 on Blackwell sm_120 (INT8 fails)
      - CTranslate2 CUDA kernels: faster than HF generate() on Blackwell
      - IndicTransToolkit: proper Bengali script normalization + SentencePiece

    Setup:
        pip install git+https://github.com/AI4Bharat/IndicTrans2.git#subdirectory=huggingface_interface
        python scripts/download_models.py --model indicTrans2-1B

    If IndicTransToolkit is not installed, falls back to raw SPM tokenization
    (lower quality but still functional).
    """

    HF_MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"
    SPM_FILENAME = "sentencepiece.bpe.model"
    DEFAULT_BEAM_SIZE: int = 5

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="indicTrans2-1B",
            model_path="models/indicTrans2-1B-ct2",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._translator: Any = None
        self._sp: Any = None
        self._processor: Any = None

    def load(self) -> None:
        import ctranslate2
        import sentencepiece as spm

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"CTranslate2 model not found at '{model_path}'. "
                "Run: python scripts/download_models.py --model indicTrans2-1B"
            )

        device = self.config.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

        # Load SPM first (needed by compute type probe)
        sp_path = model_path / self.SPM_FILENAME
        if not sp_path.exists():
            alt = model_path / "vocab.json"
            if not alt.exists():
                raise FileNotFoundError(f"SentencePiece model not found at {sp_path}")
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(sp_path))

        probe_src = (
            self._sp.encode(
                "Rabindranath Tagore is an unforgettable poet of Bengali literature.",
                out_type=str,
            )
            + ["</s>", "ben_Beng"]
        )
        compute_type = probe_compute_type(
            str(model_path),
            device,
            lambda t: t.translate_batch(
                [probe_src], target_prefix=[["eng_Latn"]], beam_size=1, max_decoding_length=20
            ),
        )

        self._translator = ctranslate2.Translator(
            str(model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=4,
        )

        # IndicTransToolkit handles script normalization + language tagging
        try:
            from IndicTransToolkit import IndicProcessor
            self._processor = IndicProcessor(inference=True)
        except ImportError:
            self._processor = None

        self._loaded = True

    def unload(self) -> None:
        self._translator = None
        self._sp = None
        self._processor = None
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self._translator is not None
        assert self._sp is not None

        if self._processor is not None:
            # IndicTransToolkit preprocessing: script normalization + sentence splitting
            preprocessed = self._processor.preprocess_batch(
                texts, src_lang=src_lang, tgt_lang=tgt_lang
            )
        else:
            preprocessed = texts

        # IndicTrans2 source format: [text_tokens..., </s>, src_lang]
        # (same NLLB-style format since IndicTrans2 also uses M2M-100 architecture)
        tokenized = [
            self._sp.encode(t, out_type=str) + ["</s>", src_lang]
            for t in preprocessed
        ]
        target_prefix = [[tgt_lang]] * len(tokenized)

        results = self._translator.translate_batch(
            tokenized,
            target_prefix=target_prefix,
            beam_size=self._effective_beam_size(),
            max_decoding_length=self.config.max_decoding_length,
            max_batch_size=self.config.max_ct2_batch_size,
        )

        decoded: list[str] = []
        for result in results:
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == tgt_lang:
                tokens = tokens[1:]
            decoded.append(self._sp.decode(tokens))

        if self._processor is not None:
            decoded = self._processor.postprocess_batch(decoded, lang=tgt_lang)

        return decoded
