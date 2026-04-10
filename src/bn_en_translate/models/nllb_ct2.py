"""NLLB-200 translator using CTranslate2 float16 — GPU-optimized."""

from __future__ import annotations

from pathlib import Path

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.utils.ct2_utils import probe_compute_type


class NLLBCt2Translator(TranslatorBase):
    """
    NLLB-200 via CTranslate2 — fastest GPU inference path.

    Uses the pre-converted model at config.model_path (e.g. models/nllb-600M-ct2).
    Requires the model to be downloaded+converted first:
        python scripts/download_models.py --model nllb-600M

    Compute type is selected automatically at load time via a real-translation probe.
    INT8 fails on Blackwell sm_120 + CUDA 12.x (CUBLAS_STATUS_NOT_SUPPORTED), so float16
    is selected on this hardware. Falls back to int8 on CPU.

    FLORES-200 bn→en: BLEU 55.3 / chrF 72.8 @ 197 ch/s, 2.0 GB VRAM (RTX 5050 float16)
    """

    DEFAULT_BEAM_SIZE: int = 4

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(model_name="nllb-600M")
        self._translator: object | None = None
        self._sp: object | None = None

    def load(self) -> None:
        import ctranslate2  # type: ignore[import-untyped]
        import sentencepiece as spm  # type: ignore[import-untyped]

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"CTranslate2 model not found at '{model_path}'. "
                "Run: python scripts/download_models.py --model nllb-600M"
            )

        device = self.config.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

        sp_path = model_path / "sentencepiece.bpe.model"
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(sp_path))  # type: ignore[union-attr]

        # Probe to find the best working compute type.
        # INT8 fails on Blackwell sm_120 + CUDA 12.x; probe catches it at load time.
        probe_src = (
            self._sp.encode(  # type: ignore[union-attr]
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
        self._loaded = True

    def unload(self) -> None:
        self._translator = None
        self._sp = None
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self._translator is not None
        assert self._sp is not None

        # NLLB source format: [tokens..., </s>, src_lang]; target prefix: [tgt_lang]
        tokenized = [
            self._sp.encode(t, out_type=str) + ["</s>", src_lang]  # type: ignore[union-attr]
            for t in texts
        ]
        target_prefix = [[tgt_lang]] * len(tokenized)

        results = self._translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self._effective_beam_size(),
            max_decoding_length=self.config.max_decoding_length,
            max_batch_size=self.config.max_ct2_batch_size,
        )

        output_texts: list[str] = []
        for result in results:
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == tgt_lang:
                tokens = tokens[1:]
            output_texts.append(self._sp.decode(tokens))  # type: ignore[union-attr]
        return output_texts
