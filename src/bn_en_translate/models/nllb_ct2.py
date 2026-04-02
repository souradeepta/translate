"""NLLB-200 translator using CTranslate2 INT8 — GPU-optimized."""

from __future__ import annotations

from pathlib import Path

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


class NLLBCt2Translator(TranslatorBase):
    """
    NLLB-200 via CTranslate2 INT8 — fastest GPU inference path.

    Uses the pre-converted model at config.model_path (e.g. models/nllb-600M-ct2).
    Requires the model to be downloaded+converted first via:
        python scripts/download_models.py --model nllb-600M

    Supports int8, float16, bfloat16 compute types on CUDA.
    Falls back to int8 on CPU if CUDA is unavailable.
    """

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

        # Load tokenizer first (needed by compute type probe)
        sp_path = model_path / "sentencepiece.bpe.model"
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(sp_path))  # type: ignore[union-attr]

        # Probe to find the best working compute type (INT8 fails on Blackwell+cu124)
        compute_type = self._best_compute_type(device, self._sp)

        self._translator = ctranslate2.Translator(
            str(model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=4,
        )

        self._loaded = True

    def _best_compute_type(self, device: str, sp: object) -> str:
        """Find the best working compute type by running a realistic translation probe.

        INT8 ops fail on Blackwell (sm_120) with CTranslate2 4.x + CUDA 12.4 cuBLAS
        when given sequences long enough to trigger INT8 tensor core matmuls.
        The probe uses a ~20-token sentence to catch this at load time.
        """
        if device == "cpu":
            return "int8"
        import ctranslate2  # type: ignore[import-untyped]
        import sentencepiece as spm  # type: ignore[import-untyped]
        assert isinstance(sp, spm.SentencePieceProcessor)

        # Use a realistic-length sentence (short probes may not trigger INT8 ops)
        probe_text = "Rabindranath Tagore is an unforgettable poet of Bengali literature."
        probe_src = sp.encode(probe_text, out_type=str) + ["</s>", "ben_Beng"]
        supported = ctranslate2.get_supported_compute_types(device)

        for ct in ("int8_float16", "int8", "float16", "bfloat16", "float32"):
            if ct not in supported:
                continue
            try:
                probe = ctranslate2.Translator(
                    str(Path(self.config.model_path)), device=device, compute_type=ct
                )
                probe.translate_batch(
                    [probe_src], target_prefix=[["eng_Latn"]], beam_size=1, max_decoding_length=20
                )
                del probe
                return ct
            except Exception:
                continue
        return "float32"

    def unload(self) -> None:
        self._translator = None
        self._sp = None
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self._translator is not None
        assert self._sp is not None

        # NLLB-200 source format: [text_tokens..., </s>, src_lang]
        # Target prefix: [tgt_lang]  (forced BOS)
        tokenized = [
            self._sp.encode(t, out_type=str) + ["</s>", src_lang]  # type: ignore[union-attr]
            for t in texts
        ]
        # Target prefix: the language token that forces the output language
        target_prefix = [[tgt_lang]] * len(tokenized)

        results = self._translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self.config.beam_size,
            max_decoding_length=self.config.max_decoding_length,
            max_batch_size=32,
        )

        output_texts: list[str] = []
        for result in results:
            tokens = result.hypotheses[0]
            # Strip the leading language token that CTranslate2 echoes back
            if tokens and tokens[0] == tgt_lang:
                tokens = tokens[1:]
            output_texts.append(self._sp.decode(tokens))  # type: ignore[union-attr]

        return output_texts
