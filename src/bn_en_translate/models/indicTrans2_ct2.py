"""IndicTrans2-1B via CTranslate2 INT8 — optimal quality + GPU efficiency."""

from __future__ import annotations

from pathlib import Path  # noqa: F401 — used in _best_compute_type

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


class IndicTrans2Ct2Translator(TranslatorBase):
    """
    AI4Bharat IndicTrans2-1B via CTranslate2 INT8.

    This is the recommended path for production use:
      - INT8 quantization: ~1.0–1.5 GB VRAM (vs ~3 GB float16)
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

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig(
            model_name="indicTrans2-1B",
            model_path="models/indicTrans2-1B-ct2",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._translator: object | None = None
        self._sp: object | None = None
        self._processor: object | None = None

    def load(self) -> None:
        import ctranslate2  # type: ignore[import-untyped]
        import sentencepiece as spm  # type: ignore[import-untyped]

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
        self._sp.load(str(sp_path))  # type: ignore[union-attr]

        compute_type = self._best_compute_type(device, self._sp)

        self._translator = ctranslate2.Translator(
            str(model_path),
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=4,
        )

        # IndicTransToolkit handles script normalization + language tagging
        try:
            from IndicTransToolkit import IndicProcessor  # type: ignore[import-untyped]
            self._processor = IndicProcessor(inference=True)
        except ImportError:
            self._processor = None

        self._loaded = True

    def _best_compute_type(self, device: str, sp: object) -> str:
        """Probe with a realistic sentence to find the best working compute type."""
        if device == "cpu":
            return "int8"
        import ctranslate2  # type: ignore[import-untyped]
        import sentencepiece as spm  # type: ignore[import-untyped]
        assert isinstance(sp, spm.SentencePieceProcessor)

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
            self._sp.encode(t, out_type=str) + ["</s>", src_lang]  # type: ignore[union-attr]
            for t in preprocessed
        ]
        target_prefix = [[tgt_lang]] * len(tokenized)

        results = self._translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self.config.beam_size,
            max_decoding_length=self.config.max_decoding_length,
        )

        decoded: list[str] = []
        for result in results:
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == tgt_lang:
                tokens = tokens[1:]
            decoded.append(self._sp.decode(tokens))  # type: ignore[union-attr]

        if self._processor is not None:
            decoded = self._processor.postprocess_batch(decoded, lang=tgt_lang)

        return decoded
