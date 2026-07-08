"""MiLMMT-46-1B translator — Xiaomi's Gemma3-based multilingual MT causal LM.

MiLMMT-46-1B (xiaomi-research/MiLMMT-46-1B-v0.1) is a 1B-parameter causal LM
fine-tuned for translation across 46 language pairs, based on Gemma3-1B.
VRAM budget: ~2 GB in bfloat16 — fits comfortably within 8 GB.

Prompt format (from model card):
    "Translate this from Bengali to English:\\nBengali: {text}\\nEnglish:"

Critical constraints:
  - AutoModelForCausalLM (decoder-only), NOT seq2seq — different generate() contract
  - tokenizer.padding_side = "left" required for correct batch generation
  - Slice output at input_len to strip the echoed prompt from generated ids
  - bfloat16 (not float16): Gemma3 was trained in bf16; float16 can produce NaN
  - do_sample=False for deterministic output

Setup:
    python scripts/download_models.py --model milmmt-46-1B
"""

from __future__ import annotations

from typing import Any

from bn_en_translate.config import REPO_ROOT, ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.models.hf_utils import (
    flash_attn_available as _flash_attn_available,  # noqa: F401
)
from bn_en_translate.models.hf_utils import free_cuda_memory, resolve_device

# Human-readable language names used in the MiLMMT prompt template.
# Keys are FLORES-200 codes; values are what the model card uses.
_LANG_NAMES: dict[str, str] = {
    "ben_Beng": "Bengali",
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "urd_Arab": "Urdu",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
}


def _resolve_attn_implementation(use_flash: bool, fallback: str = "sdpa") -> str:
    """flash_attention_2 if installed and requested; else the given fallback.

    Re-implements hf_utils.resolve_attn_implementation's logic inline (rather
    than delegating to it) so it reads the module-level `_flash_attn_available`
    name — this preserves the existing monkeypatch seam tests rely on.
    """
    if use_flash and _flash_attn_available():
        return "flash_attention_2"
    return fallback


class MiLMMTTranslator(TranslatorBase):
    """
    Xiaomi MiLMMT-46-1B translation model (Gemma3-1B based causal LM).

    Architecture: Decoder-only causal LM, 1B parameters
    HF ID: xiaomi-research/MiLMMT-46-1B-v0.1
    VRAM (bfloat16): ~2 GB — fits in 8 GB without offload
    Supports 46 language pairs including Bengali ↔ English.

    Unlike seq2seq models (NLLB, IndicTrans2), this is a causal LM:
    - Input is a structured prompt with human-readable src/tgt language names
    - Generated output includes the prompt tokens — sliced at input_len to extract translation
    - Left-padding is required for batched inference (right-padding breaks causal attention)
    - bfloat16 preferred (Gemma3 native dtype); float16 risks NaN on some inputs

    Setup:
        python scripts/download_models.py --model milmmt-46-1B
    """

    HF_MODEL_ID: str = "xiaomi-research/MiLMMT-46-1B-v0.1"
    _LOCAL_PATH: str = str(REPO_ROOT / "models/milmmt-46-1B-hf")
    DEFAULT_BEAM_SIZE: int = 1  # Greedy by default; causal LMs rarely benefit from beam search

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="milmmt-46-1b",
            model_path="",  # HF native — no CT2 conversion
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: Any = None
        self._tokenizer: Any = None

    def _build_prompts(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Format each text using the MiLMMT prompt template.

        Falls back to the raw FLORES-200 prefix if a language name is not in _LANG_NAMES,
        so that new language pairs don't hard-fail on a missing dict entry.
        """
        src_name = _LANG_NAMES.get(src_lang, src_lang.split("_")[0].capitalize())
        tgt_name = _LANG_NAMES.get(tgt_lang, tgt_lang.split("_")[0].capitalize())
        return [
            f"Translate this from {src_name} to {tgt_name}:\n{src_name}: {t}\n{tgt_name}:"
            for t in texts
        ]

    def load(self) -> None:
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bn_en_translate.utils.cuda_check import require_cuda

        model_id = self._LOCAL_PATH if Path(self._LOCAL_PATH).exists() else self.HF_MODEL_ID

        attn_impl = _resolve_attn_implementation(self.config.use_flash_attention)

        device = resolve_device(self.config.device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Left-padding is required for causal LM batch generation.
        # Causal attention is left-to-right; right-padded batches produce misaligned KV cache.
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,  # Gemma3 native dtype; float16 risks NaN
        )
        if device == "cuda":
            require_cuda(type(self).__name__)
            self._model = self._model.to("cuda")

        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        free_cuda_memory()

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch

        model_device = next(self._model.parameters()).device

        prompts = self._build_prompts(texts, src_lang, tgt_lang)
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to(model_device)

        # Record prompt length so we can strip it from the output.
        # All prompts are padded to the same length by the tokenizer.
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_decoding_length,
                num_beams=self._effective_beam_size(),
                do_sample=False,
            )

        # Strip the echoed prompt tokens — only decode what was generated after the prompt.
        new_tokens = output_ids[:, input_len:]
        return list(self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        ))
