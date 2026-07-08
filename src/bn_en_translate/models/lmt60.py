"""LMT-60-1.7B translator — NiuTrans' Qwen3-based multilingual MT causal LM.

LMT-60-1.7B (NiuTrans/LMT-60-1.7B) is a 1.7B-parameter causal LM fine-tuned
for translation across 60 languages / 234 directions, based on Qwen3-1.7B-Base.
VRAM budget: ~3.4 GB in bfloat16 — fits comfortably within 8 GB.

Prompt format (from model card, verified 2026-07-08):
    "Translate the following text from Bengali into English:\\nBengali: {text}\\nEnglish:"
wrapped in a single user chat message via tokenizer.apply_chat_template().

Critical constraints:
  - AutoModelForCausalLM (decoder-only), NOT seq2seq
  - The prompt MUST go through apply_chat_template (unlike MiLMMT's plain text)
  - tokenizer.padding_side = "left" required for correct batch generation
  - Slice output at input_len to strip the echoed prompt from generated ids
  - bfloat16 (Qwen3 native dtype); float16 risks NaN
  - Model card quickstart: num_beams=5, do_sample=False

Setup:
    python scripts/download_models.py --model lmt-60-1.7B
"""

from __future__ import annotations

from typing import Any

from bn_en_translate.config import REPO_ROOT, ModelConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.models.hf_utils import (
    free_cuda_memory,
    resolve_attn_implementation,
    resolve_device,
)

# Human-readable language names used in the LMT-60 prompt template.
# Keys are FLORES-200 codes; values are what the model card uses.
_LANG_NAMES: dict[str, str] = {
    "ben_Beng": "Bengali",
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "urd_Arab": "Urdu",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
}


class LMT60Translator(TranslatorBase):
    """
    NiuTrans LMT-60-1.7B translation model (Qwen3-1.7B based causal LM).

    Architecture: Decoder-only causal LM, 1.7B parameters
    HF ID: NiuTrans/LMT-60-1.7B
    VRAM (bfloat16): ~3.4 GB — fits in 8 GB without offload
    Supports 60 languages / 234 directions including Bengali <-> English.

    Differences from MiLMMTTranslator (otherwise the same causal-LM pattern):
    - Prompt is wrapped in a user chat message via apply_chat_template()
    - Beam search (num_beams=5) per the model card quickstart

    Setup:
        python scripts/download_models.py --model lmt-60-1.7B
    """

    HF_MODEL_ID: str = "NiuTrans/LMT-60-1.7B"
    _LOCAL_PATH: str = str(REPO_ROOT / "models/lmt-60-1.7B-hf")
    DEFAULT_BEAM_SIZE: int = 5  # Model card quickstart: num_beams=5, do_sample=False

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="lmt-60-1.7b",
            model_path="",  # HF native — no CT2 conversion
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: Any = None
        self._tokenizer: Any = None

    def _build_prompts(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Format each text using the LMT-60 prompt template (pre-chat-template).

        Falls back to the raw FLORES-200 prefix if a language name is not in
        _LANG_NAMES, so new language pairs don't hard-fail on a dict entry.
        """
        src_name = _LANG_NAMES.get(src_lang, src_lang.split("_")[0].capitalize())
        tgt_name = _LANG_NAMES.get(tgt_lang, tgt_lang.split("_")[0].capitalize())
        return [
            f"Translate the following text from {src_name} into {tgt_name}:"
            f"\n{src_name}: {t}\n{tgt_name}:"
            for t in texts
        ]

    def load(self) -> None:
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bn_en_translate.utils.cuda_check import require_cuda

        model_id = self._LOCAL_PATH if Path(self._LOCAL_PATH).exists() else self.HF_MODEL_ID

        attn_impl = resolve_attn_implementation(self.config.use_flash_attention)

        device = resolve_device(self.config.device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Left-padding is required for causal LM batch generation.
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,  # Qwen3 native dtype; float16 risks NaN
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

        # Model card: the translation instruction goes inside a user chat
        # message; the chat template supplies Qwen3's special tokens, so the
        # tokenizer call below must not add its own (add_special_tokens=False).
        chat_prompts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in self._build_prompts(texts, src_lang, tgt_lang)
        ]
        inputs = self._tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to(model_device)

        # All prompts are padded to the same length by the tokenizer.
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_decoding_length,
                num_beams=self._effective_beam_size(),
                do_sample=False,
            )

        # Strip the echoed prompt tokens — only decode what was generated.
        new_tokens = output_ids[:, input_len:]
        return list(self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        ))
