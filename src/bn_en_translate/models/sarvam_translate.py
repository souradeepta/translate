"""Sarvam-Translate — Sarvam AI + AI4Bharat's document-level translation model.

sarvamai/sarvam-translate is a Gemma3-4B-IT causal LM fine-tuned specifically
for document-level translation across the 22 official Indian languages. It is
the de-facto successor to IndicTrans2/IndicTrans3-beta (both AI4Bharat) —
AI4Bharat and Sarvam AI now jointly maintain this model.

Prompt format (from the model card, verbatim — chat template, not a raw
string): a system message "Translate the text below to {tgt_lang}." plus a
user message containing only the text to translate. This is a much more
constrained instruction than a general "preserve narrative tone" literary
prompt — deliberately so, after `OllamaTranslator`'s generic literary prompt
was found to hallucinate fabricated scenes on a real story (2026-07-09).

Critical constraints:
  - AutoModelForCausalLM (decoder-only), NOT the multimodal
    Gemma3ForConditionalGeneration class the config.json architecture field
    names — the model card's own Quickstart uses AutoModelForCausalLM, which
    resolves the text-only tower.
  - 4B params: too large for 8 GB VRAM in bf16 (~8+ GB weights alone).
    load_in_4bit=True (bitsandbytes) is the default here — probe-verified
    working on this repo's sm_120 GPU already (see docs/MODELS.md "Deferred"
    section from the MiLMMT-46-4B probe).
  - bitsandbytes quantized models cannot be moved with `.to("cuda")` after
    load — device placement must happen via `device_map` at from_pretrained
    time. We use `device_map={"": 0}` (explicit single-GPU pin), never
    `device_map="auto"` (this repo's GPU-only rule: "auto" silently permits
    CPU offload under VRAM pressure).
  - tokenizer.padding_side = "left" required for correct batch generation
    (same causal-LM constraint as MiLMMT).
  - do_sample=False for deterministic output (the model card uses
    do_sample=True, temperature=0.01 — near-greedy; False is the safer,
    fully-deterministic equivalent and matches this project's MiLMMT convention).

Setup:
    python scripts/download_models.py --model sarvam-translate
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

# Human-readable language names used in the Sarvam-Translate chat prompt.
# Keys are FLORES-200 codes; values are what the model card's examples use.
_LANG_NAMES: dict[str, str] = {
    "ben_Beng": "Bengali",
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "guj_Gujr": "Gujarati",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu",
    "asm_Beng": "Assamese",
    "ory_Orya": "Odia",
}


class SarvamTranslateTranslator(TranslatorBase):
    """
    Sarvam AI + AI4Bharat's Sarvam-Translate model (Gemma3-4B-IT based causal LM).

    Architecture: Decoder-only causal LM, 4B parameters, 4-bit quantized by default
    HF ID: sarvamai/sarvam-translate
    VRAM (4-bit bnb): ~3.5 GB (provisional — see config.MODEL_VRAM_MIB)
    Document-level translation model — the successor to IndicTrans2/IndicTrans3-beta.

    Setup:
        python scripts/download_models.py --model sarvam-translate
    """

    HF_MODEL_ID: str = "sarvamai/sarvam-translate"
    _LOCAL_PATH: str = str(REPO_ROOT / "models/sarvam-translate-hf")
    DEFAULT_BEAM_SIZE: int = 1  # Greedy; causal LMs rarely benefit from beam search

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="sarvam-translate",
            model_path="",  # HF native — no CT2 conversion
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
            load_in_4bit=True,
        )
        self._model: Any = None
        self._tokenizer: Any = None

    def _build_messages(
        self, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[list[dict[str, str]]]:
        """Build the model card's chat-style system+user messages per text.

        Only the target language name appears in the prompt (per the model
        card) — the model infers the source language from the text itself.
        """
        tgt_name = _LANG_NAMES.get(tgt_lang, tgt_lang.split("_")[0].capitalize())
        return [
            [
                {"role": "system", "content": f"Translate the text below to {tgt_name}."},
                {"role": "user", "content": t},
            ]
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

        quantization_config = None
        device_map: dict[str, int] | None = None
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            require_cuda(type(self).__name__)  # 4-bit bnb here is CUDA-only
            quantization_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # Explicit single-GPU pin — NOT "auto" (which permits silent CPU
            # offload under VRAM pressure; this repo requires GPU-only).
            # bitsandbytes quantized layers must be placed at load time; a
            # plain `.to("cuda")` after from_pretrained is not supported.
            device_map = {"": 0}

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        if device_map is None and device == "cuda":
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

        messages_batch = self._build_messages(texts, src_lang, tgt_lang)
        prompts = [
            self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_decoding_length,
            add_special_tokens=False,
        ).to(model_device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_decoding_length,
                num_beams=self._effective_beam_size(),
                do_sample=False,
            )

        new_tokens = output_ids[:, input_len:]
        return list(self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        ))
