"""Ollama-based LLM translator for high-quality literary translation."""

from __future__ import annotations

import httpx

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.base import TranslatorBase

TRANSLATION_PROMPT = """\
You are a professional literary translator.
Translate the following Bengali text to English.
Preserve the narrative tone, cultural references, and stylistic choices.
Output ONLY the English translation — no explanations, no commentary.

Bengali text:
{text}

English translation:"""


class OllamaTranslator(TranslatorBase):
    """
    Translates using a locally running Ollama model (e.g., qwen2.5:7b).

    This model produces the most natural-sounding English and handles
    literary tone, metaphors, and cultural phrases better than seq2seq models.

    Tradeoffs vs IndicTrans2:
      - ~4x slower per sentence
      - Uses 4.7 GB VRAM (less headroom for other models)
      - Non-deterministic without explicit seed

    Recommended use:
      - As an optional polishing pass after IndicTrans2 translation
      - As primary for short stories where quality > speed

    Prerequisites:
      - Ollama running: `ollama serve`
      - Model pulled: `ollama pull qwen2.5:7b-instruct-q4_K_M`
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        super().__init__()
        self.config = config or PipelineConfig()
        self._client: httpx.Client | None = None

    def load(self) -> None:
        self._client = httpx.Client(
            base_url=self.config.ollama_base_url,
            timeout=120.0,
        )
        # Verify Ollama is reachable
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.ollama_base_url}. "
                "Is `ollama serve` running?"
            ) from e
        self._loaded = True

    def unload(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        # Ollama is called one text at a time (no native batching)
        return [self._translate_one(text) for text in texts]

    def _translate_one(self, text: str) -> str:
        assert self._client is not None
        prompt = TRANSLATION_PROMPT.format(text=text)
        response = self._client.post(
            "/api/generate",
            json={
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024},
            },
        )
        response.raise_for_status()
        return response.json()["response"].strip()
