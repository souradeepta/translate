"""Compatibility tests for the explicit HuggingFace NLLB API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.nllb import NLLBTranslator

torch = pytest.importorskip("torch")


class _Inputs(dict):
    def to(self, _device: object) -> _Inputs:
        return self


class _Tokenizer:
    src_lang: str | None = None

    def __call__(self, *_args: object, **_kwargs: object) -> _Inputs:
        return _Inputs(input_ids=torch.ones((1, 2), dtype=torch.long))

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "eng_Latn"
        return 17

    def batch_decode(self, _tokens: object, **_kwargs: object) -> list[str]:
        return ["A translation."]


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.empty(0))

    def generate(self, **kwargs: object) -> torch.Tensor:
        assert kwargs["forced_bos_token_id"] == 17
        assert kwargs["num_beams"] == 4
        return torch.ones((1, 3), dtype=torch.long)


def test_nllb_uses_explicit_seq2seq_api_and_language_tokens() -> None:
    tokenizer = _Tokenizer()
    model = _Model()
    config = ModelConfig(model_name="nllb-600M", device="cpu")
    translator = NLLBTranslator(config)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=model
    ), patch("transformers.pipeline", side_effect=AssertionError("pipeline API is obsolete")):
        translator.load()
        result = translator.translate(["আমি ভালো আছি।"], "ben_Beng", "eng_Latn")

    assert result == ["A translation."]
    assert tokenizer.src_lang == "ben_Beng"
    translator.unload()
