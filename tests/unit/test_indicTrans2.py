"""Unit tests for the IndicTrans2 HF translator (attention policy pin)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bn_en_translate.config import ModelConfig


def test_indictrans2_fallback_load_uses_eager_attention(monkeypatch) -> None:
    """IndicTrans2's sdpa support is unverified — load must pass eager.

    Pins the fallback="eager" policy at the real call site (same guard style
    as MADLAD's routing test).
    """
    monkeypatch.setattr(
        "bn_en_translate.models.hf_utils.flash_attn_available", lambda: False
    )

    from bn_en_translate.models.indicTrans2 import IndicTrans2Translator

    cfg = ModelConfig(
        model_name="indicTrans2-1B",
        model_path="",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        device="cpu",
    )
    t = IndicTrans2Translator(cfg)

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()
    ), patch(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained",
        return_value=MagicMock(),
    ) as mock_from_pretrained:
        t._load_via_transformers_fallback()

    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["attn_implementation"] == "eager"
