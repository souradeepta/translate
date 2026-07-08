"""Unit tests for MADLAD-400-3B translator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bn_en_translate.config import ModelConfig


def test_madlad_import() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    assert MADLADTranslator is not None


def test_madlad_default_config() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    assert t.config.model_name == "madlad-3b"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"


def test_madlad_default_beam_size() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    assert t.DEFAULT_BEAM_SIZE == 4


def test_madlad_translate_raises_before_load() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_madlad_prepends_target_language_tag() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    result = t._build_input_texts(["আমি ভাত খাই।"], "eng_Latn")
    assert result == ["<2en> আমি ভাত খাই।"]


def test_madlad_empty_input_returns_empty() -> None:
    from bn_en_translate.models.madlad import MADLADTranslator
    t = MADLADTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._tokenizer = MagicMock()
    result = t.translate([], "ben_Beng", "eng_Latn")
    assert result == []


def test_attn_default_fallback_is_sdpa(monkeypatch) -> None:
    """Without flash-attn installed, the default fallback (no `fallback=` arg) is sdpa.

    This default suits architectures like Gemma3/MiLMMT; T5 (MADLAD) must override
    it via fallback="eager" — see test_attn_madlad_fallback_is_eager below.
    """
    import bn_en_translate.models.madlad as madlad_mod

    monkeypatch.setattr(madlad_mod, "_flash_attn_available", lambda: False)
    assert madlad_mod._resolve_attn_implementation(use_flash=True) == "sdpa"
    assert madlad_mod._resolve_attn_implementation(use_flash=False) == "sdpa"


def test_attn_uses_flash_when_available(monkeypatch) -> None:
    import bn_en_translate.models.madlad as madlad_mod

    monkeypatch.setattr(madlad_mod, "_flash_attn_available", lambda: True)
    assert madlad_mod._resolve_attn_implementation(use_flash=True) == "flash_attention_2"


def test_attn_madlad_fallback_is_eager(monkeypatch) -> None:
    """T5ForConditionalGeneration does NOT support sdpa (transformers 5.4.0) —
    MADLADTranslator.load() must pass fallback="eager" explicitly.
    """
    import bn_en_translate.models.madlad as madlad_mod

    monkeypatch.setattr(madlad_mod, "_flash_attn_available", lambda: False)
    assert madlad_mod._resolve_attn_implementation(use_flash=True, fallback="eager") == "eager"
    assert madlad_mod._resolve_attn_implementation(use_flash=False, fallback="eager") == "eager"


def test_madlad_load_passes_resolved_attn_impl_to_from_pretrained(monkeypatch) -> None:
    """load() must pass the resolver's output (not a hardcoded string) as attn_implementation.

    Patches the T5* from_pretrained classmethods directly on the real `transformers`
    module (load()'s local `from transformers import ...` resolves to the same class
    objects), and forces device="cpu" so no CUDA/download occurs.
    """
    import bn_en_translate.models.madlad as madlad_mod

    monkeypatch.setattr(madlad_mod, "_flash_attn_available", lambda: False)

    from bn_en_translate.models.madlad import MADLADTranslator

    cfg = ModelConfig(
        model_name="madlad-3b",
        model_path="models/madlad-3b-hf",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        device="cpu",
    )
    t = MADLADTranslator(cfg)

    import torch

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    tied_weight = torch.randn(4, 2)
    mock_model.shared.weight = tied_weight
    mock_model.decoder.embed_tokens.weight = tied_weight

    with patch("transformers.T5Tokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch(
             "transformers.T5ForConditionalGeneration.from_pretrained",
             return_value=mock_model,
         ) as mock_from_pretrained:
        t.load()

    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["attn_implementation"] == "eager"


def test_verify_tied_embeddings_raises_on_mismatch() -> None:
    import torch

    from bn_en_translate.models.madlad import MADLADTranslator

    class FakeEmbed:
        def __init__(self, w: torch.Tensor) -> None:
            self.weight = w

    class FakeDecoder:
        def __init__(self, w: torch.Tensor) -> None:
            self.embed_tokens = FakeEmbed(w)

    class FakeModel:
        def __init__(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
            self.shared = FakeEmbed(w1)
            self.decoder = FakeDecoder(w2)

    w = torch.randn(8, 4)
    MADLADTranslator._verify_tied_embeddings(FakeModel(w, w))  # tied: no raise

    # w + 1.0 is deterministically different from w (provably untied)
    with pytest.raises(RuntimeError, match="tied-embedding mismatch"):
        MADLADTranslator._verify_tied_embeddings(FakeModel(w, w + 1.0))


def test_madlad_load_raises_on_untied_checkpoint(monkeypatch) -> None:
    """Negative wiring test: load() must actually invoke the integrity guard.

    Without this, deleting the guard call in load() leaves every other test
    green. Also proves _loaded stays False when the guard fires.
    """
    import bn_en_translate.models.madlad as madlad_mod

    monkeypatch.setattr(madlad_mod, "_flash_attn_available", lambda: False)

    from bn_en_translate.models.madlad import MADLADTranslator

    cfg = ModelConfig(
        model_name="madlad-3b",
        model_path="models/madlad-3b-hf",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
        device="cpu",
    )
    t = MADLADTranslator(cfg)

    import torch

    mock_model = MagicMock()
    mock_model.shared.weight = torch.randn(4, 2)
    mock_model.decoder.embed_tokens.weight = torch.randn(4, 2)  # untied

    with patch("transformers.T5Tokenizer.from_pretrained", return_value=MagicMock()), \
         patch("transformers.T5ForConditionalGeneration.from_pretrained", return_value=mock_model):
        with pytest.raises(RuntimeError, match="tied-embedding mismatch"):
            t.load()

    assert t._loaded is False


def test_t5_rejects_sdpa_but_accepts_eager() -> None:
    """Empirical regression guard: T5PreTrainedModel._supports_sdpa is False in
    transformers 5.4.0 — a tiny T5 config must accept attn_implementation="eager"
    and reject "sdpa" with ValueError. No download, ~1s.
    """
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(d_model=8, d_ff=16, num_layers=1, num_heads=2, d_kv=4, vocab_size=32)

    T5ForConditionalGeneration._from_config(cfg, attn_implementation="eager")

    with pytest.raises(ValueError):
        T5ForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
