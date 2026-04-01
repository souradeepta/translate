"""Unit tests for NLLBFineTuner — all model/GPU calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> object:
    from bn_en_translate.config import FineTuneConfig, ModelConfig

    model_cfg = ModelConfig(
        model_name="nllb-600M",
        model_path="facebook/nllb-200-distilled-600M",  # HF id for fine-tuning
        device="cpu",
        compute_type="float16",
    )
    ft_cfg = FineTuneConfig(
        num_epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        output_dir=str(tmp_path / "finetuned"),
        save_steps=10,
        eval_steps=10,
        logging_steps=5,
        fp16=False,  # CPU test
    )
    return model_cfg, ft_cfg


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_finetuner_constructs_without_loading_model(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    # Should not raise even though model path doesn't exist yet
    tuner = NLLBFineTuner(model_cfg, ft_cfg)
    assert not tuner.is_loaded


def test_finetuner_is_loaded_false_before_load(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)
    assert tuner.is_loaded is False


# ---------------------------------------------------------------------------
# load() / unload()
# ---------------------------------------------------------------------------

def test_finetuner_load_sets_is_loaded(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)

    with (
        patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=MagicMock()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("peft.get_peft_model", return_value=MagicMock()),
        patch("peft.LoraConfig"),
    ):
        tuner.load()

    assert tuner.is_loaded


def test_finetuner_unload_clears_is_loaded(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)

    with (
        patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=MagicMock()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("peft.get_peft_model", return_value=MagicMock()),
        patch("peft.LoraConfig"),
    ):
        tuner.load()
        tuner.unload()

    assert not tuner.is_loaded


def test_finetuner_train_raises_if_not_loaded(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)

    with pytest.raises(RuntimeError, match="load\\(\\)"):
        tuner.train([], [], [], [])


# ---------------------------------------------------------------------------
# BLEU evaluation helper
# ---------------------------------------------------------------------------

def test_finetuner_compute_bleu_returns_float(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import compute_corpus_bleu

    hypotheses = ["The cat sat on the mat.", "A dog ran fast."]
    references = ["The cat sat on the mat.", "A dog ran fast."]
    score = compute_corpus_bleu(hypotheses, references)

    assert isinstance(score, float)
    assert 0.0 <= score <= 100.1  # sacreBLEU can return 100.000...04 for perfect match


def test_finetuner_compute_bleu_perfect_score(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import compute_corpus_bleu

    hyps = ["the quick brown fox", "jumps over the lazy dog"]
    refs = ["the quick brown fox", "jumps over the lazy dog"]
    score = compute_corpus_bleu(hyps, refs)

    assert score > 95.0  # should be ~100


def test_finetuner_compute_bleu_low_for_bad_translation() -> None:
    from bn_en_translate.training.trainer import compute_corpus_bleu

    hyps = ["completely wrong output nonsense", "garbage output here"]
    refs = ["the quick brown fox", "jumps over the lazy dog"]
    score = compute_corpus_bleu(hyps, refs)

    assert score < 10.0


# ---------------------------------------------------------------------------
# CT2 export
# ---------------------------------------------------------------------------

def test_finetuner_export_raises_if_not_loaded(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)

    with pytest.raises(RuntimeError, match="load\\(\\)"):
        tuner.export_ct2(tmp_path / "ct2_out")
