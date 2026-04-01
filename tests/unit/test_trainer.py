"""Unit tests for NLLBFineTuner — all model/GPU calls are mocked."""

from __future__ import annotations

import os
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
# DataLoader safety (parallel workers + fast tokenizer)
# ---------------------------------------------------------------------------

def test_train_sets_tokenizers_parallelism_false_on_cpu(tmp_path: Path) -> None:
    """CPU training (num_workers=4) must disable fast-tokenizer thread pool
    before workers fork to prevent deadlocks (Rust thread state after fork).
    """
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)
    tuner._use_cuda = False  # simulate CPU path

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(device=MagicMock(
        __str__=lambda _: "cpu"
    ))])

    captured_env: dict[str, str] = {}

    def fake_trainer_init(*args, **kwargs):
        captured_env["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "not_set"
        )
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(training_loss=0.5)
        return mock_trainer

    with (
        patch("transformers.Trainer", side_effect=fake_trainer_init),
        patch("transformers.Seq2SeqTrainingArguments"),
        patch("bn_en_translate.training.dataset.BengaliEnglishDataset"),
    ):
        tuner._peft_model = mock_model
        tuner._tokenizer = MagicMock()
        tuner._eval_bleu = MagicMock(return_value=30.0)  # type: ignore[method-assign]
        tuner.train(["src1"], ["tgt1"], ["vsrc1"], ["vtgt1"])

    assert captured_env.get("TOKENIZERS_PARALLELISM") == "false", (
        "TOKENIZERS_PARALLELISM must be 'false' before Trainer init "
        "to prevent fast-tokenizer deadlock in forked DataLoader workers"
    )


def test_train_num_workers_zero_on_gpu(tmp_path: Path) -> None:
    """GPU path must use num_workers=0 to avoid CUDA context in forked processes."""
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)
    tuner._use_cuda = True  # simulate GPU path

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(device=MagicMock(
        __str__=lambda _: "cuda:0"
    ))])

    captured_kwargs: dict = {}

    def fake_training_args(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = MagicMock(training_loss=0.4)

    with (
        patch("transformers.Trainer", return_value=mock_trainer),
        patch("transformers.Seq2SeqTrainingArguments", side_effect=fake_training_args),
        patch("bn_en_translate.training.dataset.BengaliEnglishDataset"),
    ):
        tuner._peft_model = mock_model
        tuner._tokenizer = MagicMock()
        tuner._eval_bleu = MagicMock(return_value=30.0)  # type: ignore[method-assign]
        tuner.train(["src1"], ["tgt1"], ["vsrc1"], ["vtgt1"])

    assert captured_kwargs.get("dataloader_num_workers") == 0, (
        "GPU path must use num_workers=0 to prevent CUDA context forking"
    )
    assert captured_kwargs.get("dataloader_prefetch_factor") is None, (
        "prefetch_factor must be None when num_workers=0 (PyTorch requires this)"
    )


# ---------------------------------------------------------------------------
# CT2 export
# ---------------------------------------------------------------------------

def test_finetuner_export_raises_if_not_loaded(tmp_path: Path) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg, ft_cfg = _make_config(tmp_path)
    tuner = NLLBFineTuner(model_cfg, ft_cfg)

    with pytest.raises(RuntimeError, match="load\\(\\)"):
        tuner.export_ct2(tmp_path / "ct2_out")
