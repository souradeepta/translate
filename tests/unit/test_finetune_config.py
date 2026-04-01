"""Tests for FineTuneConfig dataclass."""

from __future__ import annotations

import pytest


def test_default_finetune_config_is_valid() -> None:
    from bn_en_translate.config import FineTuneConfig

    cfg = FineTuneConfig()
    assert cfg.learning_rate > 0
    assert cfg.num_epochs > 0
    assert cfg.train_batch_size > 0
    assert cfg.eval_batch_size > 0
    assert 0 < cfg.lora_r <= 64
    assert cfg.lora_alpha > 0


def test_finetune_config_custom_values() -> None:
    from bn_en_translate.config import FineTuneConfig

    cfg = FineTuneConfig(
        learning_rate=5e-5,
        num_epochs=5,
        train_batch_size=4,
        lora_r=16,
        lora_alpha=32,
    )
    assert cfg.learning_rate == 5e-5
    assert cfg.num_epochs == 5
    assert cfg.train_batch_size == 4
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32


def test_finetune_config_rejects_negative_lr() -> None:
    from bn_en_translate.config import FineTuneConfig

    with pytest.raises(ValueError, match="learning_rate"):
        FineTuneConfig(learning_rate=-1e-5)


def test_finetune_config_rejects_zero_epochs() -> None:
    from bn_en_translate.config import FineTuneConfig

    with pytest.raises(ValueError, match="num_epochs"):
        FineTuneConfig(num_epochs=0)


def test_finetune_config_rejects_zero_batch_size() -> None:
    from bn_en_translate.config import FineTuneConfig

    with pytest.raises(ValueError, match="train_batch_size"):
        FineTuneConfig(train_batch_size=0)


def test_finetune_config_rejects_invalid_lora_r() -> None:
    from bn_en_translate.config import FineTuneConfig

    with pytest.raises(ValueError, match="lora_r"):
        FineTuneConfig(lora_r=0)


def test_finetune_config_output_dir_defaults_to_string() -> None:
    from bn_en_translate.config import FineTuneConfig

    cfg = FineTuneConfig()
    assert isinstance(cfg.output_dir, str)
    assert len(cfg.output_dir) > 0


def test_finetune_config_gradient_accumulation_positive() -> None:
    from bn_en_translate.config import FineTuneConfig

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        FineTuneConfig(gradient_accumulation_steps=0)


def test_finetune_config_warmup_steps_non_negative() -> None:
    from bn_en_translate.config import FineTuneConfig

    with pytest.raises(ValueError, match="warmup_steps"):
        FineTuneConfig(warmup_steps=-1)


def test_finetune_config_max_length_positive() -> None:
    from bn_en_translate.config import FineTuneConfig

    with pytest.raises(ValueError, match="max_source_length"):
        FineTuneConfig(max_source_length=0)
