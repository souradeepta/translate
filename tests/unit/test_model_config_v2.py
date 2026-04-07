"""Tests for new ModelConfig fields added in model-expansion update."""
from __future__ import annotations
import pytest
from bn_en_translate.config import ModelConfig


def test_beam_size_defaults_to_none() -> None:
    cfg = ModelConfig()
    assert cfg.beam_size is None


def test_beam_size_can_be_set_explicitly() -> None:
    cfg = ModelConfig(beam_size=5)
    assert cfg.beam_size == 5


def test_beam_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="beam_size"):
        ModelConfig(beam_size=0)


def test_beam_size_negative_raises() -> None:
    with pytest.raises(ValueError, match="beam_size"):
        ModelConfig(beam_size=-1)


def test_use_flash_attention_default_true() -> None:
    cfg = ModelConfig()
    assert cfg.use_flash_attention is True


def test_use_flash_attention_can_be_disabled() -> None:
    cfg = ModelConfig(use_flash_attention=False)
    assert cfg.use_flash_attention is False


def test_max_ct2_batch_size_default_32() -> None:
    cfg = ModelConfig()
    assert cfg.max_ct2_batch_size == 32


def test_max_ct2_batch_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_ct2_batch_size"):
        ModelConfig(max_ct2_batch_size=0)


def test_max_ct2_batch_size_negative_raises() -> None:
    with pytest.raises(ValueError, match="max_ct2_batch_size"):
        ModelConfig(max_ct2_batch_size=-1)
