"""TDD tests for configuration dataclasses."""

from __future__ import annotations

import pytest

from bn_en_translate.config import ChunkConfig, ModelConfig, PipelineConfig


# ---------------------------------------------------------------------------
# ChunkConfig
# ---------------------------------------------------------------------------


def test_chunk_config_defaults_are_valid() -> None:
    config = ChunkConfig()
    assert config.max_tokens_per_chunk == 400
    assert config.batch_size == 8
    assert config.overlap_sentences == 0


def test_chunk_config_zero_max_tokens_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens_per_chunk"):
        ChunkConfig(max_tokens_per_chunk=0)


def test_chunk_config_negative_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ChunkConfig(batch_size=-1)


def test_chunk_config_negative_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap_sentences"):
        ChunkConfig(overlap_sentences=-1)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


def test_model_config_defaults_are_valid() -> None:
    config = ModelConfig()
    assert config.model_name == "nllb-600M"
    assert config.device == "cuda"
    assert config.compute_type == "int8"
    assert config.beam_size is None


def test_model_config_invalid_device_raises() -> None:
    with pytest.raises(ValueError, match="device"):
        ModelConfig(device="tpu")


def test_model_config_invalid_compute_type_raises() -> None:
    with pytest.raises(ValueError, match="compute_type"):
        ModelConfig(compute_type="bfloat16")


def test_model_config_zero_beam_size_raises() -> None:
    with pytest.raises(ValueError, match="beam_size"):
        ModelConfig(beam_size=0)


def test_model_config_auto_device_is_valid() -> None:
    config = ModelConfig(device="auto")
    assert config.device == "auto"


def test_model_config_validate_path_raises_when_missing(tmp_path) -> None:
    config = ModelConfig(model_path=str(tmp_path / "nonexistent_model"))
    with pytest.raises(FileNotFoundError):
        config.validate_model_path()


def test_model_config_validate_path_passes_when_exists(tmp_path) -> None:
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    config = ModelConfig(model_path=str(model_dir))
    config.validate_model_path()  # should not raise


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


def test_pipeline_config_defaults() -> None:
    config = PipelineConfig()
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.chunk, ChunkConfig)
    assert config.ollama_polish is False
