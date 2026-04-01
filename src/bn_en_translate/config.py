"""Configuration dataclasses for the translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChunkConfig:
    """Controls how long stories are split before translation."""

    max_tokens_per_chunk: int = 400
    min_chunk_sentences: int = 1
    batch_size: int = 8
    overlap_sentences: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens_per_chunk <= 0:
            raise ValueError("max_tokens_per_chunk must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_chunk_sentences <= 0:
            raise ValueError("min_chunk_sentences must be positive")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")


@dataclass
class ModelConfig:
    """Configuration for the translation model."""

    model_name: str = "nllb-600M"
    model_path: str = "models/nllb-600M-ct2"
    device: str = "cuda"
    compute_type: str = "int8"
    src_lang: str = "ben_Beng"
    tgt_lang: str = "eng_Latn"
    beam_size: int = 4
    max_decoding_length: int = 512

    VALID_DEVICES = {"cuda", "cpu", "auto"}
    VALID_COMPUTE_TYPES = {"int8", "float16", "float32", "int8_float16"}

    def __post_init__(self) -> None:
        if self.device not in self.VALID_DEVICES:
            raise ValueError(f"device must be one of {self.VALID_DEVICES}, got '{self.device}'")
        if self.compute_type not in self.VALID_COMPUTE_TYPES:
            raise ValueError(
                f"compute_type must be one of {self.VALID_COMPUTE_TYPES}, got '{self.compute_type}'"
            )
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_decoding_length <= 0:
            raise ValueError("max_decoding_length must be positive")

    def validate_model_path(self) -> None:
        """Check that model_path exists on disk. Call explicitly before loading."""
        p = Path(self.model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    ollama_polish: bool = False
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
