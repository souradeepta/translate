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
class FineTuneConfig:
    """Configuration for LoRA fine-tuning of NLLB models."""

    # Optimisation
    learning_rate: float = 2e-4
    num_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    )

    # Data
    max_source_length: int = 256
    max_target_length: int = 256

    # Checkpointing / output
    output_dir: str = "models/nllb-600M-finetuned"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    fp16: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if self.max_source_length <= 0:
            raise ValueError("max_source_length must be positive")
        if self.max_target_length <= 0:
            raise ValueError("max_target_length must be positive")


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    ollama_polish: bool = False
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
