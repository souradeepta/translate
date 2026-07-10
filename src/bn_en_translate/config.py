"""Configuration dataclasses for the translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root — anchored to this file's location so path resolution is
# cwd-independent regardless of where the process is started from.
REPO_ROOT: Path = Path(__file__).parents[2]

# Canonical on-disk paths for CTranslate2 model directories.
# Single source of truth used by factory.py (routing) and download_models.py (output).
CT2_MODEL_PATHS: dict[str, str] = {
    "nllb-600m":      str(REPO_ROOT / "models/nllb-600M-ct2"),
    "nllb-1.3b":      str(REPO_ROOT / "models/nllb-1.3B-ct2"),
    "indictrans2-1b": str(REPO_ROOT / "models/indicTrans2-1B-ct2"),
    "indictrans2":    str(REPO_ROOT / "models/indicTrans2-1B-ct2"),
}

# Measured VRAM peaks (MiB) on RTX 5050 8 GB — source: monitor/observations.md
# (2026-04-10 run + 2026-07 optimization pass). Used for pre-flight checks
# before loading a second model (e.g. the Ollama polish pass). Keys are
# lower-case model names.
#
# milmmt-46-1b: 4100, not the 3,379 MiB reported on 2026-04-10 — the 2026-07
# Task 3 note (post-batching) measured 4,009 MiB standalone / 4,024 MiB
# sequential-after-nllb, rounded up to the nearest 100 like the other entries.
MODEL_VRAM_MIB: dict[str, int] = {
    "nllb-600m":         2400,
    "milmmt-46-1b":      4100,
    "seamless-medium":   4100,
    "indictrans2-1b":    3100,
    "sarvam-translate":  5000,  # 4B params, 4-bit bnb — measured 4899 MiB peak (real GPU run,
                                # after fixing a device-resolution bug that had silently run
                                # an earlier measurement on CPU), 5-sent smoke
    "krutrim-translate": 1700,  # distilled IndicTrans2 CT2, measured 1608 MiB peak, 90-sent
    "milmmt-46-4b":      8100,  # 4B params, 4-bit bnb — measured 8050 MiB peak, 90-sent —
                                # VERY TIGHT against this 8151 MiB card, ~100 MiB headroom
    "ollama-qwen2.5:7b": 4800,
    "ollama-gemma3:12b": 4700,
}


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
    model_path: str = str(REPO_ROOT / "models/nllb-600M-ct2")
    device: str = "cuda"
    # Requested compute type for CT2 backends. On sm_120 the load-time probe
    # overrides int8 with float16 (CUBLAS does not support int8 there) — this
    # value is a preference, not a guarantee. See utils/ct2_utils.probe_compute_type.
    compute_type: str = "int8"
    src_lang: str = "ben_Beng"
    tgt_lang: str = "eng_Latn"
    beam_size: int | None = None          # None = use each translator's DEFAULT_BEAM_SIZE
    max_decoding_length: int = 512
    inference_batch_size: int = 8
    use_flash_attention: bool = True      # Flash Attention 2 if flash-attn is installed
    max_ct2_batch_size: int = 32          # CT2 translate_batch max_batch_size guard
    load_in_4bit: bool = False            # bitsandbytes 4-bit quantization for HF causal LMs
                                           # too large to fit 8 GB VRAM in bf16 (e.g. 4B+ params)

    VALID_DEVICES = {"cuda", "cpu", "auto"}
    VALID_COMPUTE_TYPES = {"int8", "float16", "float32", "int8_float16"}

    def __post_init__(self) -> None:
        if self.device not in self.VALID_DEVICES:
            raise ValueError(f"device must be one of {self.VALID_DEVICES}, got '{self.device}'")
        if self.compute_type not in self.VALID_COMPUTE_TYPES:
            raise ValueError(
                f"compute_type must be one of {self.VALID_COMPUTE_TYPES}, got '{self.compute_type}'"
            )
        if self.beam_size is not None and self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_decoding_length <= 0:
            raise ValueError("max_decoding_length must be positive")
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        if self.max_ct2_batch_size <= 0:
            raise ValueError("max_ct2_batch_size must be positive")

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
    # bf16=True is required on Blackwell sm_120; fp16+GradScaler raises ValueError.
    # Keep fp16=False here — the trainer sets bf16=True explicitly.
    fp16: bool = False

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
    ollama_model: str = "gemma3:12b"
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class MonitorConfig:
    """Configuration for ResourceMonitor and RunDatabase."""

    # Sampling
    sample_interval_s: float = 2.0       # background thread wakes every N seconds
    enabled: bool = True                  # set False to make ResourceMonitor a no-op

    # Storage
    db_path: Path = field(default_factory=lambda: Path("monitor/runs.db"))

    # GPU backend preference
    gpu_backend: str = "pynvml"           # 'pynvml' | 'nvidia-smi' | 'none'

    VALID_GPU_BACKENDS = {"pynvml", "nvidia-smi", "none"}

    def __post_init__(self) -> None:
        if self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        if self.gpu_backend not in self.VALID_GPU_BACKENDS:
            raise ValueError(
                f"gpu_backend must be one of {self.VALID_GPU_BACKENDS}, "
                f"got '{self.gpu_backend}'"
            )
