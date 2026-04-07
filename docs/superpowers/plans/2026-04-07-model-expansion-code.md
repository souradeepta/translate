# Model Expansion & Inference Optimization — Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MADLAD-400-3B and SeamlessM4T-v2-medium translator backends, upgrade Ollama polish default to Gemma 3, add Flash Attention 2, per-model beam defaults, configurable batch sizes, and extend fine-tuning to any seq2seq model.

**Architecture:** All new models slot into the existing `@register_model` registry — no factory refactor needed. Config changes (`beam_size: int | None`, `use_flash_attention`, `max_ct2_batch_size`) propagate through existing `ModelConfig` dataclass. Fine-tuner is generalized by adding a `hf_model_id` parameter.

**Tech Stack:** Python 3.12, PyTorch 2.7.0+cu128, CTranslate2 4.7.1, HuggingFace Transformers 5.4.0, PEFT 0.18.1, Click, pytest

**Env before starting:**
```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
make test   # should be 186 tests, all passing
```

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/bn_en_translate/config.py` | Add `beam_size: int\|None`, `use_flash_attention`, `max_ct2_batch_size` |
| Modify | `src/bn_en_translate/models/base.py` | Add `DEFAULT_BEAM_SIZE`, `_effective_beam_size()` |
| Modify | `src/bn_en_translate/models/nllb_ct2.py` | Use `_effective_beam_size()`, `max_ct2_batch_size` |
| Modify | `src/bn_en_translate/models/nllb.py` | Set `DEFAULT_BEAM_SIZE = 4` |
| Modify | `src/bn_en_translate/models/indicTrans2_ct2.py` | Set `DEFAULT_BEAM_SIZE = 5`, `max_ct2_batch_size` |
| Modify | `src/bn_en_translate/models/indicTrans2.py` | Set `DEFAULT_BEAM_SIZE = 5`, Flash Attention 2 |
| Create | `src/bn_en_translate/models/madlad.py` | MADLAD-400-3B HF translator |
| Create | `src/bn_en_translate/models/seamless.py` | SeamlessM4T-v2 HF translator |
| Modify | `src/bn_en_translate/models/factory.py` | Register `madlad-3b`, `seamless-medium` |
| Modify | `scripts/download_models.py` | Add `madlad-3b`, `seamless-medium`, `nllb-1.3B`, `indicTrans2-1B` entries |
| Modify | `src/bn_en_translate/cli.py` | Add `--ollama-model`, update `--beam-size` default, update `--model` help |
| Modify | `src/bn_en_translate/training/trainer.py` | Rename to `Seq2SeqFineTuner`, accept `hf_model_id` |
| Modify | `scripts/finetune.py` | Add `--model` flag |
| Create | `tests/unit/test_model_config_v2.py` | Tests for new config fields |
| Create | `tests/unit/test_beam_defaults.py` | Tests for per-model beam defaults |
| Create | `tests/unit/test_madlad.py` | MADLAD translator unit tests |
| Create | `tests/unit/test_seamless.py` | SeamlessM4T translator unit tests |
| Create | `tests/unit/test_cli_ollama_model_flag.py` | `--ollama-model` CLI flag test |
| Modify | `docs/MODELS.md` | Add new models, Gemma 3, updated tables |
| Modify | `docs/ARCHITECTURE.md` | Document new config fields |
| Modify | `docs/TRAINING.md` | Document `--model` flag, `Seq2SeqFineTuner` |

---

## Task 1: Update ModelConfig — new inference fields

**Files:**
- Modify: `src/bn_en_translate/config.py`
- Test: `tests/unit/test_model_config_v2.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/unit/test_model_config_v2.py`:

```python
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
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_model_config_v2.py -v
```

Expected: FAIL — `ModelConfig` has no `use_flash_attention` or `max_ct2_batch_size`, and `beam_size` defaults to `4` not `None`.

- [ ] **Step 1.3: Update ModelConfig**

In `src/bn_en_translate/config.py`, replace the `ModelConfig` dataclass fields:

```python
@dataclass
class ModelConfig:
    """Configuration for the translation model."""

    model_name: str = "nllb-600M"
    model_path: str = "models/nllb-600M-ct2"
    device: str = "cuda"
    compute_type: str = "int8"
    src_lang: str = "ben_Beng"
    tgt_lang: str = "eng_Latn"
    beam_size: int | None = None          # None = use each translator's DEFAULT_BEAM_SIZE
    max_decoding_length: int = 512
    inference_batch_size: int = 8
    use_flash_attention: bool = True      # Flash Attention 2 if flash-attn is installed
    max_ct2_batch_size: int = 32          # CT2 translate_batch max_batch_size guard

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
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
pytest tests/unit/test_model_config_v2.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 1.5: Run the full suite to catch regressions**

```bash
make test
```

Expected: 186+ tests pass. If any existing test explicitly checks `beam_size == 4` default, update it to check `beam_size is None`.

- [ ] **Step 1.6: Commit**

```bash
git add src/bn_en_translate/config.py tests/unit/test_model_config_v2.py
git commit -m "feat(config): add beam_size=None, use_flash_attention, max_ct2_batch_size to ModelConfig"
```

---

## Task 2: Update TranslatorBase — DEFAULT_BEAM_SIZE and _effective_beam_size

**Files:**
- Modify: `src/bn_en_translate/models/base.py`
- Test: `tests/unit/test_beam_defaults.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/unit/test_beam_defaults.py`:

```python
"""Tests for per-model beam size defaults."""
from __future__ import annotations
from unittest.mock import MagicMock
from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


class _ConcreteTranslator(TranslatorBase):
    """Minimal concrete subclass for testing TranslatorBase directly."""
    DEFAULT_BEAM_SIZE = 7

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        return texts


def test_effective_beam_size_uses_config_when_set() -> None:
    cfg = ModelConfig(beam_size=3)
    t = _ConcreteTranslator(cfg)
    assert t._effective_beam_size() == 3


def test_effective_beam_size_uses_class_default_when_config_is_none() -> None:
    cfg = ModelConfig(beam_size=None)
    t = _ConcreteTranslator(cfg)
    assert t._effective_beam_size() == 7


def test_base_default_beam_size_is_4() -> None:
    # TranslatorBase.DEFAULT_BEAM_SIZE should be 4
    assert TranslatorBase.DEFAULT_BEAM_SIZE == 4
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_beam_defaults.py -v
```

Expected: FAIL — `TranslatorBase` has no `DEFAULT_BEAM_SIZE` or `_effective_beam_size`.

- [ ] **Step 2.3: Update TranslatorBase**

Replace the entire content of `src/bn_en_translate/models/base.py`:

```python
"""Abstract base class for all translation model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TranslatorBase(ABC):
    """
    Contract that all translator implementations must satisfy.

    Lifecycle:
        1. Instantiate with config.
        2. Call load() — downloads/loads model into memory.
        3. Call translate() one or more times.
        4. Call unload() to free GPU/CPU memory.
    """

    DEFAULT_BEAM_SIZE: int = 4
    """Per-model default beam size. Subclasses override this."""

    def __init__(self) -> None:
        self._loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model into memory (GPU or CPU)."""

    @abstractmethod
    def unload(self) -> None:
        """Free model from memory."""

    @abstractmethod
    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a list of texts. Called only when loaded."""

    def _effective_beam_size(self) -> int:
        """Return beam_size from config if explicitly set, else this model's DEFAULT_BEAM_SIZE."""
        config = getattr(self, "config", None)
        if config is not None and getattr(config, "beam_size", None) is not None:
            return config.beam_size  # type: ignore[union-attr]
        return self.DEFAULT_BEAM_SIZE

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """
        Translate texts from src_lang to tgt_lang.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        if not self._loaded:
            raise RuntimeError(
                "Model is not loaded. Call load() before translate()."
            )
        if not texts:
            return []
        return self._translate_batch(texts, src_lang, tgt_lang)

    def __enter__(self) -> "TranslatorBase":
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
pytest tests/unit/test_beam_defaults.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 2.5: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add src/bn_en_translate/models/base.py tests/unit/test_beam_defaults.py
git commit -m "feat(base): add DEFAULT_BEAM_SIZE and _effective_beam_size() to TranslatorBase"
```

---

## Task 3: Update NLLBCt2Translator — beam defaults + configurable batch size

**Files:**
- Modify: `src/bn_en_translate/models/nllb_ct2.py`

- [ ] **Step 3.1: Update NLLBCt2Translator**

In `src/bn_en_translate/models/nllb_ct2.py`, add `DEFAULT_BEAM_SIZE = 4` and replace the hardcoded `beam_size=self.config.beam_size` and `max_batch_size=32` calls in `_translate_batch`:

```python
class NLLBCt2Translator(TranslatorBase):
    """...(docstring unchanged)..."""

    DEFAULT_BEAM_SIZE: int = 4
```

In `_translate_batch`, replace:
```python
        results = self._translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self.config.beam_size,
            max_decoding_length=self.config.max_decoding_length,
            max_batch_size=32,
        )
```
with:
```python
        results = self._translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self._effective_beam_size(),
            max_decoding_length=self.config.max_decoding_length,
            max_batch_size=self.config.max_ct2_batch_size,
        )
```

- [ ] **Step 3.2: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 3.3: Commit**

```bash
git add src/bn_en_translate/models/nllb_ct2.py
git commit -m "feat(nllb-ct2): use _effective_beam_size() and configurable max_ct2_batch_size"
```

---

## Task 4: Update IndicTrans2Ct2Translator — beam defaults + configurable batch size

**Files:**
- Modify: `src/bn_en_translate/models/indicTrans2_ct2.py`

- [ ] **Step 4.1: Update IndicTrans2Ct2Translator**

In `src/bn_en_translate/models/indicTrans2_ct2.py`, add `DEFAULT_BEAM_SIZE = 5` to the class and update `_translate_batch`:

```python
class IndicTrans2Ct2Translator(TranslatorBase):
    """...(docstring unchanged)..."""

    HF_MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"
    SPM_FILENAME = "sentencepiece.bpe.model"
    DEFAULT_BEAM_SIZE: int = 5
```

In `_translate_batch`, replace:
```python
        results = self._translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self.config.beam_size,
            max_decoding_length=self.config.max_decoding_length,
        )
```
with:
```python
        results = self._translator.translate_batch(  # type: ignore[union-attr]
            tokenized,
            target_prefix=target_prefix,
            beam_size=self._effective_beam_size(),
            max_decoding_length=self.config.max_decoding_length,
            max_batch_size=self.config.max_ct2_batch_size,
        )
```

- [ ] **Step 4.2: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 4.3: Commit**

```bash
git add src/bn_en_translate/models/indicTrans2_ct2.py
git commit -m "feat(indictrans2-ct2): DEFAULT_BEAM_SIZE=5, configurable max_ct2_batch_size"
```

---

## Task 5: Update NLLBTranslator (HF) — beam defaults

**Files:**
- Modify: `src/bn_en_translate/models/nllb.py`

- [ ] **Step 5.1: Update NLLBTranslator**

In `src/bn_en_translate/models/nllb.py`, add `DEFAULT_BEAM_SIZE = 4` to the class and update `_translate_batch` to use `_effective_beam_size()`.

Add to class body after docstring:
```python
    DEFAULT_BEAM_SIZE: int = 4
```

In `_translate_batch`, replace `batch_size=self.config.inference_batch_size` call — that's already correct. The beam size is passed in the pipeline call; find where `beam_size` is referenced and replace with `self._effective_beam_size()`.

Looking at the current `_translate_batch`, beam size is passed via the pipeline `num_beams` field in HF pipeline — actually the HF `pipeline` call doesn't take `beam_size` in `_translate_batch`. The beam size for HF pipeline is set at pipeline init via `num_beams`. Update `load()` to use `_effective_beam_size()`:

In `load()`, the pipeline is constructed without explicit `num_beams`. Beam search is controlled at `pipeline(...)` level. Add `num_beams=self._effective_beam_size()` to the pipeline call:

Replace in `load()`:
```python
        self._pipeline = pipeline(
            "translation",
            model=model_id,
            device=device,
            src_lang=self.config.src_lang,
            tgt_lang=self.config.tgt_lang,
            max_length=self.config.max_decoding_length,
        )
```
with:
```python
        self._pipeline = pipeline(
            "translation",
            model=model_id,
            device=device,
            src_lang=self.config.src_lang,
            tgt_lang=self.config.tgt_lang,
            max_length=self.config.max_decoding_length,
            num_beams=self._effective_beam_size(),
        )
```

- [ ] **Step 5.2: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 5.3: Commit**

```bash
git add src/bn_en_translate/models/nllb.py
git commit -m "feat(nllb-hf): DEFAULT_BEAM_SIZE=4, use _effective_beam_size() in pipeline"
```

---

## Task 6: Update IndicTrans2Translator (HF) — beam defaults + Flash Attention 2

**Files:**
- Modify: `src/bn_en_translate/models/indicTrans2.py`

- [ ] **Step 6.1: Update IndicTrans2Translator**

Replace `src/bn_en_translate/models/indicTrans2.py` with:

```python
"""IndicTrans2 translator — best quality for Bengali → English."""

from __future__ import annotations
import importlib.util

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


class IndicTrans2Translator(TranslatorBase):
    """
    AI4Bharat IndicTrans2 translation model.

    Model: ai4bharat/indictrans2-indic-en-1B
    This is the recommended primary model for Bengali → English translation.

    Language codes: ben_Beng → eng_Latn (same FLORES-200 format as NLLB)

    Setup (one-time):
        pip install git+https://github.com/AI4Bharat/IndicTrans2.git#subdirectory=huggingface_interface
        # Then download the model:
        python scripts/download_models.py --model indicTrans2-1B

    VRAM usage (INT8 via CTranslate2): ~1.0–1.5 GB
    """

    HF_MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"
    DEFAULT_BEAM_SIZE: int = 5

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="indicTrans2-1B",
            model_path="models/indicTrans2-1B-ct2",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: object | None = None
        self._tokenizer: object | None = None

    def load(self) -> None:
        try:
            self._load_via_indictrans2_interface()
        except ImportError:
            self._load_via_transformers_fallback()
        self._loaded = True

    def _load_via_indictrans2_interface(self) -> None:
        from IndicTransToolkit import IndicProcessor  # type: ignore[import-untyped]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-untyped]

        attn_impl = (
            "flash_attention_2"
            if self.config.use_flash_attention and _flash_attn_available()
            else "eager"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True, attn_implementation=attn_impl
        )
        self._processor = IndicProcessor(inference=True)

        import torch  # type: ignore[import-untyped]

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model.to("cuda")  # type: ignore[union-attr]

    def _load_via_transformers_fallback(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-untyped]

        attn_impl = (
            "flash_attention_2"
            if self.config.use_flash_attention and _flash_attn_available()
            else "eager"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.HF_MODEL_ID, attn_implementation=attn_impl
        )
        self._processor = None

        import torch  # type: ignore[import-untyped]

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model.to("cuda")  # type: ignore[union-attr]

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch  # type: ignore[import-untyped]

        device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"

        if self._processor is not None:
            batch = self._processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = self._tokenizer(  # type: ignore[operator]
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
        else:
            inputs = self._tokenizer(  # type: ignore[operator]
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

        with torch.no_grad():
            generated = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                num_beams=self._effective_beam_size(),
                num_return_sequences=1,
                max_length=self.config.max_decoding_length,
            )

        decoded = self._tokenizer.batch_decode(  # type: ignore[union-attr]
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if self._processor is not None:
            decoded = self._processor.postprocess_batch(decoded, lang=tgt_lang)

        return decoded
```

- [ ] **Step 6.2: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 6.3: Commit**

```bash
git add src/bn_en_translate/models/indicTrans2.py
git commit -m "feat(indictrans2-hf): DEFAULT_BEAM_SIZE=5, Flash Attention 2 support"
```

---

## Task 7: Add MADLAD-400-3B translator

**Files:**
- Create: `src/bn_en_translate/models/madlad.py`
- Test: `tests/unit/test_madlad.py`

- [ ] **Step 7.1: Write failing tests**

Create `tests/unit/test_madlad.py`:

```python
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
    """_build_input_texts() must prepend '<2en> ' to each source text."""
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
```

- [ ] **Step 7.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_madlad.py -v
```

Expected: FAIL — `madlad` module doesn't exist yet.

- [ ] **Step 7.3: Create MADLADTranslator**

Create `src/bn_en_translate/models/madlad.py`:

```python
"""MADLAD-400-3B translator — Google's dedicated multilingual MT model."""

from __future__ import annotations
import importlib.util

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


# Map FLORES-200 language codes to MADLAD-400 target tags
_MADLAD_LANG_TAG: dict[str, str] = {
    "eng_Latn": "<2en>",
    "ben_Beng": "<2bn>",
    "hin_Deva": "<2hi>",
}


class MADLADTranslator(TranslatorBase):
    """
    Google MADLAD-400-3B translation model via HuggingFace Transformers.

    Architecture: T5-based encoder-decoder
    HF ID: google/madlad400-3b-mt
    VRAM (float16): ~3 GB
    FLORES-200 bn→en BLEU: ~36 (zero-shot)

    Source text is prefixed with the target language tag, e.g. '<2en> <bengali text>'.
    No source language tag is required.

    Setup:
        python scripts/download_models.py --model madlad-3b
    """

    HF_MODEL_ID = "google/madlad400-3b-mt"
    DEFAULT_BEAM_SIZE: int = 4

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="madlad-3b",
            model_path="models/madlad-3b-hf",
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: object | None = None
        self._tokenizer: object | None = None

    def load(self) -> None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore[import-untyped]

        model_id = self.config.model_path if (
            self.config.model_path and self.config.model_path != "models/madlad-3b-hf"
        ) else self.HF_MODEL_ID

        attn_impl = (
            "flash_attention_2"
            if self.config.use_flash_attention and _flash_attn_available()
            else "eager"
        )

        self._tokenizer = T5Tokenizer.from_pretrained(model_id)
        self._model = T5ForConditionalGeneration.from_pretrained(
            model_id, attn_implementation=attn_impl
        )

        import torch  # type: ignore[import-untyped]

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model.to("cuda")  # type: ignore[union-attr]

        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _build_input_texts(self, texts: list[str], tgt_lang: str) -> list[str]:
        """Prefix each text with the MADLAD-400 target language tag."""
        tag = _MADLAD_LANG_TAG.get(tgt_lang, "<2en>")
        return [f"{tag} {t}" for t in texts]

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch  # type: ignore[import-untyped]

        device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"

        input_texts = self._build_input_texts(texts, tgt_lang)
        inputs = self._tokenizer(  # type: ignore[operator]
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            generated = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                num_beams=self._effective_beam_size(),
                max_length=self.config.max_decoding_length,
            )

        return self._tokenizer.batch_decode(  # type: ignore[union-attr]
            generated, skip_special_tokens=True
        )
```

- [ ] **Step 7.4: Run tests**

```bash
pytest tests/unit/test_madlad.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 7.5: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 7.6: Commit**

```bash
git add src/bn_en_translate/models/madlad.py tests/unit/test_madlad.py
git commit -m "feat(madlad): add MADLAD-400-3B translator (Google T5-based multilingual MT)"
```

---

## Task 8: Add SeamlessM4T-v2 translator

**Files:**
- Create: `src/bn_en_translate/models/seamless.py`
- Test: `tests/unit/test_seamless.py`

- [ ] **Step 8.1: Write failing tests**

Create `tests/unit/test_seamless.py`:

```python
"""Unit tests for SeamlessM4T-v2 translator."""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from bn_en_translate.config import ModelConfig


def test_seamless_import() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    assert SeamlessTranslator is not None


def test_seamless_default_config() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    assert t.config.model_name == "seamless-medium"
    assert t.config.src_lang == "ben_Beng"
    assert t.config.tgt_lang == "eng_Latn"


def test_seamless_default_beam_size() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    assert t.DEFAULT_BEAM_SIZE == 5


def test_seamless_translate_raises_before_load() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    with pytest.raises(RuntimeError, match="not loaded"):
        t.translate(["test"], "ben_Beng", "eng_Latn")


def test_seamless_flores_to_seamless_lang_code() -> None:
    """FLORES-200 codes must be mapped to SeamlessM4T's short codes."""
    from bn_en_translate.models.seamless import SeamlessTranslator, _to_seamless_lang
    assert _to_seamless_lang("ben_Beng") == "ben"
    assert _to_seamless_lang("eng_Latn") == "eng"
    assert _to_seamless_lang("hin_Deva") == "hin"


def test_seamless_empty_input_returns_empty() -> None:
    from bn_en_translate.models.seamless import SeamlessTranslator
    t = SeamlessTranslator()
    t._loaded = True
    t._model = MagicMock()
    t._processor = MagicMock()
    result = t.translate([], "ben_Beng", "eng_Latn")
    assert result == []
```

- [ ] **Step 8.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_seamless.py -v
```

Expected: FAIL — `seamless` module doesn't exist yet.

- [ ] **Step 8.3: Create SeamlessTranslator**

Create `src/bn_en_translate/models/seamless.py`:

```python
"""SeamlessM4T-v2 translator — Meta's multilingual text translation model."""

from __future__ import annotations

from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase

# Map FLORES-200 codes → SeamlessM4T short codes
_SEAMLESS_LANG_MAP: dict[str, str] = {
    "ben_Beng": "ben",
    "eng_Latn": "eng",
    "hin_Deva": "hin",
    "urd_Arab": "urd",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
}


def _to_seamless_lang(flores_code: str) -> str:
    """Convert FLORES-200 language code to SeamlessM4T short code."""
    return _SEAMLESS_LANG_MAP.get(flores_code, flores_code.split("_")[0])


class SeamlessTranslator(TranslatorBase):
    """
    Meta SeamlessM4T-v2 translation model (text-only mode).

    Architecture: Custom encoder-decoder (SeamlessM4Tv2ForTextToText)
    HF ID: facebook/seamless-m4t-v2-large
    VRAM (float16): ~3.5 GB
    FLORES-200 bn→en BLEU: ~38–40

    Note: SeamlessM4T uses its own short language codes (e.g. 'ben', 'eng')
    rather than FLORES-200 codes. This class handles the mapping automatically.

    Setup:
        python scripts/download_models.py --model seamless-medium
    """

    HF_MODEL_ID = "facebook/seamless-m4t-v2-large"
    DEFAULT_BEAM_SIZE: int = 5

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(
            model_name="seamless-medium",
            model_path="",  # HF cache only — no CT2 conversion
            src_lang="ben_Beng",
            tgt_lang="eng_Latn",
        )
        self._model: object | None = None
        self._processor: object | None = None

    def load(self) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText  # type: ignore[import-untyped]

        model_id = self.HF_MODEL_ID

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = SeamlessM4Tv2ForTextToText.from_pretrained(model_id)

        import torch  # type: ignore[import-untyped]

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model.to("cuda")  # type: ignore[union-attr]

        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        import torch  # type: ignore[import-untyped]

        device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"

        seamless_src = _to_seamless_lang(src_lang)
        seamless_tgt = _to_seamless_lang(tgt_lang)

        inputs = self._processor(  # type: ignore[operator]
            text=texts,
            src_lang=seamless_src,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_tokens = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                tgt_lang=seamless_tgt,
                generate_speech=False,
                num_beams=self._effective_beam_size(),
                max_new_tokens=self.config.max_decoding_length,
            )

        return self._processor.batch_decode(  # type: ignore[union-attr]
            output_tokens, skip_special_tokens=True
        )
```

- [ ] **Step 8.4: Run tests**

```bash
pytest tests/unit/test_seamless.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 8.5: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 8.6: Commit**

```bash
git add src/bn_en_translate/models/seamless.py tests/unit/test_seamless.py
git commit -m "feat(seamless): add SeamlessM4T-v2 translator (Meta text-to-text)"
```

---

## Task 9: Register new models in factory + update download_models.py

**Files:**
- Modify: `src/bn_en_translate/models/factory.py`
- Modify: `scripts/download_models.py`

- [ ] **Step 9.1: Register new models in factory.py**

In `src/bn_en_translate/models/factory.py`, add two new factory functions after the `_make_indictrans2` block:

```python
@register_model("madlad-3b")
@register_model("madlad")
def _make_madlad(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.madlad import MADLADTranslator
    return MADLADTranslator(config.model)


@register_model("seamless-medium")
@register_model("seamless")
def _make_seamless(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.seamless import SeamlessTranslator
    return SeamlessTranslator(config.model)
```

Also update the docstring in `get_translator` to include the new model names:
```
      - "madlad-3b"       -> MADLADTranslator (Google MADLAD-400-3B)
      - "seamless-medium" -> SeamlessTranslator (Meta SeamlessM4T-v2)
```

- [ ] **Step 9.2: Update download_models.py**

Replace the `MODELS` dict in `scripts/download_models.py` with:

```python
MODELS: dict[str, dict[str, str]] = {
    "nllb-600M": {
        "hf_id": "facebook/nllb-200-distilled-600M",
        "output_dir": "models/nllb-600M-ct2",
        "quantization": "float16",
        "type": "nllb",
    },
    "nllb-1.3B": {
        "hf_id": "facebook/nllb-200-distilled-1.3B",
        "output_dir": "models/nllb-1.3B-ct2",
        "quantization": "float16",
        "type": "nllb",
    },
    "indicTrans2-1B": {
        "hf_id": "ai4bharat/indictrans2-indic-en-1B",
        "output_dir": "models/indicTrans2-1B-ct2",
        "quantization": "float16",
        "type": "indictrans2",
    },
    "madlad-3b": {
        "hf_id": "google/madlad400-3b-mt",
        "output_dir": "models/madlad-3b-hf",
        "quantization": "float16",
        "type": "hf_only",   # T5 is CT2-compatible but run via HF for simplicity
    },
    "seamless-medium": {
        "hf_id": "facebook/seamless-m4t-v2-large",
        "output_dir": "models/seamless-medium-hf",
        "quantization": "float16",
        "type": "hf_only",   # custom architecture, no CT2 conversion
    },
}
```

Add a new branch in `download_and_convert` for `hf_only` type (just snapshot_download, no CT2 conversion):

After the existing `ct2` conversion block, add:

```python
    model_type = cfg.get("type", "nllb")

    if model_type == "hf_only":
        # HuggingFace-native models: pre-download to local cache
        print(f"Pre-downloading {hf_id} to HF cache (no CT2 conversion)...")
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
            local = snapshot_download(hf_id, local_dir=str(output_dir))
            print(f"Done. Model saved to: {local}")
        except Exception as e:
            print(f"ERROR: download failed: {e}")
            sys.exit(1)
        return
```

This should be inserted before the `cmd = [...]` subprocess block, guarded by `if model_type == "hf_only": ... return`.

Also update `argparse` choices to include new model names:
```python
parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
```
(This is already dynamic from `MODELS.keys()` so no change needed.)

- [ ] **Step 9.3: Run full suite**

```bash
make test
```

Expected: all tests pass (no regression from factory changes since new models are lazily imported).

- [ ] **Step 9.4: Commit**

```bash
git add src/bn_en_translate/models/factory.py scripts/download_models.py
git commit -m "feat(factory): register madlad-3b and seamless-medium; update download_models.py"
```

---

## Task 10: Update CLI — --ollama-model flag, --beam-size default=None

**Files:**
- Modify: `src/bn_en_translate/cli.py`
- Test: `tests/unit/test_cli_ollama_model_flag.py`

- [ ] **Step 10.1: Write failing tests**

Create `tests/unit/test_cli_ollama_model_flag.py`:

```python
"""Tests for --ollama-model CLI flag."""
from __future__ import annotations
from click.testing import CliRunner
from unittest.mock import MagicMock, patch
from bn_en_translate.cli import main


def _mock_pipeline_config(captured: list) -> type:
    """Returns a PipelineConfig class that captures constructor args."""
    from bn_en_translate.config import PipelineConfig

    original_init = PipelineConfig.__init__

    def capturing_init(self, **kwargs):
        captured.append(kwargs)
        original_init(self, **kwargs)

    return capturing_init


def test_ollama_model_flag_sets_config() -> None:
    runner = CliRunner()
    captured_configs = []

    with runner.isolated_filesystem():
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write("আমি ভাত খাই।")

        with patch("bn_en_translate.cli.get_translator") as mock_get, \
             patch("bn_en_translate.cli.TranslationPipeline") as mock_pipe:

            mock_translator = MagicMock()
            mock_translator.__enter__ = lambda s: s
            mock_translator.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_translator

            mock_pipeline_instance = MagicMock()
            mock_pipe.return_value = mock_pipeline_instance

            def capture_config(config):
                captured_configs.append(config)
                return mock_translator

            mock_get.side_effect = capture_config

            result = runner.invoke(main, [
                "--input", "input.txt",
                "--output", "out.txt",
                "--ollama-model", "gemma3:12b",
            ])

    assert result.exit_code == 0, result.output
    assert len(captured_configs) == 1
    assert captured_configs[0].ollama_model == "gemma3:12b"


def test_ollama_model_flag_defaults_to_gemma3() -> None:
    runner = CliRunner()
    captured_configs = []

    with runner.isolated_filesystem():
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write("আমি ভাত খাই।")

        with patch("bn_en_translate.cli.get_translator") as mock_get, \
             patch("bn_en_translate.cli.TranslationPipeline"):

            mock_translator = MagicMock()
            mock_translator.__enter__ = lambda s: s
            mock_translator.__exit__ = MagicMock(return_value=False)

            def capture_config(config):
                captured_configs.append(config)
                return mock_translator

            mock_get.side_effect = capture_config

            result = runner.invoke(main, [
                "--input", "input.txt",
                "--output", "out.txt",
            ])

    assert result.exit_code == 0, result.output
    assert captured_configs[0].ollama_model == "gemma3:12b"
```

- [ ] **Step 10.2: Run tests to verify they fail**

```bash
pytest tests/unit/test_cli_ollama_model_flag.py -v
```

Expected: FAIL — no `--ollama-model` flag exists and default is still `qwen2.5:7b-instruct-q4_K_M`.

- [ ] **Step 10.3: Update cli.py**

Replace `src/bn_en_translate/cli.py` with:

```python
"""Command-line interface for Bengali → English story translation."""

from __future__ import annotations

import click

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.factory import get_translator
from bn_en_translate.pipeline.pipeline import TranslationPipeline
from bn_en_translate.utils.cuda_check import get_best_device


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to Bengali story file (.txt)")
@click.option("--output", "-o", "output_path", required=True, help="Path for English output file")
@click.option(
    "--model",
    "-m",
    default="nllb-600M",
    show_default=True,
    help=(
        "Translation model: nllb-600M | nllb-1.3B | indicTrans2-1B | "
        "madlad-3b | seamless-medium | ollama"
    ),
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Device: cuda | cpu | auto (auto picks cuda if available)",
)
@click.option("--batch-size", default=8, show_default=True, help="Translation batch size")
@click.option(
    "--beam-size",
    default=None,
    type=int,
    help="Beam search width (default: model-specific; NLLB=4, IndicTrans2/SeamlessM4T=5)",
)
@click.option("--ollama-polish", is_flag=True, default=False, help="Run Ollama polishing pass after translation")
@click.option(
    "--ollama-model",
    default="gemma3:12b",
    show_default=True,
    help="Ollama model tag for the polish pass (e.g. gemma3:4b, gemma3:12b, qwen2.5:7b-instruct-q4_K_M)",
)
def main(
    input_path: str,
    output_path: str,
    model: str,
    device: str,
    batch_size: int,
    beam_size: int | None,
    ollama_polish: bool,
    ollama_model: str,
) -> None:
    """Translate a Bengali story file to English using a local open-source model."""

    resolved_device = get_best_device() if device == "auto" else device
    click.echo(f"Device: {resolved_device} | Model: {model}")

    config = PipelineConfig(
        model=ModelConfig(
            model_name=model,
            device=resolved_device,
            beam_size=beam_size,
        ),
        ollama_polish=ollama_polish,
        ollama_model=ollama_model,
    )

    translator = get_translator(config)
    pipeline = TranslationPipeline(translator, config)

    click.echo(f"Loading model '{model}'...")
    with translator:
        click.echo(f"Translating '{input_path}'...")
        pipeline.translate_file(input_path, output_path)

    click.echo(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.4: Run tests**

```bash
pytest tests/unit/test_cli_ollama_model_flag.py -v
```

Expected: both tests PASS.

- [ ] **Step 10.5: Run full suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 10.6: Commit**

```bash
git add src/bn_en_translate/cli.py tests/unit/test_cli_ollama_model_flag.py
git commit -m "feat(cli): add --ollama-model flag, beam-size=None default, new model names in help"
```

---

## Task 11: Generalize fine-tuner — Seq2SeqFineTuner with --model flag

**Files:**
- Modify: `src/bn_en_translate/training/trainer.py`
- Modify: `scripts/finetune.py`

- [ ] **Step 11.1: Read current trainer.py to understand the class structure**

```bash
head -80 src/bn_en_translate/training/trainer.py
```

- [ ] **Step 11.2: Rename NLLBFineTuner to Seq2SeqFineTuner**

The trainer already uses `model_config.model_path` as the HuggingFace model ID in `load()` (line ~76: `model_id = self.model_config.model_path`). No new parameter is needed — `finetune.py` simply passes the right HF ID via `model_config.model_path`.

In `src/bn_en_translate/training/trainer.py`:

1. Change `class NLLBFineTuner:` → `class Seq2SeqFineTuner:`
2. Update the module docstring (line 1–9): replace `NLLBFineTuner` with `Seq2SeqFineTuner`
3. Add a backwards-compatibility alias at the bottom of the file:

```python
# Backwards compatibility alias — do not remove until all call sites are updated
NLLBFineTuner = Seq2SeqFineTuner
```

That's all. The logic inside is already model-agnostic.

- [ ] **Step 11.3: Update scripts/finetune.py — add --model flag**

In `scripts/finetune.py`, update the constants and argparse:

```python
MODEL_HF_IDS: dict[str, str] = {
    "nllb-600M": "facebook/nllb-200-distilled-600M",
    "madlad-3b": "google/madlad400-3b-mt",
}

MODEL_CT2_OUTPUTS: dict[str, str] = {
    "nllb-600M": "models/nllb-600M-finetuned-ct2",
    "madlad-3b": "models/madlad-3b-finetuned-ct2",
}
```

In `main()`:
```python
parser.add_argument(
    "--model",
    default="nllb-600M",
    choices=list(MODEL_HF_IDS.keys()),
    help="Base model to fine-tune (default: nllb-600M)",
)
```

And use `args.model` to look up `hf_model_id` and `ct2_output`, then pass `hf_model_id` via `model_config.model_path` (the field the trainer already reads as the HF model ID):

```python
from bn_en_translate.config import ModelConfig
from bn_en_translate.training.trainer import Seq2SeqFineTuner

hf_model_id = MODEL_HF_IDS[args.model]
ct2_output = MODEL_CT2_OUTPUTS[args.model]

# model_config.model_path is what trainer.load() uses as the HuggingFace model ID
model_cfg = ModelConfig(model_name=args.model, model_path=hf_model_id)
finetuner = Seq2SeqFineTuner(model_config=model_cfg, finetune_config=ft_config)
```

- [ ] **Step 11.4: Run full suite**

```bash
make test
```

Expected: all tests pass. The `NLLBFineTuner` alias ensures existing tests still work.

- [ ] **Step 11.5: Commit**

```bash
git add src/bn_en_translate/training/trainer.py scripts/finetune.py
git commit -m "feat(trainer): rename to Seq2SeqFineTuner, add hf_model_id param; finetune.py --model flag"
```

---

## Task 12: Update docs

**Files:**
- Modify: `docs/MODELS.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/TRAINING.md`

- [ ] **Step 12.1: Update docs/MODELS.md**

Replace the supported models table at the top with:

```markdown
| Model | HuggingFace ID | CT2 Dir | VRAM (float16) | BLEU bn→en | Speed |
|-------|---------------|---------|---------------|------------|-------|
| `nllb-600M` | `facebook/nllb-200-distilled-600M` | `models/nllb-600M-ct2/` | ~2.0 GB | ~64 (in-domain), ~22 (FLORES) | ~1000 chars/s |
| `nllb-1.3B` | `facebook/nllb-200-distilled-1.3B` | `models/nllb-1.3B-ct2/` | ~2.6 GB | ~26 (FLORES) | ~700 chars/s |
| `indicTrans2-1B` | `ai4bharat/indictrans2-indic-en-1B` | `models/indicTrans2-1B-ct2/` | ~3.0 GB | ~44 (FLORES) | ~700 chars/s |
| `madlad-3b` | `google/madlad400-3b-mt` | HF only (`models/madlad-3b-hf/`) | ~3.0 GB | ~36 (FLORES) | ~500 chars/s |
| `seamless-medium` | `facebook/seamless-m4t-v2-large` | HF only (`models/seamless-medium-hf/`) | ~3.5 GB | ~38–40 (FLORES) | ~500 chars/s |
| `ollama` | N/A (local daemon) | N/A | ~2.8–7.3 GB | subjective | ~200 chars/s |
```

Add a new **Ollama Polish Models** subsection under the Ollama section:

```markdown
## Ollama Polish Models

| Model | Ollama tag | VRAM (q4_K_M) | Notes |
|-------|-----------|---------------|-------|
| Gemma 3 4B | `gemma3:4b` | ~2.8 GB | Fast, strong multilingual |
| Gemma 3 12B | `gemma3:12b` | ~7.3 GB | **Recommended** — best literary quality |
| TowerInstruct 7B | `tower:7b` *(verify tag)* | ~4.5 GB | Translation-specialized |
| Qwen 2.5 7B | `qwen2.5:7b-instruct-q4_K_M` | ~4.7 GB | Previous default |

Select at runtime: `bn-translate --ollama-polish --ollama-model gemma3:12b ...`
```

Add **MADLAD-400** and **SeamlessM4T-v2** sections following the IndicTrans2 section, describing architecture, language codes, and download commands.

- [ ] **Step 12.2: Update docs/ARCHITECTURE.md**

Add a **Inference Configuration** subsection noting the new `ModelConfig` fields:

```markdown
## Inference Configuration

`ModelConfig` now supports:

| Field | Default | Purpose |
|-------|---------|---------|
| `beam_size` | `None` | Beam width; `None` uses the translator's `DEFAULT_BEAM_SIZE` |
| `use_flash_attention` | `True` | Enables Flash Attention 2 for HF models if `flash-attn` is installed |
| `max_ct2_batch_size` | `32` | CT2 `translate_batch()` max_batch_size guard (prevents CUDA OOM) |

Per-model beam defaults: NLLB-600M=4, NLLB-1.3B=5, IndicTrans2-1B=5, MADLAD-3B=4, SeamlessM4T-medium=5.
```

- [ ] **Step 12.3: Update docs/TRAINING.md**

Add a **Multi-model Fine-Tuning** subsection:

```markdown
## Multi-model Fine-Tuning

`NLLBFineTuner` has been renamed `Seq2SeqFineTuner` and generalized to support any seq2seq model.
`NLLBFineTuner` remains as a backwards-compatible alias.

```bash
# Fine-tune NLLB-600M (default)
python scripts/finetune.py

# Fine-tune MADLAD-400-3B
python scripts/finetune.py --model madlad-3b

# Fine-tune with custom epochs and learning rate
python scripts/finetune.py --model madlad-3b --epochs 5 --lr 1e-4
```

Supported models: `nllb-600M`, `madlad-3b`.
```

- [ ] **Step 12.4: Commit**

```bash
git add docs/MODELS.md docs/ARCHITECTURE.md docs/TRAINING.md
git commit -m "docs: update MODELS, ARCHITECTURE, TRAINING for model expansion"
```

---

## Final Verification

- [ ] **Run the full test suite one last time**

```bash
make test
make lint
make typecheck
```

Expected: all tests pass, no lint errors, no mypy errors.

- [ ] **Smoke test the CLI with a new model key (no GPU download required)**

```bash
python -c "
from bn_en_translate.models.factory import get_translator
from bn_en_translate.config import PipelineConfig, ModelConfig
# Just verify the factory resolves without error
cfg = PipelineConfig(model=ModelConfig(model_name='madlad-3b'))
print('madlad-3b factory: OK')
cfg2 = PipelineConfig(model=ModelConfig(model_name='seamless-medium'))
print('seamless-medium factory: OK')
"
```

Expected: both print "OK" without raising.
