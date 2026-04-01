# Development Guide

## Environment Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.x (optional — CPU fallback works, ~20× slower)
- WSL2 on Windows, or native Linux

### First-time Setup

```bash
# 1. Create venv
#    Note: use --without-pip if ensurepip is unavailable (Debian/Ubuntu missing python3.12-venv)
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3

# 2. Activate
source .venv/bin/activate

# 3. WSL2 only — CUDA driver path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
# Add to ~/.bashrc to make permanent

# 4. Install PyTorch (cu124 = CUDA 12.4, works on CUDA 12.x drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 5. Install project + dev deps
pip install -e ".[dev]"

# 6. Verify
make test     # 99 tests should pass in ~0.5s
```

### Verify GPU

```bash
python3 -c "
import torch, ctranslate2
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('CTranslate2:', ctranslate2.__version__, '| CUDA devices:', ctranslate2.get_cuda_device_count())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), '| Compute:', torch.cuda.get_device_capability(0))
"
```

---

## Daily Workflow

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

make test           # always run before committing
make lint           # ruff check src/ tests/
make typecheck      # mypy strict
```

---

## Test Tiers

| Tier | Command | Time | Requires |
|------|---------|------|----------|
| Unit + mock integration | `make test` | ~0.5s | nothing |
| Real NLLB-600M | `make test-slow` | ~30s | `models/nllb-600M-ct2/` |
| E2E quality (BLEU) | `make test-e2e` | ~60s+ | GPU + IndicTrans2 model |
| All | `make test-all` | ~90s+ | GPU + all models |

Run a single test file:
```bash
pytest tests/unit/test_chunker.py -v
pytest -k "test_normalize" -v
```

Run with coverage:
```bash
pytest --cov=bn_en_translate --cov-report=term-missing tests/unit/
```

### Test Structure

```
tests/
├── conftest.py                    # MockTranslator fixture, shared config
├── unit/                          # Pure logic, no GPU, no downloads
│   ├── test_chunker.py
│   ├── test_preprocessor.py
│   ├── test_postprocessor.py
│   ├── test_text_utils.py
│   ├── test_file_io.py
│   ├── test_config.py
│   ├── test_cuda_check.py
│   └── test_model_interface.py
├── integration/
│   ├── test_pipeline_mock.py      # Full pipeline, MockTranslator
│   └── test_pipeline_nllb.py      # Real NLLB-600M (marked slow)
└── e2e/
    ├── test_indicTrans2_quality.py # BLEU ≥ 25 assertion (marked e2e)
    └── test_ollama_integration.py  # Ollama API (marked e2e)
```

### Mocking CTranslate2 in Tests

```python
def test_something(mocker):
    mocker.patch("ctranslate2.get_cuda_device_count", return_value=0)  # force CPU path
    mocker.patch("ctranslate2.Translator")                              # skip model load
    mocker.patch("sentencepiece.SentencePieceProcessor")
```

---

## Code Style

All enforced by CI (ruff + mypy):

```bash
make lint       # ruff check src/ tests/ — fixes auto with `ruff check --fix`
make typecheck  # mypy --strict src/bn_en_translate/
```

Rules:
- `from __future__ import annotations` at the top of every source file
- Line length 100 chars
- Python 3.12 target
- Strict mypy — explicit return types, no implicit `Any`
- No `print()` in library code (`src/`)
- No bare `except:` — catch specific exceptions

---

## Adding a New Model Backend

Follow the CT2-first pattern:

```python
# src/bn_en_translate/models/mymodel_ct2.py
from __future__ import annotations
from pathlib import Path
from bn_en_translate.config import ModelConfig
from bn_en_translate.models.base import TranslatorBase

class MyModelCt2Translator(TranslatorBase):
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig(model_name="my-model", model_path="models/my-model-ct2")
        self._translator: object | None = None
        self._sp: object | None = None

    def load(self) -> None:
        import ctranslate2, sentencepiece as spm
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_path / "sentencepiece.bpe.model"))
        compute_type = self._best_compute_type(device, self._sp)   # probes with 20-token sentence
        self._translator = ctranslate2.Translator(str(model_path), device=device, compute_type=compute_type)
        self._loaded = True

    def unload(self) -> None:
        self._translator = None
        self._sp = None
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self._translator is not None and self._sp is not None
        # M2M/NLLB source format: tokens + [</s>, src_lang]
        tokenized = [self._sp.encode(t, out_type=str) + ["</s>", src_lang] for t in texts]
        results = self._translator.translate_batch(
            tokenized, target_prefix=[[tgt_lang]] * len(tokenized),
            beam_size=self.config.beam_size, max_decoding_length=self.config.max_decoding_length
        )
        return [self._sp.decode(r.hypotheses[0][1:] if r.hypotheses[0][0] == tgt_lang else r.hypotheses[0])
                for r in results]

    def _best_compute_type(self, device: str, sp: object) -> str:
        # See src/bn_en_translate/models/nllb_ct2.py for full implementation
        ...
```

Then register in `factory.py` and `scripts/download_models.py`.

---

## Downloading and Converting Models

```bash
# Download + convert NLLB-600M to CTranslate2 float16 (~594 MB)
python scripts/download_models.py --model nllb-600M

# Download + convert IndicTrans2-1B (~3 GB) — best Bengali quality
python scripts/download_models.py --model indicTrans2-1B

# Re-convert (overwrite)
python scripts/download_models.py --model nllb-600M --force
```

The script uses `ct2-transformers-converter` and copies the SentencePiece tokenizer into the output directory. Converted models go to `models/<name>-ct2/` (gitignored).

---

## Benchmark

```bash
# Quick smoke test (20 sentences, NLLB-600M)
python scripts/benchmark.py --models nllb-600M --sentences 20

# Full comparison
python scripts/benchmark.py --models nllb-600M nllb-1.3B indicTrans2-1B --sentences 100

# CPU only
python scripts/benchmark.py --models nllb-600M --device cpu --sentences 10
```

Output columns: Model, Backend class, BLEU, Time, chars/sec, VRAM used.

---

## Corpus

```bash
python scripts/get_corpus.py          # generates/downloads to corpus/
python scripts/get_corpus.py --force  # re-generate
```

The built-in corpus has 90 sentence pairs across 10 domains (literature, history, geography, science, education, health, everyday life, agriculture, economy, proverbs).

For FLORES-200 (1012 sentences): `pip install datasets` — the script picks it up automatically.

---

## Common Pitfalls

See [`docs/HARDWARE.md`](HARDWARE.md) for GPU-specific issues.

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUBLAS_STATUS_NOT_SUPPORTED` | INT8 ops on sm_120 + CUDA 12.4 | Use float16; `_best_compute_type()` handles this |
| "Ra Ra Ra Ra..." output | Wrong NLLB tokenisation order | Source must be `tokens + [</s>, src_lang]` |
| `Bus error` on `import torch` | PyTorch nightly on AMD CPU (AMX) | Use `2.6.0+cu124` stable |
| Corrupt torch install | Concurrent pip processes | Kill all pip, `rm -rf .venv/lib/.../torch`, reinstall |
| `ModuleNotFoundError: torch` after install | Venv not activated | `source .venv/bin/activate` |
| Slow model downloads | HF_HOME on NTFS `/mnt/c/...` | Set `HF_HOME` to Linux filesystem path |
