# Development Guide

## Environment Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.x (optional — CPU fallback works, ~20x slower)
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

# 4. Install PyTorch (cu128 = CUDA 12.8, required for sm_120 GPU training kernels)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 5. Install project + dev deps
pip install -e ".[dev]"

# 6. Verify
make test     # 186 tests should pass in ~12s
```

> **Note:** Use `cu128` (not `cu124`). PyTorch `2.7.0+cu128` ships compiled sm_120 (Blackwell) kernels required for GPU fine-tuning. `cu124` falls back to CPU for training operations on sm_120.

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

Expected output on the RTX 5050 target machine:

```
PyTorch: 2.7.0+cu128 | CUDA: True
CTranslate2: 4.7.1 | devices: 1
GPU: NVIDIA GeForce RTX 5050 Laptop GPU | Compute: (12, 0)
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
| Unit + mock integration | `make test` | ~12s | nothing |
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
│   ├── test_finetune_config.py    # FineTuneConfig validation
│   ├── test_cuda_check.py
│   ├── test_model_interface.py
│   ├── test_corpus_utils.py       # Corpus split / filter / load/save
│   ├── test_trainer.py            # NLLBFineTuner (all mocked)
│   └── test_training_dataset.py   # BengaliEnglishDataset
├── integration/
│   ├── test_pipeline_mock.py      # Full pipeline, MockTranslator
│   └── test_pipeline_nllb.py      # Real NLLB-600M (marked slow)
└── e2e/
    ├── test_indicTrans2_quality.py # BLEU >= 25 assertion (marked e2e)
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
# Quick smoke test (5 sentences, NLLB-600M) — also validates GPU
python scripts/benchmark.py --models nllb-600M --sentences 5

# Standard comparison (20 sentences)
python scripts/benchmark.py --models nllb-600M --sentences 20

# Full comparison across models
python scripts/benchmark.py --models nllb-600M nllb-1.3B indicTrans2-1B --sentences 100

# CPU only
python scripts/benchmark.py --models nllb-600M --device cpu --sentences 10
```

Output columns: Model, Backend class, BLEU, Time, chars/sec, VRAM used.

Current baseline BLEU on 90-sentence built-in corpus: **56.2** (NLLB-600M CT2 float16).

---

## Corpus

```bash
python scripts/get_corpus.py          # generates/downloads built-in 90-sentence corpus
python scripts/get_corpus.py --force  # re-generate

# Download Samanantar (8.5M Bengali-English pairs, sample 10K)
python scripts/download_corpus.py              # 10 000 pairs → corpus/samanantar/
python scripts/download_corpus.py --size 50000 # larger sample
python scripts/download_corpus.py --verify     # check alignment without re-downloading
```

### Corpus Structure

```
corpus/
├── flores200_devtest.bn.txt     # 90 Bengali sentences (built-in, 10 domains)
├── flores200_devtest.en.txt     # 90 English references
└── samanantar/                  # Downloaded from ai4bharat/samanantar
    ├── train.bn.txt / train.en.txt  # ~7 863 pairs (80%)
    ├── val.bn.txt   / val.en.txt    # ~982 pairs  (10%)
    └── test.bn.txt  / test.en.txt   # ~984 pairs  (10%)
```

---

## Running Fine-tuning

Fine-tune NLLB-600M with LoRA (parameter-efficient, ~0.76% trainable params). See [`docs/TRAINING.md`](TRAINING.md) for full details.

### Install training dependencies

```bash
pip install -e ".[train]"
# or individually:
pip install "peft>=0.12.0" "datasets>=3.0.0" "accelerate>=0.34.0"
```

### Download training corpus

```bash
python scripts/download_corpus.py   # ~10 000 pairs, takes ~30s
```

### Run fine-tuning

```bash
# Full GPU run (3 epochs on all ~7 863 training pairs, ~20-30 min on RTX 5050)
python scripts/finetune.py

# Quick smoke test (CPU, 500 pairs, 1 epoch)
python scripts/finetune.py --epochs 1 --max-train-pairs 500 --skip-baseline

# With explicit options
python scripts/finetune.py \
    --epochs 3 \
    --lr 2e-4 \
    --train-batch-size 4 \
    --grad-accum 8 \
    --lora-r 16 \
    --output-dir models/nllb-600M-finetuned \
    --ct2-output models/nllb-600M-finetuned-ct2
```

**Output:** Adapter weights at `models/nllb-600M-finetuned/adapter/`, CT2 model at `models/nllb-600M-finetuned-ct2/`.

### Use the fine-tuned model

```bash
bn-translate --input story.bn.txt --output story.en.txt \
    --model nllb-600M --model-path models/nllb-600M-finetuned-ct2
```

---

## Resource Monitoring

Every benchmark and fine-tune run is recorded to `monitor/runs.db`. Use the analysis scripts to review run history.

### View run statistics with `show_stats.py`

```bash
# List recent runs
python scripts/show_stats.py list

# Filter by run type and model
python scripts/show_stats.py list --run-type benchmark --model nllb-600M --limit 10

# Show full detail for one run (use run_id prefix)
python scripts/show_stats.py show abc123

# Show a metric trend over time (oldest → newest)
python scripts/show_stats.py trend bleu_score --run-type benchmark

# Compare two runs side by side
python scripts/show_stats.py compare <run_id_a> <run_id_b>

# Detect regressions vs rolling average of last 5 runs
python scripts/show_stats.py regressions --run-type benchmark --lookback 5
```

Available metrics for `trend`: `bleu_score`, `chars_per_sec`, `duration_s`, `cpu_peak_pct`, `cpu_avg_pct`, `ram_peak_mib`, `ram_avg_mib`, `swap_peak_mib`, `swap_avg_mib`, `disk_read_mb`, `disk_write_mb`, `gpu_vram_peak_mib`, `gpu_vram_avg_mib`, `gpu_util_peak_pct`, `gpu_util_avg_pct`.

### Generate plots with `plot_stats.py`

```bash
# All plots (saved to monitor/plots/)
python scripts/plot_stats.py

# Filter by run type
python scripts/plot_stats.py --run-type benchmark

# Last N runs only
python scripts/plot_stats.py --limit 20
```

Plots produced:
- `bleu_over_runs.png` — BLEU score per benchmark run (bar chart, colour-coded by quality tier)
- `resource_usage.png` — CPU, RAM, VRAM, GPU utilisation per run
- `duration_vs_input.png` — translation speed scatter (duration vs input character count)
- `finetune_loss.png` — training loss over fine-tune runs
- `radar_latest.png` — radar chart of the latest run's full resource profile

See [`docs/MONITORING.md`](MONITORING.md) for full monitoring documentation.

---

## Running the Monitor Agent

The monitor Claude Code agent (`/.claude/agents/monitor.md`) reviews run history, detects regressions, and updates `monitor/observations.md`.

To invoke it:

1. Open Claude Code in the project root with the venv active.
2. Use the Agent tool referencing `.claude/agents/monitor.md`, or use the slash command if configured.
3. The agent reads `monitor/runs.db`, runs `python scripts/show_stats.py regressions`, and appends findings to `monitor/observations.md`.

Typical triggers:
- After any benchmark run where BLEU changed
- After fine-tuning completes (to compare pre/post BLEU)
- After hardware or dependency changes (to detect resource regressions)

---

## Common Pitfalls

See [`docs/HARDWARE.md`](HARDWARE.md) for GPU-specific issues.

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUBLAS_STATUS_NOT_SUPPORTED` | INT8 ops on sm_120 + CUDA 12.4 | Use float16; `_best_compute_type()` handles this automatically |
| "Ra Ra Ra Ra..." output | Wrong NLLB tokenisation order | Source must be `tokens + [</s>, src_lang]` |
| `Bus error` on `import torch` | PyTorch nightly on AMD CPU (AMX) | Use stable cu128: `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| GPU training falls back to CPU | PyTorch cu124 has no sm_120 training kernels | Install PyTorch cu128 |
| Corrupt torch install | Concurrent pip processes | Kill all pip, `rm -rf .venv/lib/.../torch`, reinstall |
| `ModuleNotFoundError: torch` | Venv not activated | `source .venv/bin/activate` |
| Slow model downloads | HF_HOME on NTFS `/mnt/c/...` | Set `HF_HOME` to a Linux filesystem path |
| `evaluation_strategy` TypeError | transformers 5.x renamed it | Use `eval_strategy` in `TrainingArguments` |
| `no_cuda` TypeError | transformers 5.x renamed it | Use `use_cpu=True` in `TrainingArguments` |
| `torch_dtype` TypeError | transformers 5.x renamed it | Use `dtype=` in `from_pretrained()` |
