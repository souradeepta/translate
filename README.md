# bn-en-translate

**Local Bengali → English story translation using GPU-accelerated open-source models.**

No API keys. No cloud. Runs entirely on-device with CTranslate2 INT8/float16 quantisation.

```bash
bn-translate --input story.bn.txt --output story.en.txt --model nllb-600M
```

---

## Features

- **4-stage pipeline** — Unicode normalise → sentence-boundary chunk → GPU translate → paragraph-preserving reassemble
- **CTranslate2 backend** — float16 GPU inference via optimised CUDA kernels; auto-probes compute type at load time
- **Smart factory** — routes to CTranslate2 model when converted model exists, HuggingFace fallback otherwise
- **Multiple models** — NLLB-600M (fast), NLLB-1.3B (balanced), IndicTrans2-1B (best Bengali quality)
- **Optional literary polish** — Ollama qwen2.5:7b post-processing pass for natural English prose
- **Paragraph preservation** — output paragraph count always matches input
- **TDD test suite** — 99 unit + integration tests, ~0.5s, no GPU required

---

## Quick Start

### 1. Install

```bash
# Python 3.12+ required
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3

source .venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH   # WSL2 only

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

### 2. Download a model

```bash
# Lightweight baseline (~600 MB, converts to ~594 MB CT2)
python scripts/download_models.py --model nllb-600M

# Best quality for Bengali (~3 GB)
python scripts/download_models.py --model indicTrans2-1B
```

### 3. Translate

```bash
bn-translate --input my_story.bn.txt --output translated.en.txt --model nllb-600M
```

---

## Model Comparison

| Model | VRAM (float16) | BLEU bn→en | Speed | Notes |
|-------|---------------|------------|-------|-------|
| `nllb-600M` | ~2 GB | ~22 open-domain | Fast | ✅ Downloaded, working |
| `nllb-1.3B` | ~2.6 GB | ~26 | Medium | Download with `--model nllb-1.3B` |
| `indicTrans2-1B` | ~3 GB | ~30+ | Medium | Best for Bengali; download with `--model indicTrans2-1B` |
| `ollama` | ~4.7 GB | subjective | Slow | Optional literary polish; requires Ollama daemon |

> BLEU scores are approximate for open-domain text. On in-domain literary Bengali the working NLLB-600M model achieves BLEU ~64 on the bundled corpus.

---

## CLI Reference

```
bn-translate [OPTIONS]

Options:
  -i, --input PATH       Bengali source file (.txt, UTF-8)  [required]
  -o, --output PATH      English output file                [required]
  -m, --model TEXT       nllb-600M | nllb-1.3B | indicTrans2-1B | ollama
                         [default: nllb-600M]
  --device TEXT          cuda | cpu | auto  [default: auto]
  --batch-size INTEGER   Translation batch size  [default: 8]
  --beam-size INTEGER    Beam search width  [default: 4]
  --ollama-polish        Run Ollama literary polish pass after translation
  --help                 Show this message and exit.
```

---

## Project Structure

```
bn-en-translate/
├── src/bn_en_translate/
│   ├── cli.py                    # Click CLI entry point
│   ├── config.py                 # ChunkConfig / ModelConfig / PipelineConfig
│   ├── models/
│   │   ├── base.py               # TranslatorBase ABC
│   │   ├── factory.py            # CT2-first routing
│   │   ├── nllb_ct2.py           # NLLB via CTranslate2  ← primary GPU path
│   │   ├── indicTrans2_ct2.py    # IndicTrans2 via CTranslate2
│   │   ├── nllb.py               # NLLB via HuggingFace  ← fallback
│   │   ├── indicTrans2.py        # IndicTrans2 via HuggingFace
│   │   └── ollama_translator.py  # Ollama polish pass
│   ├── pipeline/
│   │   ├── pipeline.py           # TranslationPipeline orchestrator
│   │   ├── preprocessor.py       # NFC normalise, collapse whitespace
│   │   ├── chunker.py            # Sentence-boundary chunking ≤400 tokens
│   │   └── postprocessor.py      # Paragraph reassembly, MT artifact cleanup
│   └── utils/
│       ├── text_utils.py         # Bengali sentence split (danda ।/॥)
│       ├── file_io.py            # UTF-8 read/write
│       └── cuda_check.py         # GPU detection
├── models/                       # Downloaded CT2 models (gitignored)
├── corpus/                       # Bengali-English test corpus
├── tests/
│   ├── unit/                     # Pure logic tests (~0.5s, no GPU)
│   ├── integration/              # Pipeline tests (mock + real model)
│   └── e2e/                      # BLEU quality tests (GPU required)
├── scripts/
│   ├── download_models.py        # Download + CT2 INT8 conversion
│   ├── get_corpus.py             # Fetch/generate Bengali corpus
│   └── benchmark.py              # BLEU + speed + VRAM benchmark
└── docs/
    ├── ARCHITECTURE.md           # Pipeline design decisions
    ├── DEVELOPMENT.md            # Developer setup and workflows
    ├── MODELS.md                 # Model details and GPU notes
    └── HARDWARE.md               # WSL2 + RTX 5050 setup guide
```

---

## Running Tests

```bash
make test           # 99 unit + mock integration tests — no GPU, no downloads (~0.5s)
make test-slow      # Real NLLB-600M model (~30s)
make test-e2e       # Full quality suite — requires GPU + downloaded models
make benchmark      # BLEU + speed + VRAM for available models
```

---

## Hardware

Developed and tested on:
- **GPU:** NVIDIA RTX 5050 Laptop (Blackwell sm_120, 8 GB VRAM)
- **CPU:** AMD Ryzen 5 240 (Zen 4)
- **OS:** WSL2 Ubuntu on Windows 11
- **CUDA:** 12.8 toolkit, driver 595.79

The CT2 backend auto-detects the best working compute type at model load time (float16 on sm_120 + CUDA 12.4; INT8 is not available on this config). See [`docs/HARDWARE.md`](docs/HARDWARE.md) for full setup notes.

---

## License

MIT — see [LICENSE](LICENSE).
