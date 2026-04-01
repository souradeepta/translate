# Bengali → English Story Translation: Local Setup Guide

## What This Does

Translates Bengali text files to English using open-source AI models running
entirely on your local machine — no API tokens consumed.

**Hardware target:** Acer Nitro V16, NVIDIA RTX 5050 (8 GB VRAM), 16 GB RAM, WSL2

---

## Model Choice: Why IndicTrans2?

| Model | BLEU (bn→en) | VRAM (INT8) | Notes |
|---|---|---|---|
| **IndicTrans2-1B** | ~30+ | ~1.5 GB | Best for Bengali. Purpose-built for Indic languages. |
| NLLB-200-1.3B | ~26 | ~1.3 GB | Good multilingual, fast start |
| NLLB-200-600M | ~22 | ~0.6 GB | Lightweight fallback |
| Ollama/qwen2.5:7b | subjective | ~4.7 GB | Best literary tone — optional polish pass |
| Helsinki opus-mt-bn-en | ~12 | ~0.3 MB | Too weak for stories |

**Recommended strategy:** Use **IndicTrans2-1B** as primary translator.
Optionally enable **Ollama** (`--ollama-polish`) for literary quality refinement.
Your RTX 5050 has 8 GB VRAM — IndicTrans2 INT8 uses only ~1.5 GB, leaving
plenty of headroom.

---

## Quick Start

### 1. One-time WSL2 setup

Add to `/mnt/c/Users/<YourName>/.wslconfig` (create if missing):
```ini
[wsl2]
memory=12GB
swap=4GB
processors=8
```
Then in Windows PowerShell: `wsl --shutdown` and reopen WSL2.

### 2. Configure LD_LIBRARY_PATH (add to ~/.bashrc)

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/$USER/.cache/huggingface
```

### 3. Install Python environment

```bash
cd /home/sbisw/github/translate
sudo apt install python3.12-venv  # only needed once
python3 -m venv .venv
source .venv/bin/activate

# Run the CUDA setup script (installs PyTorch for RTX 5050 / Blackwell)
bash scripts/setup_cuda.sh

pip install -e ".[dev]"
```

### 4. Download a model

```bash
# Lightweight model (~1.2 GB) — good starting point:
python scripts/download_models.py --model nllb-600M

# Best quality model (~3 GB, requires IndicTrans2 interface):
pip install git+https://github.com/AI4Bharat/IndicTrans2.git#subdirectory=huggingface_interface
python scripts/download_models.py --model indicTrans2-1B
```

### 5. Translate a story

```bash
bn-translate --input my_story.bn.txt --output translated.en.txt --model nllb-600M
# or with the best model:
bn-translate --input my_story.bn.txt --output translated.en.txt --model indicTrans2-1B
# with Ollama literary polish:
bn-translate --input my_story.bn.txt --output translated.en.txt --model indicTrans2-1B --ollama-polish
```

---

## Architecture

```
Bengali Story File (.txt, UTF-8)
       |
       v
  [Preprocessor]          NFC normalization, whitespace cleanup
       |
       v
  [Chunker]               Split by paragraph, then sentence-bounded chunks
       |                  Max 400 tokens/chunk — fits any 512-token model
       v
  [Translator]            IndicTrans2 or NLLB via HuggingFace Transformers
       |                  GPU batching (batch_size=8 by default)
       v
  [Postprocessor]         Fix MT artifacts, reassemble paragraphs
       |
       v
  [Ollama pass]           Optional: literary polish via qwen2.5:7b
       |
       v
  English Output (.txt)
```

### Key design decisions

- **No sentence splitting mid-chunk.** The chunker always breaks at sentence
  boundaries (danda `।` / double-danda `॥`), preserving grammatical context.
- **Paragraph structure is preserved.** A 3-paragraph Bengali story produces
  a 3-paragraph English output.
- **CTranslate2 INT8.** Converting models to CTranslate2 INT8 format reduces
  VRAM usage by ~50% with minimal quality loss, enabling larger models on 8 GB.
- **Ollama is optional.** Running qwen2.5:7b for polish requires 4.7 GB VRAM,
  which conflicts with holding IndicTrans2 loaded simultaneously. The pipeline
  unloads IndicTrans2 before loading Ollama.

---

## Running Tests

```bash
# Fast unit + mock integration tests (no downloads, ~3 seconds):
make test

# Real model tests (downloads NLLB-600M, ~1.2 GB):
make test-slow

# Full E2E quality tests (requires GPU + IndicTrans2 download):
make test-e2e

# Benchmark all models with BLEU scores:
make benchmark
```

---

## TDD Approach

This project follows Test-Driven Development. Tests were written in this order:

1. **Unit tests** (`tests/unit/`) — pure logic, no model, no GPU
   - `test_chunker.py` — sentence splitting, token budgets, metadata
   - `test_preprocessor.py` — Unicode normalization
   - `test_postprocessor.py` — reassembly, MT artifact cleanup
   - `test_config.py` — validation of config dataclasses
   - `test_text_utils.py` — sentence detection, token estimation
   - `test_file_io.py` — file reading/writing, UTF-8 handling
   - `test_cuda_check.py` — GPU detection (fully mocked)
   - `test_model_interface.py` — abstract base contract

2. **Integration tests** (`tests/integration/`) — real pipeline, mock model
   - `test_pipeline_mock.py` — full pipeline with MockTranslator
   - `test_pipeline_nllb.py` — real NLLB model (marked `slow`)

3. **E2E tests** (`tests/e2e/`) — real models, quality assertions
   - `test_indicTrans2_quality.py` — BLEU ≥ 25, named entities (marked `e2e`)
   - `test_ollama_integration.py` — Ollama API (marked `e2e`)

---

## WSL2 + RTX 5050 (Blackwell) Gotchas

**1. libcuda.so location**
WSL2's CUDA driver lives in `/usr/lib/wsl/lib/`, not the CUDA toolkit path.
Always have `/usr/lib/wsl/lib` at the start of `LD_LIBRARY_PATH`.

**2. Blackwell architecture (sm_100)**
RTX 5050 uses CUDA compute capability `sm_100`. PyTorch stable may not include
pre-compiled kernels for it. If you see CUDA errors, switch to the nightly build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
```
Verify: `python3 -c "import torch; print(torch.cuda.get_device_capability(0))"`
Expected: `(10, 0)`

**3. WSL2 RAM limit**
Default WSL2 is capped at 50% of host RAM (~8 GB). Set `memory=12GB` in
`.wslconfig` before loading large models. Model loading peaks at ~3 GB RAM.

**4. HuggingFace cache speed**
If `HF_HOME` points to `/mnt/c/...`, downloads are 10-20× slower due to NTFS.
Always keep the cache on the Linux filesystem (default: `~/.cache/huggingface`).

**5. Multiprocessing**
Never use Python's default `fork` start method with CUDA in WSL2.
Add to any script using `multiprocessing`:
```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```
