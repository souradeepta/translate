# Bengali → English Translator — Claude Code Guide

## Quick Resume (read this first in any new session)

```bash
source .venv/bin/activate && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
make test          # 186 tests, ~12s — confirms env is working
python scripts/benchmark.py --models nllb-600M --sentences 5   # quick GPU smoke test
```

**State as of 2026-04-01:**
- `models/nllb-600M-ct2/` ✅ downloaded, working via CT2 float16 (BLEU 56.2 on 90-sentence corpus)
- `corpus/` ✅ 90-sentence built-in + 9,829 Samanantar pairs (train/val/test splits)
- All 186 unit/integration tests passing
- **PyTorch 2.7.0+cu128** installed — GPU training FULLY UNLOCKED (6/6 sm_120 probes pass)
- Resource monitoring: `ResourceMonitor` + `RunDatabase` (SQLite) record every benchmark/finetune run
- Next: run GPU fine-tuning `python scripts/finetune.py` (expected ~20–30 min on RTX 5050)

**Never do these (painful lessons):**
- ❌ PyTorch cu124 for training → `no kernel image` on `.ne()` and all element-wise ops on sm_120
- ✅ PyTorch 2.7.0+cu128 — all sm_120 training ops pass; AMD Ryzen SIGBUS was a corrupt-pip artifact
- ❌ PyTorch nightly → historically caused SIGBUS; use cu128 stable instead
- ❌ CTranslate2 int8/int8_float16 on CUDA → CUBLAS fail on sm_120; use float16
- ❌ `[src_lang, tokens...]` for NLLB → garbage output; use `tokens + [</s>, src_lang]`
- ❌ Multiple concurrent pip installs → corrupt install
- ❌ `models/` in .gitignore (bare) → also matches src/bn_en_translate/models/; use `/models/`
- ❌ `evaluation_strategy` in transformers ≥5.x → renamed to `eval_strategy`
- ❌ `no_cuda` in transformers ≥5.x → renamed to `use_cpu`
- ❌ `torch_dtype` in transformers ≥5.x → renamed to `dtype`
- ❌ `.half()` model weights + `fp16=True` training → GradScaler raises `ValueError: Attempting to unscale FP16 gradients`; use `bf16=True` instead (no GradScaler needed, Blackwell sm_120 supports bf16)
- ❌ CT2 `translate_batch()` with 900+ sentences at once → CUDA OOM; always pass `max_batch_size=32`

---

## Project Overview

`bn-en-translate` is a local, GPU-powered Bengali-to-English story translation system.  
**No API keys required.** Runs entirely on-device using open-source models.

**Hardware target:** Acer Nitro V16, NVIDIA RTX 5050 8 GB (Blackwell/sm_100), 16 GB RAM, WSL2 Ubuntu.

---

## Package Layout

```
src/bn_en_translate/
  cli.py                     # Click CLI entry point (bn-translate command)
  config.py                  # ChunkConfig, ModelConfig, PipelineConfig dataclasses
  models/
    base.py                  # TranslatorBase ABC — load/unload/translate contract
    factory.py               # get_translator() router
    nllb.py                  # NLLB-200-distilled-600M / 1.3B (HuggingFace pipeline)
    indicTrans2.py            # AI4Bharat IndicTrans2-1B (best quality)
    ollama_translator.py     # Ollama qwen2.5:7b-instruct literary polish pass
  pipeline/
    pipeline.py              # TranslationPipeline — orchestrates the 4-stage flow
    preprocessor.py          # normalize(): NFC Unicode, whitespace collapse
    chunker.py               # Chunker: paragraph → sentence-bounded chunks ≤400 tokens
    postprocessor.py         # reassemble(): paragraph-aware, MT artifact cleanup
  utils/
    text_utils.py            # Bengali sentence split (danda ।/॥), token estimation
    file_io.py               # read_story(), write_translation() UTF-8
    cuda_check.py            # is_cuda_available(), get_best_device(), get_free_vram_mib()

  training/
    corpus.py                # load/save/filter/split parallel corpus files
    dataset.py               # BengaliEnglishDataset (PyTorch Dataset), collate_fn
    trainer.py               # NLLBFineTuner (LoRA via PEFT + Seq2SeqTrainer), compute_corpus_bleu

tests/
  conftest.py                # MockTranslator fixture
  unit/                      # Pure logic tests — no GPU, no downloads (~8s)
  integration/               # Full pipeline with mock + real models (marked slow)
  e2e/                       # Quality assertions BLEU≥25, Ollama (marked e2e)
  fixtures/                  # Sample Bengali texts + reference translations

scripts/
  download_corpus.py         # Download Samanantar from HF, split train/val/test
  finetune.py                # LoRA fine-tune NLLB-600M + export to CT2
```

---

## Running the Project

```bash
# Activate venv
source .venv/bin/activate

# Translate a file
bn-translate --input story.bn.txt --output out.en.txt --model nllb-600M
bn-translate --input story.bn.txt --output out.en.txt --model indicTrans2-1B --ollama-polish

# Tests
make test          # fast: unit + mock integration (~3s)
make test-slow     # real NLLB model (~30s)
make test-e2e      # full quality suite (GPU + models required)
make benchmark     # BLEU comparison

# Code quality
make lint          # ruff check
make typecheck     # mypy strict
```

---

## Architecture: 4-Stage Pipeline

```
Bengali .txt  →  [Preprocessor]  →  [Chunker]  →  [Translator]  →  [Postprocessor]  →  English .txt
                  NFC + spaces      ≤400 tok        GPU batches       para reassembly
```

- Chunker always breaks at sentence boundaries (danda `।` / `॥`)
- `para_id` metadata on each `ChunkResult` drives reassembly
- CTranslate2 INT8 reduces VRAM 50% with negligible quality loss
- Ollama polish pass requires VRAM swap: IndicTrans2 is unloaded first

---

## Model Reference

| Key | HF ID | VRAM (float16) | BLEU bn→en | Status |
|-----|-------|----------------|------------|--------|
| `nllb-600M` | `facebook/nllb-200-distilled-600M` | ~2 GB | ~22–64* | **Working** — CT2 at `models/nllb-600M-ct2/` |
| `nllb-1.3B` | `facebook/nllb-200-distilled-1.3B` | ~2.6 GB | ~26 | Needs download |
| `indicTrans2-1B` | `ai4bharat/indictrans2-indic-en-1B` | ~3 GB | ~30+ | Needs download |
| `ollama` | `qwen2.5:7b-instruct-q4_K_M` via Ollama | ~4.7 GB | subjective | Optional polish |

*BLEU varies highly with corpus; 64 on built-in test pairs, ~22 on open-domain.

## Corpus

Built-in 90-sentence corpus at `corpus/` (10 domains: literature, history, geography, science,
education, health, everyday life, agriculture, proverbs). Used by benchmark.

FLORES-200 (1012 sentences): download attempted automatically, falls back to built-in.

## CT2 Model Setup

```bash
# Download + convert NLLB-600M (done — already at models/nllb-600M-ct2/)
python scripts/download_models.py --model nllb-600M

# Download IndicTrans2-1B (best quality, ~3 GB)
python scripts/download_models.py --model indicTrans2-1B

# Benchmark
python scripts/benchmark.py --models nllb-600M --sentences 20
```

---

## Environment Setup

```bash
# Activate venv (always required)
source .venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Install (if .venv missing)
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

## WSL2 / RTX 5050 Critical Notes

**GPU is RTX 5050 sm_120 (Blackwell). Key discoveries from setup:**

1. **LD_LIBRARY_PATH** must include `/usr/lib/wsl/lib` for every Python process
2. **PyTorch**: use `2.6.0+cu124` (stable). Nightly cu128 crashes with SIGBUS on AMD Ryzen+WSL2 (AMX incompatibility). GPU ops work via PTX JIT despite sm_120 warning.
3. **CTranslate2 INT8 fails on sm_120 + cu124**: `CUBLAS_STATUS_NOT_SUPPORTED` for INT8/INT8_FLOAT16. **Use `float16`** instead — same speed, works correctly.
4. **Compute type probe**: `NLLBCt2Translator._best_compute_type()` runs a real ~20-token translation probe to detect this at load time. Short probes (< ~10 tokens) don't trigger INT8 ops and give false positives.
5. **NLLB source format**: `[text_tokens..., </s>, src_lang]` — NOT `[src_lang, text_tokens...]`. Target: forced BOS = `tgt_lang`.
6. **HF_HOME** must be on Linux fs, not `/mnt/c/...`
7. **Multiprocessing**: never use `fork` with CUDA; use `spawn`
8. **VRAM**: NLLB-600M float16 = ~2 GB. IndicTrans2-1B = ~2-3 GB. Ollama qwen2.5:7b = ~4.7 GB.
9. **Python venv**: `python3-venv` package may not be installed. Use `--without-pip` + bootstrap pip via `get-pip.py`.

---

## Development Rules

- **TDD**: write unit test first, then implementation
- **No mocking internals**: mock only at system boundaries (HuggingFace, Ollama HTTP, filesystem)
- **Batch processing**: always translate via `_translate_batch()` — never loop single strings
- **Context managers**: models must implement `load()` / `unload()` and support `with translator:`
- **Paragraph preservation**: every translation must maintain the same paragraph count as input
- **Token budget**: max 400 tokens per chunk — never exceed model's 512-token context window
- **src_lang / tgt_lang**: always `ben_Beng` → `eng_Latn` for NLLB; IndicTrans2 uses its own codes

---

## Key Invariants (never break these)

1. `TranslatorBase.translate()` raises `RuntimeError` if called before `load()`
2. `Chunker.chunk()` never splits mid-sentence
3. `reassemble()` output has same paragraph count as normalized input
4. All file I/O is UTF-8 with explicit `encoding="utf-8"` — no reliance on locale

---

## Available Agents

Use these specialized agents via the Agent tool:

| Agent file | When to use |
|------------|-------------|
| `.claude/agents/coder.md` | Implementing new features, fixing bugs, refactoring |
| `.claude/agents/tester.md` | Writing or running tests, debugging test failures |
| `.claude/agents/architect.md` | Design decisions, new model integration, pipeline changes |
| `.claude/agents/monitor.md` | After any run: detect regressions, suggest optimizations, update `monitor/observations.md` |
| `.claude/agents/paper_writer.md` | After a training run: fill in results, regenerate figures, update `paper/ieee_paper.tex` |
| `.claude/agents/survey_writer.md` | When new Bengali NMT papers publish: search, extract BLEU, update `paper/survey_paper.tex` |
| `.claude/agents/docs_writer.md` | After API/config/hardware changes: keep `docs/` in sync with codebase |

---

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/project:translate` | Translate a Bengali file end-to-end |
| `/project:test` | Run the full test matrix |
| `/project:lint` | Lint + typecheck |
