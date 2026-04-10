# Bengali → English Translator — Claude Code Guide

## Quick Resume (read this first in any new session)

```bash
source .venv/bin/activate && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
make test          # 212 tests, ~27s — confirms env is working
python scripts/benchmark.py --models nllb-600M --sentences 5   # quick GPU smoke test
make papers        # regenerate figures + compile all 4 PDFs
```

**State as of 2026-04-10:**
- `models/nllb-600M-ct2/` ✅ CT2 float16 — **BLEU 55.3 / chrF 72.8** on FLORES-200 90-sentence
- `models/seamless-medium-hf/` ✅ HF float16, `.to("cuda")` — **BLEU 67.0 / chrF 80.2** on FLORES-200
- `models/madlad-3b-hf/` ✅ downloaded but **EXCLUDED** — local checkpoint weight mismatch causes garbage output; 8 GB VRAM insufficient for 3B float16 (CPU offload → degenerate sequences)
- `corpus/` ✅ 90-sentence built-in + 9,829 Samanantar pairs (train/val/test splits)
- `paper/pdf/` — run `make papers` to rebuild all 4 PDFs (tectonic, no sudo needed)
- All 212 unit/integration tests passing
- **PyTorch 2.7.0+cu128** — GPU training FULLY UNLOCKED (6/6 sm_120 probes pass)
- LoRA fine-tune done: 2.46h, 3 epochs, eval_loss 1.992, post-FT BLEU 0.17 (open-domain)
- Papers: `paper/ieee_paper.tex`, `paper/survey_paper.tex`, `paper/ieee_transactions_paper.tex`, `paper/acm_paper.tex`
- Figures: `python scripts/gen_paper_figures.py` or `make figures` → `paper/figures/`

---

## Never Do These (Painful Lessons)

### PyTorch / CUDA
- ❌ PyTorch cu124 for **training** → `no kernel image` on `.ne()` and all element-wise ops on sm_120
- ✅ PyTorch **2.7.0+cu128** — all sm_120 training ops pass; AMD Ryzen SIGBUS was a corrupt-pip artifact
- ❌ PyTorch nightly → historically caused SIGBUS; use cu128 stable instead
- ❌ Multiple concurrent pip installs → corrupt `.so` files, SIGBUS
- ❌ CTranslate2 int8/int8_float16 on CUDA → CUBLAS fail on sm_120; use `float16`

### NLLB / CTranslate2
- ❌ `[src_lang, tokens...]` for NLLB → garbage output; use `tokens + [</s>, src_lang]`
- ❌ CT2 `translate_batch()` with 900+ sentences at once → CUDA OOM; always pass `max_batch_size=32`

### MADLAD-3B
- ❌ `tie_word_embeddings=False` → randomises encoder embedding matrix; leave at default (True)
- ❌ `max_length` in generate() → counts input + output tokens; use `max_new_tokens=256`
- ❌ `device_map="auto"` on 8 GB VRAM → CPU offload → ~30 s/sentence → degenerate sequences
- ⚠️ Local checkpoint at `models/madlad-3b-hf/` has `shared.weight ≠ decoder.embed_tokens.weight` — garbage output even on CPU. **Do not benchmark MADLAD-3B** until checkpoint is re-downloaded cleanly.

### Seamless M4T
- ❌ `SeamlessM4Tv2ForTextToText` does NOT support `device_map="auto"` → use `.to("cuda")` after float16 load
- ❌ `generate_speech=False` in `.generate()` → `ValueError: not used by the model` (it's for the combined text+speech model only)
- ❌ Processor call without `padding=True, truncation=True` → `ValueError` on batches with unequal-length inputs
- ✅ Load pattern: `from_pretrained(model_id, dtype=torch.float16)` then `.to("cuda")`
- ✅ VRAM for text-only inference: ~3.9 GB — fits in 8 GB without offload

### HuggingFace Transformers 5.x
- ❌ `evaluation_strategy` → renamed to `eval_strategy`
- ❌ `no_cuda` → renamed to `use_cpu`
- ❌ `torch_dtype=` in `from_pretrained` → renamed to `dtype=`
- ❌ `trust_remote_code=True` in `load_dataset()` → no longer supported, remove it

### Training
- ❌ `.half()` model weights + `fp16=True` → `ValueError: Attempting to unscale FP16 gradients`; use `bf16=True` (no GradScaler, Blackwell supports it)
- ❌ `dataloader_prefetch_factor=2` with `num_workers=0` → `ValueError`; set to `None` when workers=0
- ❌ CT2 export missing SPM: copy `sentencepiece.bpe.model` from source CT2 dir to export dir

### Miscellaneous
- ❌ `pynvml` package → deprecated; use `nvidia-ml-py` (same `import pynvml` API, no code changes)
- ❌ RunDatabase schema migration → new columns added after deploy; use `_apply_migrations()` with `ALTER TABLE ADD COLUMN`
- ❌ `models/` in .gitignore (bare) → also matches `src/bn_en_translate/models/`; use `/models/`
- ❌ `\usepackage{pgfplotsset}` in LaTeX → `pgfplotsset` is NOT a .sty file; it's a command inside pgfplots. Only use `\usepackage{pgfplots}` then `\pgfplotsset{compat=1.18}`
- ✅ Paper compilation: `tectonic` is installed at `~/.local/bin/tectonic` — no sudo needed. Use `make papers`

---

## Benchmark Results (FLORES-200, 90 sentences, RTX 5050, 2026-04-10)

| Model | BLEU | chrF | VRAM | Speed | Notes |
|-------|------|------|------|-------|-------|
| NLLB-600M CT2 float16 | **55.3** | **72.8** | 2.0 GB | 197 ch/s | Primary model |
| Seamless-medium float16 | **67.0** | **80.2** | 3.9 GB | 32 ch/s | Best quality |
| MADLAD-3B float16 | ~~0.0~~ | — | 8.1 GB | ~2 ch/s | Excluded — checkpoint corrupted |

---

## Project Overview

`bn-en-translate` is a local, GPU-powered Bengali-to-English story translation system.
**No API keys required.** Runs entirely on-device using open-source models.

**Hardware target:** Acer Nitro V16, NVIDIA RTX 5050 8 GB (Blackwell/sm_120), 16 GB RAM, WSL2 Ubuntu.

---

## Package Layout

```
src/bn_en_translate/
  cli.py                     # Click CLI entry point (bn-translate command)
  config.py                  # ChunkConfig, ModelConfig, PipelineConfig dataclasses
  models/
    base.py                  # TranslatorBase ABC — load/unload/translate contract
    factory.py               # get_translator() router
    nllb.py                  # NLLB-200-distilled-600M / 1.3B (CTranslate2 CT2 float16)
    madlad.py                # MADLAD-400-3B (HF T5, device_map=auto — EXCLUDED, see above)
    seamless.py              # SeamlessM4Tv2 text-only (.to("cuda"), no device_map)
    indicTrans2.py           # AI4Bharat IndicTrans2-1B (CT2, best quality, needs download)
    ollama_translator.py     # Ollama literary polish pass
  pipeline/
    pipeline.py              # TranslationPipeline — orchestrates the 4-stage flow
    preprocessor.py          # normalize(): NFC Unicode, whitespace collapse
    chunker.py               # Chunker: paragraph → sentence-bounded chunks ≤400 tokens
    postprocessor.py         # reassemble(): paragraph-aware, MT artifact cleanup
  utils/
    text_utils.py            # Bengali sentence split (danda ।/॥), token estimation
    file_io.py               # read_story(), write_translation() UTF-8
    cuda_check.py            # is_cuda_available(), get_best_device(), get_free_vram_mib()
    monitor.py               # ResourceMonitor (psutil + pynvml daemon thread)
    run_db.py                # RunDatabase (SQLite, upsert, schema migrations)
  training/
    corpus.py                # load/save/filter/split parallel corpus files
    dataset.py               # BengaliEnglishDataset (PyTorch), collate_fn
    trainer.py               # NLLBFineTuner (LoRA via PEFT + Seq2SeqTrainer)

tests/
  conftest.py                # MockTranslator fixture
  unit/                      # Pure logic tests — no GPU, no downloads (~8s)
  integration/               # Full pipeline with mock + real models (marked slow)
  e2e/                       # Quality assertions BLEU≥25, Ollama (marked e2e)
  fixtures/                  # Sample Bengali texts + reference translations

scripts/
  benchmark.py               # Multi-model BLEU/chrF benchmark + RunDatabase logging
  download_models.py         # Download + convert models to CT2
  download_corpus.py         # Download Samanantar from HF, split train/val/test
  finetune.py                # LoRA fine-tune NLLB-600M + export to CT2
  gen_paper_figures.py       # Generate all 6 paper figures at 300 DPI (academic style)
  show_stats.py              # Query RunDatabase (list/show/trend/compare/regressions)
  plot_stats.py              # matplotlib charts from runs.db

paper/
  ieee_paper.tex             # IEEE conference paper
  survey_paper.tex           # Comparative survey of Bengali NMT
  ieee_transactions_paper.tex # IEEE Transactions journal version
  acm_paper.tex              # ACM TALLIP version
  figures/                   # PNGs generated by gen_paper_figures.py
  pdf/                       # Compiled PDFs (gitignored — run: make papers)
```

---

## Running the Project

```bash
# Activate venv
source .venv/bin/activate && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Translate a file
bn-translate --input story.bn.txt --output out.en.txt --model nllb-600M
bn-translate --input story.bn.txt --output out.en.txt --model seamless-medium
bn-translate --input story.bn.txt --output out.en.txt --model indicTrans2-1B --ollama-polish

# Tests
make test          # fast: unit + mock integration (~27s)
make test-slow     # real NLLB model (~60s)
make test-e2e      # full quality suite (GPU + models required)

# Benchmarks
python scripts/benchmark.py --models nllb-600M --sentences 5   # quick smoke test
python scripts/benchmark.py --models nllb-600M seamless-medium --sentences 90

# Papers (figures → PDF)
make figures       # regenerate paper/figures/*.png (300 DPI)
make papers        # figures + compile all 4 PDFs via tectonic → paper/pdf/

# Run history
python scripts/show_stats.py list
python scripts/show_stats.py show <run_id>
python scripts/show_stats.py trend bleu_score nllb-600M

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
- CTranslate2 float16 is the standard inference backend for NLLB/IndicTrans2
- HF native (.to("cuda") float16) for Seamless; device_map="auto" for nothing (not supported)
- Ollama polish pass requires VRAM swap: translator is unloaded first

---

## Model Reference

| Key | HF ID | Backend | VRAM | FLORES BLEU/chrF | Status |
|-----|-------|---------|------|------------------|--------|
| `nllb-600M` | `facebook/nllb-200-distilled-600M` | CT2 float16 | 2.0 GB | **55.3 / 72.8** (measured) | ✅ Working |
| `seamless-medium` | `facebook/seamless-m4t-v2-large` | HF float16 `.to(cuda)` | 3.9 GB | **67.0 / 80.2** (measured) | ✅ Working |
| `nllb-1.3B` | `facebook/nllb-200-distilled-1.3B` | CT2 float16 | 2.6 GB | 33.4 (published) | Needs download |
| `indicTrans2-1B` | `ai4bharat/indictrans2-indic-en-1B` | CT2 float16 | 3.0 GB | 41.4 (published) | Needs download |
| `madlad-3b` | `google/madlad400-3b-mt` | HF T5 float16 | 8.1 GB | ❌ EXCLUDED | Checkpoint corrupted |
| `ollama` | `gemma3:12b` via Ollama | HTTP | ~4.7 GB | subjective | Optional polish |

---

## Corpus

Built-in 90-sentence FLORES-200 devtest at `corpus/flores200_devtest.{bn,en}.txt`.
Custom 90-sentence corpus (10 domains) at `corpus/custom_90.{bn,en}.txt`.
Samanantar 9,829 pairs at `corpus/samanantar/{train,val,test}.{bn,en}.txt`.

---

## Environment Setup

```bash
# Activate venv (always required)
source .venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Install (if .venv missing)
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3
# CUDA 12.8 for sm_120 training:
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,gpu,train,monitor]"
```

## WSL2 / RTX 5050 Critical Notes

**GPU is RTX 5050 sm_120 (Blackwell). Key discoveries from setup:**

1. **LD_LIBRARY_PATH** must include `/usr/lib/wsl/lib` for every Python process using CUDA
2. **PyTorch**: use `2.7.0+cu128` (stable). cu124 works for inference but NOT for training on sm_120 (element-wise ops fail). cu128 is required for all sm_120 training ops.
3. **CTranslate2 INT8 fails on sm_120**: `CUBLAS_STATUS_NOT_SUPPORTED`. Use `float16`.
4. **Compute type probe**: runs a real ~20-token translation to detect INT8 failure at load time. Short probes (<10 tokens) give false positives.
5. **NLLB source format**: `[text_tokens..., </s>, src_lang]` — NOT `[src_lang, text_tokens...]`.
6. **HF_HOME** must be on Linux fs, not `/mnt/c/...`
7. **Multiprocessing**: never use `fork` with CUDA; use `spawn`
8. **SeamlessM4T**: `device_map="auto"` not supported — use `.to("cuda")` after float16 load
9. **Paper compilation**: `tectonic` at `~/.local/bin/tectonic` — works without sudo

---

## Development Rules

- **TDD**: write unit test first, then implementation
- **No mocking internals**: mock only at system boundaries (HuggingFace, Ollama HTTP, filesystem)
- **Batch processing**: always translate via `_translate_batch()` — never loop single strings
- **Context managers**: models must implement `load()` / `unload()` and support `with translator:`
- **Paragraph preservation**: every translation must maintain the same paragraph count as input
- **Token budget**: max 400 tokens per chunk — never exceed model's 512-token context window
- **Commits**: many small logical commits; conventional prefixes (feat/fix/docs/test/refactor/perf/paper); no Claude attribution in commits or code

---

## Key Invariants (never break these)

1. `TranslatorBase.translate()` raises `RuntimeError` if called before `load()`
2. `Chunker.chunk()` never splits mid-sentence
3. `reassemble()` output has same paragraph count as normalized input
4. All file I/O is UTF-8 with explicit `encoding="utf-8"` — no reliance on locale

---

## Available Agents

| Agent file | When to use |
|------------|-------------|
| `.claude/agents/coder.md` | Implementing new features, fixing bugs, refactoring |
| `.claude/agents/tester.md` | Writing or running tests, debugging test failures |
| `.claude/agents/architect.md` | Design decisions, new model integration, pipeline changes |
| `.claude/agents/monitor.md` | After any run: detect regressions, update `monitor/observations.md` |
| `.claude/agents/paper_writer.md` | After benchmarks/training: update `paper/ieee_paper.tex` |
| `.claude/agents/survey_writer.md` | When new Bengali NMT papers publish: update `paper/survey_paper.tex` |
| `.claude/agents/docs_writer.md` | After API/config/hardware changes: keep `docs/` in sync |

---

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/project:translate` | Translate a Bengali file end-to-end |
| `/project:test` | Run the full test matrix |
| `/project:lint` | Lint + typecheck |
