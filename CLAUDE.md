# Bengali → English Translator — Claude Code Guide

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

tests/
  conftest.py                # MockTranslator fixture
  unit/                      # Pure logic tests — no GPU, no downloads (~3s)
  integration/               # Full pipeline with mock + real models (marked slow)
  e2e/                       # Quality assertions BLEU≥25, Ollama (marked e2e)
  fixtures/                  # Sample Bengali texts + reference translations
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

| Key | HF ID | VRAM INT8 | BLEU bn→en |
|-----|-------|-----------|------------|
| `nllb-600M` | `facebook/nllb-200-distilled-600M` | 0.6 GB | ~22 |
| `nllb-1.3B` | `facebook/nllb-200-distilled-1.3B` | 1.3 GB | ~26 |
| `indicTrans2-1B` | `ai4bharat/indictrans2-indic-en-1B` | 1.5 GB | ~30+ |
| `ollama` | `qwen2.5:7b-instruct-q4_K_M` via Ollama | 4.7 GB | subjective |

---

## WSL2 / RTX 5050 Critical Notes

1. **LD_LIBRARY_PATH** must include `/usr/lib/wsl/lib` — CUDA driver lives there in WSL2
2. **Blackwell (sm_100)** — use PyTorch nightly if stable fails: `pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128`
3. **HF_HOME** must be on Linux fs, not `/mnt/c/...` (10-20× speed difference)
4. **Multiprocessing** — never use `fork` with CUDA; use `spawn`
5. **VRAM headroom** — IndicTrans2 INT8 uses ~1.5 GB; Ollama qwen2.5:7b uses ~4.7 GB (can't run both at once)

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

---

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/project:translate` | Translate a Bengali file end-to-end |
| `/project:test` | Run the full test matrix |
| `/project:lint` | Lint + typecheck |
