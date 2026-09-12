# Handoff Notes — Bengali Translation Research Project

## Project Status

**Repository:** `/home/sbisw/github/translate`  
**Branch:** `main` (up to date with origin)  
**Last Commit:** 601ce93 — "data: add translation inputs, candidate source texts, and model outputs"  
**Date:** 2026-09-12

**Project Description:**  
Local Bengali → English story translation using GPU-accelerated open-source models. No API keys required; runs entirely on-device with CTranslate2 INT8/float16 quantization and HuggingFace model backends. Includes a 4-stage pipeline (normalize → chunk → translate → reassemble) and comparative benchmarks across NLLB, Seamless, IndicTrans2, and MiLMMT models.

## What's in `inputs/`

### Source Material
- **Bengali Story (source):**
  - `RAA -যে আসে অন্ধকারে- 1703.docx` — original DOCX file
  - `story.bn.txt` — extracted Bengali source as UTF-8 plain text

### Model Translation Outputs (3 variants with metrics)
1. `story.milmmt.en.txt` + `story.milmmt.en.txt.metrics.txt`  
   (MiLMMT-46-1B translation + BLEU/chrF metrics)

2. `story.milmmt4b.en.txt` + `story.milmmt4b.en.txt.metrics.txt`  
   (MiLMMT-46-4B translation + metrics — best BLEU found at 68.5/81.8)

3. `story.seamless.en.txt` + `story.seamless.en.txt.metrics.txt`  
   (Seamless-M4T-v2 translation + metrics)

### Metadata
- `TRANSLATION_REPORT.md` — summary of translation runs, model comparison, and quality observations

### Candidate Texts (for future translation)
- `candidates/CANDIDATES.md` — index and description of candidate source texts
- `candidates/jibito-o-mrito.bn.txt` — Bengali source text 1
- `candidates/kankal.bn.txt` — Bengali source text 2
- `candidates/postmaster.bn.txt` — Bengali source text 3

## Excluded Files

**Zone.Identifier artifact:** `inputs/RAA -যে আসে অন্ধকারে- 1703.docx:Zone.Identifier` (Windows metadata, 25 bytes) was intentionally excluded from git and remains untracked locally.

## Current Work Status

**No other pending work-in-progress as of 2026-09-12.**

- All paper PDFs (IEEE conference, IEEE transactions, survey, ACM TALLIP) are compiled and up to date in `paper/pdf/`.
- All 294 unit and integration tests passing.
- GPU models (`nllb-600M`, `seamless-medium`, `milmmt-46-1b`, `milmmt-46-4b`) verified and benchmarked.
- Local environment fully configured (PyTorch 2.7.0+cu128, CTranslate2, Ollama on WSL2 / RTX 5050).

## Resumption Instructions

```bash
cd /home/sbisw/github/translate
source .venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Verify environment
make test          # 294 tests, ~27s
python scripts/benchmark.py --models nllb-600M --sentences 5

# Inspect project
make papers        # Rebuild all PDFs
```

See `README.md` and `CLAUDE.md` for detailed architecture, model reference, and development guidelines.

---

**Repository is fully resumable from git alone.** All project data, source texts, model outputs, and metadata are committed. Local `.claude/settings.local.json` is intentionally excluded (tooling config, not project content).
