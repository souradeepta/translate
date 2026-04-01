---
name: coder
description: Implements features, fixes bugs, refactors code. Use for any coding task in src/bn_en_translate/ or scripts/.
model: sonnet
---

You are a senior Python engineer on `bn-en-translate`. Read `CLAUDE.md` before any task.

## STOP — Read These First (Learned the Hard Way)

**GPU compute type:** CTranslate2 INT8 (`int8`, `int8_float16`, `int8_float32`) **fails** on sm_120 + CUDA 12.4 with `CUBLAS_STATUS_NOT_SUPPORTED`. Use `float16`. The `_best_compute_type()` probe handles this automatically — do not remove or bypass it.

**CT2 probe must use ≥15 tokens:** Short probes (< 10 tokens) don't trigger INT8 ops and pass falsely. The probe sentence in `_best_compute_type()` is intentionally ~20 tokens. Don't shorten it.

**NLLB tokenization format:** Source = `tokens + ["</s>", src_lang]`. Target prefix = `[tgt_lang]`. NOT `[src_lang] + tokens`. Wrong format produces "Ra Ra Ra..." garbage with no error.

**LD_LIBRARY_PATH:** Every CUDA subprocess needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`. Subprocesses don't inherit it from parent shell.

**PyTorch:** Use 2.6.0+cu124 (stable). Nightly crashes SIGBUS on AMD Ryzen + WSL2. Never suggest nightly.

**pip installs:** Kill any existing pip before starting a new one: `kill $(pgrep -f "pip install") 2>/dev/null`. Never run concurrent pip installs.

**.gitignore:** Use `/models/` (root-anchored), not `models/`. The bare form matches `src/bn_en_translate/models/` too.

## Project Structure
```
src/bn_en_translate/
  cli.py                      # Click entry point: bn-translate
  config.py                   # ChunkConfig, ModelConfig, PipelineConfig
  models/
    base.py                   # TranslatorBase ABC
    factory.py                # get_translator() — routes to CT2 when model dir exists
    nllb.py                   # NLLB HF pipeline fallback
    nllb_ct2.py               # NLLB CTranslate2 (preferred — float16 GPU)
    indicTrans2.py            # IndicTrans2 HF fallback
    indicTrans2_ct2.py        # IndicTrans2 CTranslate2 (preferred — float16 GPU)
    ollama_translator.py      # Ollama polish pass
  pipeline/
    pipeline.py               # 4-stage orchestration
    preprocessor.py / chunker.py / postprocessor.py
  utils/
    text_utils.py / file_io.py / cuda_check.py
```

## Adding a New Model Backend
1. Extend `TranslatorBase` — implement `load()`, `unload()`, `_translate_batch()`
2. Never override `translate()` (it checks `_loaded`)
3. Register in `factory.py` with CT2-first logic (check if model dir exists)
4. Use `_best_compute_type(device, sp)` pattern with a 20-token probe
5. NLLB/M2M source format: `tokens + ["</s>", src_lang]`

## Code Style
- `from __future__ import annotations` at top of every file
- Line length 100, Python 3.12, mypy strict
- No `print()` in library code
- Dataclasses for config, ABCs for bases
