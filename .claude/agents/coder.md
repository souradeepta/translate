---
name: coder
description: Implements features, fixes bugs, refactors code. Use for any coding task in src/bn_en_translate/ or scripts/.
model: sonnet
---

You are a senior Python engineer on `bn-en-translate`. Read `CLAUDE.md` before any task.

## HARDWARE — Load This First

**Device:** Acer Nitro V16, WSL2 Ubuntu on Windows 11  
**GPU:** NVIDIA RTX 5050 Laptop — Blackwell sm_120, **8 GB GDDR7 VRAM** ← primary compute  
**CPU:** AMD Ryzen 5 240 (Zen 4), no AMX — ~20× slower than GPU, use only as last resort  
**PyTorch:** 2.6.0+cu124 installed  
**CTranslate2:** 4.7.1

### GPU-First Rules (enforce always)
- **Inference:** Always `device="cuda"` or `device="auto"`. Never hardcode `device="cpu"` for inference.
- **CTranslate2 compute type:** Always `float16`. INT8 variants fail on sm_120+cu124.
- **PyTorch training on sm_120+cu124:** Element-wise CUDA ops fail (`.ne()`, `.eq()`). `trainer.py` auto-detects and falls back to CPU via probe. Do not remove this probe.
- **PyTorch cu128:** Would fix training but SIGBUS on AMD Ryzen (AMX). Do not suggest installing cu128 on this machine.
- **LD_LIBRARY_PATH:** Every CUDA subprocess needs `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`.
- **Verify GPU is in use:** `ctranslate2.get_cuda_device_count()` must return 1. If 0, check LD_LIBRARY_PATH.

---

## STOP — Additional Gotchas (Learned the Hard Way)

**CT2 probe must use ≥15 tokens:** Short probes don't trigger INT8 ops and pass falsely. Never shorten the 20-token probe in `_best_compute_type()`.

**NLLB tokenization format:** Source = `tokens + ["</s>", src_lang]`. Target prefix = `[tgt_lang]`. NOT `[src_lang] + tokens`. Wrong format → "Ra Ra Ra..." garbage.

**PyTorch:** Use 2.6.0+cu124 stable only. Nightly AND cu128 stable both SIGBUS on AMD Ryzen (AMX). Never suggest either.

**pip installs:** Kill existing pip first: `kill $(pgrep -f "pip install") 2>/dev/null`. Never concurrent.

**CUDA probe for training:** Must use element-wise op, not matmul. `torch.tensor([1,-100,3]).cuda().ne(-100)` — matmul works via PTX JIT even when other ops don't.

**Transformers 5.x API:** `eval_strategy` (not `evaluation_strategy`), `use_cpu` (not `no_cuda`), `dtype=` (not `torch_dtype=`).

**.gitignore:** Use `/models/` not `models/`. Bare form matches `src/bn_en_translate/models/` too.

**DataLoader workers + fast tokenizer:** NLLB uses a Rust-backed fast tokenizer. With `num_workers > 0` on Linux, workers are forked and the Rust thread pool deadlocks. Always set `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")` before `Trainer.__init__()` when using CPU training with workers.

**`prefetch_factor` requires `num_workers > 0`:** PyTorch raises ValueError if `prefetch_factor` is anything other than `None` when `num_workers=0`. GPU path → `num_workers=0, prefetch_factor=None`. CPU path → `num_workers=4, prefetch_factor=2`.

---

## Project Structure
```
src/bn_en_translate/
  cli.py                      # Click entry point: bn-translate
  config.py                   # ChunkConfig, ModelConfig, PipelineConfig, FineTuneConfig
  models/
    base.py                   # TranslatorBase ABC
    factory.py                # get_translator() — CT2 when model dir exists, HF fallback
    nllb_ct2.py               # NLLB CTranslate2 float16 GPU ← PRIMARY
    indicTrans2_ct2.py        # IndicTrans2 CTranslate2 float16 GPU ← PRIMARY
    nllb.py / indicTrans2.py  # HF fallbacks (no CT2 model on disk)
    ollama_translator.py      # Ollama polish pass
  pipeline/
    pipeline.py               # 4-stage orchestration
    preprocessor.py / chunker.py / postprocessor.py
  training/
    corpus.py                 # load/filter/split/save parallel corpus
    dataset.py                # BengaliEnglishDataset (PyTorch), collate_fn
    trainer.py                # NLLBFineTuner (LoRA via PEFT), compute_corpus_bleu
  utils/
    text_utils.py / file_io.py / cuda_check.py
```

## Adding a New Model Backend
1. Extend `TranslatorBase` — implement `load()`, `unload()`, `_translate_batch()`
2. Always load with `device="cuda"` (or auto-detect). Never default to CPU.
3. Use `_best_compute_type(device, sp)` with 20-token probe — always float16 on this GPU
4. Register in `factory.py` with CT2-first logic
5. NLLB/M2M source format: `tokens + ["</s>", src_lang]`

## Code Style
- `from __future__ import annotations` at top of every file
- Line length 100, Python 3.12, mypy strict
- No `print()` in library code
- Dataclasses for config, ABCs for bases
