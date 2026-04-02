# Documentation Index

`bn-en-translate` is a local, GPU-powered Bengali-to-English story translation system. It runs entirely on-device (no API keys required) using a 4-stage pipeline: Unicode normalisation, sentence-bounded chunking, CTranslate2 float16 GPU inference with NLLB-200-distilled-600M, and paragraph-aware reassembly. Optional literary polish is available via a local Ollama instance. The system also supports LoRA fine-tuning of the NLLB model on parallel Bengali-English corpora, with resource monitoring for every run persisted to SQLite.

---

## Quick Start

Three commands to get running:

```bash
# 1. Activate environment (PyTorch cu128 for sm_120 GPU training)
source .venv/bin/activate && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 2. Verify everything works (186 tests, ~12s)
make test

# 3. Translate a Bengali file
bn-translate --input story.bn.txt --output story.en.txt --model nllb-600M
```

Model must be downloaded first if not already present:

```bash
python scripts/download_models.py --model nllb-600M
```

---

## Documentation Files

| File | Description |
|------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture: component overview, 4-stage pipeline, UML class hierarchy, sequence diagram, layer descriptions, key invariants, data flow, VRAM budget, and extension points |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Developer setup and daily workflow: environment setup, test tiers, code style, adding model backends, downloading models, running benchmarks, fine-tuning, resource monitoring, and common pitfalls |
| [TRAINING.md](TRAINING.md) | Fine-tuning guide: training pipeline diagram, LoRA explanation, `FineTuneConfig` parameter reference, step-by-step fine-tuning instructions, CT2 export details, bf16 vs fp16 rationale, VRAM budget, and evaluation |
| [MONITORING.md](MONITORING.md) | Resource monitoring: architecture diagram, `ResourceMonitor` internals, `ResourceSummary` field reference, `RunDatabase` schema and query examples, `show_stats.py` CLI reference, `plot_stats.py` chart types, regression thresholds, and monitor agent usage |
| [HARDWARE.md](HARDWARE.md) | Hardware-specific notes for the RTX 5050 sm_120 (Blackwell): WSL2 CUDA setup, INT8 limitations, AMD Ryzen AMX notes, `.wslconfig` tuning, VRAM budget table |
| [MODELS.md](MODELS.md) | Model reference: download instructions, BLEU benchmarks, VRAM requirements, language codes, and per-model notes |
| [GLOSSARY.md](GLOSSARY.md) | AI/ML terminology reference: transformer architecture, M2M-100, attention, LoRA/PEFT, bf16/fp16, CTranslate2, beam search, quantization, BLEU, CUDA, WSL2, and more — all defined in the context of this project |

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_models.py` | Download and convert NLLB or IndicTrans2 models to CTranslate2 format |
| `scripts/download_corpus.py` | Download Samanantar Bengali-English corpus from HuggingFace (splits into train/val/test) |
| `scripts/get_corpus.py` | Generate or download the built-in 90-sentence evaluation corpus |
| `scripts/benchmark.py` | Run BLEU benchmark across one or more models; records results to `monitor/runs.db` |
| `scripts/finetune.py` | Run LoRA fine-tuning of NLLB-600M; records results to `monitor/runs.db` |
| `scripts/show_stats.py` | Query `monitor/runs.db`: list runs, show detail, view trends, compare runs, detect regressions |
| `scripts/plot_stats.py` | Generate PNG performance charts from `monitor/runs.db` into `monitor/plots/` |

---

## Source Layout

```
src/bn_en_translate/
  cli.py                    # bn-translate command (Click)
  config.py                 # ChunkConfig, ModelConfig, PipelineConfig, FineTuneConfig, MonitorConfig
  models/
    base.py                 # TranslatorBase ABC
    factory.py              # get_translator() router
    nllb_ct2.py             # NLLBCt2Translator (CTranslate2 float16, primary)
    nllb.py                 # NLLBTranslator (HuggingFace fallback)
    indicTrans2.py          # IndicTrans2Translator
    ollama_translator.py    # OllamaTranslator (literary polish)
  pipeline/
    pipeline.py             # TranslationPipeline
    preprocessor.py         # normalize()
    chunker.py              # Chunker, ChunkResult
    postprocessor.py        # reassemble()
  training/
    corpus.py               # Corpus load/save/filter/split
    dataset.py              # BengaliEnglishDataset
    trainer.py              # NLLBFineTuner, compute_corpus_bleu
  utils/
    monitor.py              # ResourceMonitor, ResourceSummary, ResourceSample
    run_db.py               # RunDatabase (SQLite)
    text_utils.py           # Bengali sentence splitting, token estimation
    file_io.py              # read_story(), write_translation()
    cuda_check.py           # is_cuda_available(), get_best_device(), get_free_vram_mib()
```
