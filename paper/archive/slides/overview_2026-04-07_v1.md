---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
  }
  h1 { color: #1a237e; }
  h2 { color: #283593; }
  code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
  blockquote { border-left: 4px solid #1a237e; padding-left: 12px; color: #444; }
---

# bn-en-translate

**Fully offline Bengali → English translation on consumer GPU**

No API keys. No cloud. RTX 5050 (8 GB) only.

---

## The Problem

- Bengali: 234 million native speakers, 7th most spoken language
- Existing translation tools: Google Translate, DeepL (cloud, privacy risk, cost)
- No production-quality **offline** Bengali → English translator existed
- Best open-source models need hardware-specific deployment expertise

---

## The Solution

```bash
# Translate a story
bn-translate --input story.bn.txt --output story.en.txt --model indicTrans2-1B

# Literary polish via Gemma 3
bn-translate --input story.bn.txt --output story.en.txt \
  --model indicTrans2-1B --ollama-polish --ollama-model gemma3:12b
```

Fully offline. No API keys. Runs on 8 GB VRAM.

---

## Pipeline: 4 Stages

```
Bengali .txt
  → [Preprocessor]  NFC Unicode + whitespace cleanup
  → [Chunker]       Split at sentence boundaries, ≤400 tokens/chunk
  → [Translator]    GPU batch inference (CTranslate2 float16 or HF)
  → [Postprocessor] Reassemble paragraphs, fix MT artifacts
  → English .txt
```

**Key invariant:** output has exactly the same paragraph count as input.

---

## Supported Models

| Model | BLEU (FLORES) | VRAM | Speed |
|-------|-------------|------|-------|
| NLLB-200-600M | 22 | 2.0 GB | ~1000 chars/s |
| NLLB-200-1.3B | 26 | 2.6 GB | ~700 chars/s |
| IndicTrans2-1B | 44 | 3.0 GB | ~700 chars/s |
| MADLAD-400-3B | 36 | 3.0 GB | ~500 chars/s |
| SeamlessM4T-v2 | ~39 | 3.5 GB | ~500 chars/s |
| Gemma 3 12B (Ollama) | subjective | 7.3 GB | ~200 chars/s |

---

## Results

**In-domain (90-sentence curated corpus, 10 domains):**

- Overall BLEU: **65.2**
- Best domain: Health **80.6**
- Worst domain: News **4.7**
- Throughput: **97–100 chars/s**, 82–84% GPU utilisation

> Single-number BLEU hides 75-point domain variance — always evaluate on your target domain.

---

## LoRA Fine-Tuning

- Fine-tune any seq2seq model on your own corpus
- Uses PEFT LoRA — only **<1% of parameters** are trainable
- **GPU training unlocked** on RTX 5050 (Blackwell sm_120, bf16)

```bash
# Fine-tune NLLB-600M on Samanantar corpus (2.46 hours)
python scripts/finetune.py --model nllb-600M --epochs 3

# Fine-tune MADLAD-400-3B
python scripts/finetune.py --model madlad-3b --epochs 3
```

---

## Hardware: RTX 5050 (Blackwell sm_120)

We resolved **6 previously undocumented** deployment constraints:

1. CT2 INT8 → `CUBLAS_STATUS_NOT_SUPPORTED` → **use float16**
2. SIGBUS (misdiagnosed as AMX) → **corrupt pip install**
3. CUDA + fork deadlock → **use spawn**
4. fp16/GradScaler conflict → **use bf16**
5. DataLoader prefetch issue → **prefetch_factor=2**
6. CT2 OOM (900+ sentences) → **max_batch_size=32**

---

## Resource Monitoring

Every run (benchmark, fine-tune, translate) is recorded:

```bash
# View recent runs
python scripts/show_stats.py list --limit 10

# Compare two runs
python scripts/show_stats.py compare run_A run_B

# Detect regressions
python scripts/show_stats.py regressions
```

Metrics: BLEU, duration, peak VRAM, avg GPU%, input size, model.

---

## Testing

**212 unit and integration tests**, all passing.

```bash
make test         # fast: unit + mock integration (~12s)
make test-slow    # real NLLB model (~30s)
make test-e2e     # full quality suite (GPU required)
```

TDD discipline: test first, then implementation. No mocking internals.

---

## Architecture at a Glance

```
src/bn_en_translate/
  cli.py              → Click CLI (bn-translate command)
  config.py           → PipelineConfig, ModelConfig dataclasses
  models/             → Translator backends (NLLB, IT2, MADLAD, Seamless, Ollama)
  pipeline/           → 4-stage pipeline orchestration
  training/           → Seq2SeqFineTuner (LoRA + Seq2SeqTrainer)
  utils/              → CUDA check, resource monitor, text utils

scripts/
  benchmark.py        → BLEU benchmarking with ResourceMonitor
  finetune.py         → LoRA fine-tuning with --model flag
  download_models.py  → Download + convert models to CT2/HF
```

---

## Open Source

**GitHub:** `github.com/souradeepta/translate`

- Apache 2.0 license
- Full documentation in `docs/`
- IEEE paper: `paper/ieee_paper.tex`
- Survey paper: `paper/survey_paper.tex`
- Slide decks: `paper/slides/`

*No API keys. No cloud. Fully reproducible.*
