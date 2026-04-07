# Model Expansion & Inference Optimization Design

**Date:** 2026-04-07  
**Status:** Approved  
**Scope:** New translation models, Ollama polish upgrade, inference tuning, fine-tuning extension, paper revision tracking, slide decks

---

## 1. Goals and Priorities

1. **Translation quality** — BLEU/chrF scores above all else
2. **Speed/throughput** — chars/sec for bulk story translation
3. **Research completeness** — all viable models appear in survey/IEEE paper
4. **VRAM budget** — 8 GB RTX 5050 sm_120 (Blackwell, WSL2)

Scope: new model backends + Ollama polish upgrade + inference optimizations + fine-tuning extension + paper revision tracking + slide decks.

---

## 2. New Seq2Seq Models

Two new translator files, registered via `@register_model`. MADLAD-400-3B uses CTranslate2; SeamlessM4T-v2 runs directly via HuggingFace (custom architecture, not CT2-convertible).

### 2.1 MADLAD-400-3B

| Property | Value |
|----------|-------|
| HF ID | `google/madlad400-3b-mt` |
| Architecture | T5-based encoder-decoder |
| FLORES-200 bn→en BLEU | ~36 (zero-shot) |
| CT2 float16 VRAM | ~3 GB |
| File | `src/bn_en_translate/models/madlad.py` |
| Registry key | `madlad-3b` |

**Tokenization:** T5 tokenizer. Target language specified as forced BOS token `<2en>`. No SentencePiece language prefix in source — different from NLLB.

**Download:** `python scripts/download_models.py --model madlad-3b`  
Converts to CT2 float16 at `models/madlad-3b-ct2/`.

### 2.2 SeamlessM4T-v2-medium

| Property | Value |
|----------|-------|
| HF ID | `facebook/seamless-m4t-v2-large` |
| Architecture | Custom encoder-decoder (`SeamlessM4Tv2ForTextToText`) |
| FLORES-200 bn→en BLEU | ~38–40 |
| HF float16 VRAM | ~3.5 GB |
| File | `src/bn_en_translate/models/seamless.py` |
| Registry key | `seamless-medium` |

**Tokenization:** Uses `AutoProcessor` with explicit `src_lang="ben"` / `tgt_lang="eng"`. Not a standard seq2seq pipeline — requires custom `_translate_batch` implementation using `SeamlessM4Tv2ForTextToText.generate()`.

**Download:** `python scripts/download_models.py --model seamless-medium`  
HuggingFace pipeline (no CT2 conversion — model uses its own optimized path).

### 2.3 Long-overdue downloads

Also trigger download of already-coded-but-missing models:
- `nllb-1.3B` → `models/nllb-1.3B-ct2/`
- `indicTrans2-1B` → `models/indicTrans2-1B-ct2/`

This gives a complete 5-model benchmark: nllb-600M, nllb-1.3B, indicTrans2-1B, madlad-3b, seamless-medium.

---

## 3. Ollama Polish Model Upgrade

No new code beyond a CLI flag.

### 3.1 CLI flag
Add `--ollama-model` to `cli.py` (currently the model is only settable via `PipelineConfig` programmatically). Routes to the existing `PipelineConfig.ollama_model` string field.

### 3.2 Recommended models

| Model | Ollama tag | VRAM (q4_K_M) | Notes |
|-------|-----------|---------------|-------|
| Gemma 3 4B | `gemma3:4b` | ~2.8 GB | Fast, strong multilingual |
| Gemma 3 12B | `gemma3:12b` | ~7.3 GB | Best literary quality; leaves ~0.7 GB headroom |
| TowerInstruct 7B | `tower:7b` *(verify Ollama tag — may need manual GGUF import)* | ~4.5 GB | Translation-specialized instruction model |
| Qwen 2.5 7B | `qwen2.5:7b-instruct-q4_K_M` | ~4.7 GB | Current default — remains supported |

**New default:** `gemma3:12b` (updated in `PipelineConfig.ollama_model`).

**VRAM note:** With SeamlessM4T-medium (3.5 GB) + Gemma 3 12B (7.3 GB) = 10.8 GB — exceeds 8 GB. Pipeline already auto-unloads the translator before the Ollama pass; this remains the contract.

---

## 4. Inference Optimizations

### 4.1 Flash Attention 2

For HuggingFace-loaded models (IndicTrans2, MADLAD-400, SeamlessM4T):

```python
attn_implementation = "flash_attention_2" if _flash_attn_available() else "eager"
model = AutoModelForSeq2SeqLM.from_pretrained(..., attn_implementation=attn_implementation)
```

Gated behind `_flash_attn_available()` helper that checks `importlib.util.find_spec("flash_attn")`. Falls back silently. Added to `ModelConfig` as `use_flash_attention: bool = True`.

**Expected gain:** ~1.5–2× throughput on sequences >128 tokens, no quality change.

### 4.2 Per-model beam size defaults

Each translator class defines `DEFAULT_BEAM_SIZE: int`. `TranslatorBase._translate_batch` uses `self.config.beam_size` only if the user explicitly set it (not the dataclass default); otherwise falls back to `DEFAULT_BEAM_SIZE`.

| Model | `DEFAULT_BEAM_SIZE` | Rationale |
|-------|--------------------|-----------| 
| NLLB-600M | 4 | Current, good balance |
| NLLB-1.3B | 5 | Larger capacity |
| MADLAD-3B | 4 | T5 standard |
| SeamlessM4T-medium | 5 | Meta recommendation |
| IndicTrans2-1B | 5 | AI4Bharat recommendation |

**Implementation note:** Distinguish "user set" from "dataclass default" by adding `beam_size: int | None = None` to `ModelConfig`, defaulting to `None` (meaning "use translator default"). Validators updated accordingly.

### 4.3 Dynamic batch size config

Promote hardcoded `max_batch_size=32` in CT2 translators to `ModelConfig.max_ct2_batch_size: int = 32`. Non-CT2 models use `inference_batch_size` as before.

---

## 5. Fine-Tuning Extension

Done in two phases:

### Phase 1 — Benchmark all 5 models
Run `scripts/benchmark.py --models nllb-600M nllb-1.3B indicTrans2-1B madlad-3b seamless-medium` on the 90-sentence corpus and the Samanantar test set. The best performer by BLEU becomes the fine-tuning target.

### Phase 2 — Extend `trainer.py`
Add `--model` flag to `scripts/finetune.py`. `NLLBFineTuner` is renamed `Seq2SeqFineTuner` and generalized to accept any seq2seq model ID. MADLAD-400 uses the same LoRA + `Seq2SeqTrainer` pattern (T5 architecture is PEFT-compatible). `FineTuneConfig.output_dir` already accepts arbitrary paths.

Fine-tuning extended to all seq2seq models after Phase 1 identifies the winner.

---

## 6. Paper Revision Tracking

### 6.1 Directory structure

```
paper/
  archive/
    ieee_paper_2026-04-02_v1.tex
    survey_paper_2026-04-02_v1.tex
    ieee_paper_v1_to_v2.diff
    survey_paper_v1_to_v2.diff
    slides/
      (versioned slide snapshots)
  slides/
    ieee_slides.tex
    survey_slides.tex
    overview.md
    survey_reveal.html
  ieee_paper.tex
  survey_paper.tex
  PAPER_REVISIONS.md
```

### 6.2 PAPER_REVISIONS.md format

Each revision entry:

```markdown
## v2 — YYYY-MM-DD (short description)
### ieee_paper.tex
- <bullet: what changed>
### survey_paper.tex
- <bullet: what changed>
### Slides
- <bullet: what changed>
Archived: paper/archive/ieee_paper_YYYY-MM-DD_v1.tex
Diff: paper/archive/ieee_paper_v1_to_v2.diff
```

### 6.3 Process
Before any paper or slide edit:
1. Copy current `.tex`/`.md`/`.html` files to `paper/archive/` with date+version stamp
2. After edits, generate `.diff` files with `diff -u old new > archive/..._vX_to_vY.diff`
3. Append entry to `PAPER_REVISIONS.md`

The `paper_writer` agent instructions are updated to follow this process.

---

## 7. Slide Decks

### 7.1 IEEE paper slides (`paper/slides/ieee_slides.tex`)
**Format:** LaTeX Beamer, same bibliography as `ieee_paper.tex`  
**Sections:** Motivation → Pipeline architecture → Model comparison table → BLEU results (domain breakdown) → Fine-tuning results → Hardware deployment challenges → Conclusion  
**Estimated:** ~20 slides

### 7.2 Survey slides (`paper/slides/survey_slides.tex`)
**Format:** LaTeX Beamer  
**Sections:** Bengali NLP landscape → Survey scope (2019–2025) → BLEU trend chart → Pareto frontier → System-by-system comparison → Research gaps → Future directions  
**Estimated:** ~25 slides

### 7.3 Project overview (`paper/slides/overview.md`)
**Format:** Marp markdown  
**Audience:** General technical audience, no LaTeX required  
**Content:** 10–15 slides: what the project does, hardware, models, results, how to run  

### 7.4 Interactive survey (`paper/slides/survey_reveal.html`)
**Format:** Reveal.js  
**Content:** Survey slides with syntax-highlighted code snippets and interactive Pareto chart (D3.js or Chart.js inline)  
**Estimated:** ~20 slides

---

## 8. Docs Updates

| File | Changes |
|------|---------|
| `docs/MODELS.md` | Add MADLAD-400-3B, SeamlessM4T-v2-medium rows; add Gemma 3 Ollama options; update compute type table |
| `docs/ARCHITECTURE.md` | Document `use_flash_attention`, `max_ct2_batch_size`, per-model beam defaults |
| `docs/TRAINING.md` | Document `--model` flag on `finetune.py`; rename `NLLBFineTuner` → `Seq2SeqFineTuner` |

---

## 9. Tests

- `tests/unit/test_madlad.py` — MockTranslator pattern, no GPU
- `tests/unit/test_seamless.py` — MockTranslator pattern, no GPU
- `tests/unit/test_cli_ollama_model_flag.py` — `--ollama-model gemma3:12b` routes correctly
- `tests/unit/test_beam_defaults.py` — per-model beam defaults applied when `beam_size=None`
- `tests/unit/test_model_config.py` — updated for `beam_size: int | None`, `max_ct2_batch_size`, `use_flash_attention`

---

## 10. Key Invariants (unchanged)

1. `TranslatorBase.translate()` raises `RuntimeError` if called before `load()`
2. `Chunker.chunk()` never splits mid-sentence
3. `reassemble()` output has same paragraph count as normalized input
4. All file I/O is UTF-8 with explicit `encoding="utf-8"`
5. Pipeline auto-unloads translator before Ollama pass (VRAM constraint)
