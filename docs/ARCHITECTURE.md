# Architecture

## Pipeline Overview

```
Bengali .txt (UTF-8)
        │
        ▼
┌─────────────────┐
│  Preprocessor   │  unicodedata.normalize("NFC"), collapse whitespace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunker      │  Split by paragraph → sentence boundaries (danda ।/॥)
└────────┬────────┘  Max 400 tokens/chunk. Returns ChunkResult(text, para_id, chunk_id)
         │
         ▼
┌─────────────────┐
│   Translator    │  CTranslate2 float16 GPU (preferred)
└────────┬────────┘  HuggingFace pipeline (fallback if CT2 model missing)
         │           Batch size = 8 by default
         ▼
┌─────────────────┐
│ Postprocessor   │  Group by para_id, join sentences, fix MT artifacts
└────────┬────────┘  (doubled articles, leading spaces)
         │
         ▼  [optional]
┌─────────────────┐
│  Ollama Polish  │  qwen2.5:7b-instruct — literary tone refinement
└────────┬────────┘  Only if --ollama-polish flag set
         │
         ▼
English .txt (UTF-8)
```

---

## Layer Responsibilities

### Config Layer (`config.py`)

Three dataclasses validated at construction time — no lazy validation:

```python
ChunkConfig(max_tokens_per_chunk=400, batch_size=8, min_chunk_sentences=1)
ModelConfig(model_name="nllb-600M", device="cuda", compute_type="int8", beam_size=4)
PipelineConfig(model=ModelConfig(), chunk=ChunkConfig(), ollama_polish=False)
```

`ModelConfig.device` accepts `"auto"` — resolved to `"cuda"` or `"cpu"` at load time.

### Model Layer (`models/`)

**Hierarchy:**

```
TranslatorBase (ABC)
├── NLLBTranslator          — HuggingFace pipeline, no CT2 model needed
├── NLLBCt2Translator       — CTranslate2 float16, preferred for NLLB
├── IndicTrans2Translator   — HuggingFace + optional IndicTransToolkit
├── IndicTrans2Ct2Translator — CTranslate2 float16, preferred for IndicTrans2
└── OllamaTranslator        — HTTP to local Ollama daemon
```

**Lifecycle contract** (all implementations must follow):

```python
translator.load()          # download / load into VRAM
translations = translator.translate(texts, src_lang, tgt_lang)
translator.unload()        # free VRAM

# Or equivalently:
with translator:
    translations = translator.translate(...)
```

Calling `translate()` before `load()` raises `RuntimeError`.

**Factory routing** (`factory.py`):

```
get_translator(config)
├── model_name = "nllb-600M" or "nllb-1.3B"
│   ├── models/nllb-{N}-ct2/ exists? → NLLBCt2Translator
│   └── else                          → NLLBTranslator (HF)
├── model_name = "indicTrans2-1B"
│   ├── models/indicTrans2-1B-ct2/ exists? → IndicTrans2Ct2Translator
│   └── else                                → IndicTrans2Translator (HF)
└── model_name = "ollama"              → OllamaTranslator
```

### Pipeline Layer (`pipeline/`)

`TranslationPipeline` composes the stages. It is stateless between calls — the same instance can translate multiple files.

```python
pipeline = TranslationPipeline(translator, config)
with translator:
    result_a = pipeline.translate(text_a)
    result_b = pipeline.translate(text_b)   # translator stays loaded
```

**Chunker invariants:**
- Never splits mid-sentence (splits only at danda `।`, `॥`, or end-of-paragraph)
- Each chunk ≤ `max_tokens_per_chunk` (default 400) to fit 512-token model context
- Token estimation: ~4.5 chars/token (Bengali multi-byte UTF-8 average)
- Returns `ChunkResult(text, para_id, chunk_id)` — `para_id` drives reassembly

**Postprocessor invariants:**
- Output paragraph count = input paragraph count (always)
- Strips leading/trailing whitespace per paragraph
- Collapses repeated articles (`the the` → `the`)

---

## CTranslate2 Source Tokenisation Format

NLLB-200 and IndicTrans2 both use the M2M-100 architecture. The correct CT2 source format is:

```
[text_tokens..., </s>, src_lang_token]
```

**Target prefix:** `[tgt_lang_token]` (forces output language as BOS)

Wrong format (`[src_lang, text_tokens...]`) produces degenerate looping output with no error raised.

Language token positions in the shared vocabulary:
- `ben_Beng` → index 256026
- `eng_Latn` → index 256047

---

## Compute Type Selection

On CUDA, `NLLBCt2Translator._best_compute_type()` probes each compute type in preference order by running a real ~20-token translation:

```
int8_float16 → int8 → float16 → bfloat16 → float32
```

Short probes (< 10 tokens) do not trigger INT8 cuBLAS matmuls and produce false positives. The probe uses an English sentence of ~20 tokens.

**Observed behaviour on RTX 5050 sm_120 + CTranslate2 4.7.1 + CUDA 12.4:**
- `int8`, `int8_float16`, `int8_float32` → `CUBLAS_STATUS_NOT_SUPPORTED`
- `float16`, `bfloat16`, `float32` → ✅ working

Selected at load time: **float16**.

---

## VRAM Budget (8 GB total)

| Configuration | VRAM at load | VRAM during inference | Headroom |
|---|---|---|---|
| NLLB-600M CT2 float16 | ~2.0 GB | ~2.1 GB | ~5.9 GB |
| NLLB-1.3B CT2 float16 | ~2.6 GB | ~2.7 GB | ~5.3 GB |
| IndicTrans2-1B CT2 float16 | ~3.0 GB | ~3.1 GB | ~4.9 GB |
| Ollama qwen2.5:7b | ~4.7 GB | ~4.7 GB | ~3.3 GB |
| IndicTrans2 + Ollama | ~7.7 GB | OOM risk | ⚠️ Unload before switching |

The pipeline unloads the translator before running the Ollama polish pass when `--ollama-polish` is set.

---

## Extension Points

### Adding a new model backend

1. Create `src/bn_en_translate/models/<name>_ct2.py` extending `TranslatorBase`
2. Implement `load()`, `unload()`, `_translate_batch()`
3. In `load()`: load SPM first, call `_best_compute_type(device, sp)`, then load CT2 Translator
4. Use the M2M source format: `tokens + ["</s>", src_lang]`
5. Register in `factory.py` with CT2-first path check
6. Add to `scripts/download_models.py` MODELS dict
7. Write unit test for tokenisation format and compute type probe

### Adding a new pipeline stage

1. Write a pure function in `pipeline/` (e.g., `quality_filter.py`)
2. Add a field to `PipelineConfig` if it needs configuration
3. Wire into `TranslationPipeline.translate()` between existing stages
4. Unit test the function independently; integration test via mock pipeline
