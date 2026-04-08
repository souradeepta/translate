# Models

## Supported Models

| Key | HuggingFace ID | Backend | VRAM (float16) | FLORES BLEU bn→en | Default Beams | Status |
|-----|---------------|---------|---------------|-------------------|---------------|--------|
| `nllb-600M` | `facebook/nllb-200-distilled-600M` | CT2 float16 | ~2.0 GB | 22 | 4 | Working — `models/nllb-600M-ct2/` |
| `nllb-1.3B` | `facebook/nllb-200-distilled-1.3B` | CT2 float16 | ~2.6 GB | 26 | 4 | Needs download |
| `indicTrans2-1B` | `ai4bharat/indictrans2-indic-en-1B` | CT2 float16 | ~3.0 GB | 44 | 5 | Needs download |
| `madlad-3b` | `google/madlad400-3b-mt` | HF float16 | ~3.0 GB | 36 | 4 | Needs download |
| `seamless-medium` | `facebook/seamless-m4t-v2-large` | HF float16 | ~3.5 GB | ~39 | 5 | Needs download |
| `ollama` | N/A (local daemon) | Ollama HTTP | ~4.7 GB | subjective | N/A | Optional polish |

> FLORES BLEU scores reflect zero-shot performance on FLORES-200 Bengali→English.  
> The in-domain 90-sentence corpus BLEU for NLLB-600M is ~64 — substantially higher due to corpus overlap with literary Bengali.  
> "Default Beams" is each model's `DEFAULT_BEAM_SIZE`; override with `ModelConfig(beam_size=N)`.

---

## Per-Model Beam Size Defaults

Each translator class declares a `DEFAULT_BEAM_SIZE` class variable. When `ModelConfig.beam_size` is `None` (the default), `_effective_beam_size()` returns this class-level default. Pass an explicit `beam_size` in `ModelConfig` to override for all models.

| Model | `DEFAULT_BEAM_SIZE` |
|-------|-------------------|
| `NLLBCt2Translator` | 4 |
| `IndicTrans2Translator` | 5 |
| `MADLADTranslator` | 4 |
| `SeamlessTranslator` | 5 |

---

---

## NLLB-200

**Architecture:** M2M-100 (seq2seq Transformer)
**License:** CC-BY-NC 4.0
**Languages:** 200 languages — covers Bengali (`ben_Beng`) and English (`eng_Latn`)

### Variants

| Variant | Parameters | CT2 float16 VRAM | Notes |
|---------|-----------|-----------------|-------|
| distilled-600M | 600M | ~2.0 GB | Fastest; best for development |
| distilled-1.3B | 1.3B | ~2.6 GB | Better quality, still fits easily in 8 GB |

### Language Token IDs (shared vocabulary)

```
ben_Beng → index 256026
eng_Latn → index 256047
```

Verify with:
```bash
python3 -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('models/nllb-600M-ct2/sentencepiece.bpe.model')
print(sp.piece_to_id('ben_Beng'))   # should be 256026
print(sp.piece_to_id('eng_Latn'))   # should be 256047
"
```

### CTranslate2 Source Format

```
source tokens: [token1, token2, ..., </s>, src_lang]
target prefix: [tgt_lang]
```

**Wrong format** (`[src_lang, token1, ...]`) produces looping garbage like "Ra Ra Ra Ra...".

### Download and Convert

```bash
python scripts/download_models.py --model nllb-600M
python scripts/download_models.py --model nllb-1.3B
```

Script uses `ct2-transformers-converter` and copies the SentencePiece `.model` file into the output directory.

---

## IndicTrans2-1B

**Architecture:** M2M-100 variant, optimized for Indic languages
**Source:** AI4Bharat
**License:** MIT
**Languages:** 22 Indic languages ↔ English

Best model for Bengali specifically — purpose-trained on Indic language pairs with more Bengali data than NLLB.

### IndicTransToolkit (optional)

The `IndicTransToolkit` library provides an `IndicProcessor` for script normalization and transliteration:

```bash
pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git
```

When installed, `IndicTrans2Ct2Translator.load()` picks it up automatically for pre/post-processing. Without it, raw SentencePiece tokenization is used (still works, slightly lower quality).

### Source Language Code

IndicTrans2 uses `ben_Beng` (same as NLLB) for Bengali.

### Download and Convert

```bash
python scripts/download_models.py --model indicTrans2-1B
```

This downloads ~3 GB from HuggingFace and converts to CT2 float16 (~3 GB on disk).

### Flash Attention 2

`IndicTrans2Translator` and `MADLADTranslator` will use Flash Attention 2 if the `flash-attn` package is installed and `ModelConfig.use_flash_attention=True` (default). This reduces memory bandwidth and improves throughput on Ampere and later GPUs.

```bash
pip install flash-attn --no-build-isolation
```

Without `flash-attn` installed, both translators fall back to the standard `"eager"` attention implementation with no quality difference.

---

## MADLAD-400-3B

**Architecture:** T5-based encoder-decoder  
**Source:** Google Research  
**License:** Apache 2.0  
**Languages:** 400 languages — supports Bengali and English  
**HF ID:** `google/madlad400-3b-mt`

Source text is prefixed with a target-language tag (e.g. `<2en> <bengali text>`). No source language tag is required. The class handles this prefix automatically.

### Download

```bash
python scripts/download_models.py --model madlad-3b
```

This uses `snapshot_download` from `huggingface_hub` — no CTranslate2 conversion is performed. The model is stored at `models/madlad-3b-hf/` and loaded at inference time as a standard HuggingFace model in float16.

---

## SeamlessM4T-v2

**Architecture:** Custom encoder-decoder (`SeamlessM4Tv2ForTextToText`)  
**Source:** Meta AI  
**License:** CC-BY-NC 4.0  
**Languages:** 100+ languages — supports Bengali and English  
**HF ID:** `facebook/seamless-m4t-v2-large`

SeamlessM4T uses short language codes (`ben`, `eng`) rather than FLORES-200 codes (`ben_Beng`, `eng_Latn`). `SeamlessTranslator` handles the mapping automatically — callers always pass FLORES-200 codes.

### Download

```bash
python scripts/download_models.py --model seamless-medium
```

Stored at `models/seamless-medium-hf/`. Loaded via HuggingFace Transformers; no CT2 conversion.

---

## HF Native vs CTranslate2 Backends

| Backend | Models | Inference path | Quantization |
|---------|--------|---------------|-------------|
| CTranslate2 (CT2) | `nllb-600M`, `nllb-1.3B`, `indicTrans2-1B` | `ctranslate2.Translator` | float16 (INT8 fails on sm_120) |
| HuggingFace native | `madlad-3b`, `seamless-medium` | `transformers.generate()` | float16 (auto, via `.to("cuda")`) |

MADLAD and Seamless do not have a CT2 conversion path — their architectures (T5 and SeamlessM4T) are fully supported by standard HuggingFace Transformers inference.

---

## Ollama (Literary Polish)

Not a translation model — used as a post-processing step to improve prose quality.

**Default model:** `gemma3:12b` (CLI default as of this branch)  
**VRAM:** ~4.7 GB (gemma3:12b Q4)  
**Purpose:** Converts raw MT output into fluent English prose

### Setup

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull gemma3:12b

# Or pull a lighter alternative
ollama pull gemma3:4b

# Verify daemon is running
curl http://localhost:11434/api/tags
```

### Usage

```bash
# Use default Ollama model (gemma3:12b)
bn-translate --input story.bn.txt --output story.en.txt --model nllb-600M --ollama-polish

# Override Ollama model via --ollama-model flag
bn-translate --input story.bn.txt --output story.en.txt \
    --model indicTrans2-1B --ollama-polish \
    --ollama-model gemma3:4b

# Use the older qwen2.5 model
bn-translate --input story.bn.txt --output story.en.txt \
    --model nllb-600M --ollama-polish \
    --ollama-model qwen2.5:7b-instruct-q4_K_M
```

### VRAM Note

Running IndicTrans2-1B (3 GB) + Ollama (4.7 GB) simultaneously = ~7.7 GB, close to OOM on 8 GB cards. The pipeline automatically unloads the translator before starting the Ollama pass.

---

## Compute Type Selection

At model load time, `_best_compute_type(device, sp)` probes each compute type in order:

```
int8_float16 → int8 → float16 → bfloat16 → float32
```

The probe runs a real ~20-token translation (not just a forward pass). Short probes (< 10 tokens) do not trigger the problematic INT8 cuBLAS matmuls on some hardware and give false positives.

**Results on RTX 5050 sm_120 + CTranslate2 4.7.1 + CUDA 12.4:**

| Compute Type | Result |
|---|---|
| `int8_float16` | `CUBLAS_STATUS_NOT_SUPPORTED` |
| `int8` | `CUBLAS_STATUS_NOT_SUPPORTED` |
| `int8_float32` | `CUBLAS_STATUS_NOT_SUPPORTED` |
| `float16` | ✅ Working |
| `bfloat16` | ✅ Working |
| `float32` | ✅ Working |

Selected: **float16** (best speed/memory trade-off among working types).

---

## Model Storage

Converted models go to `models/` in the project root (gitignored). Each CT2 model directory contains:

```
models/nllb-600M-ct2/
├── model.bin                  # CT2 binary (float16, ~594 MB)
├── shared_vocabulary.json     # token → index mapping
└── sentencepiece.bpe.model    # SentencePiece tokenizer (~4 MB)
```

The `sentencepiece.bpe.model` is copied from the HuggingFace cache by `download_models.py` since `ct2-transformers-converter` doesn't include it by default.
