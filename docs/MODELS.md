# Models

## Supported Models

| Model | HuggingFace ID | CT2 Dir | VRAM (float16) | BLEU bn→en | Speed |
|-------|---------------|---------|---------------|------------|-------|
| `nllb-600M` | `facebook/nllb-200-distilled-600M` | `models/nllb-600M-ct2/` | ~2.0 GB | ~64 (in-domain), ~22 (open) | ~1000 chars/s |
| `nllb-1.3B` | `facebook/nllb-200-distilled-1.3B` | `models/nllb-1.3B-ct2/` | ~2.6 GB | ~26 (open) | ~700 chars/s |
| `indicTrans2-1B` | `ai4bharat/indictrans2-indic-en-1B` | `models/indicTrans2-1B-ct2/` | ~3.0 GB | ~30+ (open) | ~700 chars/s |
| `ollama` | N/A (local daemon) | N/A | ~4.7 GB | subjective | ~200 chars/s |

> BLEU scores are approximate. In-domain literary Bengali BLEU for NLLB-600M is ~64 on the bundled 90-sentence corpus; open-domain numbers reflect FLORES-200 estimates.

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

---

## Ollama (Literary Polish)

Not a translation model — used as a post-processing step to improve prose quality.

**Model:** `qwen2.5:7b-instruct` (or any instruction-tuned model available locally)
**VRAM:** ~4.7 GB
**Purpose:** Converts raw MT output into fluent English prose

### Setup

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull qwen2.5:7b-instruct

# Verify daemon is running
curl http://localhost:11434/api/tags
```

### Usage

```bash
bn-translate --input story.bn.txt --output story.en.txt --model nllb-600M --ollama-polish
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
