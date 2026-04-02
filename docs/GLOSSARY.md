# Glossary

AI/ML terminology as it applies to the `bn-en-translate` project.

---

## Model Architecture

### Transformer

The neural network architecture underlying all models in this project. Introduced in "Attention Is All You Need" (Vaswani et al., 2017). Transformers process tokens in parallel (unlike RNNs which process sequentially) by applying self-attention: each token attends to all other tokens in the sequence simultaneously.

### Seq2Seq (Sequence-to-Sequence)

A model that maps an input sequence of tokens to an output sequence of tokens, where the two sequences may have different lengths. Translation is a seq2seq task: a Bengali sentence of N tokens maps to an English sentence of M tokens, where N ≠ M in general. In this project, `NLLBFineTuner` uses `transformers.AutoModelForSeq2SeqLM`.

### Encoder-Decoder

The two-part structure of seq2seq transformers. The **encoder** reads the entire input sequence and produces a sequence of hidden-state vectors (the context). The **decoder** generates the output sequence one token at a time, attending to the encoder context at each step. NLLB-200 and IndicTrans2 both use this architecture.

### M2M-100

The multilingual translation architecture that NLLB-200 is based on. "Many-to-Many" — trained to translate directly between any pair of 200 languages without routing through English as an intermediate pivot. Uses shared encoder and decoder across all language pairs, with language tokens (`ben_Beng`, `eng_Latn`) to specify the input and output languages. IndicTrans2 also follows this architecture.

### Attention Mechanism

The core operation of a transformer layer. Given a query `Q`, keys `K`, and values `V`, attention computes `softmax(QK^T / sqrt(d_k)) × V`. This produces a weighted sum of value vectors, where the weights reflect how "relevant" each key is to the query. Higher weights mean more attention. In translation, the decoder attends to encoder states to decide which parts of the source to focus on for each output token.

### Multi-Head Attention

Running the attention mechanism `h` times in parallel with different learned projections, then concatenating the results. NLLB-600M uses multi-head attention with 8 heads. Each head learns to attend to different linguistic patterns (syntactic structure, named entities, positional relationships). The LoRA adapters in this project target the four projection matrices within the attention layers: `q_proj`, `k_proj`, `v_proj`, `out_proj`.

### Language Token

A special vocabulary token that identifies the language of a sequence. For NLLB-200: `ben_Beng` (Bengali, Bengali script, vocabulary index 256026) and `eng_Latn` (English, Latin script, vocabulary index 256047). During inference, the source language token appears at the end of the source token sequence (`tokens + ["</s>", "ben_Beng"]`), and the target language token is forced as the first output token (BOS) to tell the decoder which language to generate.

### BOS / EOS Tokens

**BOS** (Beginning of Sequence): the special token placed at the start of the decoder's input during generation. In NLLB-200, the BOS is forced to be the target language token (e.g., `eng_Latn`), which controls the output language. **EOS** (End of Sequence): the token `</s>` that signals the end of a sequence. In NLLB source format, `</s>` appears before the source language token: `[text_tokens..., </s>, src_lang]`.

### Tokenization

The process of splitting raw text into tokens (subword units) that the model's vocabulary can represent. This project uses SentencePiece BPE tokenization (see below).

### SentencePiece

A language-agnostic subword tokenizer used by NLLB-200 and IndicTrans2. SentencePiece operates directly on Unicode characters without requiring pre-tokenized text, making it well-suited for scripts like Bengali that use different word-boundary conventions than European languages. The tokenizer model file is `sentencepiece.bpe.model`, co-located with the CT2 model weights.

### BPE (Byte Pair Encoding)

The subword segmentation algorithm used by SentencePiece in this project. BPE starts with individual characters and iteratively merges the most frequent adjacent pair into a new subword unit. The resulting vocabulary of ~250K units covers all 200 NLLB languages. Bengali words are typically represented as 2–5 subword tokens.

---

## Training

### Fine-tuning vs Pre-training

**Pre-training** trains a model from random weights on a large, general-purpose dataset (NLLB-200 was pre-trained on 1.6 billion sentence pairs across 200 languages). Pre-training requires months of compute on hundreds of GPUs. **Fine-tuning** continues training a pre-trained model on a smaller, domain-specific dataset to improve performance on a narrower task. In this project, we fine-tune NLLB-600M on ~7 863 Bengali-English pairs from the Samanantar corpus to improve translation quality for literary text.

### LoRA (Low-Rank Adaptation)

A parameter-efficient fine-tuning method (Hu et al., 2021). Instead of modifying the original weight matrices, LoRA injects trainable rank-decomposition matrices alongside the frozen weights. For a weight `W ∈ R^(d×k)`, LoRA adds `ΔW = B × A` where `A ∈ R^(r×k)`, `B ∈ R^(d×r)`, and rank `r << min(d, k)`. Only `A` and `B` are trained.

**`lora_r` (rank):** Controls the number of dimensions in the adapter matrices. Higher rank gives more expressive capacity but uses more VRAM and compute. The default `r=16` gives ~0.76% trainable parameters for NLLB-600M.

**`lora_alpha` (scaling factor):** Scales the adapter contribution by `alpha / r`. With `alpha=32, r=16`, the scale is 2.0. Higher alpha makes the adapter updates larger relative to the frozen base weights. Setting `alpha = 2 × r` is a common heuristic.

### PEFT (Parameter-Efficient Fine-Tuning)

The HuggingFace library that implements LoRA and other adapter methods. Key classes used in this project: `LoraConfig` (configures the adapter), `get_peft_model()` (wraps a base model with the adapters), `merge_and_unload()` (merges the trained adapters into the base weights for deployment).

### Gradient Accumulation

Running several forward/backward passes before calling the optimizer step, accumulating gradients from each micro-batch. Effective batch size = `train_batch_size × gradient_accumulation_steps`. With `train_batch_size=8` and `gradient_accumulation_steps=4`, the effective batch size is 32, without requiring 32 samples to fit in VRAM simultaneously. Used to simulate larger batches on limited-VRAM hardware.

### bf16 / fp16

**fp16** (16-bit floating point, IEEE 754 half precision): 5-bit exponent, 10-bit mantissa. Range ~6×10^-8 to 65504. Prone to underflow for small gradients. Requires GradScaler for stable training.

**bf16** (bfloat16): 8-bit exponent, 7-bit mantissa. Same exponent range as float32 (range ~10^-38 to 3.4×10^38), but lower precision. Does not require GradScaler. This project uses bf16 for training on CUDA (RTX 5050 sm_120 supports bf16 natively in PyTorch `2.7.0+cu128`).

### GradScaler

PyTorch's automatic mixed precision (AMP) tool for fp16 training. Because fp16 has a small exponent range, gradients can underflow to zero. GradScaler multiplies the loss by a large scale factor before backpropagation, then divides the gradients back down before the optimizer step. bf16's wider exponent range eliminates this problem, so `GradScaler` is not needed and is not used. Using fp16 AMP with mixed-dtype LoRA models (float32 base + float16 adapters) can produce incorrect scale factors and NaN gradients — another reason to prefer bf16.

### Overfitting

When a model learns the training data too closely and performs worse on unseen data. Signs: training loss decreases but validation loss increases (loss divergence). Mitigations used in this project: LoRA dropout (`lora_dropout=0.1`), weight decay, loading the best checkpoint by `eval_loss` (`load_best_model_at_end=True`).

### Learning Rate

The step size for gradient descent updates. Too high: unstable training, loss diverges. Too low: training stalls. LoRA fine-tuning typically uses a higher LR than full fine-tuning (e.g., `2e-4` vs `5e-5`) because only a small fraction of parameters are being updated.

### Warmup Steps

The number of gradient steps during which the learning rate linearly increases from 0 to the target learning rate. Prevents large weight updates at the start of training when gradients are noisy. Default: 100 steps.

---

## Inference

### CTranslate2 (CT2)

An inference-optimised runtime for transformer models, developed by OpenNMT. CTranslate2 compiles models to an efficient binary format and implements hardware-specific kernels for batched inference. Key advantages over HuggingFace pipeline inference: 2–4x faster, lower VRAM usage, supports quantization. Models must be converted using `ct2-transformers-converter` before use. This project uses CT2 as the primary inference backend for NLLB-600M.

### Beam Search

A decoding algorithm for seq2seq models that maintains the N most probable partial sequences (the "beam") at each decoding step. At each step, each hypothesis is extended by all possible next tokens, and the top N are kept. `beam_size=4` is the default in this project. Higher beam sizes generally improve translation quality but increase latency linearly. Beam size 1 is greedy decoding.

### Quantization

Representing model weights in a lower-precision numeric format to reduce memory usage and increase inference speed. This project uses float16 quantization for CT2 models (`--quantization float16`). INT8 quantization would halve VRAM further but fails on the RTX 5050 sm_120 GPU due to a missing cuBLAS kernel (`CUBLAS_STATUS_NOT_SUPPORTED`). See `_best_compute_type()` for the runtime probe that selects the best available compute type.

### Batch Inference

Translating multiple sequences simultaneously in a single forward pass. CTranslate2's `translate_batch()` processes a list of tokenized sequences together, using GPU parallelism. The pipeline uses `batch_size=8` (chunks sent to the translator at once) and CT2's internal `max_batch_size=32` (maximum sequences processed in a single CT2 call).

### max_batch_size

The maximum number of sequences CTranslate2 will process in a single GPU call (`max_batch_size=32` in `NLLBCt2Translator._translate_batch()`). If the input batch is larger, CT2 splits it internally. This prevents VRAM spikes from large batches. The effective GPU batch is bounded by this value.

---

## Evaluation

### BLEU Score

Bilingual Evaluation Understudy. The standard automatic metric for machine translation quality. BLEU measures the overlap between n-grams in the model's output and the reference translation, with a brevity penalty for short outputs. Scores range from 0 to 100 (higher is better). This project uses SacreBLEU for reproducible computation.

Rough quality tiers for Bengali-English:
- < 20: Poor, major errors, hard to understand
- 20–40: Adequate, main meaning conveyed, noticeable errors
- 40–60: Good, mostly correct, minor errors
- > 60: Near-human, on in-domain test sets

Current baseline: BLEU ~56.2 on the built-in 90-sentence corpus.

### Corpus BLEU vs Sentence BLEU

**Corpus BLEU** (`sacrebleu.corpus_bleu()`) computes a single BLEU score over the entire test set, accumulating n-gram counts across all sentences before computing the final score. This is the standard and is more statistically stable. **Sentence BLEU** computes a score for each sentence individually and averages them — not used here because it inflates scores for short sentences.

### Why BLEU Is Low for Literary Text

BLEU measures surface-level n-gram overlap. Literary translation involves many factors that BLEU does not capture well:
- **Named entities:** Bengali proper nouns transliterated differently from the reference still produce a valid translation but score zero.
- **Style:** Paraphrases that preserve meaning but use different wording score low even if the translation is excellent.
- **Cultural references:** An accurate contextual translation of an idiom will not match the reference literal translation.
- **Single reference:** BLEU is most reliable with multiple reference translations. This project uses one reference per sentence.

BLEU is used for regression detection (catching quality drops) rather than as an absolute quality measure.

---

## Infrastructure

### CUDA

NVIDIA's parallel computing platform and programming model for GPU acceleration. CUDA kernels are programs compiled for a specific GPU architecture (compute capability). This project requires CUDA 12.x drivers.

### sm_120 (Blackwell compute capability)

The compute capability version of the NVIDIA RTX 5050 Laptop GPU (Blackwell architecture). The `sm_120` designation means streaming multiprocessor architecture version 12.0. PyTorch `2.7.0+cu128` ships pre-compiled kernels for sm_120. Older PyTorch builds fall back to PTX JIT compilation, which works for inference but historically missed some training op implementations.

### VRAM vs RAM

**VRAM** (Video RAM): memory on the GPU die, used to store model weights and activation tensors during GPU computation. Fast but limited — the RTX 5050 has 8 GB. Models must fit in VRAM during inference; training requires additional space for gradients and optimizer states. **RAM** (system memory): CPU-accessible memory, 16 GB on the target machine. Used for data loading, tokenization, and CPU-side operations.

### WSL2

Windows Subsystem for Linux 2. A Linux environment running on Windows via a lightweight virtual machine. This project runs in WSL2 on Windows 11. The NVIDIA GPU is shared between Windows and WSL2 via a special driver bridge at `/usr/lib/wsl/lib/libcuda.so.1`. The `LD_LIBRARY_PATH` must include `/usr/lib/wsl/lib` for both PyTorch and CTranslate2 to find the CUDA runtime. Without it, both silently fall back to CPU with no error message.

### LoRA Adapter

The set of trained `A` and `B` matrices produced by LoRA fine-tuning. Stored separately from the base model weights in `models/nllb-600M-finetuned/adapter/` as small `.bin` files (a few MB, vs ~1.2 GB for the full model). The adapter can be applied to the base model at inference time without merging, or merged permanently for deployment.

### CT2 Export / Model Conversion

The process of converting a HuggingFace transformers model to CTranslate2 format. Done by `ct2-transformers-converter` (command line) or `python -m ctranslate2.tools.transformers`. The converter reads the HuggingFace model config and weight files, re-packages the weights in CT2's binary format, and optionally quantizes them. The output directory contains CTranslate2 model binary files plus the SentencePiece tokenizer file. This conversion is required once per model version and produces a static artifact that can be loaded quickly at inference time.
