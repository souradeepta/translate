# AI Primer: Every Concept in the bn-en-translate Papers

**Audience:** Computer science students who know Python and basic calculus but have not studied deep learning or NLP formally.

**Purpose:** Explain — from first principles — every mathematical, AI, and programming concept that appears in the `ieee_paper.tex` system paper and `survey_paper.tex` comparative survey. By the end of this document you should be able to read both papers and understand every equation, table, and design decision.

**Reading time:** ~3–4 hours end to end. Each section is self-contained; use the table of contents to skip what you already know.

---

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
   - 1.1 Vectors and Matrices
   - 1.2 Matrix Multiplication
   - 1.3 Probability and the Softmax Function
   - 1.4 Logarithms in ML
   - 1.5 Gradient Descent
   - 1.6 The Chain Rule (Backpropagation)
2. [Representing Language as Numbers](#2-representing-language-as-numbers)
   - 2.1 Tokenization and Vocabulary
   - 2.2 Byte-Pair Encoding (BPE)
   - 2.3 SentencePiece
   - 2.4 Embeddings
   - 2.5 Positional Encoding
3. [The Transformer Architecture](#3-the-transformer-architecture)
   - 3.1 Why Not RNNs?
   - 3.2 Self-Attention from Scratch
   - 3.3 Multi-Head Attention
   - 3.4 Feed-Forward Sub-Layer
   - 3.5 Layer Normalization and Residual Connections
   - 3.6 The Encoder Stack
   - 3.7 The Decoder Stack and Cross-Attention
   - 3.8 The Full Encoder-Decoder for Translation
4. [Neural Machine Translation](#4-neural-machine-translation)
   - 4.1 Sequence-to-Sequence Formulation
   - 4.2 Beam Search Decoding
   - 4.3 Language Tokens and Forced BOS
   - 4.4 NLLB-200 Token Format
5. [Multilingual and Massively Multilingual NMT](#5-multilingual-and-massively-multilingual-nmt)
   - 5.1 Shared Multilingual Vocabulary
   - 5.2 Cross-Lingual Transfer
   - 5.3 Mixture of Experts (MoE)
   - 5.4 NLLB-200, M2M-100, mBART-50
   - 5.5 IndicTrans2 and Script-Unified Tokenization
6. [Training Large Models](#6-training-large-models)
   - 6.1 The Cross-Entropy Loss
   - 6.2 The Adam Optimizer
   - 6.3 Learning Rate Scheduling and Warmup
   - 6.4 Gradient Accumulation
   - 6.5 Mixed Precision Training (fp16 and bf16)
   - 6.6 Why bf16 and Not fp16 on This Hardware
7. [Parameter-Efficient Fine-Tuning and LoRA](#7-parameter-efficient-fine-tuning-and-lora)
   - 7.1 Why Full Fine-Tuning Is Expensive
   - 7.2 Low-Rank Matrix Factorization
   - 7.3 The LoRA Equations Step by Step
   - 7.4 Which Layers to Adapt
   - 7.5 LoRA Hyperparameters: r, alpha, target_modules
   - 7.6 PEFT Library
   - 7.7 Exporting LoRA Weights Back to a Full Model
8. [Efficient Inference: CTranslate2 and Quantization](#8-efficient-inference-ctranslate2-and-quantization)
   - 8.1 Why Inference Is Different from Training
   - 8.2 Quantization: INT8 and float16
   - 8.3 CTranslate2 Architecture
   - 8.4 Why INT8 Fails on sm_120
   - 8.5 Batching and Max Batch Size
9. [Evaluation Metrics](#9-evaluation-metrics)
   - 9.1 BLEU — Full Derivation
   - 9.2 The Brevity Penalty
   - 9.3 SacreBLEU and Why Tokenization Matters
   - 9.4 chrF
   - 9.5 TER
   - 9.6 COMET
   - 9.7 Human Evaluation and Its Limits
10. [Datasets and Benchmarks](#10-datasets-and-benchmarks)
    - 10.1 FLORES-200
    - 10.2 Samanantar
    - 10.3 WMT Bengali-English
    - 10.4 Domain Diversity and Its Effect on BLEU
11. [Bengali Linguistics for NLP](#11-bengali-linguistics-for-nlp)
    - 11.1 Script and Unicode
    - 11.2 Word Order (SOV vs SVO)
    - 11.3 Agglutinative Morphology
    - 11.4 Zero Pronouns and Pro-Drop
    - 11.5 Code-Switching
12. [GPU Architecture and CUDA](#12-gpu-architecture-and-cuda)
    - 12.1 Why GPUs for ML
    - 12.2 CUDA and Compute Capability
    - 12.3 VRAM vs RAM
    - 12.4 Tensor Cores and Blackwell sm_120
    - 12.5 WSL2 and LD_LIBRARY_PATH
13. [The Four-Stage Translation Pipeline](#13-the-four-stage-translation-pipeline)
    - 13.1 Preprocessor: NFC Unicode Normalization
    - 13.2 Chunker: Token Budget and Sentence Boundaries
    - 13.3 Translator: Batched GPU Inference
    - 13.4 Postprocessor: Paragraph Reassembly
14. [Resource Monitoring and Run Databases](#14-resource-monitoring-and-run-databases)
    - 14.1 Why Monitor Resources?
    - 14.2 CPU and RAM Profiling with psutil
    - 14.3 GPU Monitoring with pynvml
    - 14.4 SQLite for Run History
    - 14.5 Regression Detection
15. [Software Engineering Concepts](#15-software-engineering-concepts)
    - 15.1 Test-Driven Development (TDD)
    - 15.2 The Strategy Pattern
    - 15.3 Context Managers
    - 15.4 Daemon Threads
16. [Scale, Efficiency, and the Pareto Frontier](#16-scale-efficiency-and-the-pareto-frontier)
    - 16.1 Scaling Laws
    - 16.2 Marginal Returns per Billion Parameters
    - 16.3 The Quality–VRAM Pareto Frontier
17. [Putting It All Together: Reading the Papers](#17-putting-it-all-together-reading-the-papers)

---

## 1. Mathematical Foundations

### 1.1 Vectors and Matrices

A **vector** is an ordered list of numbers. In ML, vectors represent things: a word embedding is a vector, a hidden state is a vector, a probability distribution is a vector.

```
v = [0.2, -1.4, 0.8, 3.1]   # a 4-dimensional vector
```

A **matrix** is a 2D grid of numbers. Dimensions are written rows × columns.

```
W = [[1, 0, 2],
     [0, 3, 1]]   # a 2×3 matrix
```

The **transpose** W^T flips rows and columns: a 2×3 matrix becomes a 3×2 matrix.

### 1.2 Matrix Multiplication

If A is m×k and B is k×n, then C = A·B is m×n. Each element C[i,j] is the dot product of row i of A and column j of B:

```
C[i,j] = sum(A[i,t] * B[t,j] for t in range(k))
```

**Why this matters for transformers:** Every layer in a transformer is fundamentally a sequence of matrix multiplications. When you hear "the attention computation is O(n²)", it's because each token's query is dot-producted with every token's key — n queries times n keys.

### 1.3 Probability and the Softmax Function

A **probability distribution** over N outcomes is a list of N non-negative numbers that sum to 1.

The **softmax** function converts any list of real numbers into a probability distribution:

```
softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for j in range(N))
```

Example: `softmax([2, 1, 0.1])` = `[0.659, 0.242, 0.099]`. The largest input gets the highest probability; all outputs sum to 1.

**Why this matters:** The final layer of a transformer language model outputs a raw score (logit) for every word in the vocabulary. Softmax converts these to probabilities, and the model picks (or samples from) this distribution to choose the next token.

### 1.4 Logarithms in ML

The **natural logarithm** `log(x)` is the inverse of `exp(x)`. Key properties:

- `log(a * b) = log(a) + log(b)` — multiplying probabilities becomes adding log-probabilities
- `log(x)` for 0 < x ≤ 1 gives values in (−∞, 0] — small probabilities give very negative logs

In ML, we almost always work in log space because:
1. Multiplying many small probabilities underflows to zero in floating point
2. Adding log-probabilities is numerically stable
3. The BLEU formula, cross-entropy loss, and beam search all use log-probabilities

### 1.5 Gradient Descent

Suppose you have a function `L(θ)` (the loss) that you want to minimize over parameters `θ` (millions of floating-point numbers representing the model's weights).

The **gradient** ∇L(θ) is a vector of partial derivatives — it points in the direction of steepest increase of L.

**Gradient descent** updates θ by taking a small step in the *opposite* direction:

```
θ ← θ - α · ∇L(θ)
```

where `α` is the **learning rate** (a small positive number like 0.0002).

Repeat this for many steps over many batches of training data and the loss (hopefully) decreases.

**Stochastic Gradient Descent (SGD):** Instead of computing the gradient over the entire dataset (expensive), compute it over a small random batch of examples. Noisy but fast.

### 1.6 The Chain Rule (Backpropagation)

The chain rule from calculus: if `z = f(y)` and `y = g(x)`, then `dz/dx = (dz/dy) · (dy/dx)`.

A neural network is a composition of many functions (layers). **Backpropagation** applies the chain rule repeatedly backward through all layers to compute the gradient of the loss with respect to every parameter. PyTorch does this automatically — you call `loss.backward()` and it fills in `.grad` for every parameter tensor.

The key insight: gradients flow backward through the same path that data flows forward. This is why both the forward pass and backward pass require GPU memory simultaneously — you need to keep all intermediate activations around to compute the backward pass.

---

## 2. Representing Language as Numbers

### 2.1 Tokenization and Vocabulary

Neural networks work with numbers, not text. **Tokenization** is the process of splitting a string into a sequence of integer IDs that the model can process.

A **vocabulary** is a fixed mapping from string pieces to integers. Example:
```
"Hello" → 12843
"world" → 6243
"." → 4
```

The vocabulary size determines the dimensionality of the model's output layer. NLLB-200 uses a vocabulary of ~256,000 tokens.

### 2.2 Byte-Pair Encoding (BPE)

**BPE** is the most common tokenization algorithm for neural NMT. It works by:

1. Start with a vocabulary of all individual characters.
2. Count the most frequent adjacent pair of tokens in the training corpus.
3. Merge that pair into a single new token.
4. Repeat until the vocabulary reaches the desired size.

Result: common words become single tokens; rare words are split into subword pieces.

```
"unhappiness" → ["un", "happiness"]
"खुश" (happy in Hindi) → ["खु", "श"]  # subword split for a non-English script
```

BPE handles out-of-vocabulary words gracefully because any word can be decomposed into known subword pieces, down to individual characters in the worst case.

### 2.3 SentencePiece

**SentencePiece** is a language-agnostic tokenizer (used by NLLB-200 and mBART-50) that:
- Operates on raw Unicode text without pre-tokenization
- Treats whitespace as a regular character (prefixed with `▁`)
- Supports both BPE and unigram language model algorithms

The key practical difference from standard BPE: SentencePiece tokenization is invertible (you can reconstruct the original string exactly from the token IDs), which matters for evaluation. When the papers refer to "SentencePiece normalization" in SacreBLEU, they mean computing BLEU after running the SentencePiece tokenizer consistently on both the hypothesis and reference.

### 2.4 Embeddings

Once a word is tokenized to an integer ID, it is looked up in an **embedding matrix** E of shape [vocabulary_size × d_model]. Each row is a learned vector of dimension d_model (e.g., 1024 for NLLB-600M).

```python
embedding_vector = E[token_id]  # shape: [d_model]
```

Embeddings are learned during training: similar words end up with similar vectors because they appear in similar contexts. The embedding table for a 256K vocabulary at d_model=1024 and float16 is 256000 × 1024 × 2 bytes ≈ 512 MB — a significant fraction of the model's VRAM footprint.

### 2.5 Positional Encoding

Transformers process all tokens in parallel (unlike RNNs which process left to right). But word order matters in language: "cat ate mouse" ≠ "mouse ate cat". Positional encoding injects order information by adding a position-dependent vector to each embedding.

The original paper (Vaswani et al., 2017) used sinusoidal functions:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

NLLB-200 uses learned positional embeddings (each position has a learnable vector), which works well up to the maximum sequence length seen during training (512 tokens for NLLB).

---

## 3. The Transformer Architecture

### 3.1 Why Not RNNs?

Before transformers, **Recurrent Neural Networks (RNNs)** and their variants (LSTMs, GRUs) were standard for NMT. An RNN processes tokens one at a time, maintaining a hidden state vector that summarizes everything seen so far.

**Problems with RNNs:**
1. **Sequential computation:** Token t cannot be processed until token t-1 is done. Can't parallelise across sequence positions → slow training.
2. **Vanishing gradients:** Gradients diminish as they flow backward through many steps, making it hard to learn long-range dependencies.
3. **Fixed-size context:** The hidden state has a fixed dimension regardless of sentence length, creating an information bottleneck.

Transformers solve all three by replacing sequential recurrence with parallel self-attention.

### 3.2 Self-Attention from Scratch

Self-attention lets every token in a sequence directly attend to every other token. The output for each token is a weighted sum of all other tokens' value vectors, where the weights reflect relevance.

**Step 1: Project each token's embedding into Q, K, V**

For each token embedding `x_i` (a vector of size d_model), compute three vectors using learned weight matrices:

```
q_i = x_i · W_Q    # query:  "what am I looking for?"
k_i = x_i · W_K    # key:    "what do I offer?"
v_i = x_i · W_V    # value:  "what information do I carry?"
```

All three weight matrices are d_model × d_k (where d_k = d_model / num_heads).

**Step 2: Compute attention scores**

For token i, compute a raw score against every token j:

```
score(i, j) = dot(q_i, k_j) / sqrt(d_k)
```

The `/ sqrt(d_k)` prevents scores from growing large when d_k is large (which would push softmax into a saturated, low-gradient region).

**Step 3: Normalise into attention weights**

```
a(i, j) = softmax over j of score(i, j)
```

This gives a probability distribution: how much token i should "attend to" token j.

**Step 4: Compute the output**

```
output_i = sum(a(i, j) * v_j  for j in range(n))
```

The output for token i is the weighted average of all value vectors. Tokens that are more relevant to token i get higher weight.

**Full matrix form (efficient computation):**

Stacking all tokens into matrices Q (n×d_k), K (n×d_k), V (n×d_v):

```
Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
```

This is computed as a single GPU kernel, fully parallelised.

### 3.3 Multi-Head Attention

A single attention head captures one "type" of relationship. **Multi-head attention** runs h attention heads in parallel, each with different W_Q, W_K, W_V projections, then concatenates their outputs:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
```

where `head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)`.

In practice: one head might learn syntactic dependencies (subject-verb agreement), another semantic similarity, another coreference. The concatenation followed by W_O mixes these diverse representations.

NLLB-200-600M uses 16 attention heads with d_model=1024, so each head has d_k = 64.

### 3.4 Feed-Forward Sub-Layer

After the attention sub-layer, each transformer layer applies a position-wise **feed-forward network (FFN)** independently to each token:

```
FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2
```

The inner dimension d_ff is typically 4 × d_model (so 4096 for NLLB-600M). The `max(0, ...)` is the ReLU activation (or GELU in more recent models). This sub-layer provides non-linearity and allows each token to process information from the attention sub-layer.

### 3.5 Layer Normalization and Residual Connections

**Layer normalization** normalizes the activations within a layer to have zero mean and unit variance:

```
LayerNorm(x) = γ · (x - mean(x)) / sqrt(var(x) + ε) + β
```

where γ and β are learned scale and shift parameters. This stabilizes training by preventing activations from growing or shrinking uncontrollably.

**Residual connections** (skip connections) add the input of a sub-layer directly to its output:

```
x = x + SubLayer(LayerNorm(x))
```

This means gradients can flow directly from later layers to earlier layers without passing through the sub-layer at all — solving the vanishing gradient problem that plagued deep networks.

### 3.6 The Encoder Stack

The encoder consists of N identical layers (NLLB-600M uses 12 encoder layers). Each layer has two sub-layers:

1. Multi-head **self**-attention (every source token attends to every other source token)
2. Position-wise feed-forward network

Input → [Embedding + Positional Encoding] → Layer_1 → Layer_2 → ... → Layer_12 → **encoder output** (a sequence of n vectors of size d_model, one per source token)

The encoder output represents the source sentence as a rich, contextualized sequence of vectors.

### 3.7 The Decoder Stack and Cross-Attention

The decoder also has N layers, but each layer has **three** sub-layers:

1. **Masked** multi-head self-attention (each generated token attends only to previously generated tokens — causal masking prevents "looking ahead")
2. **Cross-attention**: queries come from the decoder, keys and values come from the encoder output. This is how the decoder "reads" the source sentence.
3. Position-wise feed-forward network

The cross-attention is what makes translation possible: at each decoding step, the decoder attends to the encoder's representation of the source to decide what to translate next.

### 3.8 The Full Encoder-Decoder for Translation

```
Source tokens [বাংলা ইনপুট] 
    → Encoder → context vectors
    
Target tokens so far [English output so far]
    → Decoder (using encoder context) → logits over vocabulary
    → Softmax → probability distribution over next token
    → Argmax / Sample → next token
```

During training, the target tokens are the reference translation, and the loss is computed against the reference at every position (teacher forcing). During inference, each generated token is fed back as input for the next step.

---

## 4. Neural Machine Translation

### 4.1 Sequence-to-Sequence Formulation

NMT models the conditional probability P(Y | X) where:
- X = source sentence (sequence of tokens x_1, ..., x_m)
- Y = target sentence (sequence of tokens y_1, ..., y_n)

This is factored autoregressively:

```
P(Y | X) = P(y_1 | X) · P(y_2 | y_1, X) · P(y_3 | y_1, y_2, X) · ...
         = ∏ P(y_t | y_<t, X)
```

The transformer decoder computes each factor P(y_t | y_<t, X) by attending to the encoded source X and all previously generated tokens y_<t.

Training maximises log P(Y | X) over all (X, Y) training pairs, which is equivalent to minimising the cross-entropy loss.

### 4.2 Beam Search Decoding

**Greedy decoding** picks the single most probable token at each step. This is fast but misses better overall sequences that might require a lower-probability first token.

**Beam search** maintains the top-k partial hypotheses at each step (k is the beam size, set to 4 in the papers):

1. Start with one hypothesis: `[<BOS>]`
2. Expand each hypothesis by appending every possible next token, score = log P(token | hypothesis)
3. Keep only the top-k scored hypotheses
4. Repeat until all k hypotheses end with `<EOS>`
5. Return the highest-scoring complete hypothesis

A beam size of 4 strikes a good balance between quality and speed; larger beams give marginally better quality but require 4× more memory and compute per step.

### 4.3 Language Tokens and Forced BOS

NLLB-200 is a multilingual model that translates between 200 languages. It needs to know:
- Which language is the source (so it can properly encode it)
- Which language to generate (so it knows what vocabulary to use in the decoder)

**Source language token:** appended at the end of the source sequence: `[tokens..., </s>, src_lang_token]`. Note: the language token comes *after* the end-of-sequence token — this is a quirk of the M2M-100/NLLB architecture.

**Target language token:** provided as the forced first token of the decoder (forced BOS = beginning of sequence). The decoder sees `[tgt_lang_token]` and generates the translation in that language.

Getting this wrong is a common mistake. Prepending the language token to the source (instead of appending after `</s>`) produces garbage output — this is explicitly documented as a lesson learned in the papers.

### 4.4 NLLB-200 Token Format

```python
# CORRECT format for NLLB-200 source:
source_tokens = tokenizer.encode("বাংলা টেক্সট") + ["</s>", "ben_Beng"]

# WRONG (produces garbage):
source_tokens = ["ben_Beng"] + tokenizer.encode("বাংলা টেক্সট")

# Target: force the decoder to start with the target language token:
decoder_input = ["eng_Latn"]
```

In CTranslate2, this maps to:
```python
translator.translate_batch(
    [source_tokens],
    target_prefix=[["eng_Latn"]]  # forced BOS
)
```

---

## 5. Multilingual and Massively Multilingual NMT

### 5.1 Shared Multilingual Vocabulary

A multilingual model uses a single vocabulary that covers all supported languages simultaneously. NLLB-200's vocabulary has ~256K tokens including subwords from all 200 languages plus special language-ID tokens.

Trade-off: each individual language gets fewer vocabulary slots than a dedicated bilingual model would (vocabulary competition), but the model gains from shared representations and cross-lingual transfer.

### 5.2 Cross-Lingual Transfer

Languages share grammatical structures, concepts, and cognates. A model trained on 200 languages learns representations that capture these cross-lingual similarities. This benefits low-resource languages: Bengali can borrow from the model's knowledge of Hindi, Sanskrit, and other Indic languages that share script and vocabulary.

**Cross-lingual transfer** is the phenomenon where improving the model's performance on a high-resource language also improves its performance on related low-resource languages, even without adding any new data for the low-resource language.

### 5.3 Mixture of Experts (MoE)

The NLLB-200-54B model uses **Mixture of Experts**. Instead of one large feed-forward layer per transformer layer, MoE has N expert sub-networks (each a separate FFN) plus a **gating network** that routes each token to the top-k experts.

```
MoE_FFN(x) = sum(gate(x)[i] * Expert_i(x)  for i in top_k_experts)
```

This dramatically increases the model's parameter count (54B vs 3.3B) without proportionally increasing compute: each token only activates 2 out of 512 experts per layer. The "active parameter count" for any single forward pass is much smaller than 54B.

**Why this matters:** The NLLB-200-54B achieves 41.8 BLEU on Bengali-English while requiring 200+ GB VRAM to store all expert weights simultaneously — far beyond any consumer GPU.

### 5.4 NLLB-200, M2M-100, mBART-50

| Model | Architecture | Training objective | Bengali data |
|-------|-------------|-------------------|--------------|
| mBART-50 | Encoder-Decoder Transformer | Multilingual denoising autoencoding (reconstruct masked/shuffled sentences) | CC25 web data |
| M2M-100 | Encoder-Decoder (M2M-100 arch) | Supervised parallel text, many-to-many | CCAligned + WikiMatrix |
| NLLB-200 | Encoder-Decoder + MoE for 54B | Supervised parallel text + data mining | 100M+ mined pairs |

mBART-50's denoising pre-training gives it strong generalisation but lower zero-shot translation quality than NLLB-200 (14 vs 30.5 BLEU on FLORES-200) because it was not trained directly on translation pairs.

### 5.5 IndicTrans2 and Script-Unified Tokenization

IndicTrans2 makes two key architectural choices beyond the standard multilingual transformer:

1. **Script-unified tokenizer:** All 22 official Indian scripts (Devanagari, Bengali, Tamil, etc.) are normalised to a shared representation before tokenization. This means the model can more easily learn cross-Indic-script relationships — e.g., that "শিক্ষা" (Bengali for "education") and "शिक्षा" (Hindi for "education") are cognates.

2. **BPCC training data:** The Bharat Parallel Corpus Collection contains curated Indic-specific parallel text, including literary, legal, and government document domains that are underrepresented in NLLB-200's web-mined data.

Result: IndicTrans2-1B achieves 41.4 BLEU on FLORES-200, outperforming NLLB-200-3.3B (37.2 BLEU) at one-quarter the parameter count.

---

## 6. Training Large Models

### 6.1 The Cross-Entropy Loss

For each position t in the target sequence, the model outputs a probability distribution P(y | context). The **cross-entropy loss** at position t is:

```
L_t = -log P(y_t | y_<t, X)
```

This is the negative log-probability of the correct next token. A perfect model would assign probability 1.0 to the correct token, giving L_t = -log(1) = 0. A random model over a 256K vocabulary would assign probability 1/256000 ≈ 0.0000039, giving L_t ≈ 12.4.

The training loss reported in the papers (final train_loss = 9.098) is the average cross-entropy loss over all positions in the training batch, summed across the sequence. Higher is worse.

The eval loss (1.992 at epoch 3) being much lower than train loss (9.098) is expected: the eval loss is measured after 3 epochs of adaptation, while the train loss reflects the difficulty of individual training steps including early steps when the model is far from converged.

### 6.2 The Adam Optimizer

**Adam** (Adaptive Moment Estimation) improves on plain SGD by maintaining running averages of both the gradient and the squared gradient for each parameter:

```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t          # first moment (mean of gradients)
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t²          # second moment (mean of squared gradients)

m̂_t = m_t / (1 - β_1^t)   # bias-corrected first moment
v̂_t = v_t / (1 - β_2^t)   # bias-corrected second moment

θ_t = θ_{t-1} - α · m̂_t / (sqrt(v̂_t) + ε)
```

Standard settings: β_1=0.9, β_2=0.999, ε=1e-8. The fine-tuning in the papers uses α=2×10⁻⁴.

**Key insight:** Each parameter gets its own adaptive learning rate. Parameters with consistently large gradients get a smaller effective step size (v̂_t is large, so `α / sqrt(v̂_t)` is small). Rarely updated parameters get a larger effective step size. This works well for sparse problems like NLP where different parameters are relevant in different contexts.

### 6.3 Learning Rate Scheduling and Warmup

Starting with a large learning rate causes instability in the early steps when the model's gradients are large and noisy. **Linear warmup** solves this:

- For the first `warmup_steps` steps, linearly increase α from 0 to α_max.
- Then decay α (linearly, cosine, or constant depending on the scheduler).

In the fine-tuning run: warmup = 74 steps (≈ one-tenth of the first epoch), α_max = 2×10⁻⁴, then linear decay to 0 over the remaining 664 steps.

### 6.4 Gradient Accumulation

GPU memory limits the batch size. With batch_size=8 but a desired effective batch size of 32 (to stabilise gradient estimates), use gradient accumulation with accum_steps=4:

```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accum_steps  # scale loss
    loss.backward()                     # accumulate gradients
    if (i + 1) % accum_steps == 0:
        optimizer.step()                # update weights every 4 batches
        optimizer.zero_grad()
```

The gradient is accumulated for 4 steps before an optimizer step. This is mathematically equivalent to training with batch size 32 but uses only 8 examples' worth of VRAM at any given time.

### 6.5 Mixed Precision Training (fp16 and bf16)

Floating point numbers are stored with different precision:

| Format | Bits | Exponent bits | Mantissa bits | Range | Precision |
|--------|------|---------------|---------------|-------|-----------|
| float32 | 32 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits |
| float16 | 16 | 5 | 10 | ±65504 | ~3 decimal digits |
| bfloat16 | 16 | 8 | 7 | ±3.4×10³⁸ | ~2 decimal digits |

**Mixed precision training** keeps model weights in float32 but performs forward/backward passes in float16 (or bf16), reducing memory ~2× and increasing speed on tensor cores that natively support 16-bit operations.

**fp16 training with GradScaler:** Because float16 can't represent very small gradient values (underflows to 0), PyTorch's `GradScaler` multiplies the loss by a large scale factor before the backward pass, then divides gradients back before the optimizer step. This keeps gradients in a representable range.

### 6.6 Why bf16 and Not fp16 on This Hardware

**The problem with fp16 on this project:**

The LoRA fine-tuner originally tried:
```python
model.half()                            # cast weights to float16
args = TrainingArguments(fp16=True)     # use fp16 AMP
```

This raises `ValueError: Attempting to unscale FP16 gradients` because GradScaler assumes model parameters are in float32. If you've already halved the weights to float16, the unscale step is incoherent.

**The solution: bf16**

bfloat16 has the same exponent range as float32 (8 exponent bits vs float16's 5), so gradient underflow is not an issue — no GradScaler needed. The RTX 5050 (Blackwell sm_120) has native bf16 tensor core support.

```python
model = model.float()                   # keep weights in float32
args = TrainingArguments(bf16=True)     # use bf16 AMP (no GradScaler)
```

---

## 7. Parameter-Efficient Fine-Tuning and LoRA

### 7.1 Why Full Fine-Tuning Is Expensive

Fine-tuning a 600M parameter model requires:
- Storing the model weights: 600M × 4 bytes (float32) = 2.4 GB
- Storing the gradient for every weight: another 2.4 GB
- Storing the Adam optimizer state (two momentum vectors per parameter): another 4.8 GB
- Total: ~10 GB just for model + gradients + optimizer

The RTX 5050 has 8 GB VRAM. Full fine-tuning is impossible. Parameter-efficient fine-tuning freezes most of the model and adds a small number of trainable parameters.

### 7.2 Low-Rank Matrix Factorization

A **rank-r approximation** of a matrix W ∈ ℝ^(d×k) expresses it as the product of two smaller matrices:

```
W ≈ B · A    where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d, k)
```

The number of parameters in the approximation is r·d + r·k = r(d+k), compared to d·k for the full matrix. With r=16 and d=k=1024, this is 16×2048 = 32,768 vs 1,048,576 — a 32× reduction.

The intuition: weight *updates* during fine-tuning have low intrinsic rank. The model doesn't need to move in all d×k directions of weight space; fine-tuning is mostly adjusting along a low-dimensional subspace.

### 7.3 The LoRA Equations Step by Step

LoRA (Low-Rank Adaptation) adds a low-rank bypass to existing weight matrices without modifying them:

**Original layer:**
```
h = W_0 · x
```

**LoRA-adapted layer:**
```
h = W_0 · x  +  ΔW · x
  = W_0 · x  +  (B · A) · x
  = W_0 · x  +  B · (A · x)
```

With the scaling factor:
```
h = W_0 · x  +  (α/r) · B · A · x
```

Where:
- W_0 ∈ ℝ^(d×k): the **frozen** pre-trained weight (never updated during fine-tuning)
- B ∈ ℝ^(d×r): initialized to all zeros (so ΔW = 0 at the start of training)
- A ∈ ℝ^(r×k): initialized with a Gaussian distribution
- r = 16: the rank (a hyperparameter controlling capacity)
- α = 32: the scaling factor; effectively scales the learning rate for LoRA parameters
- α/r = 32/16 = 2: the effective multiplier applied to ΔW

**Why initialize B to zeros?** At the start of fine-tuning, the LoRA modification is zero, so the model behaves exactly as the pre-trained model. This is important: if B were random, the model would start with corrupted outputs and training would be unstable.

**After training**, you can merge B·A back into W_0:
```
W_merged = W_0 + (α/r) · B · A
```

This produces a standard weight matrix with no extra inference overhead. CTranslate2 export uses this merged form.

### 7.4 Which Layers to Adapt

LoRA is applied to the attention projection matrices: query (W_Q), key (W_K), value (W_V), and output (W_O) in each attention layer. These are the most influential parameters for how the model attends to and integrates information.

With r=16 applied to all 4 projections in all 12 encoder + 12 decoder layers of NLLB-600M:

```
Trainable params = 4 projections × 24 layers × r × (d_in + d_out) per LoRA module
                 ≈ 4 × 24 × 16 × (1024 + 1024)
                 ≈ 3.1M (encoder self-attention only)

Total with decoder cross-attention and self-attention included: ≈ 4.7M
```

4.7M trainable out of 619.8M total = 0.76%. The optimizer state for 4.7M parameters requires only 4.7M × 3 × 4 bytes ≈ 57 MB — easily fits in VRAM.

### 7.5 LoRA Hyperparameters

| Hyperparameter | Value used | Meaning |
|---|---|---|
| `r` (rank) | 16 | Dimension of the low-rank subspace |
| `lora_alpha` | 32 | Scaling factor; effective scale = α/r = 2 |
| `lora_dropout` | 0.1 | Dropout applied to A·x before multiplying with B |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "out_proj"]` | Which weight matrices to adapt |
| `bias` | `"none"` | Don't add LoRA to bias terms |

**Practical rule:** r=16 with α=2r is a strong default for NMT fine-tuning. Lower r (8) uses less memory but may underfit. Higher r (32, 64) adds capacity but increases memory and overfitting risk on small datasets.

### 7.6 PEFT Library

The HuggingFace **PEFT** (Parameter-Efficient Fine-Tuning) library wraps any HuggingFace model with LoRA adapters in 5 lines of code:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# "trainable params: 4,718,592 || all params: 619,827,200 || trainable%: 0.76%"
```

PEFT handles the insertion of LoRA modules, freezing all non-LoRA parameters, and merging adapters for export.

### 7.7 Exporting LoRA Weights Back to a Full Model

After training, the LoRA adapters exist as separate B and A matrices. To use the fine-tuned model with CTranslate2 (which expects a standard HuggingFace model format), merge and export:

```python
model = model.merge_and_unload()   # merges B·A into W_0, removes LoRA modules
model.save_pretrained("nllb-finetuned-hf/")  # saves as standard HF checkpoint
# then convert:
# ct2-transformers-converter --model nllb-finetuned-hf --output nllb-finetuned-ct2
```

---

## 8. Efficient Inference: CTranslate2 and Quantization

### 8.1 Why Inference Is Different from Training

During inference (translating a document):
- No gradient computation — backprop is not needed
- No optimizer state — no Adam m/v vectors
- Model weights are read-only — they don't change

This means inference needs far less memory than training (no gradient buffer, no optimizer state). But for production translation of long documents, **throughput** (characters per second) matters. CTranslate2 is a C++/CUDA library specifically optimised for Transformer inference, 2–4× faster than running the HuggingFace pipeline at the same batch size.

### 8.2 Quantization: INT8 and float16

**Quantization** reduces model precision to save memory and increase speed.

**float16 quantization:** Stores all weights as 16-bit floats. Cuts model size from ~2.4 GB (float32) to ~1.2 GB. All arithmetic is in float16. Small precision loss on most operations.

**INT8 quantization:** Stores weights as 8-bit integers. Cuts size to ~600 MB. Before each matrix multiplication, a scale factor converts INT8 weights to float for the actual computation, then converts back. This requires **INT8 CUDA kernels** in cuBLAS.

**The sm_120 INT8 problem:** CUDA 12.4's cuBLAS does not include INT8 tensor core kernels for Blackwell sm_120 hardware (RTX 5050). Attempting INT8 raises `CUBLAS_STATUS_NOT_SUPPORTED`. **float16 is the correct compute type** for this hardware — same throughput, no kernel support gap.

### 8.3 CTranslate2 Architecture

CTranslate2 converts a HuggingFace model to its own binary format (model.bin) and applies:
1. **Weight transposition:** Pre-transposes weight matrices so matrix multiplications use the optimal memory access pattern on GPU.
2. **Kernel fusion:** Combines adjacent operations (e.g., layer norm + attention projection) into single kernels, reducing memory bandwidth.
3. **Dynamic batching:** Sorts inputs by length, pads minimally, and applies `max_batch_size` to cap peak VRAM usage.

### 8.4 Why INT8 Fails on sm_120

GPU compute capability (sm_XY where X.Y is the version) determines which CUDA operations are available:
- sm_80 (Ampere, RTX 3090): INT8 tensor core ops in cuBLAS — fully supported
- sm_89 (Ada Lovelace, RTX 4090): same
- sm_120 (Blackwell, RTX 5050): cuBLAS 12.4 INT8 tensor core ops not yet compiled for sm_120

The fix: a runtime **compute type probe** attempts translation with INT8, INT8_FLOAT16, then float16, catching `CUBLAS_STATUS_NOT_SUPPORTED` exceptions. This is done at model load time with a 20-token probe (short probes don't trigger INT8 matmuls and give false positives).

### 8.5 Batching and Max Batch Size

**Batch processing** translates multiple sentences simultaneously. The GPU executes them in parallel on thousands of CUDA cores, achieving much higher utilisation than one-at-a-time translation.

**Why `max_batch_size` matters:** Calling `translate_batch()` with 900 sentences allocates GPU memory for all 900 input/output sequences simultaneously before any decoding starts. At 28 tokens average × 900 sequences × 1024 hidden dim × 2 bytes = ~50 MB just for the input; with intermediate activations and beam hypotheses, the full allocation can exceed 8 GB VRAM.

Setting `max_batch_size=32` tells CTranslate2 to split the 900 sentences into sub-batches of ≤32 internally, processing them sequentially. Peak VRAM stays within budget, throughput remains near-optimal (GPU is still busy).

---

## 9. Evaluation Metrics

### 9.1 BLEU — Full Derivation

**BLEU** (Bilingual Evaluation Understudy) is the standard automatic metric for NMT, introduced by Papineni et al. (2002). It measures how similar the system's output (hypothesis) is to a human reference translation.

**Step 1: Modified n-gram precision**

For each n from 1 to 4, count how many n-grams in the hypothesis appear in the reference:

```
# Example:
hypothesis = "the cat sat on the mat"
reference  = "the cat sat on the mat"

# Unigrams (n=1):
hypothesis has: {the:2, cat:1, sat:1, on:1, the:2, mat:1}
# Count clipped at reference counts:
p_1 = 6/6 = 1.0  (all 6 words match reference)

# Bigrams (n=2):
hypothesis bigrams: {(the,cat):1, (cat,sat):1, (sat,on):1, (on,the):1, (the,mat):1}
p_2 = 5/5 = 1.0
```

**Clipping** prevents inflating the score by repeating a common word: if the reference has "the" once but the hypothesis has "the" three times, only count one match.

**Step 2: The BLEU formula**

```
BLEU = BP · exp( (1/4) · sum(log p_n for n in 1..4) )
     = BP · (p_1 · p_2 · p_3 · p_4)^(1/4)   # geometric mean of n-gram precisions
```

The geometric mean punishes any n that is 0 — a zero 4-gram precision gives BLEU=0.

**Step 3: Brevity Penalty (BP)**

Pure n-gram precision rewards short translations (fewer words = less chance of getting an n-gram wrong). The brevity penalty compensates:

```
BP = 1                    if hypothesis_length > reference_length
BP = exp(1 - r/c)         if hypothesis_length ≤ reference_length
```

where c = total hypothesis characters/words and r = effective reference length.

**Interpreting BLEU scores:**

| BLEU | Quality |
|------|---------|
| < 10 | Barely usable |
| 10–20 | Gist understandable |
| 20–30 | Good quality |
| 30–40 | High quality |
| 40–50 | Near human (for some pairs) |
| > 50 | Human or near-human (typically in-domain) |

The 56.2 BLEU in the papers is high because it is measured on a curated in-domain corpus where the model performs strongly. The same model scores ~30.5 on the open-domain FLORES-200 benchmark.

### 9.2 The Brevity Penalty

An intuitive example: if the reference is "the big cat sat on the mat" (7 words) and the hypothesis is "cat sat" (2 words), every bigram in the hypothesis matches the reference — p_1=1, p_2=1. Without BP, BLEU=1.0 for this terrible translation!

BP = exp(1 - 7/2) = exp(-2.5) ≈ 0.082. Final BLEU ≈ 0.082 × 1.0 = 8.2%. Much more appropriate.

### 9.3 SacreBLEU and Why Tokenization Matters

Standard BLEU gives different results depending on how you tokenize the text before counting n-grams. A space-tokenized hypothesis and a SentencePiece-tokenized reference have near-zero n-gram overlap even for semantically correct translations — because "বাংলাদেশ" tokenized by SentencePiece might be split into ["▁বাং", "লাদেশ"] while the reference keeps it as one unit.

**SacreBLEU** (Post, 2018) solves this by bundling a specific tokenizer into the metric, making BLEU scores reproducible and comparable across papers. The `flores200` tokenizer is the standard for NLLB-200 evaluations.

**The 0.15 BLEU anomaly:** The raw sacreBLEU score on Samanantar (0.15) results from computing BLEU with the default tokenizer on hypotheses generated by a SentencePiece-based model, without applying the same SentencePiece normalization to the references. It is a measurement artifact, not a quality regression.

### 9.4 chrF

**chrF** (character F-score) measures similarity at the character level rather than word level. This is particularly useful for morphologically rich languages like Bengali, where small spelling variations (due to morphological inflection) cause word-level mismatches even when the meaning is correct.

```
chrF_β = (1 + β²) · chrP · chrR / (β² · chrP + chrR)
```

Where chrP and chrR are character n-gram precision and recall computed over all n up to a maximum (typically 6).

With β=2, recall is weighted twice as heavily as precision. chrF correlates better with human judgment than BLEU for Indic languages because it handles the many surface forms of a single concept (e.g., different honorific forms of the same verb in Bengali).

### 9.5 TER

**TER** (Translation Edit Rate) measures the minimum number of edits (insertions, deletions, substitutions, and phrase shifts) needed to transform the hypothesis into the reference, normalised by reference length:

```
TER = (# edits) / (# words in reference)
```

Lower TER is better. Unlike BLEU, TER is fully symmetric and has an intuitive interpretation as "what fraction of words need to be fixed". However, it doesn't capture paraphrastic correctness and is less widely used for Bengali NMT.

### 9.6 COMET

**COMET** (Crosslingual Optimised Metric for Evaluation of Translation) uses a pre-trained multilingual model (XLM-R) as a judge. It takes the source, hypothesis, and reference as inputs and outputs a quality score.

```
COMET_score = f(source, hypothesis, reference)
```

where f is a learned regression model fine-tuned on human quality annotations (Direct Assessment scores from professional translators).

COMET correlates better with human judgment than BLEU or chrF, but:
1. Requires GPU inference (another large model)
2. Scores are not directly comparable to BLEU (different scale)
3. Less widely reported in the Bengali NMT literature

The survey paper notes this as a gap: none of the surveyed Bengali NMT papers compute COMET consistently enough for cross-system comparison.

### 9.7 Human Evaluation and Its Limits

Automatic metrics correlate imperfectly with human quality judgments. The literary translation paper (Khan et al.) cited in the survey found that human raters preferred IndicTrans2 translations of Tagore stories over NLLB-200 translations despite only a 3-BLEU-point difference.

For literary and idiomatic Bengali prose, BLEU is particularly unreliable because:
- Multiple valid English phrasings exist (paraphrase diversity)
- Cultural idioms have no literal translation
- Register and style (formal/informal/poetic) are invisible to n-gram matching

The proverb example in the papers illustrates this: the Bengali proverb "যে রাঁধে সে চুলও বাঁধে" (roughly "she who cooks also ties her hair") gets translated literally to garbage, scoring near-zero BLEU and also being semantically wrong.

---

## 10. Datasets and Benchmarks

### 10.1 FLORES-200

**FLORES-200** (Few-shot Learning Across 200 Languages Benchmark) is the standard open-domain benchmark for massively multilingual translation. It was created for the NLLB-200 paper by Facebook AI.

- **1,012 sentences** taken from English Wikipedia articles on diverse topics
- Professionally translated into 200 languages including Bengali (`ben_Beng`)
- The `devtest` split (1,012 sentences) is universally used for single-number BLEU comparisons
- All BLEU scores in the survey's comparison table (Table II) use FLORES-200 `devtest` with `flores200` tokenization unless explicitly noted

**Why it matters:** Because every major paper reports FLORES-200 BLEU, it is the only truly apples-to-apples comparison across systems. A system that only reports WMT or in-domain BLEU cannot be directly compared.

### 10.2 Samanantar

**Samanantar** is the largest publicly available Indic parallel corpus (Ramesh et al., 2022):
- 49.7 million sentence pairs across 11 Indic languages
- Bengali-English subset: ~9 million sentence pairs
- Sourced from: websites, news archives, Wikipedia, government documents

The fine-tuning in the papers uses a 9,829-pair subset of Samanantar after quality filtering:
- Length filter: 10–500 characters per segment (removes very short noisy pairs and very long documents)
- 80/10/10 train/val/test split

**Why only 9,829 out of 9,000,000?** This is a deliberate small-scale experiment to demonstrate LoRA fine-tuning works on consumer hardware within a 2-hour training budget. Fine-tuning on all 9M pairs would take ~100× longer and is planned as future work.

### 10.3 WMT Bengali-English

The **WMT** (Workshop on Machine Translation) shared tasks are annual competitions that define standard training/test data and evaluation protocols. WMT 2020 included Bengali-English as a low-resource language pair:
- Training data: ~2M sentence pairs (news domain)
- Test set: ~1000 news sentences
- Top system (University of Edinburgh): 21.7 BLEU

WMT scores are not directly comparable to FLORES-200 scores because the test domains differ (news vs Wikipedia). The papers explicitly note this when comparing across benchmarks.

### 10.4 Domain Diversity and Its Effect on BLEU

The 75-point gap between Health (80.6 BLEU) and News (4.7 BLEU) in the same model on the same evaluation run demonstrates a critical property of NMT: **BLEU depends heavily on domain match between training data and test data**.

NLLB-200 was trained predominantly on web and Wikipedia text. The custom corpus in this project has:
- Health domain: formal medical text → high overlap with training data → high BLEU
- News domain: named entities, rapidly changing proper nouns → low overlap → near-zero BLEU

This means **a single-number BLEU is misleading** without domain breakdown. A model reporting 56.2 BLEU "overall" is averaging a 80.6 and a 4.7 — very different practical utilities depending on your use case.

---

## 11. Bengali Linguistics for NLP

### 11.1 Script and Unicode

Bengali (Bangla) is written in the **Bengali script**, encoded in Unicode at U+0980–U+09FF. Key Unicode concepts:

- **Code point:** A unique number assigned to each character. "ক" is U+0995.
- **Grapheme cluster:** What a user perceives as a single character may be multiple code points. "কা" (ka + vowel sign aa) is U+0995 + U+09BE.
- **NFC normalization:** Unicode has multiple ways to represent the same visual character (precomposed vs decomposed forms). NFC (Canonical Decomposition followed by Canonical Composition) collapses these to a canonical form. Without NFC, "কা" from one source may not match "কা" from another even though they look identical.

The preprocessor in `bn-en-translate` applies NFC with `unicodedata.normalize("NFC", text)` before chunking. This is essential because web-sourced Bengali text frequently has encoding inconsistencies.

### 11.2 Word Order (SOV vs SVO)

English is **SVO** (Subject-Verb-Object): "The cat **ate** the mouse."
Bengali is **SOV** (Subject-Object-Verb): "বিড়ালটি ইঁদুরটি **খেল**।" (literally: "The cat the mouse ate.")

For NMT, this requires long-range reordering: the encoder must read and encode the entire Bengali sentence before the decoder can correctly place the verb in the English output. Transformers handle this naturally via attention; phrase-based statistical MT struggled with Bengali-English because it relied on local reordering rules.

### 11.3 Agglutinative Morphology

Bengali is **agglutinative**: grammatical relationships are expressed by attaching suffixes to root words. A single Bengali verb form can encode tense, aspect, mood, person, number, and honorificity:

- "খাই" (I eat, present) vs "খাইলাম" (I ate, past) vs "খাইতেছি" (I am eating, progressive)
- "খান" (you eat, formal/respectful) vs "খাও" (you eat, familiar) vs "খা" (you eat, intimate/rude)

BPE tokenizers fragment these morphologically complex forms into 3–5 subword pieces:
```
"খাইতেছিলেন" → ["খাই", "তেছি", "লেন"]
```

This means the model must learn to reconstruct the morphological parse from fragments. IndicTrans2's larger Indic-specific vocabulary helps by giving more frequent Bengali morphemes their own tokens.

### 11.4 Zero Pronouns and Pro-Drop

Bengali is a **pro-drop** language: subject pronouns are frequently omitted because the verb form encodes enough information to infer the subject:

- "খাচ্ছি" can mean "I am eating" — the "I" is implicit in the verb suffix "-চ্ছি"

English is not pro-drop. The NMT model must hallucinate (insert) a pronoun that doesn't appear in the Bengali source. Getting the gender or person wrong (translating an implicit "she" as "he") is a common and difficult-to-detect error in Bengali NMT outputs.

### 11.5 Code-Switching

Informal Bengali text (social media, messaging) frequently mixes Bengali script and Roman-script English within the same sentence:

"আজকে meeting টা really boring ছিল" (Today the meeting was really boring)

Standard NMT models trained on formal parallel text struggle with:
1. Roman-script fragments not in the SentencePiece vocabulary
2. Mixed-script tokenization producing unexpected token sequences
3. The model trying to "translate" English words that should be passed through unchanged

IndicTrans2 partially addresses this through its unified vocabulary, but code-mixed Bengali NMT remains an open research problem.

---

## 12. GPU Architecture and CUDA

### 12.1 Why GPUs for ML

A CPU has 8–64 high-clock-speed cores optimised for sequential, branch-heavy code. A GPU (RTX 5050) has thousands of smaller cores (CUDA cores) running at lower clock speed but able to execute thousands of parallel operations simultaneously.

Matrix multiplication — the dominant operation in neural network inference and training — is embarrassingly parallel: each output element C[i,j] = dot(A[i,:], B[:,j]) can be computed independently. A GPU can compute thousands of these simultaneously, achieving 100–200× speedup over CPU for ML workloads.

### 12.2 CUDA and Compute Capability

**CUDA** (Compute Unified Device Architecture) is NVIDIA's programming model for GPU computation. You write GPU kernels in C++/CUDA, compile them, and execute them on the GPU.

**Compute capability sm_XY** is a version number for GPU microarchitectures:
- sm_86 = Ampere (RTX 3090)
- sm_89 = Ada Lovelace (RTX 4090)
- sm_120 = Blackwell (RTX 5050, RTX 5090)

A CUDA kernel compiled for sm_86 may not run on sm_120. PyTorch ships pre-compiled kernels for specific compute capabilities; a `cu128` (CUDA 12.8) build includes sm_120 kernels, while a `cu124` build does not. This is why PyTorch `cu128` was needed for training on the RTX 5050 — the `cu124` build lacked sm_120 kernels for element-wise operations like `.ne()`.

### 12.3 VRAM vs RAM

**VRAM** (Video RAM, GPU memory): fast, on-GPU memory. Everything the GPU operates on must be in VRAM:
- Model weights: ~2 GB for NLLB-600M float16
- Activations (forward pass): ~0.5 GB during inference
- Gradients + optimizer state: ~5 GB during training
- Input/output token buffers

**RAM** (system memory): slower, much larger (16 GB on this machine). The CPU operates here. PyTorch can move tensors between RAM and VRAM with `.to("cuda")` and `.to("cpu")`.

The LoRA training strategy loads the full model in RAM (float32, ~2.4 GB) first, wraps with LoRA, then moves to VRAM. This is why RAM usage during training reaches ~7.2 GB.

### 12.4 Tensor Cores and Blackwell sm_120

**Tensor Cores** are specialised hardware units in NVIDIA GPUs that perform matrix multiply-accumulate operations in a single clock cycle on 4×4 matrix tiles. They are the reason GPUs are so fast for ML.

Blackwell tensor cores natively support:
- float16 matrix multiplication (HGEMM)
- bfloat16 matrix multiplication
- float32 accumulation

INT8 tensor core operations (INT8 IGEMM) require specific cuBLAS kernel support that was not included in the CUDA 12.4 distribution for sm_120. This is why CTranslate2 INT8 mode fails — it calls the cuBLAS INT8 GEMM API, which is unavailable, raising `CUBLAS_STATUS_NOT_SUPPORTED`.

### 12.5 WSL2 and LD_LIBRARY_PATH

**WSL2** (Windows Subsystem for Linux 2) runs a real Linux kernel in a lightweight VM on Windows. CUDA drivers are provided by the Windows GPU driver but exposed to WSL2 through `/usr/lib/wsl/lib/`.

When Python spawns a new subprocess (e.g., a DataLoader worker or a CTranslate2 inference process), it does **not** automatically inherit the parent's environment variables. The `LD_LIBRARY_PATH=/usr/lib/wsl/lib` setting — which tells the dynamic linker where to find libcuda.so — must be set explicitly in every subprocess that calls CUDA.

Failure to do so produces `libcuda.so.1: cannot open shared object file` errors even though the parent process works fine.

---

## 13. The Four-Stage Translation Pipeline

The `bn-en-translate` system processes documents through four stateless stages connected by a `para_id` metadata tag.

### 13.1 Preprocessor: NFC Unicode Normalization

```python
import unicodedata

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)  # canonical Unicode form
    text = re.sub(r"[ \t]+", " ", text)        # collapse whitespace
    text = text.strip()
    return text
```

Why NFC specifically: Bengali web text frequently contains code points that look identical but have different internal representations. NFC ensures consistent tokenization by normalising to the composed form (single code point where possible).

### 13.2 Chunker: Token Budget and Sentence Boundaries

The chunker solves two problems:
1. NLLB-200 has a 512-token context window. Bengali prose paragraphs can be much longer.
2. Splitting mid-sentence corrupts grammar and creates unnatural translation boundaries.

**Solution:** Split at Bengali sentence-ending punctuation (danda `।` / double danda `॥`), and accumulate sentences into chunks until reaching 400 tokens (80% of 512 — leaving headroom for the source language token and any expansion during subword tokenization).

Each chunk carries a `para_id` integer matching its source paragraph, used later for reassembly.

### 13.3 Translator: Batched GPU Inference

```python
results = translator.translate_batch(
    [tokens + ["</s>", "ben_Beng"] for tokens in all_chunks],
    target_prefix=[["eng_Latn"]] * len(all_chunks),
    beam_size=4,
    max_batch_size=32       # prevents OOM on 8 GB VRAM
)
```

CTranslate2 internally sorts chunks by length, pads to the maximum length in the sub-batch, and processes all chunks in the sub-batch through the GPU simultaneously.

### 13.4 Postprocessor: Paragraph Reassembly

```python
def reassemble(chunks: list[ChunkResult]) -> str:
    paragraphs = defaultdict(list)
    for chunk in sorted(chunks, key=lambda c: c.para_id):
        paragraphs[chunk.para_id].append(chunk.translation)
    
    result = "\n\n".join(
        " ".join(paragraphs[pid]) for pid in sorted(paragraphs)
    )
    assert result.count("\n\n") == input_para_count - 1   # invariant
    return result
```

The invariant — output paragraph count equals input paragraph count — is verified by both the postprocessor and a dedicated unit test. This ensures that a 5-paragraph Bengali story produces a 5-paragraph English story, not more or fewer.

---

## 14. Resource Monitoring and Run Databases

### 14.1 Why Monitor Resources?

Without monitoring:
- You don't know if a new model change caused a VRAM regression
- You can't tell if a batch size increase improved throughput or just increased memory pressure
- Quality regressions (BLEU drops) go unnoticed until someone reads the output

The `ResourceMonitor` + `RunDatabase` subsystem records every benchmark and training run to a SQLite database, enabling automated regression detection.

### 14.2 CPU and RAM Profiling with psutil

```python
import psutil

process = psutil.Process()
cpu_percent = process.cpu_percent(interval=1.0)    # % CPU used
ram_mib = process.memory_info().rss / 1024**2      # resident set size
swap_mib = psutil.swap_memory().used / 1024**2
```

The `ResourceMonitor` samples these every 2 seconds in a daemon background thread and computes peak and average values over the run duration.

### 14.3 GPU Monitoring with pynvml

```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)

vram_used_mib = mem_info.used // 1024**2
gpu_util_pct = util.gpu
```

pynvml wraps NVIDIA's Management Library (NVML), which provides the same data shown by `nvidia-smi`.

### 14.4 SQLite for Run History

```python
import sqlite3

conn = sqlite3.connect("monitor/runs.db")
conn.execute("""
    INSERT OR REPLACE INTO runs (
        run_id, timestamp, model, bleu, throughput_chars_s,
        vram_peak_mib, gpu_util_avg_pct, cpu_util_avg_pct, duration_s
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (...))
```

SQLite is a serverless, file-based relational database. The `runs.db` file stores every benchmark and fine-tuning run. Queries like "find runs where BLEU dropped more than 5% compared to the previous run of the same model" provide automated regression alerts.

### 14.5 Regression Detection

The monitor agent reads `runs.db` and computes:

```python
# Regression: BLEU dropped more than 5% vs prior run of same model
prev_bleu = db.query("SELECT bleu FROM runs WHERE model=? ORDER BY timestamp DESC LIMIT 1 OFFSET 1")
current_bleu = db.query("SELECT bleu FROM runs WHERE model=? ORDER BY timestamp DESC LIMIT 1")
if (prev_bleu - current_bleu) / prev_bleu > 0.05:
    alert("BLEU regression detected!")
```

---

## 15. Software Engineering Concepts

### 15.1 Test-Driven Development (TDD)

**TDD** is a development practice where you write the test *before* writing the implementation:

1. Write a failing test that specifies the desired behaviour
2. Write the minimum implementation to pass the test
3. Refactor while keeping all tests green

The project has 186 tests covering:
- **Unit tests:** Pure logic (tokenizer output, chunker behaviour, BLEU calculation) — no GPU, no downloads
- **Integration tests:** Full pipeline with mock models (verify paragraph counts, batch sizes)
- **End-to-end tests:** Real GPU, real model — verify BLEU ≥ 25 on built-in corpus

**Key invariants enforced by tests:**
- `TranslatorBase.translate()` raises `RuntimeError` if called before `load()`
- `Chunker.chunk()` never splits mid-sentence (test on corpus of 100 sentences)
- `reassemble()` output paragraph count equals input paragraph count

### 15.2 The Strategy Pattern

The model backend uses the **Strategy design pattern**: a `TranslatorBase` abstract class defines a common interface (`load()`, `unload()`, `translate()`), and different concrete implementations (NLLB CT2, IndicTrans2, Ollama) are swapped in without changing the pipeline.

```python
# Abstract interface
class TranslatorBase(ABC):
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def translate(self, texts: list[str]) -> list[str]: ...

# Concrete implementations
class NLLBCt2Translator(TranslatorBase): ...
class IndicTrans2Translator(TranslatorBase): ...

# Factory — caller doesn't care which implementation
translator = get_translator(model_name="nllb-600M")
```

### 15.3 Context Managers

Python **context managers** ensure resources are acquired and released correctly:

```python
with translator:      # calls translator.__enter__() → loads model to GPU
    results = translator.translate(chunks)
# translator.__exit__() is called here even if an exception occurred → unloads model
```

This guarantees VRAM is freed even if translation raises an error. Without the context manager, a crashed translation would leave the model in VRAM and the next operation would fail with OOM.

### 15.4 Daemon Threads

The `ResourceMonitor` uses a **daemon thread** — a background thread that:
1. Runs concurrently with the main thread (samples CPU/GPU every 2 seconds)
2. Is automatically killed when the main thread exits (unlike regular threads, which prevent the process from exiting)
3. Cannot block process cleanup even if the monitor code hangs

```python
import threading

class ResourceMonitor:
    def __enter__(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def __exit__(self, *args):
        self._stop_event.set()
        self._thread.join(timeout=5)   # wait up to 5s for clean shutdown
```

If the process is killed (SIGKILL), the daemon thread dies automatically — no zombie processes.

---

## 16. Scale, Efficiency, and the Pareto Frontier

### 16.1 Scaling Laws

**Scaling laws** (Kaplan et al., 2020) describe how model performance improves with more compute, more data, and more parameters. For language models, loss decreases as a power law:

```
L(N) ≈ A / N^α   # loss as a function of parameter count N
```

The NLLB-200 data in the survey illustrates this for Bengali BLEU:

| Model | Parameters | FLORES-200 BLEU |
|-------|------------|-----------------|
| 600M  | 6×10⁸     | 30.5 |
| 1.3B  | 1.3×10⁹   | 33.4 |
| 3.3B  | 3.3×10⁹   | 37.2 |
| 54B   | 5.4×10¹⁰  | 41.8 |

Each doubling of parameters gives diminishing BLEU returns: +9.7 then +3.8 then +4.6 BLEU for roughly ×2, ×2.5, ×16 parameter increases. The log-linear relationship breaks down — scaling alone does not solve Bengali NMT.

### 16.2 Marginal Returns per Billion Parameters

A cleaner way to measure scaling efficiency:

```
ΔBLEU/B = (BLEU_new - BLEU_old) / (params_new - params_old)
```

| Step | ΔBLEU | ΔParams (B) | ΔBLEU/B |
|------|-------|-------------|---------|
| NLLB 600M → 1.3B | +2.9 | +0.7B | **+4.1** |
| NLLB 1.3B → 3.3B | +3.8 | +2.0B | **+1.9** |
| NLLB 3.3B → 54B  | +4.6 | +50.7B| **+0.09** |
| NLLB 600M → IT2-200M | +7.4 | −0.4B | **+37.0** |

The last row is the key insight: switching to a smaller, language-specialized model yields +37 BLEU per billion parameters — 400× more efficient than scaling within NLLB. This is not because the model is larger; it's because the training data is better matched to the language pair.

### 16.3 The Quality–VRAM Pareto Frontier

A **Pareto frontier** is the set of options where no improvement in one dimension (BLEU) is possible without worsening another (VRAM). Systems on the frontier are called **Pareto-optimal**.

```
VRAM (GB) vs FLORES-200 BLEU:

42 |         IT2-1B ★
40 |     IT2-200M ★
38 |
36 |
32 |                         NLLB-3.3B
30 |      NLLB-600M ★
28 |  M2M-100-418M
22 |
16 |
14 |  mBART-50
   +--------------------------
     0   1   2   3   4   5  13  (VRAM GB)

★ = Pareto-optimal
```

IndicTrans2-200M (0.8 GB, 37.9 BLEU) is Pareto-optimal: no other system achieves 37+ BLEU with less than 0.8 GB VRAM. IndicTrans2-1B (3.2 GB, 41.4 BLEU) is Pareto-optimal at a higher quality tier.

NLLB-600M CT2 (1.9 GB, 30.5 BLEU) is Pareto-dominated by IndicTrans2-200M in raw quality — but NLLB-600M CT2 is the only fully-validated option for RTX 5050 at this project's stage.

---

## 17. Putting It All Together: Reading the Papers

Now that you have all the building blocks, here is a guide to reading both papers with full comprehension.

### Reading `ieee_paper.tex` (System Paper)

**Abstract:** Understand the claim structure:
- "BLEU score of 56.2 on a curated 90-sentence corpus" — see §9.1 (BLEU), §10.4 (domain effects)
- "LoRA fine-tuning via PEFT with bf16 precision" — see §7 (LoRA), §6.5–6.6 (bf16)
- "Six hardware-specific constraints on Blackwell sm_120" — see §12 (GPU)
- "186 unit and integration tests" — see §15.1 (TDD)

**Section III (System Architecture):**
- The TikZ diagram shows the 4-stage pipeline — see §13
- The `_best_compute_type` probe code — see §8.4
- The LoRA forward pass equation `h = W₀x + (α/r)BAx` — see §7.3

**Section V (Experimental Results):**
- Table II (benchmark) — 97 chars/s, 82% GPU util — see §8.5 (batching)
- Table III (per-domain BLEU) — Health 80.6 vs News 4.7 — see §10.4 (domain effects)
- Table VI (GPU fine-tuning) — train_loss 9.098, eval_loss 1.992 — see §6.1 (cross-entropy)
- The 0.15 sacreBLEU anomaly — see §9.3 (SacreBLEU tokenization)

### Reading `survey_paper.tex` (Survey Paper)

**Section II (Background):**
- BLEU equation `BLEU = BP · exp(Σ wₙ log pₙ)` — see §9.1
- chrF equation — see §9.4
- SentencePiece discussion — see §2.3

**Section III (Survey of Systems):**
- Each system subsection maps to §5 (multilingual NMT) of this primer
- The API cost table (Table II) — see §16.3 (Pareto analysis)

**Section IV (Comparative Analysis):**
- Table VI (marginal returns) — see §16.2
- Pareto frontier figure — see §16.3
- BLEU trend figure — see §9.1 for interpreting score levels
- "Cross-System Comparative Analysis" subsection — pulls together §5, §7, §9, §10

**Section V (Bengali-Specific Challenges):**
- Named entity translation — see §11.1–11.2
- Morphological complexity — see §11.3
- Zero pronouns — see §11.4
- Code-switching — see §11.5

---

## Quick Reference: Key Numbers

| Quantity | Value | Where explained |
|----------|-------|-----------------|
| NLLB-600M parameters | 619.8M | §5.4 |
| LoRA trainable parameters | 4.7M (0.76%) | §7.4 |
| LoRA rank r | 16 | §7.5 |
| LoRA alpha α | 32 (scale = α/r = 2) | §7.3 |
| NLLB context window | 512 tokens | §13.2 |
| Chunk token budget | 400 (80% of 512) | §13.2 |
| Batch size for inference | 8 (optimal for 8 GB VRAM) | §8.5 |
| Max batch size for CT2 | 32 | §8.5 |
| Beam size | 4 | §4.2 |
| BLEU on custom corpus | 56.2 | §9.1 |
| BLEU on FLORES-200 (backbone) | 30.5 | §9.3 |
| Eval loss after 3 epochs | 1.992 | §6.1 |
| Training time (GPU bf16) | 8,852s / 2.46h | §6.5 |
| VRAM: NLLB-600M float16 | ~2 GB | §12.3 |
| VRAM: Full training peak | ~8 GB | §12.3 |
| Throughput | 97 chars/s | §13.3 |
| GPU utilisation (inference) | 82% avg / 97% peak | §8.1 |

---

*This document covers every concept referenced in `paper/ieee_paper.tex` and `paper/survey_paper.tex`. For implementation details, see `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, and the source code in `src/bn_en_translate/`.*
