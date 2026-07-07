# Performance Baseline — 2026-07-07 (pre-optimization)

FLORES-200 devtest, 90 sentences, RTX 5050 8 GB, WSL2.
Command: `python scripts/benchmark.py --models nllb-600M milmmt-46-1b seamless-medium --sentences 90`
Benchmark loop: UNBATCHED (one sentence per pipeline.translate call) — this is the loop Task 1 replaces.

| Model | BLEU | chrF | Time (s) | ch/s | VRAM peak (MiB) |
|-------|------|------|----------|------|-----------------|
| nllb-600M | 55.3 | 72.8 | 24.7 | 153 | 2701 |
| milmmt-46-1b | 65.0 | 79.3 | 65.1 | 58 | 3620 |
| seamless-medium | 67.0 | 80.2 | 83.1 | 46 | 4457 |

Acceptance gate for all Phase 1 changes: BLEU within ±0.3 of this table per model.

**Caveat (discovered during Task 1):** the `Time (s)` / `ch/s` columns above conflate
model *load* time (including cold-page-cache disk reads) with translate time — the
benchmark's timer starts before `with translator:` (i.e. before `translator.load()`).
On a cold cache, load time can dominate the total and swamp any translate-side
speedup. Task 1 splits these into separate `Load` and `Time` columns so translate-only
throughput is comparable run to run regardless of disk-cache state.

---

## After Task 1 (batched loop via `translate_sentences()`)

Warm A/B, same session, same page cache, back-to-back runs — isolates the batching
effect from load-time/disk-cache noise:

```bash
python scripts/benchmark.py --models nllb-600M milmmt-46-1b seamless-medium --sentences 90 --no-batch
python scripts/benchmark.py --models nllb-600M milmmt-46-1b seamless-medium --sentences 90
```

### Run A — `--no-batch` (one sentence per `pipeline.translate()` call, pre-Task-1 loop)

| Model | BLEU | chrF | Load (s) | Translate (s) | ch/s |
|-------|------|------|----------|----------------|------|
| nllb-600M | 55.3 | 72.8 | 5.2 | 9.9 | 383 |
| milmmt-46-1b | 65.0 | 79.3 | 9.2 | 51.0 | 74 |
| seamless-medium | 67.0 | 80.2 | 96.0 | 51.5 | 73 |

### Run B — batched (`pipeline.translate_sentences()`, new default)

| Model | BLEU | chrF | Load (s) | Translate (s) | ch/s | Speedup (translate-only) |
|-------|------|------|----------|----------------|------|---------------------------|
| nllb-600M | 55.3 | 72.8 | 23.9 | 1.8 | 2108 | 5.5x |
| milmmt-46-1b | 64.7 | 79.4 | 16.8 | 11.6 | 326 | 4.4x |
| seamless-medium | 67.0 | 80.2 | 50.6 | 12.6 | 300 | 4.1x |

BLEU gate: nllb 55.3→55.3 (Δ0.0), milmmt 65.0→64.7 (Δ−0.3, exactly at the inclusive
±0.3 edge — small padding/attention interaction on the HF causal-LM backend, chrF
improved 79.3→79.4), seamless 67.0→67.0 (Δ0.0). **Gate passes for all three models.**

Load-time and total-wall-clock figures still vary run to run with OS page-cache state
(compare Run A's seamless load of 96.0s against Run B's 50.6s — same code, same
model, different cache warmth) — only the `Translate (s)` / `ch/s` columns above are
attributable to the batching change in `translate_sentences()`.

---

## After Task 2 (SDPA attention fallback, replaces `eager`)

flash-attn is never installed on this machine (sm_120/WSL2), so the fallback path
always won. `attn_implementation` for MiLMMT (Gemma3) now resolves to `"sdpa"`
(PyTorch's built-in `scaled_dot_product_attention`) instead of `"eager"` when
flash-attn is unavailable, via `_resolve_attn_implementation(use_flash, fallback)`
in both `milmmt.py` and `madlad.py`. MADLAD (T5) keeps `fallback="eager"` — T5
rejects sdpa in transformers 5.4.0 (`_supports_sdpa=False`, ValueError at load;
caught in code review). MADLAD is also excluded from benchmarks (corrupted
checkpoint); seamless.py never set `attn_implementation` and is untouched.

```bash
python scripts/benchmark.py --models milmmt-46-1b --sentences 90
```

| Model | BLEU | chrF | Load (s) | Translate (s) | ch/s | vs. Task 1 (eager) ch/s |
|-------|------|------|----------|----------------|------|--------------------------|
| milmmt-46-1b | 64.8 | 79.4 | 7.3 | 9.5 | 399 | 326 → 399 (+22%) |

BLEU gate: 65.0 (original) → 64.8 (Δ−0.2, within the ±0.3 gate, ≥64.7 floor).
chrF unchanged at 79.4. SDPA is faster than eager as expected — no regression,
no `DONE_WITH_CONCERNS` needed.
