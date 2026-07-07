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
