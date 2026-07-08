## 2026-07-07 — optimization pass Phase 1 / benchmark measurement learnings (branch: perf/optimization-pass)

**Measurement flaw found and fixed (benchmark.py):**
- The benchmark timer started BEFORE `with translator:`, so model load time — including cold-page-cache disk reads (up to 5.0 GB for seamless) — was counted as translation time. All historical Time/ch/s figures (including the 2026-04-10 table below and CLAUDE.md's table) conflate load+translate and depend on page-cache warmth; BLEU/chrF are unaffected.
- Fixed in commit 6d56b34: `Load` column added, `ch/s` is now translate-only. Load time remains cache-dependent (seamless load: 96.0 s cold vs 50.6 s warm, same code) and must never be used for cross-run comparisons.

**Batching result (Task 1, warm A/B, translate-only):**
- nllb-600M 383 → 2,108 ch/s (5.5×), milmmt-46-1b 74 → 326 ch/s (4.4×), seamless 73 → 300 ch/s (4.1×) at batch_size=8.
- BLEU parity: nllb Δ0.0, seamless Δ0.0, milmmt Δ−0.3 (65.0 → 64.7, chrF improved 79.3 → 79.4). Left-padded batch generation on causal LMs can shift BLEU a few tenths — expected, not a regression.
- The single-sentence path is preserved behind `--no-batch` for future A/B comparisons.

**SDPA attention result (Task 2, commit 04ddf9a):**
- MiLMMT fallback eager → sdpa: 326 → 399 ch/s (+22%) at BLEU 64.8 / chrF 79.4 (gate ≥64.7 passed). flash-attn remains uninstallable on sm_120/WSL2, so SDPA is the effective default for architectures that support it. indicTrans2.py (HF fallback path) still has the old eager ternary — scheduled for the hf_utils refactor.
- **Review catch (Critical):** T5 rejects `attn_implementation="sdpa"` in transformers 5.4.0 (`_supports_sdpa=False` → ValueError at load), so blindly applying the same fallback to MADLAD made its load() crash — caught by code review's tiny-config CPU instantiation probe, fixed with a per-architecture `fallback` parameter (MADLAD → eager). Lesson: an attention-resolver helper must take a per-architecture fallback; a 1-second `_from_config` smoke test catches unsupported attn kernels without any download.

**Inter-model VRAM residue eliminated (Task 3, commit 5f82047):**
- `reset_cuda_state()` (empty_cache + reset_peak_memory_stats) now runs between models in multi-model benchmarks. Validation: milmmt vram_peak sequential-after-nllb 4,024 MiB vs standalone 4,009 MiB (Δ15 MiB) — previously residue inflated the second model's reading by ~300 MiB (the 2026-04-10 false WARNING). Pre-2026-07 multi-model vram_peak rows in runs.db should not be used as regression baselines.

**Length-sorted batching + Phase 1 exit (Task 4, commits 20ca052/3860a26):**
- `_translate_in_batches` now sorts by input length and restores order via index scatter. BLEU gate: nllb 55.27 / milmmt 65.24 (above the 65.0 baseline) / seamless 67.00 — all pass; BLEU parity is the load-bearing proof that order restoration is correct end-to-end.
- **Phase 1 cumulative (translate-only ch/s, warm):** nllb 383 → ~2,072 (5.4×), milmmt 74 → ~366 (4.9×), seamless 73 → ~317 (4.3×; corrected from an ungrounded 302 by spec review — commit 34a574e).
- **Speed-noise learning:** repeated identical-code milmmt runs spanned 146-375 ch/s in one session (host load >2; slowest run mid-sequence, so session noise, not cache warming). Single-run ch/s deltas under ~2× are not attributable on this machine — use BLEU for correctness gates and multi-run medians for speed claims.
- **Plan-authoring lesson:** the plan's prescribed test input ordering passed trivially against unfixed code (first batch happened to be pre-sorted). TDD's red-phase check caught it; test inputs must be constructed to fail against the current behavior, not just describe the desired one.

**Process learning (agent tooling):**
- A watcher loop of the form `until ! pgrep -f "scripts/benchmark.py --models ..."` deadlocks: the pattern matches the watcher's own command line. Use a narrower pattern (`"python scripts/benchmark.py"`), poll the log for a sentinel line, or run the benchmark in the foreground with a large timeout.

---

## 2026-04-10T00:00:00Z — benchmark / nllb-600M + milmmt-46-1B + seamless-medium (branch: feat/new-models-indicTrans2-milmmt46)

**Regressions:**
- nllb-600M: VRAM peak rose ~307 MiB (2,048 MiB baseline → 2,355 MiB current) — WARNING (threshold: 200 MiB). Single-run observation; multi-model sequential benchmark may cause inter-model VRAM residue to inflate this reading. Monitor over next 2–3 solo runs before acting.
- nllb-600M: chars_per_sec 197 → 191 (~3.0% drop) — within threshold (15%), no regression.
- nllb-600M: BLEU 55.3 → 55.3 — no regression.
- seamless-medium: VRAM 3,994 MiB → 4,096 MiB (+102 MiB) — within threshold (200 MiB), no regression.
- seamless-medium: chars_per_sec 32 → 31 (~3.1% drop) — within threshold, no regression.
- seamless-medium: BLEU 67.0 → 67.0 — no regression.
- milmmt-46-1B: first run, no prior baseline — regression check skipped.

**Patterns detected:**
- benchmark.py runs all three models sequentially in a single invocation. `torch.cuda.empty_cache()` is not called after each `with translator:` block exits. Residual VRAM fragments from an earlier model may inflate peak readings for the next model in the same run. This is the most likely cause of the nllb-600M VRAM WARNING above.
- nllb-600M VRAM increase is a single data point — monotonic-increase pattern (5+ runs required) not yet confirmed. Requires 2–3 more solo-model benchmark runs to distinguish residue inflation from a genuine allocation growth.
- milmmt-46-1B at 28 ch/s is comparable to seamless-medium (31 ch/s) — both are HF native float-precision models running via `.to("cuda")`. Speed is expected in this range.
- No swap pressure detected. All three models fit individually within the 8 GB VRAM budget.

**Optimization suggestions:**
- /home/sbisw/github/translate/scripts/benchmark.py line 41: After `with translator:` exits, add `import torch; torch.cuda.empty_cache()` before the next model iteration begins. This prevents residual allocations from contaminating VRAM peak readings for subsequent models in multi-model benchmark runs.
- /home/sbisw/github/translate/src/bn_en_translate/models/nllb_ct2.py lines 71–77: `inter_threads=1, intra_threads=4` is already set — no action needed. Note: inter_threads=1 means only one translation batch runs at a time; if you add concurrent requests in future, raise inter_threads to 2.
- /home/sbisw/github/translate/src/bn_en_translate/config.py line 29: `ChunkConfig.batch_size=8` is current. No swap pressure detected; do not reduce. If milmmt or seamless OOM occurs in longer stories, reduce to 4 as a first step.

**VRAM budget check (7.5 GB usable ceiling):**
- nllb-600M (2.3 GB) + Ollama qwen2.5:7b (4.7 GB) = 7.0 GB — SAFE (0.5 GB headroom).
- milmmt-46-1B (3.3 GB) + Ollama qwen2.5:7b (4.7 GB) = 8.0 GB — WARNING: exceeds 7.5 GB. Do not run milmmt + Ollama concurrently. Unload milmmt before starting Ollama polish pass.
- seamless-medium (4.0 GB) + Ollama qwen2.5:7b (4.7 GB) = 8.7 GB — WARNING: exceeds 7.5 GB. Do not run seamless + Ollama concurrently. Unload seamless before starting Ollama polish pass.
- IndicTrans2-1B CT2 float16 (est. 3.1 GB) + Ollama (4.7 GB) = 7.8 GB — WARNING: exceeds 7.5 GB. Same constraint applies.
- Safe Ollama polish combinations: nllb-600M only.

**New model summary (milmmt-46-1B, first run):**
- BLEU 65.0 / chrF 79.3 — strong result, within 2 BLEU points of seamless-medium (67.0 / 80.2).
- VRAM 3.3 GB — 0.7 GB lower than seamless-medium. Useful if seamless-medium is unavailable.
- 28 ch/s — similar speed tier to seamless-medium (31 ch/s). Both are HF native inference, not CT2.
- No prior baseline — establish this run as the milmmt reference point.

**Resource snapshot:**
- nllb-600M: BLEU 55.3 (baseline 55.3), Duration ~N/A, VRAM peak 2,355 MiB, chars/s 191
- milmmt-46-1B: BLEU 65.0 (no prior), VRAM peak 3,379 MiB, chars/s 28 (first baseline)
- seamless-medium: BLEU 67.0 (baseline 67.0), VRAM peak 4,096 MiB, chars/s 31
- RAM: no swap pressure reported
- CPU: not reported in this run
