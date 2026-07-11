## 2026-07-10 — new-model evaluation (milmmt-46-4b) — ACCEPTED, opt-in only (best BLEU/chrF of all models tried)

**New-model evaluation:**

- **milmmt-46-4b** (xiaomi-research/MiLMMT-46-4B-v0.1, 4-bit bitsandbytes quantization, 90-sentence FLORES, run `272694b8409a`): BLEU 68.55 / chrF 81.8, 34.0 ch/s translate-only, duration 305.9 s, VRAM peak 8,050 MiB (of 8,151 MiB total — **~101 MiB headroom, ~99% of card**), GPU util peak 100% / avg 19%, RAM peak 7,225 MiB, swap peak 654 MiB, CPU peak 99.7% / avg 50.2%.
- Beats every model previously benchmarked: seamless-medium (67.0/80.2), milmmt-46-1b (65.2/79.3), nllb-600M (55.3/72.8), and all rejected candidates. New best-quality result for this project.
- Verified on the real 90-paragraph story (not just FLORES): no hallucination, paragraph count matched exactly, corrected mistranslations the 1B sibling made (e.g. "lion" → "jackals").
- **Decision: KEPT, opt-in only** (`--model milmmt-46-4b`), NOT made the default. VRAM headroom of ~100 MiB is inside the noise band of a single extra CUDA context or concurrent process — any other GPU activity (Ollama, a second benchmark, a stray notebook kernel) risks OOM. `milmmt-46-1b` remains the default for routine use.
- Companion 5-sentence smoke run (`c35b7faef15d`): BLEU 88.90, VRAM peak only 6,852 MiB, chars/s 60.0. **Flag as small-sample artifact — do not use for quality ranking or as a rolling-average input.** Same small-sample BLEU inflation mechanism documented for lmt-60-1.7b (73.2 vs 63.8) and the nllb/milmmt-1b 5-sentence rows in the 2026-07-08 entry below. The lower VRAM peak on the smoke run (6,852 vs 8,050 MiB) is expected — VRAM scales with the volume of text held in flight — not a discrepancy to investigate.

**Regressions vs incumbents:** nllb-600M (55.3), milmmt-46-1b (65.2), seamless-medium (67.0) were **not re-run** on 2026-07-10 — no new data exists for those three models, so there is nothing to flag as a regression against their 2026-07-08 baselines. milmmt-46-4b itself has no prior baseline (first-ever run of this model), so no regression check applies to it either. `show_stats.py regressions --lookback 5` (no `--model` filter) does fire four WARNINGs (`duration_s`, `gpu_vram_peak_mib`, `ram_peak_mib`, `chars_per_sec`) comparing the `272694b8409a` row against a "prior 5" window built from sarvam-translate and krutrim-translate rows — this is the same cross-model rolling-average false-alarm mechanism already documented on 2026-07-08 (point 1 under Regressions); it is not a genuine regression and should be disregarded. Filtering `--model milmmt-46-4b` correctly returns no output (fewer than the 3 same-model prior runs required for a baseline).

**Patterns detected:**

- VRAM budget: milmmt-46-4b alone consumes essentially the entire 7.5 GB usable ceiling. Per the VRAM budget table, it cannot be run alongside Ollama (4.7 GB) or any second model — this is the reason it is opt-in-only, confirmed by today's measured 8,050 MiB peak.
- `swap_peak_mib > 0` on both milmmt-46-4b runs (654 MiB final, 536 MiB smoke) — part of a session-wide swap pattern shared with the 2026-07-09 entry below; see that entry's note on likely host/WSL2-level (not code-level) cause.
- GPU util avg 19% despite peak 100% on the final run — consistent with wall-clock time spent on 4-bit dequantization/decode setup and idle gaps between sequential per-sentence generations, not an undersized batch. **Do not raise `ChunkConfig.batch_size`** for this model — VRAM is already at ~99% capacity; any larger in-flight batch risks the exact OOM this model is opt-in to avoid.

**Optimization suggestions:**

- No source-code changes indicated for milmmt-46-4b — its resource profile is a function of the 4-bit quantized 4B parameter count, not a chunking/threading inefficiency. Leave `ChunkConfig.batch_size=8` unchanged for this model (`src/bn_en_translate/config.py`).
- Reiterate the 2026-07-08 tooling gap: `RunDatabase`/`show_stats.py regressions` (`scripts/show_stats.py`, `regressions` command) still has no `num_sentences`/`is_smoke` column. The false-alarm firing above is at least the third occurrence of this exact gap in three days (2026-07-08 nllb/milmmt smoke rows, 2026-07-08 cross-model comparison, now this run) — recommend prioritizing the schema fix before the next new-model evaluation session adds more noise to the rolling window.

**Resource snapshot:**
- BLEU: 68.55 (final, 90-sentence), prior avg: none (first run of this model); smoke row 88.90 excluded from any average per the flag above
- Duration: 305.9s
- VRAM peak: 8,050 MiB (of 8,151 MiB), GPU util peak: 100%
- RAM peak: 7,225 MiB, Swap: 654 MiB
- CPU avg: 50.2%

---

## 2026-07-09 — new-model evaluation (sarvam-translate, krutrim-translate) — both REJECTED, checkpoints deleted

**New-model evaluations — both REJECTED (FLORES BLEU adequate; both failed on real text):**

- **sarvam-translate** (sarvamai/sarvam-translate, Gemma3-4B + AI4Bharat, 4-bit bnb, 90-sentence FLORES final run `a4ac5158189d`): BLEU 55.67 / chrF 74.3, 139.0 ch/s translate-only, duration 162.4 s, VRAM peak 4,880 MiB, GPU util peak 72% / avg 14%, RAM peak 3,850 MiB, swap peak 977 MiB, CPU avg 19.0%. FLORES score is roughly on par with nllb-600M (55.3) and comfortably above krutrim (44.9). **Rejected anyway**: on the real 90-paragraph story it degenerated into a repetition loop — ~62% of output paragraphs repeated filler text instead of translating. FLORES BLEU did not predict this failure mode. Checkpoint deleted.
  - Four earlier same-day runs recorded during integration/debugging (not independent quality data points): `4f99daef363a` (46.5 s, BLEU 63.90, chars/s 7.0, VRAM 1,385 MiB — duration and throughput point to a 5-sentence smoke run; flag as small-sample) and three 90-sentence-scale iterations at markedly lower throughput than the final run (`07077bf1a8b7`: 28 ch/s, 1,410 MiB VRAM; `da917ad103ae`: 33 ch/s, 4,926 MiB VRAM; `e01d97576642`: 12 ch/s, 4,899 MiB VRAM) — all well below the final run's 139 ch/s, consistent with harness fixes made mid-session rather than four independent benchmark results. Only `a4ac5158189d` should be treated as the canonical sarvam-translate result.
- **krutrim-translate** (krutrim-ai-labs/Krutrim-Translate, CT2 float16, 90-sentence FLORES final run `782553d37359`): BLEU 44.87 / chrF 73.1, 7,050 ch/s, duration 5.3 s, VRAM peak 1,608 MiB, GPU util peak 4% / avg 4%, RAM peak 3,386 MiB, swap peak 987 MiB, CPU avg 12.5%. Below the nllb-600M incumbent (55.3) on BLEU alone — and separately, its internal `ben_Beng eng_Latn` language tag leaked into ~41% of real-story output paragraphs, a correctness bug independent of the BLEU gap. Checkpoint deleted.
  - `826554d8642c` (BLEU 0.00, 4.5 s) is the run that first surfaced the tag-leakage failure at full severity — a genuine bug reproduction, not noise; kept as evidence of the defect, not a candidate for averaging.
  - `56131eabb68a` (5-sentence smoke, BLEU 64.77, VRAM 1,613 MiB): **flag as small-sample artifact**, do not use for ranking — same mechanism as sarvam's smoke row above and the lmt-60-1.7b/nllb/milmmt-1b smoke rows on 2026-07-08.

**Regressions vs incumbents:** nllb-600M (55.3), milmmt-46-1b (65.2), seamless-medium (67.0) were **not re-run** on 2026-07-09 — nothing to compare, no regression possible for those three. Both sarvam-translate and krutrim-translate are first-time model entries in `runs.db`, so `show_stats.py regressions --model <name>` correctly returns no output for either (fewer than 3 same-model prior runs required for a baseline). The unfiltered `regressions --lookback 5` command should not be run across these mixed-model rows without `--model` — see the cross-model false-alarm mechanism already documented on 2026-07-08 and reiterated in the 2026-07-10 entry above.

**Patterns detected:**

- **Small-sample BLEU inflation, again:** both new models show the same 5-sentence-smoke-inflates-BLEU pattern as every prior new-model evaluation this project has run (lmt-60-1.7b +9.3 pts, hunyuan-mt-7b +6.6 pts on 2026-07-08; sarvam +8.2 pts, krutrim +19.9 pts here). This is now a consistent, reproducible artifact across 5+ independent model evaluations — reiterating the 2026-07-08 tooling recommendation with added urgency: the gap is large and directionally consistent enough that any dashboard or report reading `runs.db` unfiltered will systematically overstate quality for whichever model was smoke-tested most recently.
- **Elevated swap on every run this session (both models, all 8 runs: 624-994 MiB peak)** — materially different from the 2026-07-08 close-out entry, which reported swap ≤5 MiB on the three production models and dismissed 100-126 MiB swap on old smoke rows as WSL2 background noise. Here the swap magnitude is 5-10x larger and appears uniformly regardless of model size or VRAM footprint: krutrim-translate (1.4-1.6 GB VRAM, the lightest model in the whole project) shows the *highest* swap of the session (987-994 MiB), while heavier sarvam-translate (4.9 GB VRAM) shows comparable or lower swap (624-977 MiB). Swap magnitude does not track VRAM/RAM footprint or `ChunkConfig.batch_size`, which argues against a code-level fix (the diagnostic table's default suggestion of reducing `batch_size` from 8 to 4 is unlikely to help here) and toward host/WSL2-level memory pressure — e.g. `vmmem` fragmentation or a lingering process from earlier work — as the more likely cause. Recommend checking `free -h` and WSL2 memory state (`wsl --shutdown` and cold-restart if it recurs) before the next benchmark session, rather than touching `ChunkConfig`. The same pattern continued into the 2026-07-10 milmmt-46-4b runs (536-654 MiB swap), suggesting the condition persisted across sessions rather than being a one-off.
- `cpu_avg_pct` stayed low across all runs this session (11-50%) — no CPU-bound data-loading bottleneck; the `nllb_ct2.py` `inter_threads`/`intra_threads` settings checked on 2026-07-08 remain adequate and are not implicated by krutrim's CT2 backend today.

**Optimization suggestions:**

- No source-code changes indicated by sarvam-translate or krutrim-translate data — both are excluded models with deleted checkpoints; their resource profiles are moot going forward.
- Reiterate (now a repeated finding, see 2026-07-10 entry above): add `num_sentences`/`is_smoke` to the `RunDatabase` schema and exclude smoke rows from rolling averages by default in `scripts/show_stats.py` (`regressions` command).
- New: investigate the session-wide swap pattern at the host/WSL2 level (not a `bn_en_translate` code change) before the next benchmark session — `free -h` / `wsl --shutdown` as a first check, per the pattern note above.

**Resource snapshot:**
- BLEU: sarvam-translate 55.67 (final), krutrim-translate 44.87 (final); prior avg: none (both first-run models)
- Duration: sarvam 162.4s, krutrim 5.3s
- VRAM peak: sarvam 4,880 MiB (GPU util peak 72%), krutrim 1,608 MiB (GPU util peak 4%)
- RAM peak: sarvam 3,850 MiB (swap 977 MiB), krutrim 3,386 MiB (swap 987 MiB)
- CPU avg: sarvam 19.0%, krutrim 12.5%

---

## 2026-07-08 — new-model evaluation (lmt-60-1.7b, hunyuan-mt-7b) + close-out benchmark + MADLAD re-download verdict (branch: perf/optimization-pass)

**New-model evaluations — both REJECTED:**

- **lmt-60-1.7b** (90-sentence FLORES, run `a63da1133a8c`): BLEU 63.84 / chrF 78.3, 86 ch/s, duration 139.2 s, VRAM peak 6,634.9 MiB, GPU util peak 100% / avg 14%, RAM peak 5,430 MiB, CPU avg 31.9%, `num_beams=5`. Below seamless-medium (67.0) and milmmt-46-1b (65.2) on BLEU while costing ~2 GB more VRAM than either — rejected on quality-per-VRAM.
- lmt-60-1.7b **5-sentence smoke** (run `432433753d67`): BLEU **73.18** vs the 90-sentence figure of **63.84** — a 9.3-point gap from sample size alone. **Never gate a model accept/reject decision on a 5-sentence BLEU number**; the smoke test exists to catch load/crash failures, not to rank quality. This is the same small-sample-inflation pattern documented for nllb-600M and milmmt-46-1b below.
- **hunyuan-mt-7b** (Q4 via Ollama, 90-sentence FLORES, run `80a3c494ddcd`): BLEU 54.7 / chrF 74.7, 112 ch/s, duration 34.0 s, VRAM peak 6,614.4 MiB, GPU util peak 89% / avg 59%. Below all three production models on BLEU (nllb 55.3, milmmt 65.2, seamless 67.0) while using the most VRAM of any accepted or candidate model — rejected. A companion 5-sentence run (`526fd3c565da`) scored BLEU 61.3 — same inflation pattern, disregarded per the rule above.
- Both rejected models sit at ~6.6 GB VRAM standalone — each alone consumes essentially the full 7.5 GB usable budget, leaving no headroom for Ollama polish or any second model. This is a second, independent reason neither is viable even ignoring BLEU.

**Ollama CPU-fallback gotcha (operational, not captured as a runs.db row):**

- If VRAM is already occupied (e.g., by a prior model that wasn't unloaded) when Ollama loads a model, Ollama silently falls back to 100% CPU inference instead of erroring — observed at ~5.4 tok/s CPU vs ~58.5 tok/s GPU for the same model, a >10× slowdown. One benchmark attempt today timed out this way; because the pipeline's Ollama HTTP client has a 120 s timeout and the CPU-bound cold load exceeded it, the run never completed and consequently never wrote a row to `runs.db` — this incident is invisible to `show_stats.py` and must be logged here instead.
- **Mitigation:** run `ollama ps` before trusting any Ollama-backed benchmark number, to confirm the model is actually resident on GPU (`PROCESSOR` column shows `100% GPU`, not `100% CPU`). Also be aware the first request after a fresh `ollama pull` can exceed the translator's 120 s HTTP timeout purely from cold load — retry once before concluding the model is broken.

**Close-out three-model run at BLEU parity (all at 90-sentence FLORES, translate-only ch/s per the Task-1 Load/translate split):**

- nllb-600M (`2713ab956b97`): BLEU 55.27 @ 2,346 ch/s, VRAM peak 2,552 MiB — matches the 55.3 April baseline.
- milmmt-46-1b (`43d3473c7721`): BLEU 65.24 @ 401 ch/s, VRAM peak 3,557 MiB — +0.2 vs the April baseline of 65.0; this is the known left-padded-batch-generation effect documented in the 2026-07-07 entry above, well within the BLEU gate.
- seamless-medium (`de028b8d7954`): BLEU 67.00 @ 372 ch/s, VRAM peak 4,630 MiB — exact match to the 67.0 April baseline.
- All three confirm Phase 1's batching + SDPA + length-sorting changes are BLEU-neutral-to-positive at 90-sentence scale; this run is the new post-optimization baseline for `runs.db` comparisons going forward.

**MADLAD-3B re-download verdict:**

- Re-downloaded `models/madlad-3b-hf/` clean from source and re-ran the tied-embedding guard (`shared.weight == decoder.embed_tokens.weight` check immediately after `from_pretrained()`, before the model is marked loaded). The guard still fails on the fresh download — the corruption is **at the source checkpoint**, not a local artifact of a prior bad download/interrupted transfer. MADLAD-3B is now **permanently excluded**; the local checkpoint has been deleted to reclaim disk. No `runs.db` row exists for this attempt (the guard raises before a benchmark run is instantiated, so nothing reaches `RunDatabase`) — the only DB evidence of MADLAD's failure remains the three April rows (`c715cc7b7daf`, `d0417e625e3d`, `315a147013a3`, all BLEU 0.0), which should continue to be read as "excluded," not as a live regression target.

**Regressions — none genuine; two false-alarm mechanisms found in the tooling itself:**

1. `python scripts/show_stats.py regressions --lookback 5` (no `--model` filter) flagged WARNING on `duration_s` (52.96 vs prior_avg 40.12), `ram_peak_mib` (5,674 vs 2,738), and `chars_per_sec` (372 vs 607) against the latest row. That latest row is the seamless-medium close-out run, and the "prior 5" it's compared against are a mix of hunyuan-mt-7b, lmt-60-1.7b, milmmt-46-1b, and nllb-600M rows — completely different models with different resource profiles. Comparing seamless (heavier, slower) against a rolling average dominated by lighter/faster models is not a regression signal; it's a model-identity mismatch. **Always pass `--model <name>` when checking regressions**, or the rolling average is meaningless.
2. Filtered per-model (`--model nllb-600M`), regressions still reported WARNING+CRITICAL on `bleu_score` (55.27 vs prior_avg 63.93), and `--model milmmt-46-1b` reported WARNING (65.24 vs prior_avg 66.54). Root cause: the "prior 5" window contains 5-sentence smoke-test rows (`af333ae2e76c` and `37e068100f82`, both nllb-600M BLEU 76.91 on 5 sentences; `7b4a41c2d10d`, milmmt-46-1b BLEU 71.70 on 5 sentences) interleaved with genuine 90-sentence FLORES rows. Small-sample BLEU inflation (same mechanism as lmt-60-1.7b's 73.2-vs-63.8 gap above) drags the rolling average up, making the next real 90-sentence run look like a regression when it is in fact flat (nllb 55.27 vs April 55.3; milmmt 65.24 vs April 65.0, +0.2 known effect). **No genuine BLEU, VRAM, or throughput regression exists in today's data** once smoke-test rows and cross-model comparisons are excluded.
- **Tooling suggestion:** `RunDatabase` / `show_stats.py regressions` has no concept of sample size (`num_sentences` is not stored or filtered on). Recommend adding a `num_sentences` (or `is_smoke`) column and excluding smoke runs from the rolling-average window by default — this is the second independent false-alarm mechanism found in this tool in two days (see also the pre-6d56b34 load+translate conflation and pre-July inter-model VRAM residue caveats already on record) and will keep recurring until fixed at the schema level.
- Minor, sub-threshold observation: both nllb 5-sentence smoke rows (`af333ae2e76c` swap_peak 126 MiB, `37e068100f82` swap_peak 108 MiB) show nonzero swap despite RAM peak only ~2 GB of 11 GB total (~18% utilization) — almost certainly WSL2/OS-level swap noise rather than code-driven RAM pressure, since no other run today (including the heavier lmt-60/hunyuan runs at 5.4-6.6 GB VRAM and matching RAM growth) shows swap activity. Not actioned; flagged for pattern-tracking only.

**Optimization suggestions:**
- `/home/sbisw/github/translate/scripts/show_stats.py` (`regressions` command, around line 274-310): add a `num_sentences`/`is_smoke` filter so smoke-test rows never enter the rolling-average baseline, and consider defaulting `--model` to required (or emitting a loud caveat) when comparing across heterogeneous model rows.
- No source-code (`nllb_ct2.py`, `chunker.py`, `config.py`) changes indicated by today's data — `inter_threads=1, intra_threads=4` (nllb_ct2.py:76-77) and `ChunkConfig.batch_size=8` (config.py:45) remain correct for current CPU/GPU utilization profiles; none of today's `cpu_avg_pct` readings exceed 40% and no `gpu_util_peak_pct < 50` pattern persists across repeated same-model runs.

**Resource snapshot (close-out run, new post-Phase-1 baseline):**
- BLEU: nllb 55.27 (April baseline 55.3), milmmt 65.24 (April baseline 65.0, +0.2 known effect), seamless 67.00 (April baseline 67.0)
- Duration: nllb 6.9 s, milmmt 16.0 s, seamless 53.0 s (translate-only-adjacent per Task-1 Load/translate split)
- VRAM peak: nllb 2,552 MiB, milmmt 3,557 MiB, seamless 4,630 MiB; GPU util peak 50% / 40% / 73% respectively
- RAM peak: nllb 2,117 MiB, milmmt 3,007 MiB, seamless 5,674 MiB; Swap: ≤5 MiB across all three (negligible)
- CPU avg: nllb 9.2%, milmmt 19.1%, seamless 38.4%

---

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
