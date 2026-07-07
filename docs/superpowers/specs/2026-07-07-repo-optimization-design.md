# Repo Optimization & Performance Pass — Design

**Date:** 2026-07-07
**Status:** Approved by user (scope: all four areas + new-model research + docs-as-you-go)
**Branch:** `perf/optimization-pass` (off `feat/new-models-indicTrans2-milmmt46`)

## Goal

Make bn-en-translate measurably faster and cleaner without regressing translation
quality: fix the un-batched benchmark loop, use faster attention kernels, enforce
the VRAM budget in code, retire quality debt, complete the model roster, and
evaluate newer models that could beat Seamless-medium (BLEU 67.0) on the
RTX 5050 8 GB. User and developer docs are updated in the same commit as each
change they describe.

## Evidence (what's wrong today)

1. `scripts/benchmark.py:40` runs `[pipeline.translate(t) for t in bengali_texts]`
   — every FLORES sentence is a batch of 1. HF models (MiLMMT 28 ch/s,
   Seamless 31 ch/s) never batch. Largest performance lever in the repo.
2. `milmmt.py` falls back to `eager` attention when flash-attn is absent
   (it always is on sm_120/WSL2). PyTorch `sdpa` is faster and always available.
3. No `torch.cuda.empty_cache()` / `reset_peak_memory_stats()` between models in
   multi-model benchmark runs → inter-model VRAM residue already produced one
   false regression WARNING (see `monitor/observations.md` 2026-04-10).
4. HF batches pad every sentence to the batch max; no length-sorted batching
   (CT2 does this internally; the HF paths don't).
5. MADLAD loads a known-corrupted checkpoint (`shared.weight ≠
   decoder.embed_tokens.weight`) and silently emits garbage — no guard.
6. HF load/unload/device-resolution boilerplate is copy-pasted across
   `seamless.py`, `milmmt.py`, `madlad.py`, `indicTrans2.py`.
7. `ModelConfig` defaults contradict reality: `compute_type="int8"` (probe always
   picks float16 on this GPU), `model_path` is relative while `CT2_MODEL_PATHS`
   is REPO_ROOT-anchored.
8. VRAM budget (model + Ollama > 7.5 GB combinations) documented only in
   markdown, not enforced in code.
9. Roster gaps: indicTrans2-1B blocked on gated-repo login; MADLAD-3B needs a
   clean re-download and is borderline on 8 GB.

## Approach decision

Phased sequential execution with before/after benchmark measurement per change
(chosen over big-bang and over parallel subagents: perf changes need individual
attribution, and the single GPU serializes benchmarking anyway).

## Phase 1 — Inference performance (measured)

Baseline first: run the 90-sentence FLORES benchmark on nllb-600M,
milmmt-46-1b, seamless-medium; snapshot BLEU/chrF/ch/s/VRAM to
`docs/perf_baseline_2026-07-07.md` before touching code.

1. **Batch the benchmark loop.** Normalize each sentence, then call
   `translator.translate(batch)` in `ChunkConfig.batch_size` groups, preserving
   1:1 hypothesis↔reference alignment. Sentences are <400 tokens so chunking is
   unnecessary. Keep the per-sentence path available behind
   `--no-batch` for comparability checks.
2. **SDPA attention fallback.** `milmmt.py` (and `madlad.py`): attention
   fallback `eager` → `sdpa` when flash-attn is unavailable. Verify Seamless
   already uses its default optimal path.
3. **Benchmark VRAM hygiene.** After each model's run in `benchmark.py`:
   `torch.cuda.empty_cache()` + `torch.cuda.reset_peak_memory_stats()` so
   per-model peak readings are honest.
4. **Length-sorted batching** in `pipeline._translate_in_batches`: sort chunk
   texts by estimated token length, translate, restore original order.
   Cuts padding waste on HF paths; harmless on CT2.

Acceptance gate per change: BLEU within ±0.3 of baseline per model, ch/s
recorded, `make test` green. One commit per change with its measurement in the
commit message.

## Phase 2 — Code quality

1. **MADLAD integrity guard**: at `load()`, compare `shared.weight` and
   `decoder.embed_tokens.weight`; raise `RuntimeError` with re-download
   instructions instead of emitting garbage. TDD with a mocked model.
2. **Shared HF mixin**: extract common load/unload/device-resolution/
   empty_cache boilerplate from the four HF model files.
3. **Config honesty**: `ModelConfig.compute_type` default documented/aligned
   with probe behavior; `model_path` anchored via `REPO_ROOT`.
4. **Repo hygiene**: commit `monitor/observations.md`; `make lint` and
   `make typecheck` green.

## Phase 3 — VRAM budget enforcement

Encode the per-model VRAM table (from `monitor/observations.md`) in
`config.py` or `cuda_check.py`. Before the Ollama polish pass: assert the
translator is unloaded, check `get_free_vram_mib()` against the Ollama model's
requirement, raise a clear `RuntimeError` naming the safe combinations instead
of OOM-ing mid-run. Unit-tested with mocks; no GPU needed for tests.

## Phase 4 — Model roster completion + new-model evaluation

**Blocked-on-user items** (surfaced at plan delivery):
- indicTrans2-1B: user must accept terms at
  huggingface.co/ai4bharat/indictrans2-indic-en-1B and run
  `! huggingface-cli login`. Then: download, CT2-convert, benchmark.
- MADLAD-3B: one clean re-download attempt behind the new integrity guard;
  if 8 GB still can't hold it, it stays excluded (guard documents why).

**New-model candidates (web research, 2026-07-07), priority order:**

| Candidate | Bengali | Fits 8 GB? | Integration path | Why |
|-----------|---------|------------|------------------|-----|
| MiLMMT-46-4B-v0.1 | ✅ (46 langs) | bf16 8.6 GB ✗ → 4-bit quant ~3 GB ✅ | Same family/prompt as existing MiLMMT-46-1B translator | 1B sibling already scores 65.0; 12B beats Seed-X-7B on FLORES+ xx→en; 4B is the sweet spot |
| Hunyuan-MT-7B | ✅ | bf16 16 GB ✗ → GGUF Q4_K_M ~5 GB via Ollama ✅ | Reuse OllamaTranslator with MT prompt | WMT25 winner in 30/31 language pairs |
| NiuTrans LMT-60-1.7B | ✅ (60 langs) | bf16 ~3.4 GB ✅ native | New HF causal-LM translator (pattern exists) | 4B variant beats Aya-101-13B; 1.7B fits natively without quantization |
| GemmaX2-28-2B | ✅ | ✅ | — | Skipped: superseded by MiLMMT-46 (same lab, newer) |
| Seed-X-7B | unconfirmed | quant only | — | Skipped: Bengali not in documented 28 languages |

Evaluation protocol: integrate one candidate at a time, benchmark on the same
90-sentence FLORES set, log to RunDatabase, invoke monitor agent, and only keep
models that beat an incumbent on BLEU or on BLEU-per-GB. Update CLAUDE.md,
README, docs/MODELS.md, and paper tables with measured results.

## Docs-as-you-go (cross-cutting, user request)

Every phase's final commit updates the docs that the change touches:
- **User docs**: `README.md` (usage/flags), `docs/MODELS.md` (roster/results)
- **Developer docs**: `docs/ARCHITECTURE.md` (mixin, VRAM enforcement),
  `docs/DEVELOPMENT.md` (benchmark flags, measurement protocol),
  `docs/MONITORING.md` (VRAM hygiene), `docs/HARDWARE.md` (budget table)
- `CLAUDE.md` state block after each phase.
No separate docs phase — stale docs fail the phase's acceptance gate.

## Testing strategy

TDD for all new logic (integrity guard, VRAM check, sorted batching, batched
benchmark path). Full `make test` (217+) per phase. Benchmark before/after per
perf change with BLEU parity gate ±0.3. `make lint` + `make typecheck` green at
each phase boundary. Checkpoint commit per logical unit, pushed per phase.

## Success criteria

- HF-model benchmark wall-clock ≥2× faster with BLEU parity (±0.3)
- Zero false VRAM regression warnings in multi-model runs
- MADLAD garbage-output failure mode impossible (guard raises instead)
- Ollama polish OOM impossible (pre-flight check raises instead)
- Roster table has measured numbers for every non-excluded model
- At least one new candidate benchmarked; kept only if it beats an incumbent
- Docs accurate to shipped behavior at every phase boundary
