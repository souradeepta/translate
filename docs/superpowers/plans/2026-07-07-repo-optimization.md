# Repo Optimization & Performance Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bn-en-translate benchmarks ≥2× faster on HF models with BLEU parity (±0.3), enforce the VRAM budget in code, remove quality debt (dead `--ollama-polish` flag, unguarded corrupt MADLAD checkpoint, copy-pasted HF boilerplate), and evaluate newer models (Hunyuan-MT-7B, NiuTrans LMT-60, MiLMMT-46-4B) against the Seamless-medium incumbent (BLEU 67.0).

**Architecture:** Phased sequential execution on branch `perf/optimization-pass`. Every perf change is measured against a committed baseline before the next change lands. Docs are updated in the same commit as the change they describe (user request: docs-as-you-go). Spec: `docs/superpowers/specs/2026-07-07-repo-optimization-design.md`.

**Tech Stack:** Python 3.11, PyTorch 2.7.0+cu128, transformers 5.x, CTranslate2 (float16 on sm_120), sacrebleu, pytest, Ollama.

**Environment (every session, before anything else):**
```bash
cd /home/sbisw/github/translate
source .venv/bin/activate && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

**Hard rules (from CLAUDE.md — violations are plan failures):**
- GPU-only inference: never add CPU fallback; raise `RuntimeError` when CUDA is required but missing.
- No `device_map="auto"` anywhere except the existing MADLAD code (and Task 5 does not add it elsewhere).
- Conventional commits, one concern per commit, no Claude attribution lines.
- `make test` must pass before every commit.

---

## Phase 0 — Branch & Baseline

### Task 0: Create branch and snapshot the performance baseline

**Files:**
- Create: `docs/perf_baseline_2026-07-07.md`

- [ ] **Step 0.1: Create the branch**

```bash
git checkout -b perf/optimization-pass
```

- [ ] **Step 0.2: Confirm the environment works**

Run: `make test`
Expected: `217 passed` (approximately, ~27 s). If this fails, STOP — fix the environment first.

- [ ] **Step 0.3: Run the 90-sentence baseline benchmark (GPU, ~10-15 min)**

```bash
python scripts/benchmark.py --models nllb-600M milmmt-46-1b seamless-medium --sentences 90 2>&1 | tee /tmp/claude-1000/-home-sbisw-github-translate/bd1797e0-21ca-4b99-b212-4e9af9db3b03/scratchpad/baseline_bench.txt
```

Expected: BLEU ≈ 55.3 (nllb), ≈ 65.0 (milmmt), ≈ 67.0 (seamless). If any model errors, STOP and report.

- [ ] **Step 0.4: Write the baseline doc**

Create `docs/perf_baseline_2026-07-07.md` with the measured table (fill in the actual numbers from Step 0.3 — the values below are the expected prior measurements, replace them):

```markdown
# Performance Baseline — 2026-07-07 (pre-optimization)

FLORES-200 devtest, 90 sentences, RTX 5050 8 GB, WSL2.
Command: `python scripts/benchmark.py --models nllb-600M milmmt-46-1b seamless-medium --sentences 90`
Benchmark loop: UNBATCHED (one sentence per pipeline.translate call) — this is the loop Task 1 replaces.

| Model | BLEU | chrF | Time (s) | ch/s | VRAM peak (MiB) |
|-------|------|------|----------|------|-----------------|
| nllb-600M | 55.3 | 72.8 | <measured> | 191 | 2355 |
| milmmt-46-1b | 65.0 | 79.3 | <measured> | 28 | 3379 |
| seamless-medium | 67.0 | 80.2 | <measured> | 31 | 4096 |

Acceptance gate for all Phase 1 changes: BLEU within ±0.3 of this table per model.
```

- [ ] **Step 0.5: Commit**

```bash
git add docs/perf_baseline_2026-07-07.md
git commit -m "perf: snapshot pre-optimization benchmark baseline"
```

---

## Phase 1 — Inference Performance (measured per change)

### Task 1: Batched sentence translation in the benchmark

The benchmark currently does `[pipeline.translate(t) for t in bengali_texts]` — batch size 1 for every sentence. Add `TranslationPipeline.translate_sentences()` (batched, order-preserving, 1:1 in/out) and use it in `scripts/benchmark.py` with a `--no-batch` escape hatch.

**Files:**
- Modify: `src/bn_en_translate/pipeline/pipeline.py`
- Modify: `scripts/benchmark.py:40` (the list-comprehension loop) and the arg parser
- Test: `tests/unit/test_pipeline_sentences.py` (new)

- [ ] **Step 1.1: Write the failing tests**

Create `tests/unit/test_pipeline_sentences.py`:

```python
"""Tests for TranslationPipeline.translate_sentences (batched, 1:1, order-preserving)."""

from __future__ import annotations

from bn_en_translate.config import ChunkConfig, PipelineConfig
from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.pipeline.pipeline import TranslationPipeline


class RecordingTranslator(TranslatorBase):
    """Mock that records every batch it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        self.batches.append(list(texts))
        return [f"[MOCK] {t}" for t in texts]


def _make_pipeline(batch_size: int = 3) -> tuple[TranslationPipeline, RecordingTranslator]:
    translator = RecordingTranslator()
    translator.load()
    config = PipelineConfig(chunk=ChunkConfig(batch_size=batch_size))
    return TranslationPipeline(translator, config), translator


def test_translate_sentences_one_to_one_and_ordered() -> None:
    pipeline, _ = _make_pipeline()
    sentences = [f"বাক্য {i}।" for i in range(7)]
    out = pipeline.translate_sentences(sentences)
    assert len(out) == 7
    for i, o in enumerate(out):
        assert f"বাক্য {i}" in o


def test_translate_sentences_batches_by_batch_size() -> None:
    pipeline, translator = _make_pipeline(batch_size=3)
    pipeline.translate_sentences([f"বাক্য {i}।" for i in range(7)])
    assert [len(b) for b in translator.batches] == [3, 3, 1]


def test_translate_sentences_normalizes_input() -> None:
    pipeline, translator = _make_pipeline()
    pipeline.translate_sentences(["  বাক্য\t\tএক।  "])
    # normalize() collapses runs of spaces/tabs and strips
    assert translator.batches[0][0] == "বাক্য এক।"


def test_translate_sentences_empty_list() -> None:
    pipeline, translator = _make_pipeline()
    assert pipeline.translate_sentences([]) == []
    assert translator.batches == []
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `pytest tests/unit/test_pipeline_sentences.py -v`
Expected: FAIL with `AttributeError: 'TranslationPipeline' object has no attribute 'translate_sentences'`

- [ ] **Step 1.3: Implement `translate_sentences`**

In `src/bn_en_translate/pipeline/pipeline.py`, add after the `translate` method (reuses the existing `_translate_in_batches` and `normalize` import):

```python
    def translate_sentences(self, sentences: list[str]) -> list[str]:
        """
        Translate pre-split sentences with true batching, 1:1 in/out.

        Unlike translate(), this does no chunking or reassembly — each input
        string maps to exactly one output string, in order. Intended for
        benchmark corpora where hypothesis/reference alignment must be exact.
        Inputs are normalized the same way translate() normalizes documents.
        """
        if not sentences:
            return []
        normalized = [normalize(s) for s in sentences]
        return self._translate_in_batches(normalized)
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest tests/unit/test_pipeline_sentences.py -v`
Expected: 4 PASS

- [ ] **Step 1.5: Use it in the benchmark with a `--no-batch` escape hatch**

In `scripts/benchmark.py`:

Add to `benchmark_model` signature (after `device: str = "auto"`):

```python
def benchmark_model(
    model_name: str,
    bengali_texts: list[str],
    references: list[str],
    device: str = "auto",
    batched: bool = True,
) -> dict:  # type: ignore[type-arg]
```

Replace the line `hypotheses = [pipeline.translate(t) for t in bengali_texts]` with:

```python
                if batched:
                    hypotheses = pipeline.translate_sentences(bengali_texts)
                else:
                    hypotheses = [pipeline.translate(t) for t in bengali_texts]
```

In `main()`, add the flag after the `--sentences` argument:

```python
    parser.add_argument("--no-batch", action="store_true",
                        help="Translate one sentence at a time (pre-2026-07 behavior)")
```

and pass it through in the model loop:

```python
        r = benchmark_model(model_name, bn_texts, en_refs, device=args.device,
                            batched=not args.no_batch)
```

- [ ] **Step 1.6: Full test suite**

Run: `make test`
Expected: all pass (221 = 217 + 4 new)

- [ ] **Step 1.7: Measure (GPU) — BLEU parity gate**

```bash
python scripts/benchmark.py --models nllb-600M milmmt-46-1b seamless-medium --sentences 90
```

Compare against `docs/perf_baseline_2026-07-07.md`:
- BLEU within ±0.3 per model → PASS. If BLEU drops >0.3 on an HF model, the padding/attention interaction is at fault — investigate with `--no-batch` A/B before proceeding (systematic-debugging skill).
- Record new ch/s and wall-clock. Expect milmmt and seamless to improve substantially; nllb (CT2 batches internally) to improve modestly.

- [ ] **Step 1.8: Update docs (same commit)**

- `docs/DEVELOPMENT.md`: document `translate_sentences()` and the `--no-batch` flag in the benchmarking section.
- `docs/perf_baseline_2026-07-07.md`: append a "After Task 1 (batched loop)" table with the new measurements.

- [ ] **Step 1.9: Commit (include measurements in the message body)**

```bash
git add src/bn_en_translate/pipeline/pipeline.py scripts/benchmark.py tests/unit/test_pipeline_sentences.py docs/DEVELOPMENT.md docs/perf_baseline_2026-07-07.md
git commit -m "perf(benchmark): batch sentence translation via translate_sentences()" \
  -m "90-sentence FLORES: milmmt <old> -> <new> ch/s, seamless <old> -> <new> ch/s, BLEU deltas <values> (all within ±0.3 gate)."
```

### Task 2: SDPA attention fallback for MiLMMT and MADLAD

flash-attn is never installed on this machine (sm_120/WSL2), so the `eager` fallback always wins. PyTorch's built-in `sdpa` kernel is faster and always available in torch 2.7.

**Files:**
- Modify: `src/bn_en_translate/models/milmmt.py:100-104`
- Modify: `src/bn_en_translate/models/madlad.py:82-86`
- Test: `tests/unit/test_milmmt.py`, `tests/unit/test_madlad.py` (add one test each; if an existing test asserts `"eager"`, update it)

- [ ] **Step 2.1: Write the failing tests**

First check for existing attention tests: `grep -n "eager\|attn" tests/unit/test_milmmt.py tests/unit/test_madlad.py`. Update any that assert `"eager"` as the fallback. Then add to `tests/unit/test_milmmt.py`:

```python
def test_attn_fallback_is_sdpa(monkeypatch) -> None:
    """Without flash-attn installed, the fallback must be sdpa, not eager."""
    import bn_en_translate.models.milmmt as milmmt_mod

    monkeypatch.setattr(milmmt_mod, "_flash_attn_available", lambda: False)
    assert milmmt_mod._resolve_attn_implementation(use_flash=True) == "sdpa"
    assert milmmt_mod._resolve_attn_implementation(use_flash=False) == "sdpa"


def test_attn_uses_flash_when_available(monkeypatch) -> None:
    import bn_en_translate.models.milmmt as milmmt_mod

    monkeypatch.setattr(milmmt_mod, "_flash_attn_available", lambda: True)
    assert milmmt_mod._resolve_attn_implementation(use_flash=True) == "flash_attention_2"
```

And the equivalent two tests in `tests/unit/test_madlad.py` against `bn_en_translate.models.madlad`.

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest tests/unit/test_milmmt.py tests/unit/test_madlad.py -v -k attn`
Expected: FAIL with `AttributeError: module ... has no attribute '_resolve_attn_implementation'`

- [ ] **Step 2.3: Implement**

In `src/bn_en_translate/models/milmmt.py`, add below `_flash_attn_available()`:

```python
def _resolve_attn_implementation(use_flash: bool) -> str:
    """flash_attention_2 if installed and requested; else PyTorch SDPA.

    SDPA (scaled_dot_product_attention) is always available in torch>=2.0 and is
    significantly faster than eager. flash-attn is not installable on sm_120/WSL2
    as of 2026-07, so SDPA is the effective default on this machine.
    """
    if use_flash and _flash_attn_available():
        return "flash_attention_2"
    return "sdpa"
```

In `load()`, replace the `attn_impl = (...)` conditional with:

```python
        attn_impl = _resolve_attn_implementation(self.config.use_flash_attention)
```

Make the identical change in `src/bn_en_translate/models/madlad.py` (add the same `_resolve_attn_implementation` function and replace its `attn_impl = (...)` block). Note: Task 6 will deduplicate these two copies into `hf_utils.py` — that's intentional sequencing (working code first, then refactor).

- [ ] **Step 2.4: Run tests**

Run: `pytest tests/unit/test_milmmt.py tests/unit/test_madlad.py -v` then `make test`
Expected: all PASS

- [ ] **Step 2.5: Measure (GPU) — milmmt only (madlad is excluded from benchmarks)**

```bash
python scripts/benchmark.py --models milmmt-46-1b --sentences 90
```

Gate: BLEU within ±0.3 of the post-Task-1 measurement. Record ch/s delta. (Seamless is untouched — `SeamlessM4Tv2ForTextToText` picks its own default attention; verify with `grep -n "attn" src/bn_en_translate/models/seamless.py` that we never force eager there. No change expected.)

- [ ] **Step 2.6: Update docs + commit**

Update `docs/MODELS.md` MiLMMT section (attention: SDPA default, flash-attn if installed) and append measurement to `docs/perf_baseline_2026-07-07.md`.

```bash
git add src/bn_en_translate/models/milmmt.py src/bn_en_translate/models/madlad.py tests/unit/test_milmmt.py tests/unit/test_madlad.py docs/MODELS.md docs/perf_baseline_2026-07-07.md
git commit -m "perf(models): fall back to SDPA attention instead of eager" \
  -m "milmmt 90-sentence: <old> -> <new> ch/s, BLEU delta <value>."
```

### Task 3: Per-model CUDA state reset in multi-model benchmark runs

Fixes the false VRAM regression warnings flagged in `monitor/observations.md` (inter-model residue).

**Files:**
- Modify: `src/bn_en_translate/utils/cuda_check.py`
- Modify: `scripts/benchmark.py` (model loop in `main()`)
- Test: `tests/unit/test_cuda_check.py`

- [ ] **Step 3.1: Write the failing test**

Add to `tests/unit/test_cuda_check.py`:

```python
def test_reset_cuda_state_no_cuda_is_noop() -> None:
    """Must never raise, even without CUDA (CI, CPU-only envs)."""
    from bn_en_translate.utils.cuda_check import reset_cuda_state

    reset_cuda_state()  # should not raise


def test_reset_cuda_state_calls_torch(monkeypatch) -> None:
    import sys
    from unittest.mock import MagicMock

    from bn_en_translate.utils import cuda_check

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cuda_check.reset_cuda_state()

    fake_torch.cuda.empty_cache.assert_called_once()
    fake_torch.cuda.reset_peak_memory_stats.assert_called_once()
```

- [ ] **Step 3.2: Run to verify failure**

Run: `pytest tests/unit/test_cuda_check.py -v -k reset`
Expected: FAIL with `ImportError: cannot import name 'reset_cuda_state'`

- [ ] **Step 3.3: Implement in `cuda_check.py`**

```python
def reset_cuda_state() -> None:
    """Release cached CUDA allocations and reset peak-memory statistics.

    Call between models in multi-model runs so each model's VRAM peak reading
    starts from a clean slate (residual allocations from a previous model
    otherwise inflate the next model's reported peak — see
    monitor/observations.md 2026-04-10).
    Safe no-op when torch or CUDA is unavailable.
    """
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass
```

- [ ] **Step 3.4: Wire into the benchmark loop**

In `scripts/benchmark.py` `main()`, inside `for model_name in args.models:` — call after each model finishes (i.e., after the `print` calls for the result, as the last statement of the loop body):

```python
        from bn_en_translate.utils.cuda_check import reset_cuda_state
        reset_cuda_state()
```

(Put the import at the top of `main()` rather than inside the loop.)

- [ ] **Step 3.5: Tests + measure**

Run: `make test` → all pass.
GPU check: `python scripts/benchmark.py --models nllb-600M milmmt-46-1b --sentences 5` — confirm no errors and that the second model's monitor VRAM line is plausible (~3.3 GB, not inflated by nllb residue).

- [ ] **Step 3.6: Update docs + commit**

Update `docs/MONITORING.md`: note that multi-model runs reset CUDA state between models and that pre-2026-07 multi-model VRAM peaks are inflated (cross-reference the 2026-04-10 observation).

```bash
git add src/bn_en_translate/utils/cuda_check.py scripts/benchmark.py tests/unit/test_cuda_check.py docs/MONITORING.md
git commit -m "fix(benchmark): reset CUDA cache and peak stats between models" \
  -m "Prevents inter-model VRAM residue from inflating peak readings (monitor 2026-04-10 false WARNING)."
```

### Task 4: Length-sorted batching in the pipeline

HF models pad every batch to its longest member. Sorting by estimated length before batching (and restoring order after) cuts wasted decode steps. CT2 sorts internally, so NLLB is unaffected.

**Files:**
- Modify: `src/bn_en_translate/pipeline/pipeline.py` (`_translate_in_batches`)
- Test: `tests/unit/test_pipeline_sentences.py` (extend)

- [ ] **Step 4.1: Write the failing tests**

Add to `tests/unit/test_pipeline_sentences.py`:

```python
def test_batches_are_length_sorted_but_output_order_restored() -> None:
    pipeline, translator = _make_pipeline(batch_size=2)
    # Mixed lengths, deliberately unsorted
    sentences = ["ছোট।", "এটি একটি অনেক অনেক অনেক লম্বা বাংলা বাক্য যা চলতেই থাকে।", "মাঝারি বাক্য।"]
    out = pipeline.translate_sentences(sentences)
    # Output order must match input order exactly
    assert [o.replace("[MOCK] ", "") for o in out] == [
        "ছোট।",
        "এটি একটি অনেক অনেক অনেক লম্বা বাংলা বাক্য যা চলতেই থাকে।",
        "মাঝারি বাক্য।",
    ]
    # Each batch must be internally ordered shortest-to-longest input
    for batch in translator.batches:
        lengths = [len(t) for t in batch]
        assert lengths == sorted(lengths)


def test_document_translate_still_preserves_paragraphs(mock_translator) -> None:
    """Regression guard: sorting inside _translate_in_batches must not break reassembly."""
    from bn_en_translate.pipeline.pipeline import TranslationPipeline

    mock_translator.load()
    pipeline = TranslationPipeline(mock_translator)
    text = "প্রথম অনুচ্ছেদ।\n\nদ্বিতীয় অনুচ্ছেদ যা একটু লম্বা।\n\nতৃতীয়।"
    result = pipeline.translate(text)
    assert result.count("\n\n") == 2
```

- [ ] **Step 4.2: Run to verify failure**

Run: `pytest tests/unit/test_pipeline_sentences.py -v -k sorted`
Expected: `test_batches_are_length_sorted_but_output_order_restored` FAILS on the sorted-lengths assertion (current code batches in input order). The paragraph test should already pass — it's the regression guard.

- [ ] **Step 4.3: Implement**

Replace `_translate_in_batches` in `pipeline.py`:

```python
    def _translate_in_batches(self, texts: list[str]) -> list[str]:
        """Translate texts in batches, length-sorted to minimize padding waste.

        HF models pad each batch to its longest member; grouping similar
        lengths cuts wasted decode steps. Original order is restored before
        returning, so callers (and reassemble()) see 1:1 positional mapping.
        CT2 backends sort internally — this is harmless there.
        """
        from bn_en_translate.utils.text_utils import estimate_tokens

        batch_size = self.config.chunk.batch_size
        order = sorted(range(len(texts)), key=lambda i: estimate_tokens(texts[i]))
        results: list[str] = [""] * len(texts)

        for start in range(0, len(order), batch_size):
            index_batch = order[start : start + batch_size]
            batch = [texts[i] for i in index_batch]
            translated = self.translator.translate(
                batch,
                src_lang=self.config.model.src_lang,
                tgt_lang=self.config.model.tgt_lang,
            )
            for idx, out in zip(index_batch, translated):
                results[idx] = out

        return results
```

(Move the `estimate_tokens` import to the module top with the other imports.)

- [ ] **Step 4.4: Run tests**

Run: `pytest tests/unit/test_pipeline_sentences.py -v` then `make test`
Expected: all PASS (chunker/postprocessor/integration tests confirm no reassembly breakage)

- [ ] **Step 4.5: Measure (GPU) — full gate**

```bash
python scripts/benchmark.py --models nllb-600M milmmt-46-1b seamless-medium --sentences 90
```

Gate: BLEU within ±0.3 of baseline per model. Record ch/s. Expect a further improvement on milmmt/seamless (FLORES sentence lengths vary ~2×), none on nllb.

- [ ] **Step 4.6: Update docs + commit**

Update `docs/ARCHITECTURE.md` pipeline section (length-sorted batching, order restoration invariant) and append measurements to `docs/perf_baseline_2026-07-07.md` with a final Phase-1 summary row (total speedup vs baseline).

```bash
git add src/bn_en_translate/pipeline/pipeline.py tests/unit/test_pipeline_sentences.py docs/ARCHITECTURE.md docs/perf_baseline_2026-07-07.md
git commit -m "perf(pipeline): length-sorted batching with order restoration" \
  -m "90-sentence FLORES: <deltas>. Phase 1 cumulative: milmmt <baseline> -> <now> ch/s, seamless <baseline> -> <now> ch/s."
git push -u origin perf/optimization-pass
```

---

## Phase 2 — Code Quality

### Task 5: MADLAD tied-embeddings integrity guard

The local checkpoint's `shared.weight != decoder.embed_tokens.weight` corruption currently produces silent garbage. Fail loudly at load time instead.

**Files:**
- Modify: `src/bn_en_translate/models/madlad.py`
- Test: `tests/unit/test_madlad.py`

- [ ] **Step 5.1: Write the failing tests**

Add to `tests/unit/test_madlad.py`:

```python
def test_verify_tied_embeddings_raises_on_mismatch() -> None:
    import pytest
    import torch

    from bn_en_translate.models.madlad import MADLADTranslator

    class FakeEmbed:
        def __init__(self, w: "torch.Tensor") -> None:
            self.weight = w

    class FakeDecoder:
        def __init__(self, w: "torch.Tensor") -> None:
            self.embed_tokens = FakeEmbed(w)

    class FakeModel:
        def __init__(self, w1: "torch.Tensor", w2: "torch.Tensor") -> None:
            self.shared = FakeEmbed(w1)
            self.decoder = FakeDecoder(w2)

    w = torch.randn(8, 4)
    MADLADTranslator._verify_tied_embeddings(FakeModel(w, w))  # tied: no raise

    with pytest.raises(RuntimeError, match="tied-embedding mismatch"):
        MADLADTranslator._verify_tied_embeddings(FakeModel(w, torch.randn(8, 4)))
```

- [ ] **Step 5.2: Run to verify failure**

Run: `pytest tests/unit/test_madlad.py -v -k tied`
Expected: FAIL with `AttributeError: ... has no attribute '_verify_tied_embeddings'`

- [ ] **Step 5.3: Implement**

In `madlad.py`, add to `MADLADTranslator`:

```python
    @staticmethod
    def _verify_tied_embeddings(model: object) -> None:
        """Detect the known corrupt-checkpoint failure mode at load time.

        A healthy T5 MT checkpoint has shared.weight tied to
        decoder.embed_tokens.weight. The local madlad-3b-hf checkpoint was
        observed with untied (randomised) weights, which produces degenerate
        output (BLEU 0) with no error. Compare a slice — full 3B comparison
        would be slow and the corruption randomises the whole matrix.
        """
        import torch  # type: ignore[import-untyped]

        shared = model.shared.weight  # type: ignore[attr-defined]
        decoder = model.decoder.embed_tokens.weight  # type: ignore[attr-defined]
        if not torch.equal(shared[:64].float().cpu(), decoder[:64].float().cpu()):
            raise RuntimeError(
                "MADLAD checkpoint tied-embedding mismatch: shared.weight != "
                "decoder.embed_tokens.weight. This checkpoint produces garbage "
                "output. Re-download cleanly: rm -rf models/madlad-3b-hf && "
                "python scripts/download_models.py --model madlad-3b"
            )
```

In `load()`, call it immediately after `from_pretrained(...)` completes (before `self._loaded = True`):

```python
        self._verify_tied_embeddings(self._model)
```

- [ ] **Step 5.4: Run tests + commit**

Run: `pytest tests/unit/test_madlad.py -v` then `make test` → all PASS.

Update `docs/MODELS.md` MADLAD row: "load() verifies tied embeddings and raises on the known corruption instead of emitting garbage."

```bash
git add src/bn_en_translate/models/madlad.py tests/unit/test_madlad.py docs/MODELS.md
git commit -m "fix(madlad): fail loudly on tied-embedding checkpoint corruption"
```

### Task 6: Extract shared HF helpers into `hf_utils.py`

Dedupe `_flash_attn_available` / `_resolve_attn_implementation` (now in milmmt + madlad), device resolution, and the identical `unload()` bodies in the four HF-native models.

**Files:**
- Create: `src/bn_en_translate/models/hf_utils.py`
- Modify: `src/bn_en_translate/models/milmmt.py`, `madlad.py`, `seamless.py`, `indicTrans2.py`
- Test: `tests/unit/test_hf_utils.py` (new)

- [ ] **Step 6.1: Write the failing tests**

Create `tests/unit/test_hf_utils.py`:

```python
"""Tests for shared HF model helpers."""

from __future__ import annotations

from bn_en_translate.models import hf_utils


def test_resolve_attn_sdpa_fallback(monkeypatch) -> None:
    monkeypatch.setattr(hf_utils, "flash_attn_available", lambda: False)
    assert hf_utils.resolve_attn_implementation(use_flash=True) == "sdpa"
    assert hf_utils.resolve_attn_implementation(use_flash=False) == "sdpa"


def test_resolve_attn_flash_when_available(monkeypatch) -> None:
    monkeypatch.setattr(hf_utils, "flash_attn_available", lambda: True)
    assert hf_utils.resolve_attn_implementation(use_flash=True) == "flash_attention_2"


def test_resolve_device_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(
        "bn_en_translate.utils.cuda_check.get_best_device", lambda: "cuda"
    )
    assert hf_utils.resolve_device("auto") == "cuda"
    assert hf_utils.resolve_device("cuda") == "cuda"
    assert hf_utils.resolve_device("cpu") == "cpu"


def test_free_cuda_memory_never_raises() -> None:
    hf_utils.free_cuda_memory()
```

- [ ] **Step 6.2: Run to verify failure**

Run: `pytest tests/unit/test_hf_utils.py -v`
Expected: FAIL with `ImportError: cannot import name 'hf_utils'`

- [ ] **Step 6.3: Implement `src/bn_en_translate/models/hf_utils.py`**

```python
"""Shared helpers for HuggingFace-native translator implementations.

Used by milmmt.py, madlad.py, seamless.py, indicTrans2.py — keep these
free of model-specific logic.
"""

from __future__ import annotations

import importlib.util


def flash_attn_available() -> bool:
    """True if the flash-attn package is importable."""
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attn_implementation(use_flash: bool) -> str:
    """flash_attention_2 if installed and requested; else PyTorch SDPA.

    SDPA is always available in torch>=2.0 and is significantly faster than
    eager. flash-attn is not installable on sm_120/WSL2 as of 2026-07.
    """
    if use_flash and flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def resolve_device(config_device: str) -> str:
    """Resolve 'auto' to the best available device; pass through otherwise."""
    from bn_en_translate.utils.cuda_check import get_best_device

    return get_best_device() if config_device == "auto" else config_device


def free_cuda_memory() -> None:
    """Release cached CUDA allocations. Safe no-op without torch/CUDA."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
```

- [ ] **Step 6.4: Refactor the four models to use it**

In each of `milmmt.py`, `madlad.py`, `seamless.py`, `indicTrans2.py`:
- Delete the local `_flash_attn_available` and `_resolve_attn_implementation` definitions (where present) and the `importlib.util` import if now unused.
- Replace `attn_impl = ...` with `attn_impl = resolve_attn_implementation(self.config.use_flash_attention)` (milmmt, madlad, indicTrans2 — seamless doesn't set one).
- Replace the `device = get_best_device() if self.config.device == "auto" else self.config.device` pattern with `device = resolve_device(self.config.device)`.
- Replace each `unload()`'s try/except torch block with a call to `free_cuda_memory()`:

```python
    def unload(self) -> None:
        self._model = None
        self._tokenizer = None  # or self._processor for seamless
        self._loaded = False
        free_cuda_memory()
```

- Import at top of each: `from bn_en_translate.models.hf_utils import free_cuda_memory, resolve_attn_implementation, resolve_device` (only the names each file uses).
- Keep the Task 2 unit tests passing by aliasing in milmmt.py and madlad.py — the tests monkeypatch module attributes, so re-export in each file:

```python
from bn_en_translate.models.hf_utils import flash_attn_available as _flash_attn_available  # noqa: F401
```

and change the local `_resolve_attn_implementation` to a thin wrapper that respects the monkeypatched `_flash_attn_available`:

```python
def _resolve_attn_implementation(use_flash: bool) -> str:
    if use_flash and _flash_attn_available():
        return "flash_attention_2"
    return "sdpa"
```

(Yes, this keeps a 3-line wrapper per file — that is the cost of module-level monkeypatch seams; do not delete the Task 2 tests.)

- [ ] **Step 6.5: Run everything**

Run: `pytest tests/unit/test_hf_utils.py -v` then `make test`
Expected: all PASS. Then `make lint && make typecheck` — fix any fallout.

- [ ] **Step 6.6: GPU smoke test + docs + commit**

```bash
python scripts/benchmark.py --models milmmt-46-1b --sentences 5
```
Expected: normal output, no load errors.

Update `docs/ARCHITECTURE.md` models section: hf_utils.py responsibilities.

```bash
git add src/bn_en_translate/models/hf_utils.py src/bn_en_translate/models/milmmt.py src/bn_en_translate/models/madlad.py src/bn_en_translate/models/seamless.py src/bn_en_translate/models/indicTrans2.py tests/unit/test_hf_utils.py docs/ARCHITECTURE.md
git commit -m "refactor(models): extract shared HF helpers into hf_utils"
```

### Task 7: Config honesty — anchor model_path, document compute_type

`ModelConfig.model_path` defaults to a cwd-relative string while everything else is `REPO_ROOT`-anchored; `compute_type="int8"` default is misleading (the probe always selects float16 on this GPU).

**Files:**
- Modify: `src/bn_en_translate/config.py:47-49`
- Test: `tests/unit/test_config.py`, `tests/unit/test_model_config_v2.py` (check for assertions on the old defaults first: `grep -n "model_path\|compute_type" tests/unit/test_config.py tests/unit/test_model_config_v2.py`)

- [ ] **Step 7.1: Write/adjust the test**

Add to `tests/unit/test_config.py` (and update any existing assertion that expects the relative path):

```python
def test_model_path_default_is_repo_root_anchored() -> None:
    from pathlib import Path

    from bn_en_translate.config import REPO_ROOT, ModelConfig

    config = ModelConfig()
    assert Path(config.model_path).is_absolute()
    assert Path(config.model_path) == REPO_ROOT / "models/nllb-600M-ct2"
```

- [ ] **Step 7.2: Run to verify failure**

Run: `pytest tests/unit/test_config.py -v -k repo_root`
Expected: FAIL (`models/nllb-600M-ct2` is not absolute)

- [ ] **Step 7.3: Implement**

In `config.py`, change the two `ModelConfig` field lines:

```python
    model_path: str = str(REPO_ROOT / "models/nllb-600M-ct2")
    device: str = "cuda"
    # Requested compute type for CT2 backends. On sm_120 the load-time probe
    # overrides int8 with float16 (CUBLAS does not support int8 there) — this
    # value is a preference, not a guarantee. See utils/ct2_utils.probe_compute_type.
    compute_type: str = "int8"
```

- [ ] **Step 7.4: Run tests, fix fallout, commit**

Run: `make test` — any test constructing `ModelConfig()` and asserting the relative path must be updated to the anchored form.
Run: `make lint && make typecheck`

Update `docs/DEVELOPMENT.md` config section (model_path is REPO_ROOT-anchored; compute_type is a preference the probe may override).

```bash
git add src/bn_en_translate/config.py tests/unit/test_config.py tests/unit/test_model_config_v2.py docs/DEVELOPMENT.md
git commit -m "fix(config): anchor model_path to REPO_ROOT; document compute_type probe override"
```

### Task 8: Repo hygiene

**Files:**
- Add: `monitor/observations.md` (currently untracked)

- [ ] **Step 8.1: Commit the monitor observations**

```bash
git add monitor/observations.md
git commit -m "docs(monitor): add 2026-04-10 benchmark observations"
```

- [ ] **Step 8.2: Lint + typecheck clean**

Run: `make lint && make typecheck`
Fix every finding in code touched by this plan; pre-existing findings in untouched files get fixed only if trivial (one line). Commit as `chore: lint and typecheck fixes` if anything changed.

---

## Phase 3 — Wire the Ollama polish pass with VRAM enforcement

`--ollama-polish` is currently a dead flag: parsed in `cli.py`, stored in `PipelineConfig`, never applied (verified 2026-07-07: no consumer of `config.ollama_polish` exists outside cli/config/tests). Implement it properly: translate → unload translator → VRAM pre-flight → polish → write. The VRAM budget table from `monitor/observations.md` becomes code.

### Task 9: VRAM budget table + pre-flight check

**Files:**
- Modify: `src/bn_en_translate/config.py` (add `MODEL_VRAM_MIB`)
- Modify: `src/bn_en_translate/utils/cuda_check.py` (add `ensure_vram_available`)
- Test: `tests/unit/test_cuda_check.py`, `tests/unit/test_config.py`

- [ ] **Step 9.1: Write the failing tests**

Add to `tests/unit/test_cuda_check.py`:

```python
def test_ensure_vram_available_raises_when_insufficient(monkeypatch) -> None:
    import pytest

    from bn_en_translate.utils import cuda_check

    monkeypatch.setattr(cuda_check, "get_free_vram_mib", lambda: 1000)
    with pytest.raises(RuntimeError, match="polish pass"):
        cuda_check.ensure_vram_available(4800, context="Ollama polish pass")


def test_ensure_vram_available_passes_when_sufficient(monkeypatch) -> None:
    from bn_en_translate.utils import cuda_check

    monkeypatch.setattr(cuda_check, "get_free_vram_mib", lambda: 6000)
    cuda_check.ensure_vram_available(4800, context="Ollama polish pass")  # no raise
```

Add to `tests/unit/test_config.py`:

```python
def test_model_vram_table_has_known_models() -> None:
    from bn_en_translate.config import MODEL_VRAM_MIB

    for key in ("nllb-600m", "milmmt-46-1b", "seamless-medium", "ollama-qwen2.5:7b"):
        assert MODEL_VRAM_MIB[key] > 0
```

- [ ] **Step 9.2: Run to verify failure**

Run: `pytest tests/unit/test_cuda_check.py tests/unit/test_config.py -v -k vram`
Expected: FAIL (missing name imports)

- [ ] **Step 9.3: Implement**

In `config.py`, below `CT2_MODEL_PATHS`:

```python
# Measured VRAM peaks (MiB) on RTX 5050 8 GB — source: monitor/observations.md
# 2026-04-10 run. Used for pre-flight checks before loading a second model
# (e.g. the Ollama polish pass). Keys are lower-case model names.
MODEL_VRAM_MIB: dict[str, int] = {
    "nllb-600m":         2400,
    "milmmt-46-1b":      3400,
    "seamless-medium":   4100,
    "indictrans2-1b":    3100,
    "ollama-qwen2.5:7b": 4800,
    "ollama-gemma3:12b": 4700,
}
```

In `cuda_check.py`:

```python
def ensure_vram_available(required_mib: int, context: str) -> None:
    """Raise RuntimeError if free VRAM is below required_mib.

    Pre-flight for loading a second model (e.g. Ollama polish after
    translation). Failing here with a clear message beats a CUDA OOM
    mid-run. GPU-only rule: we never fall back to CPU.
    """
    free = get_free_vram_mib()
    if free < required_mib:
        raise RuntimeError(
            f"{context}: needs ~{required_mib} MiB VRAM but only {free} MiB free. "
            "Unload the translation model first, or use a smaller polish model. "
            "Safe combination on 8 GB: nllb-600M + Ollama."
        )
```

- [ ] **Step 9.4: Run tests + commit**

Run: `make test` → all PASS.

Update `docs/HARDWARE.md`: VRAM budget table now lives in `config.MODEL_VRAM_MIB` and is enforced by `ensure_vram_available()`.

```bash
git add src/bn_en_translate/config.py src/bn_en_translate/utils/cuda_check.py tests/unit/test_cuda_check.py tests/unit/test_config.py docs/HARDWARE.md
git commit -m "feat(vram): encode measured VRAM budget table and pre-flight check"
```

### Task 10: Implement the polish pass in the CLI flow

**Files:**
- Modify: `src/bn_en_translate/pipeline/pipeline.py` (add `polish_with_ollama`)
- Modify: `src/bn_en_translate/cli.py`
- Test: `tests/unit/test_polish_pass.py` (new)

- [ ] **Step 10.1: Write the failing tests**

Create `tests/unit/test_polish_pass.py`:

```python
"""Tests for the Ollama polish pass orchestration (mocked Ollama)."""

from __future__ import annotations

import pytest

from bn_en_translate.config import PipelineConfig
from bn_en_translate.pipeline.pipeline import polish_with_ollama


class FakeOllama:
    """Stands in for OllamaTranslator: records lifecycle and inputs."""

    def __init__(self) -> None:
        self.loaded = False
        self.polished: list[str] = []

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        assert self.loaded, "polish called before load()"
        self.polished.extend(texts)
        return [f"POLISHED: {t}" for t in texts]


def test_polish_preserves_paragraph_count(monkeypatch) -> None:
    fake = FakeOllama()
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline._make_ollama", lambda config: fake
    )
    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline.ensure_vram_available",
        lambda required_mib, context: None,
    )
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    result = polish_with_ollama(text, PipelineConfig())
    assert result.count("\n\n") == 2
    assert result.startswith("POLISHED: ")
    assert not fake.loaded  # unloaded afterwards


def test_polish_raises_on_low_vram(monkeypatch) -> None:
    def _raise(required_mib: int, context: str) -> None:
        raise RuntimeError(f"{context}: needs {required_mib} MiB")

    monkeypatch.setattr(
        "bn_en_translate.pipeline.pipeline.ensure_vram_available", _raise
    )
    with pytest.raises(RuntimeError, match="polish"):
        polish_with_ollama("text", PipelineConfig())
```

- [ ] **Step 10.2: Run to verify failure**

Run: `pytest tests/unit/test_polish_pass.py -v`
Expected: FAIL with `ImportError: cannot import name 'polish_with_ollama'`

- [ ] **Step 10.3: Implement in `pipeline.py`**

Add at module level (below the `TranslationPipeline` class):

```python
def _make_ollama(config: PipelineConfig) -> TranslatorBase:
    """Seam for tests — constructs the real OllamaTranslator."""
    from bn_en_translate.models.ollama_translator import OllamaTranslator

    return OllamaTranslator(config)


def polish_with_ollama(english_text: str, config: PipelineConfig) -> str:
    """Run the Ollama literary polish pass over translated English text.

    Per-paragraph so paragraph structure survives (key invariant #3).
    Caller must have unloaded the translation model first — this checks the
    Ollama model's VRAM requirement and raises rather than OOM-ing.
    """
    from bn_en_translate.config import MODEL_VRAM_MIB
    from bn_en_translate.utils.text_utils import split_paragraphs

    required = MODEL_VRAM_MIB.get(
        f"ollama-{config.ollama_model.split('-')[0].split('_')[0]}", 4800
    )
    ensure_vram_available(required, context="Ollama polish pass")

    paragraphs = split_paragraphs(english_text)
    ollama = _make_ollama(config)
    ollama.load()
    try:
        polished = ollama.translate(paragraphs, src_lang="eng_Latn", tgt_lang="eng_Latn")
    finally:
        ollama.unload()
    return "\n\n".join(polished)
```

Add the import at the top of `pipeline.py`:

```python
from bn_en_translate.utils.cuda_check import ensure_vram_available
```

Note: the VRAM key lookup is deliberately forgiving — unknown Ollama tags fall back to 4800 MiB (the largest measured Ollama footprint). Simplify the key derivation if it fights you: `MODEL_VRAM_MIB.get(f"ollama-{config.ollama_model}", 4800)` with table keys matching the two documented tags is also acceptable — pick one and keep the test green.

- [ ] **Step 10.4: Wire into `cli.py`**

Replace the `with translator:` block in `main()`:

```python
    translator = get_translator(config)
    pipeline = TranslationPipeline(translator, config)

    click.echo(f"Loading model '{model}'...")
    with translator:
        click.echo(f"Translating '{input_path}'...")
        result = pipeline.translate_file(input_path, output_path)
    # translator is now unloaded — VRAM is free for the polish model

    if ollama_polish:
        from bn_en_translate.pipeline.pipeline import polish_with_ollama
        from bn_en_translate.utils.file_io import write_translation

        click.echo(f"Polishing with Ollama ({ollama_model})...")
        polished = polish_with_ollama(result, config)
        write_translation(polished, output_path)

    click.echo(f"Done. Output written to: {output_path}")
```

- [ ] **Step 10.5: Run tests**

Run: `pytest tests/unit/test_polish_pass.py -v` then `make test`
Expected: all PASS

- [ ] **Step 10.6: End-to-end check (GPU + Ollama running; skip gracefully if Ollama absent)**

```bash
ollama list >/dev/null 2>&1 && bn-translate --input tests/fixtures/sample_short.bn.txt --output /tmp/claude-1000/-home-sbisw-github-translate/bd1797e0-21ca-4b99-b212-4e9af9db3b03/scratchpad/polish_test.en.txt --model nllb-600M --ollama-polish || echo "Ollama not running — skipped e2e, unit tests cover the logic"
```

- [ ] **Step 10.7: Update docs + commit**

- `README.md`: `--ollama-polish` now actually runs (was silently ignored before — say so in a "Fixed" note), safe VRAM combinations.
- `docs/ARCHITECTURE.md`: polish-pass sequence diagram/description (translate → unload → pre-flight → polish).

```bash
git add src/bn_en_translate/pipeline/pipeline.py src/bn_en_translate/cli.py tests/unit/test_polish_pass.py README.md docs/ARCHITECTURE.md
git commit -m "feat(cli): wire --ollama-polish pass with VRAM pre-flight (was a dead flag)"
git push
```

---

## Phase 4 — Model roster completion + new-model evaluation

Run each model task independently; benchmark on the same 90-sentence FLORES set; log to RunDatabase; invoke the monitor agent after each benchmark. Keep a model only if it beats an incumbent on BLEU or BLEU-per-GB. After each kept/rejected model, update `docs/MODELS.md`, `README.md`, `CLAUDE.md` benchmark table, and paper tables (`paper_writer` agent).

### Task 11: indicTrans2-1B — BLOCKED ON USER

- [ ] **Step 11.1 (USER ACTION):** Accept terms at https://huggingface.co/ai4bharat/indictrans2-indic-en-1B while logged in, then run in-session: `! huggingface-cli login`
- [ ] **Step 11.2:** `python scripts/download_models.py --model indicTrans2-1B` (downloads + CT2-converts; verify `models/indicTrans2-1B-ct2/sentencepiece.bpe.model` exists afterwards — known gotcha: SPM must be copied into the CT2 dir)
- [ ] **Step 11.3:** `python scripts/benchmark.py --models indicTrans2-1B --sentences 5` (smoke), then `--sentences 90` (full)
- [ ] **Step 11.4:** Update `docs/MODELS.md`, `CLAUDE.md` table, `README.md`; dispatch monitor agent; commit `feat(models): benchmark indicTrans2-1B CT2 float16 (BLEU <x>)`.

### Task 12: MADLAD-3B clean re-download — one attempt, guard decides

- [ ] **Step 12.1:** Check disk first (`df -h ~`; the download is ~11 GB). If <20 GB free, skip and record why in `docs/MODELS.md`.
- [ ] **Step 12.2:** `rm -rf models/madlad-3b-hf && python scripts/download_models.py --model madlad-3b`
- [ ] **Step 12.3:** Load test: `python -c "from bn_en_translate.config import ModelConfig; from bn_en_translate.models.madlad import MADLADTranslator; t = MADLADTranslator(); t.load(); print('guard passed'); t.unload()"`
  - Guard raises → checkpoint corrupt at source; MADLAD stays EXCLUDED; document in `docs/MODELS.md`; done.
  - Guard passes → `python scripts/benchmark.py --models madlad-3b --sentences 5`. If output is coherent but ch/s < 5 (CPU offload), MADLAD stays EXCLUDED on VRAM grounds; document. Only if coherent AND usable speed: full 90-sentence run and table updates.
- [ ] **Step 12.4:** Commit whichever outcome: `docs(models): MADLAD-3B re-download verdict — <kept|excluded: reason>`.

### Task 13: New model — Hunyuan-MT-7B via Ollama (Q4_K_M GGUF, ~5 GB)

WMT25 winner in 30/31 language pairs; Bengali confirmed supported. bf16 (~16 GB) and fp8 (~8 GB) don't fit the RTX 5050 — the Q4_K_M GGUF through the existing Ollama plumbing does.

**Files:**
- Modify: `src/bn_en_translate/models/ollama_translator.py` (prompt override support)
- Modify: `src/bn_en_translate/models/factory.py` (register `hunyuan-mt-7b`)
- Modify: `src/bn_en_translate/config.py` (`MODEL_VRAM_MIB` entry after measuring)
- Test: `tests/unit/test_hunyuan_ollama.py` (new)

- [ ] **Step 13.1:** Pull the model (network, ~5 GB): `ollama pull demonbyron/HY-MT1.5-7B:Q4_K_M` — if that tag is unavailable, search `ollama search hunyuan` and use the best Q4 tag; record the exact tag used.
- [ ] **Step 13.2: Write the failing test**

Create `tests/unit/test_hunyuan_ollama.py`:

```python
"""Hunyuan-MT via Ollama: factory routing and prompt format."""

from __future__ import annotations

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.factory import get_translator
from bn_en_translate.models.ollama_translator import OllamaTranslator


def test_factory_routes_hunyuan_to_ollama_translator() -> None:
    config = PipelineConfig(model=ModelConfig(model_name="hunyuan-mt-7b"))
    translator = get_translator(config)
    assert isinstance(translator, OllamaTranslator)
    # Hunyuan-MT prompt format (model card): plain instruction, no chat template
    assert "Translate the following segment into English" in translator.prompt_template
```

- [ ] **Step 13.3:** Run: `pytest tests/unit/test_hunyuan_ollama.py -v` — expect FAIL (unknown model name / no `prompt_template` attribute).
- [ ] **Step 13.4: Implement.** In `ollama_translator.py`, make the prompt a constructor parameter:

```python
    def __init__(
        self,
        config: PipelineConfig | None = None,
        prompt_template: str | None = None,
        model_tag: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PipelineConfig()
        self.prompt_template = prompt_template or TRANSLATION_PROMPT
        self.model_tag = model_tag or self.config.ollama_model
        self._client: httpx.Client | None = None
```

Use `self.prompt_template.format(text=text)` and `self.model_tag` in `_translate_one`. Add the Hunyuan prompt at module level (exact format from the Hunyuan-MT model card — verify against https://huggingface.co/tencent/Hunyuan-MT-7B at implementation time):

```python
HUNYUAN_MT_PROMPT = """\
Translate the following segment into English, without additional explanation.

{text}"""
```

In `factory.py`:

```python
@register_model("hunyuan-mt-7b")
@register_model("hunyuan")
def _make_hunyuan(config: PipelineConfig) -> TranslatorBase:
    from bn_en_translate.models.ollama_translator import (
        HUNYUAN_MT_PROMPT,
        OllamaTranslator,
    )
    return OllamaTranslator(
        config,
        prompt_template=HUNYUAN_MT_PROMPT,
        model_tag="demonbyron/HY-MT1.5-7B:Q4_K_M",  # update to the tag actually pulled
    )
```

- [ ] **Step 13.5:** Run: `pytest tests/unit/test_hunyuan_ollama.py -v` then `make test` — all PASS.
- [ ] **Step 13.6:** Smoke then full benchmark (Ollama serving): `python scripts/benchmark.py --models hunyuan-mt-7b --sentences 5`, inspect the preview line for coherent English; then `--sentences 90`.
- [ ] **Step 13.7:** Verdict vs incumbents (seamless 67.0 BLEU / 4.0 GB): keep if BLEU > 67.0 or BLEU-per-GB beats milmmt. Add measured VRAM to `MODEL_VRAM_MIB` (nvidia-smi while running). Update `docs/MODELS.md`, `CLAUDE.md`, `README.md`; monitor agent; commit `feat(models): Hunyuan-MT-7B Q4 via Ollama (BLEU <x>)` or `docs(models): Hunyuan-MT-7B evaluated and rejected (<reason>)`.

### Task 14: New model — NiuTrans LMT-60-1.7B (native bf16, ~3.4 GB)

60 languages including Bengali; the 4B sibling beats Aya-101-13B. 1.7B fits natively — no quantization risk. Qwen3-based causal LM, so the translator mirrors `milmmt.py`.

**Files:**
- Create: `src/bn_en_translate/models/lmt60.py` (copy the structure of `milmmt.py` exactly: left-padding, bf16, prompt slice at `input_len`, `resolve_attn_implementation`, `resolve_device`, `free_cuda_memory`)
- Modify: `src/bn_en_translate/models/factory.py`, `scripts/download_models.py` (download entry), `src/bn_en_translate/config.py` (VRAM entry after measuring)
- Test: `tests/unit/test_lmt60.py` (mirror `tests/unit/test_milmmt.py`'s structure: prompt format, factory routing, padding side)

- [ ] **Step 14.1:** Check the model card prompt format FIRST: https://huggingface.co/NiuTrans/LMT-60-1.7B — the prompt template below must be replaced with the card's exact format before writing the test (LLM MT models are prompt-brittle; MiLMMT lesson).
- [ ] **Step 14.2:** TDD the translator: failing tests (prompt build, factory routing, left-padding assertion) → implement `LMT60Translator` cloned from `MiLMMTTranslator` with `HF_MODEL_ID = "NiuTrans/LMT-60-1.7B"`, `_LOCAL_PATH = str(REPO_ROOT / "models/lmt-60-1.7B-hf")`, the verified prompt template, and Gemma-isms removed if the card differs → `make test` green.
- [ ] **Step 14.3:** Download (`python scripts/download_models.py --model lmt-60-1.7B` after adding the entry), smoke (5 sentences, inspect preview), full 90-sentence benchmark.
- [ ] **Step 14.4:** Same verdict/docs/commit protocol as Step 13.7.

### Task 15: New model — MiLMMT-46-4B, 4-bit (OPTIONAL — attempt only if Tasks 13-14 both reject)

bf16 is 8.6 GB (doesn't fit); 4-bit via bitsandbytes ~3 GB. Risk: bitsandbytes kernels on sm_120 are unproven on this machine — timebox to one attempt.

- [ ] **Step 15.1:** Probe bitsandbytes on sm_120 before anything else: `python -c "import torch, bitsandbytes; print(bitsandbytes.__version__)"` and a 10-line 4-bit load of the existing MiLMMT-46-1B checkpoint. Any kernel error → abandon task, record in `docs/MODELS.md` ("4B needs 4-bit; bitsandbytes unverified on sm_120"), done.
- [ ] **Step 15.2:** If the probe passes: extend `milmmt.py` with an optional `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` path gated on a new `ModelConfig.load_in_4bit: bool = False`, register `milmmt-46-4b`, download, smoke, benchmark, verdict — same protocol as 13.7.

### Task 16: Close out

- [ ] **Step 16.1:** Final full benchmark table (all kept models, 90 sentences) → update `CLAUDE.md` state block + benchmark table, `docs/MODELS.md`, `README.md`, `docs/perf_baseline_2026-07-07.md` final summary.
- [ ] **Step 16.2:** Dispatch paper_writer agent to refresh paper tables with new measured results.
- [ ] **Step 16.3:** `make test && make lint && make typecheck && make papers` — all green.
- [ ] **Step 16.4:** Commit, push, then invoke superpowers:finishing-a-development-branch (merge/PR decision).

---

## Blocked-on-user summary (surface these when delivering the plan)

1. **Task 11** cannot start until the user accepts the gated-repo terms and runs `! huggingface-cli login`.
2. **Task 13** needs Ollama running (`ollama serve`) and ~5 GB disk for the pull.
3. Everything else (Tasks 0-10, 12, 14-16) is executable without user input.
