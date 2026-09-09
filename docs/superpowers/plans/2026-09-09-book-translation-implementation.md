# Book Translation System — Detailed Implementation Plan

**Date:** 2026-09-09

**Implementer:** GPT-Luna

**Reviewer:** GPT-Terra

**Specification:** `docs/superpowers/specs/2026-09-09-book-translation-design.md`

## Operating Instructions

Implement in the task order below. Each task is independently reviewable and must
leave the repository green. Use test-first development for state transitions,
segmentation, QA rules, and import/export behavior. Do not rewrite or delete the
user's existing `.claude/settings.local.json` or untracked `inputs/` files.

At the start of every task:

```bash
cd /home/sbisw/github/translate
source .venv/bin/activate
git status --short
```

At every task boundary run:

```bash
make test
make lint
make typecheck
git diff --check
```

Run model/GPU tests only at the named gates. Never make test success depend on a
network download. Do not commit model weights, copyrighted book text, generated
working projects, SQLite databases, or review exports.

GPT-Terra reviews each phase before GPT-Luna starts the next. A review result is one
of `APPROVED`, `CHANGES_REQUESTED`, or `BLOCKED`, with findings categorized as
blocker/high/medium/low and linked to file/line evidence.

## Baseline Recorded During Planning

- `make test` equivalent (`pytest -m 'not slow and not e2e'`) passes.
- `mypy src` passes in strict mode.
- An unfiltered `pytest` run has eight real-model failures: four IndicTrans2 tokenizer
  incompatibilities and four NLLB/Transformers pipeline incompatibilities; Ollama is
  skipped when not running.
- The CLI's `batch_size` argument is not wired into `PipelineConfig.chunk`.
- Ruff across `src tests scripts` has pre-existing script findings; the Makefile lint
  target currently checks only `src/ tests/`.
- The worktree already has unrelated user changes. Preserve them.

## Phase 0 — Baseline Reliability and Contract Corrections

### Task 0.1: Freeze a book-quality baseline

**Create:**

- `docs/book_quality_baseline_2026-09-09.md`
- `tests/fixtures/book/consistency_source.bn.txt`
- `tests/fixtures/book/consistency_reference.en.txt`
- `tests/fixtures/book/expected_terms.json`

**Work:**

- Select redistributable synthetic/public-domain snippets covering recurring names,
  honorifics, dialogue, pronouns, a number/date, one cultural term, a long sentence,
  scene breaks, and two chapters.
- Record current outputs from `milmmt-46-4b` and one lower-VRAM model when available.
- Score chrF/BLEU only where references exist. Manually annotate consistency and
  coverage failures using stable categories from the spec.
- Record exact command, model revision, package versions, hardware, generation
  parameters, and whether output came from cache.

**Tests:** fixture licensing/provenance metadata exists; source/reference block counts
match; expected term occurrences point to valid source blocks.

**Acceptance:** baseline can be reproduced without the private/untracked `inputs/`
directory. If no GPU is available, commit fixtures and protocol with model results
explicitly marked pending.

### Task 0.2: Fix current CLI configuration drift

**Modify:**

- `src/bn_en_translate/cli.py`
- `src/bn_en_translate/models/factory.py`
- `tests/unit/test_cli_ollama_model_flag.py` (rename to `test_cli.py` if practical)
- `README.md`

**Work:**

- Construct `ChunkConfig(batch_size=batch_size)` and pass it into `PipelineConfig`.
- Validate positive batch size through Click and retain dataclass validation.
- Expose a public `supported_model_names()` from the registry and use a Click choice
  or generated help text so CLI help cannot drift from the factory.
- Update factory docstrings for all registered production and alias names.

**Tests:** invoke CLI with `--batch-size 3`, capture config, and assert both
`config.chunk.batch_size` and model config values. Assert every registry model is
accepted or intentionally hidden with a documented reason.

**Acceptance:** no ignored CLI flags; existing command/output behavior is unchanged.

### Task 0.3: Make test tiers truthful and repair backend compatibility

**Modify:**

- `pyproject.toml`
- `Makefile`
- `src/bn_en_translate/models/nllb.py`
- `src/bn_en_translate/models/indicTrans2.py`
- relevant integration/e2e tests and compatibility documentation

**Work:**

- Make default bare `pytest` exclude `slow`, `e2e`, and `gpu`; provide explicit
  Makefile targets for each tier and a deliberate `test-all` target.
- Pin a tested Transformers range or adapt NLLB away from the removed generic
  `pipeline("translation")` API.
- Either pin the IndicTrans2-compatible Transformers version or isolate its remote
  tokenizer behind a tested compatibility adapter. Prefer supported public APIs.
- Ensure e2e tests resolve models through `get_translator()` unless testing a backend
  implementation specifically.
- Add a startup compatibility report with actionable versions, never silent fallback.

**Acceptance:** fast tests are hermetic; model tests fail only for missing declared
prerequisites; one available local backend completes a smoke translation.

**GPT-Terra Phase 0 gate:** verify no broad dependency downgrade silently breaks other
backends, CLI compatibility is retained, and test markers reflect actual resource
requirements.

## Phase 1 — Book Domain Model and Project Store

### Task 1.1: Add immutable book schema

**Create:**

- `src/bn_en_translate/book/__init__.py`
- `src/bn_en_translate/book/schema.py`
- `src/bn_en_translate/book/serialization.py`
- `tests/unit/book/test_schema.py`
- `tests/unit/book/test_serialization.py`

**Implement:**

- `BlockKind`, `InlineRun`, `BookMetadata`, `BookBlock`, `Chapter`, and
  `BookDocument` from the spec.
- Deterministic ID helpers and SHA-256 source hashes over normalized canonical JSON.
- Re-import reconciliation that preserves project IDs using source-format locators,
  then hash/neighbor sequence alignment; ambiguous matches produce a dry-run report
  and abort without mutation.
- Schema validation: unique IDs, contiguous ordinals, known chapter references,
  source hash correctness, and all blocks represented exactly once.
- Versioned JSON/JSONL serialization with explicit UTF-8 and deterministic key order.
- A migration dispatch point that rejects unknown future versions.

**Tests:** round trip every block kind and Unicode edge case; duplicate/missing IDs;
empty blocks; deterministic serialization; tampered hash; unknown schema version.

**Acceptance:** serialize-deserialize equality and byte-stable repeat serialization.

### Task 1.2: Implement the persistent project store

**Create:**

- `src/bn_en_translate/book/project.py`
- `src/bn_en_translate/book/store.py`
- `src/bn_en_translate/book/migrations/0001_initial.sql`
- `tests/unit/book/test_project.py`
- `tests/unit/book/test_store.py`

**Implement:**

- `BookProject.create/open`, safe path validation, project layout, and config loading.
- SQLite schema from the spec with foreign keys, WAL mode, busy timeout, and explicit
  transactions.
- Repositories for units, attempts, runs, context assets, and QA findings.
- Attempt-dependency rows for source units, selected prior targets, assets, summaries,
  prompts, and config, plus graph-based direct/transitive stale propagation.
- Compare-and-set status transitions; leases with owner/expiry; append-only attempts;
  selected/approved candidate pointers.
- Content/config/context hashes and stale-marking rules.
- Atomic YAML/JSON writes and atomic final file replacement.
- Migration runner with backup and rollback behavior.

**State-transition tests:**

- legal path: pending -> running -> drafted -> revised -> approved;
- illegal transition rejection;
- expired lease recovery and active lease exclusion;
- crash after model response but before selection (attempt remains recoverable);
- resume cache hit when all hashes match;
- source change marks generated candidates stale but retains approvals for review;
- changed source/asset invalidates exactly its recorded direct and transitive
  dependents while an unrelated chapter remains cache-valid;
- in-place re-import preserves ID, insertion allocates a new ID without renumbering
  existing blocks, and ambiguous reconciliation is non-mutating;
- concurrent claim permits only one worker;
- transaction rollback under injected exception.

**Acceptance:** a fault-injection test kills/reopens a project at each persistence
boundary without losing committed work or selecting partial output.

### Task 1.3: Add TXT importer and exporter on the schema

**Create:**

- `src/bn_en_translate/book/formats/__init__.py`
- `src/bn_en_translate/book/formats/base.py`
- `src/bn_en_translate/book/formats/text.py`
- `tests/unit/book/formats/test_text.py`

**Implement:**

- Reader/writer protocols and extension registry.
- TXT import as one chapter by default, with optional heading/scene-break detection.
- Preserve blank and scene-break blocks and original newline style in attributes.
- Export selected target per block without benchmark headers or metadata in prose.

**Acceptance:** import/export/re-import preserves block kinds, order, and translatable
block count; all source characters are represented by a block or separator mapping.

**GPT-Terra Phase 1 gate:** inspect schema evolution, transaction boundaries, stable
ID determinism, stale semantics, and whether immutable/human-approved records can be
accidentally overwritten.

## Phase 2 — Structure-Preserving Document Formats

### Task 2.1: Add DOCX import

**Dependencies:** add `python-docx` under a `book` optional dependency.

**Create/modify:**

- `src/bn_en_translate/book/formats/docx.py`
- `tests/unit/book/formats/test_docx.py`
- generated programmatic DOCX fixtures (built in tests, not opaque binaries)

**Implement:**

- Map Word heading styles to chapters/headings; normal paragraphs, quotes, and list
  items to block kinds; asterism-only paragraphs to scene breaks.
- Preserve inline bold/italic/underline and hyperlinks as `InlineRun` data.
- Preserve paragraph style name, list metadata, and document core properties.
- Extract footnotes/endnotes only if `python-docx` exposes them reliably; otherwise
  emit a blocker warning and preserve references. Do not silently drop them.
- Reject password-protected/corrupt files cleanly.

**Tests:** multi-chapter file, mixed runs, hyperlink, list, quote, blank paragraph,
scene break, tables warning, note warning, Bengali Unicode, deterministic IDs.

### Task 2.2: Add DOCX export and semantic round-trip validation

**Implement:**

- Rebuild DOCX from the schema and selected target text while preserving semantic
  styles and inline emphasis spans where alignment permits.
- For a translated block whose inline boundaries cannot be reliably projected, retain
  block-level style and record a non-blocking `inline_style_projection` QA finding
  rather than guessing offsets.
- Embed source block IDs in custom metadata or a sidecar manifest, not visible prose.
- Re-import the produced DOCX and compare chapter/block kind/order coverage.

**Acceptance:** no missing/reordered blocks; supported style semantics survive; export
is atomic; unsupported structures are reported.

### Task 2.3: Add EPUB support after DOCX is stable

**Dependencies:** add `ebooklib` and a bounded HTML parser dependency under `epub`.

**Create:**

- `src/bn_en_translate/book/formats/epub.py`
- `tests/unit/book/formats/test_epub.py`

**Implement:** parse spine order and semantic XHTML; map headings, paragraphs, quotes,
lists, emphasis, links, and notes; preserve package metadata and non-text assets; block
active scripts/external fetches; export valid EPUB and run `epubcheck` when installed.

**Acceptance:** spine/chapter/block order round trip, links/notes resolve, images are
preserved byte-for-byte, and no source paragraph disappears.

**GPT-Terra Phase 2 gate:** use adversarial DOCX/EPUB fixtures and verify unsupported
features are visible findings rather than data loss.

## Phase 3 — Segmentation and Model Capability Layer

### Task 3.1: Introduce translator capabilities and token counting

**Create/modify:**

- `src/bn_en_translate/models/capabilities.py`
- `src/bn_en_translate/models/base.py`
- each production model adapter
- `tests/unit/test_model_capabilities.py`

**Implement:** immutable capabilities from the spec; exact tokenizer count methods for
HF/CT2 adapters; conservative fallback on `TranslatorBase`; model-specific input and
output limits; capability validation before a book stage starts.

**Acceptance:** every registered production model reports capabilities; tests verify
advertised limits against tokenizer behavior; unsupported context is not silently
discarded.

### Task 3.2: Replace character-estimate book chunking

**Create:**

- `src/bn_en_translate/book/segmenter.py`
- `tests/unit/book/test_segmenter.py`

**Implement:** `TranslationUnit`, `Segment`, and `ContextReference`; tokenizer-aware
budget accounting; sentence/clause/phrase/hard fallback splitting; continuation
metadata; separate context-only overlap; deterministic packing within chapters and
scenes; no cross-heading merge.

**Mandatory tests:** one sentence over limit; punctuation-free text; emoji/combining
marks; dialogue; source exactly at limit; context consumes reserve; no duplicated
overlap; every source span covered exactly once; reassembly property test.

**Acceptance:** for exact counters, no request exceeds its backend input limit. For
fallback counters, a documented safety margin is applied and provenance says
`estimated`.

### Task 3.3: Correct legacy chunker claims

**Modify:** legacy `chunker.py`, its tests, config, README, and architecture docs.

Either implement `min_chunk_sentences`/`overlap_sentences` correctly without duplicate
output, or deprecate/remove them. Add a real oversized-sentence test. Do not claim a
hard 400-token invariant while permitting oversized chunks.

**GPT-Terra Phase 3 gate:** independently property-test no loss/duplication and inspect
each adapter's claimed token limits.

## Phase 4 — Analysis, Glossary, Cast, and Context

### Task 4.1: Add human-editable project assets

**Create:**

- `src/bn_en_translate/book/assets.py`
- `src/bn_en_translate/book/config.py`
- JSON schemas or equivalent strict validators
- `tests/unit/book/test_assets.py`
- `tests/unit/book/test_config.py`

**Implement:** typed glossary, cast, style guide, summaries, lock state, scope, source,
confidence, provenance, YAML read/write, duplicate/alias conflict validation, and
merge semantics where locked human values always win.

**Acceptance:** generated analysis cannot mutate locked fields; invalid/ambiguous
aliases produce actionable errors; serialization preserves comments if the selected
YAML library supports it, otherwise preserve a generated/user section boundary.

### Task 4.2: Implement deterministic candidate extraction

**Create:**

- `src/bn_en_translate/book/analyze.py`
- `tests/unit/book/test_analyze.py`

**Implement:** Bengali proper-token/frequency candidates, repeated multiword terms,
numbers/dates, quoted aliases, honorifics, first occurrence, chapter distribution,
and concordance snippets. Keep this stage fast and model-independent.

**Acceptance:** recurring candidates are extracted with source IDs and no passage is
sent to a model.

### Task 4.3: Add optional model-assisted analysis

**Create:** versioned analysis prompt and strict response schema.

Run chapter-sized analysis windows, merge suggestions by normalized forms, preserve
uncertainty, and never overwrite locked assets. Summaries must be labeled generated
and may not be treated as more authoritative than source text.

**Tests:** malformed JSON, missing IDs, conflicting aliases, prompt injection in book
text, retry/retain behavior, and locked-value precedence with a fake model.

### Task 4.4: Build deterministic context packets

**Create:**

- `src/bn_en_translate/book/context.py`
- `tests/unit/book/test_context.py`

**Implement:** relevance scoring, priority order from the spec, strict token budget,
previous approved/revised blocks, next-source look-ahead, context hashes, and a debug
rendering that contains IDs but redacts full text unless explicitly requested.

**Acceptance:** same inputs produce byte-identical packet/hash; current source and
locked rules are never truncated; context-only units cannot appear as requested
translation IDs.

**GPT-Terra Phase 4 gate:** test locked-value precedence, prompt-injection boundaries,
token budgeting, and deterministic context selection.

## Phase 5 — Resumable Draft Translation

### Task 5.1: Implement stage runner and scheduler

**Create:**

- `src/bn_en_translate/book/runner.py`
- `src/bn_en_translate/book/stages/base.py`
- `src/bn_en_translate/book/stages/draft.py`
- `tests/unit/book/test_runner.py`
- `tests/integration/book/test_draft_resume.py`

**Implement:** project run creation, unit leasing, source-order iteration, bounded
batching of independent draft requests, per-window persistence, progress callbacks,
SIGINT-safe shutdown, resume matching hashes, `--continue-on-error`, and clear stage
summary.

Use an adapter over existing `TranslatorBase`; do not duplicate backend model loading.
Load one model once per stage and always unload in `finally`.

**Fault-injection tests:** failure before request, during request, after response,
during attempt write, and after attempt before status selection. Verify exactly which
calls repeat after resume.

### Task 5.2: Add dynamic OOM recovery

On recognized CUDA OOM, unload/reset state, halve batch size once, and retry. Persist
the adjustment in the run manifest. Never silently move to CPU. Non-OOM exceptions
must not trigger batch-size guessing.

**Acceptance:** mocked OOM proves one bounded retry and no lost attempts; terminal OOM
names model, request tokens, batch/window size, free/required VRAM, and resume command.

### Task 5.3: Add `bn-book init`, `inspect`, and `translate`

**Create:**

- `src/bn_en_translate/book_cli.py`
- console entry point `bn-book`
- `tests/unit/book/test_cli.py`

Commands must support `--help`, project validation, dry-run request/token estimates,
progress, resume by default, and explicit `--restart-stage` confirmation. Never infer
destructive project replacement from an existing directory.

**GPT-Terra Phase 5 gate:** interrupt/resume a fake 100-block book, verify no completed
calls repeat, inspect manifests for reproducibility, and check all model lifecycles.

## Phase 6 — Source-Grounded Literary Revision

### Task 6.1: Version prompts and response schemas

**Create:**

- `src/bn_en_translate/book/prompts/revise_v1.txt`
- `src/bn_en_translate/book/prompts/analyze_v1.txt`
- `src/bn_en_translate/book/responses.py`
- `tests/unit/book/test_responses.py`

Implement exact-ID JSON validation, duplicate/unexpected/missing ID rejection,
non-empty target validation, uncertainty notes, safe raw-response retention, prompt
hashes, and source-as-untrusted-data delimiters.

### Task 6.2: Implement revision stage

**Create:**

- `src/bn_en_translate/book/stages/revise.py`
- `tests/integration/book/test_revision.py`

Use adjacent windows within chapter/scene, Bengali source, MT draft, context packet,
locked assets, and previous selected targets. Load revision model only after the draft
model is unloaded. Persist candidates before selection. Never touch approved units.

Validation path:

1. HTTP/backend success;
2. strict response schema and exact ID set;
3. deterministic high-risk QA (empty, Bengali residue, numbers, glossary);
4. select valid revision, otherwise retain draft and mark `needs_review`;
5. bounded repair retry for schema failures only.

### Task 6.3: Deprecate, do not repurpose, legacy polish

Update `--ollama-polish` docs to explain that it is paragraph-local legacy behavior.
Optionally fix its prompt to an English editing prompt, but do not call it equivalent
to book revision. Emit a deprecation warning pointing book users to `bn-book revise`.

**GPU acceptance gate:** run draft then revision sequentially on at least one chapter;
peak VRAM stays within measured budget, process does not hold both models, and resume
works after interruption between stages.

**GPT-Terra Phase 6 gate:** adversarially test malformed output, invented IDs, missing
units, source prompt injection, number changes, empty revision, locked-name violation,
and approved-unit immutability.

## Phase 7 — QA and Human Review

### Task 7.1: Build QA framework and deterministic rules

**Create:**

- `src/bn_en_translate/book/qa/base.py`
- `src/bn_en_translate/book/qa/structure.py`
- `src/bn_en_translate/book/qa/coverage.py`
- `src/bn_en_translate/book/qa/consistency.py`
- `src/bn_en_translate/book/qa/typography.py`
- `src/bn_en_translate/book/qa/report.py`
- corresponding unit tests

Each rule returns typed findings and never mutates translations. Implement all
deterministic checks in FR-8. Calibrate length-ratio thresholds from development data
and keep them configurable. Deduplicate findings deterministically.

**Acceptance:** mutation tests inject every required defect and the expected rule,
severity, and block IDs appear; clean fixtures have no blocker findings.

### Task 7.2: Add review export/import

**Create:**

- `src/bn_en_translate/book/review.py`
- HTML template and static CSS package resources
- `tests/unit/book/test_review.py`

Export self-contained HTML plus JSONL with source/draft/revision/selected text,
findings, glossary context, status, ID, and source hash. Escape all content. Import
validates schema, ID, source hash, and edit status transactionally. Human approvals
create append-only attempts with `source=human`.

**Security tests:** source containing HTML/script is escaped; spreadsheet-formula-like
text is safe in any CSV option; stale corrections are quarantined; duplicate IDs abort
the transaction.

### Task 7.3: Add translation-memory suggestions

Store approved exact matches and optional normalized fuzzy candidates project-locally.
Never auto-apply fuzzy matches. Exact application must respect chapter scope, glossary
version, and source hash; every use is recorded.

**GPT-Terra Phase 7 gate:** verify QA is pure, blocker export policy works, review
artifacts are safe, and human corrections always outrank generated attempts.

## Phase 8 — Export, Observability, and End-to-End CLI

### Task 8.1: Implement export policy

Add `bn-book export` for TXT/DOCX and later EPUB. Run QA first; refuse unresolved
blockers unless `--allow-blockers` is explicit. Write atomically and produce a sidecar
manifest containing source hash, selected attempt IDs, model/prompt/config hashes,
unresolved finding counts, and override state.

### Task 8.2: Extend monitoring

**Modify:** `utils/run_db.py`, `utils/monitor.py`, stats scripts, docs.

Add book stage metadata without breaking existing rows. Prefer a versioned JSON
details column or normalized child table over a wide collection of nullable fields.
Track stage metrics listed in FR-11. Add migrations and tests for old database upgrade.

### Task 8.3: Implement orchestration command

`bn-book run` executes import if needed, analysis, draft, revision, QA, and export.
Human asset edits remain an explicit pause unless `--accept-suggestions` is passed.
Every stage is resumable and independently rerunnable. Print the exact resume command
on failure.

### Task 8.4: Documentation

Update:

- `README.md` with story vs book quick starts;
- `docs/ARCHITECTURE.md` with book components and state flow;
- `docs/DEVELOPMENT.md` with migrations, fixtures, and test tiers;
- `docs/MODELS.md` with draft/revision capability matrix;
- `docs/MONITORING.md` with stage metrics;
- `docs/INDEX.md` and `CLAUDE.md` state summary.

**GPT-Terra Phase 8 gate:** run an end-to-end fake-model DOCX workflow, inspect atomic
failure behavior and manifest reproducibility, then run a local-model chapter smoke.

## Phase 9 — Document-Level Evaluation and Release Gate

### Task 9.1: Add book benchmark harness

**Create:**

- `scripts/benchmark_book.py`
- `src/bn_en_translate/evaluation/book_metrics.py`
- `tests/unit/evaluation/test_book_metrics.py`
- `docs/BOOK_EVALUATION.md`

Evaluate direct draft, draft+legacy polish (for comparison), and draft+book revision.
Report all metrics from section 11 of the spec, per chapter and aggregate. Keep test
set provenance and split declarations machine-readable.

### Task 9.2: Run ablations

On the literary development set compare:

- direct best draft model;
- draft + glossary only;
- draft + context only;
- draft + glossary/context + bilingual revision;
- revision window/context sizes that fit the hardware.

Do not select based on BLEU alone. Record quality, latency, VRAM, edit effort, and QA
errors. Freeze chosen defaults only after results are reviewed.

### Task 9.3: Conduct blind human review

Use at least two Bengali/English reviewers where available. Randomize system labels,
use the MQM-style rubric, adjudicate critical accuracy disagreements, and report
inter-reviewer agreement. Do not expose model names until scoring is complete.

### Task 9.4: Release decision

The release candidate must satisfy the spec's release quality gate and all structural,
resume, review, and export acceptance criteria. If quality targets fail, ship the
infrastructure behind an experimental flag and record the failed criterion; do not
claim production book quality.

**GPT-Terra final gate:** independently reproduce fast tests, typecheck, lint, fake
end-to-end workflow, interrupted resume, DOCX round trip, QA mutation suite, one GPU
chapter, and benchmark report. Review manifests and corpus licenses before approval.

## Requirements Traceability

| Spec requirement | Primary implementation tasks | Required review gate |
|---|---|---|
| FR-1 structure-aware import | 1.1, 1.3, 2.1, 2.3 | Phases 1 and 2 |
| FR-2 project/provenance/resume | 1.2, 5.1, 8.1 | Phases 1, 5, and 8 |
| FR-3 analysis assets | 4.1, 4.2, 4.3 | Phase 4 |
| FR-4 tokenizer-aware segmentation | 3.1, 3.2, 3.3 | Phase 3 |
| FR-5 capability-aware drafting | 3.1, 5.1, 5.2 | Phases 3 and 5 |
| FR-6 context packets | 4.4, 6.2 | Phases 4 and 6 |
| FR-7 grounded literary revision | 6.1, 6.2, 6.3 | Phase 6 |
| FR-8 automated QA | 7.1, 8.1, 9.1 | Phases 7, 8, and 9 |
| FR-9 human review | 7.2, 7.3 | Phase 7 |
| FR-10 format-preserving export | 1.3, 2.2, 2.3, 8.1 | Phases 2 and 8 |
| FR-11 observability | 5.1, 8.2, 9.1 | Phases 5, 8, and 9 |

Every FR must have passing tests and reviewer evidence at its final listed gate. A
phase cannot be approved by documentation or mocks alone when its acceptance gate
explicitly requires a real document or local model run.

## Optional Phase 10 — Literary Fine-Tuning (Only After Evaluation)

Do not begin with fine-tuning. First collect licensed, document-aligned literary data
and human corrections, then prove prompting/context/revision limitations through the
Phase 9 ablations.

If justified:

- create train/dev/holdout splits by book/author, never random sentence split;
- prevent neighboring passages and repeated editions from crossing splits;
- retain preceding-context fields and structural tags;
- track dataset license and source provenance per record;
- train LoRA adapters against the best viable base model;
- compare against prompt/revision baseline using the blind book gate;
- reject adapters that gain BLEU but worsen names, omissions, or human MQM scores.

## Cross-Cutting Review Checklist for GPT-Terra

For every phase, answer all applicable questions:

### Correctness and data integrity

- Can any source block disappear, duplicate, reorder, or map to the wrong target?
- Can a crash leave a selected partial response or corrupt project/export?
- Are all state transitions and migrations transactional and tested?
- Do stable IDs and hashes behave predictably after source edits?
- Can generated output overwrite a human-approved correction?

### Translation quality

- Is current Bengali source present in every revision decision?
- Are glossary/cast locks authoritative and context size bounded?
- Are ambiguity and uncertainty preserved rather than invented away?
- Are checks document-level, not only paragraph-count or sentence BLEU checks?
- Is every quality claim backed by a representative book benchmark or human review?

### Model and resource behavior

- Are capabilities truthful for each backend?
- Are token budgets calculated with the active tokenizer where possible?
- Are draft and revision models loaded sequentially and always unloaded?
- Is OOM recovery bounded, observable, and GPU-only?
- Does resume avoid paid/slow duplicate model calls?

### Security and privacy

- Is source text treated as untrusted prompt data and escaped in reports?
- Are remote endpoints opt-in and conspicuous?
- Are logs, raw responses, and manifests appropriately scoped?
- Are corpus licenses and redistribution permissions documented?

### Maintainability

- Are format, storage, model, stage, and QA concerns behind narrow interfaces?
- Are schema/prompt/config versions explicit?
- Do tests use fakes and generated fixtures rather than GPU/network by default?
- Are CLI help, registry capabilities, docs, and shipped behavior generated from or
  checked against one source of truth?

## Final Definition of Done

- [ ] All specification acceptance criteria pass.
- [ ] TXT and DOCX book workflows are production-ready; EPUB milestone status is
      explicit and truthful.
- [ ] A 100+ chapter/block fake run survives interruption and resumes without
      retranslating completed units.
- [ ] QA detects every required mutation and clean fixtures have no blocker findings.
- [ ] Human correction import is stale-safe, transactional, and authoritative.
- [ ] Export is atomic, structure-preserving, and blocked by unresolved blockers.
- [ ] The 8 GB GPU chapter workflow loads models sequentially and records peak VRAM.
- [ ] Document-level benchmark and blind review meet the release gate, or the feature
      is clearly marked experimental.
- [ ] `make test`, `make lint`, `make typecheck`, and `git diff --check` pass.
- [ ] Documentation and CLI help match the implemented capability set.
- [ ] GPT-Terra records final `APPROVED` with no unresolved blocker/high findings.
