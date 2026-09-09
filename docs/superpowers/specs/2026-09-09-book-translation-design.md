# Book Translation System — Product and Architecture Specification

**Date:** 2026-09-09

**Status:** Proposed for implementation

**Implementer:** GPT-Luna

**Reviewer:** GPT-Terra

**Companion plan:** `docs/superpowers/plans/2026-09-09-book-translation-implementation.md`

## 1. Goal

Evolve `bn-en-translate` from a stateless Bengali-to-English story translator into
a local-first, resumable system capable of producing consistent, reviewable draft
translations of full books while preserving document structure and fitting the
project's primary RTX 5050 8 GB environment.

The system must improve book-level quality, not merely sentence-level fluency. In
particular, it must preserve names, relationships, terminology, narrative voice,
dialogue style, cultural terms, and formatting across chapters; detect omissions
and uncontrolled additions; survive interruption; and expose aligned source and
target text for human review.

## 2. Current-State Assessment

### 2.1 Strengths to retain

- A clean `TranslatorBase` contract and registry support many local model backends.
- Normalization, chunking, translation, and reassembly are separated and unit tested.
- Translation is batched and order is restored after length sorting.
- Paragraph count is treated as an invariant in the current text workflow.
- Resource monitoring and benchmark history already exist.
- The fast test suite passes and strict mypy passes on the source package.
- The repository contains a real 90-paragraph Bengali story and outputs from three
  models, providing the seed for a document-level regression fixture.

### 2.2 Critical gaps for books

1. **No book/document domain model.** The pipeline accepts one string and reduces it
   to `ChunkResult(text, para_id, ...)`. Chapters, headings, scene breaks, quotations,
   lists, footnotes, italics, and source locations do not exist in the data model.
2. **No cross-chunk state.** `TranslationPipeline` is explicitly stateless. Names,
   honorifics, pronouns, terminology, and prior editorial decisions cannot influence
   later chunks or chapters.
3. **The polish pass is ungrounded.** It sends English paragraphs independently to
   a prompt that says to translate Bengali. It has no Bengali source, glossary,
   neighboring text, stable IDs, schema validation, or addition/omission safeguards.
4. **Chunk limits are estimates, not guarantees.** A fixed `4.5 chars/token` estimate
   is shared across unlike tokenizers. A single sentence longer than the budget is
   emitted intact. `overlap_sentences` and `min_chunk_sentences` are configured but
   not implemented.
5. **Only UTF-8 text files are supported.** The checked-in DOCX was manually extracted
   with an ad-hoc script, and the translation report says the production run also
   used an ad-hoc runner. No format round trip exists.
6. **No resumability or provenance.** A book is translated in one process, retained
   in memory, and written at the end. Output writes are not atomic. There are no unit
   statuses, source hashes, attempts, prompt/model versions, or restart semantics.
7. **Insufficient quality controls.** Existing checks cover paragraph count, basic
   UTF-8 output, Bengali residue, one proper noun, and sentence-level BLEU. They do
   not check coverage, numbers, quotations, names across chapters, glossary rules,
   repeated output, hallucinations, or narrative consistency.
8. **Evaluation does not represent books.** FLORES is useful for model screening but
   consists of independent sentences. The bundled Samanantar sample is generic and
   sentence aligned. The real story has no reference translation and only informal
   subjective observations.
9. **Operational and documentation drift exists.** The CLI accepts `--batch-size`
   but does not put it into `ChunkConfig`; CLI model help and factory documentation
   omit registered models. The full local test run currently has eight real-model
   failures caused by installed Transformers incompatibilities, although the fast
   suite and source type check pass.
10. **Stored outputs demonstrate book-level errors.** The same protagonist appears
    as `Man`, `Manab`, `human`, and `people`; `Geeta` becomes `Gita`; person and tense
    shift within nearby paragraphs; untranslated material such as `naib naib ch`
    survives. These are precisely the failures a document-level layer must prevent.

## 3. Scope

### 3.1 In scope

- Bengali source to English target, with language fields kept extensible.
- TXT and DOCX import/export in the first production release; EPUB after the core
  model is stable.
- Chapters, headings, body paragraphs, scene breaks, block quotes, list items,
  footnotes/endnotes where the source format exposes them, and inline emphasis.
- A persistent book project containing aligned source blocks, drafts, revisions,
  human edits, context assets, QA findings, and reproducibility metadata.
- Sequential model loading so draft MT and LLM revision fit an 8 GB GPU.
- Automatic glossary/cast suggestions plus human-editable locked decisions.
- Resumable draft and revision stages with bounded retries and atomic persistence.
- Source-grounded literary revision and book-level automated QA.
- Side-by-side review artifacts and import of approved human corrections.
- Document-level evaluation and regression gates.

### 3.2 Non-goals for the first production release

- OCR or page-layout reconstruction from scanned PDFs.
- Desktop GUI or multi-user web application.
- Fully autonomous publication-ready translation without human review.
- Training a foundation model from scratch.
- Exact preservation of Word pagination, tracked changes, macros, or arbitrary
  publisher-specific EPUB CSS.
- Cloud APIs. Provider interfaces may be extensible, but the default workflow stays
  local and no source text leaves the machine.

## 4. Product Workflow

The user-facing workflow is a persistent project, not a single output file:

```text
import source
    -> inspect structure
    -> analyze cast/terms/style
    -> edit and lock glossary/style guide
    -> create MT draft
    -> source-grounded literary revision
    -> automated QA
    -> human review/corrections
    -> final QA
    -> DOCX/EPUB/TXT export
```

The full workflow must also be available as one resumable command.

Proposed CLI:

```bash
bn-book init novel.bn.docx --project work/novel
bn-book inspect work/novel
bn-book analyze work/novel --model gemma3:12b
bn-book translate work/novel --model milmmt-46-4b
bn-book revise work/novel --model gemma3:12b
bn-book qa work/novel
bn-book review-export work/novel --format html
bn-book corrections-import work/novel reviewed.jsonl
bn-book export work/novel --format docx --output novel.en.docx

# Resumable orchestration of all non-human stages:
bn-book run novel.bn.docx --project work/novel \
  --draft-model milmmt-46-4b --revision-model gemma3:12b
```

The existing `bn-translate` command remains supported for short TXT inputs and uses
the existing `TranslationPipeline`. It must not silently switch behavior.

## 5. Functional Requirements

### FR-1: Structure-aware import

- Parse source files into a `BookDocument` without translating during import.
- Assign deterministic stable IDs based on document order and structural location,
  not random UUIDs.
- Preserve chapter order, block order, block kind, text, inline runs, list level,
  note references, and source metadata needed for export.
- Preserve intentional blank/scene-break blocks. Repeated blank lines must not be
  collapsed before structural parsing.
- Reject unsupported/corrupt input with a clear error before creating a partial
  project, or mark the import manifest failed atomically.

### FR-2: Persistent project and provenance

- Persist immutable imported source records and source hashes.
- Persist every generated candidate as an attempt; never overwrite an approved
  human translation.
- Persist the exact dependency set for each attempt: source blocks, glossary/cast/
  style asset versions, summaries, prior selected targets, prompt, and configuration.
  Cache validity is determined from these dependencies, not only a global run hash.
- Record model name/revision, backend, generation parameters, prompt template
  version/hash, context hash, source hash, start/end time, status, and error.
- Statuses: `pending`, `running`, `drafted`, `revised`, `needs_review`, `approved`,
  `failed`, and `stale`.
- Restart skips completed records whose source/config/context hashes still match.
- A stale `running` lease is recoverable after a configurable timeout.
- Writes use transactions and final exports use temp-file-plus-rename semantics.

### FR-3: Book analysis assets

- Generate a book synopsis, chapter summaries, character/cast entries, term entries,
  and style observations as suggestions.
- Every character entry supports Bengali forms, chosen English form, aliases,
  gender/pronoun notes when known, relationships, honorific policy, first occurrence,
  and `locked`/`suggested` state.
- Every glossary entry supports Bengali source forms, required/preferred/forbidden
  English forms, case sensitivity, notes, scope (book/chapter), and lock state.
- A human can edit these assets before translation. Machine analysis must never
  overwrite locked values.
- Analysis output must identify uncertainty rather than inventing facts.

### FR-4: Tokenizer-aware segmentation

- Segmentation operates on translatable `BookBlock` records and never loses IDs.
- The active backend exposes input token counting and context/output limits through
  capabilities; a conservative fallback estimator is allowed only when exact token
  counting is unavailable and must be labeled in provenance.
- Reserve tokens for prompt/context and expected output before selecting source text.
- Long paragraphs may split at Bengali sentence boundaries. Long sentences must
  split using ordered fallbacks (clause punctuation, phrase/whitespace, then a hard
  tokenizer-aware split) with explicit continuation metadata.
- Reassembly must be lossless with respect to unit IDs and source coverage.
- Context overlap is represented separately from text to translate, so overlapping
  source is never duplicated in output.

### FR-5: Capability-aware drafting

- Add immutable `TranslatorCapabilities`: exact token counting availability, maximum
  input/output tokens, batching, contextual prompting, glossary constraints, JSON
  output, deterministic seed, and supported language pairs.
- Classic seq2seq/CT2 models produce faithful drafts from source-only segments.
- Prompted translation models may receive a bounded `ContextPacket` but must return
  translations keyed by requested unit IDs.
- A backend must not be given context or constraints it cannot honor. Unsupported
  features fail validation or are handled by a later stage, never silently ignored.
- Draft scheduling may length-sort independent requests for throughput, but persisted
  results and all context construction follow source order.

### FR-6: Context packet

For each revision window, build a deterministic, size-bounded context packet from:

1. book title/author and translation policy;
2. locked style-guide rules;
3. relevant locked glossary and cast entries;
4. chapter summary and local scene summary;
5. previous approved/revised target blocks, normally two;
6. current Bengali source blocks and their MT drafts;
7. optionally the next source block for look-ahead, clearly marked as context-only.

Selection must be relevance-first and auditable. Locked glossary/cast rules outrank
summaries, and current source/draft text must never be truncated to fit optional
context. The packet and its component hashes are stored with each attempt.

### FR-7: Source-grounded literary revision

- Replace the English-only polish behavior for books with a bilingual revision pass.
- Revise windows of adjacent blocks within one chapter/scene, with stable unit IDs.
- Prompt goals: semantic fidelity first; consistent names/pronouns/terms; natural
  literary English; preserve ambiguity, tone, tense, dialogue, paragraph boundaries,
  and cultural policy; do not summarize, sanitize, explain, or add facts.
- The response uses strict JSON containing exactly the requested unit IDs and one
  target string per ID. Validate schema, ID set, non-empty output, and limits.
- Retry transient transport failures and malformed/schema-invalid replies with
  bounded exponential backoff. Do not retry deterministic semantic QA failures
  indefinitely; retain the attempt and mark `needs_review`.
- If revision fails, retain the MT draft. Never replace a valid draft with empty or
  invalid output.
- Human-approved units are immutable unless the user explicitly unlocks them.

### FR-8: Automated QA

Produce machine-readable JSON and human-readable HTML reports. Findings include
severity, rule, source unit IDs, evidence, and suggested action.

Required deterministic checks:

- structural ID/order/count agreement;
- empty or duplicate target blocks;
- Bengali-script residue outside allowlisted terms;
- source/target length-ratio outliers using corpus-calibrated bounds;
- number, date, currency, URL, and proper-token preservation;
- paired quote/bracket balance and dialogue marker preservation;
- glossary required/forbidden-form compliance;
- character-name and alias consistency;
- suspicious repeated phrases or outputs;
- missing sentence/continuation parts after segmentation and reassembly;
- export round-trip block coverage.

Optional model-based QA must be clearly separated from deterministic findings and
must cite source and target unit IDs. It cannot auto-approve or silently rewrite.

### FR-9: Human review loop

- Export side-by-side source, draft, revised target, QA findings, glossary notes, and
  status keyed by stable ID to HTML and JSONL.
- Import corrections only when IDs and source hashes match; reject or quarantine
  stale edits.
- Approved corrections feed a project-local translation memory and take precedence
  over generated candidates on future resume/re-export.
- Offer a report of repeated/similar source phrases where a correction may need to
  propagate, but require confirmation before propagation.

### FR-10: Format-preserving export

- TXT export preserves chapter and block order using configured separators.
- DOCX export preserves semantic styles, headings, lists, emphasis, block quotes,
  scene breaks, and notes to the extent represented by the internal schema.
- EPUB export preserves spine order, chapter boundaries, semantic HTML, emphasis,
  links, and notes; regenerate package metadata safely.
- Export refuses by default when blocker-level QA findings exist. `--allow-blockers`
  is an explicit override recorded in the manifest.
- Embed or accompany output with a reproducibility manifest; never put benchmark
  metric headers into the literary text itself.

### FR-11: Observability

- Add run types `book_analyze`, `book_draft`, `book_revise`, `book_qa`, and
  `book_export` to monitoring.
- Track blocks/characters/tokens processed, cache hit rate, retries, failed units,
  QA counts by severity, model load time, stage duration, throughput, and VRAM.
- Logs include project and unit IDs but not entire book passages by default.

## 6. Proposed Domain Model

Use frozen dataclasses for values at package boundaries and SQLite rows for mutable
workflow state. JSON serialization must be versioned.

```python
class BlockKind(StrEnum):
    TITLE = "title"
    CHAPTER_HEADING = "chapter_heading"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BLOCK_QUOTE = "block_quote"
    LIST_ITEM = "list_item"
    SCENE_BREAK = "scene_break"
    FOOTNOTE = "footnote"
    ENDNOTE = "endnote"
    BLANK = "blank"

@dataclass(frozen=True)
class InlineRun:
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    href: str | None = None

@dataclass(frozen=True)
class BookBlock:
    block_id: str
    chapter_id: str
    ordinal: int
    kind: BlockKind
    source_text: str
    source_hash: str
    runs: tuple[InlineRun, ...] = ()
    attrs: Mapping[str, JSONValue] = field(default_factory=dict)

@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    ordinal: int
    title: str | None
    block_ids: tuple[str, ...]

@dataclass(frozen=True)
class BookDocument:
    schema_version: int
    document_id: str
    metadata: BookMetadata
    chapters: tuple[Chapter, ...]
    blocks: tuple[BookBlock, ...]
```

Initial stable ID format: `c{chapter_ordinal:04d}-b{block_ordinal:06d}`. Importing
identical content with the same importer version produces identical IDs and hashes.
IDs are then project identities, not regenerated labels. Explicit re-import reconciles
blocks by a source-format locator when available, then by sequence alignment of hashes
and neighboring anchors. An in-place edit retains its block ID and changes its hash;
an insertion receives a new monotonic ID without renumbering later blocks. Ambiguous
reconciliation aborts with a report and does not mutate the project.

### 6.1 Project layout

```text
work/novel/
  project.yaml              # human-editable policy and model configuration
  source.jsonl              # immutable imported blocks
  structure.json            # metadata, chapters, format map, schema version
  glossary.yaml             # locked and suggested terms
  cast.yaml                 # names, aliases, pronouns, relationships
  style-guide.yaml          # voice, register, dialogue and cultural policies
  state.sqlite3             # attempts, selected translations, statuses, QA
  manifests/
    <run-id>.json
  reports/
    qa.json
    qa.html
    review.html
    review.jsonl
  exports/
```

Generated files are written atomically. `source.jsonl` and `structure.json` are only
replaced by an explicit re-import operation.

### 6.2 State tables

Minimum tables:

- `project_meta(key, value)` including schema and importer versions.
- `units(block_id, source_hash, status, selected_attempt_id, approved_attempt_id,
  lease_owner, lease_expires_at, updated_at)`.
- `attempts(id, block_id, stage, model, model_revision, prompt_version,
  source_hash, context_hash, config_hash, target_text, raw_response, status,
  error_type, error_message, started_at, finished_at)`.
- `attempt_dependencies(attempt_id, dependency_kind, dependency_key,
  dependency_hash)`; every cache hit revalidates all rows, and changed dependencies
  mark directly and transitively dependent generated attempts stale.
- `context_assets(kind, asset_key, value_json, locked, source, updated_at)`.
- `qa_findings(id, run_id, rule, severity, block_ids_json, evidence_json,
  status, created_at)`.
- `runs(run_id, stage, config_hash, status, started_at, finished_at, summary_json)`.

Foreign keys must be enabled. Migrations are monotonic, transactional, backed up,
and tested from every released schema version.

## 7. Service Interfaces

```python
class DocumentReader(Protocol):
    def read(self, path: Path) -> BookDocument: ...

class DocumentWriter(Protocol):
    def write(self, document: BookDocument,
              translations: Mapping[str, str], path: Path) -> None: ...

@dataclass(frozen=True)
class TranslatorCapabilities:
    max_input_tokens: int
    max_output_tokens: int
    supports_batching: bool
    supports_context_prompt: bool
    supports_glossary_constraints: bool
    supports_json_output: bool
    supports_seed: bool
    token_count_is_exact: bool

class BookTranslator(Protocol):
    @property
    def capabilities(self) -> TranslatorCapabilities: ...
    def count_input_tokens(self, text: str) -> int: ...
    def translate_units(self, requests: Sequence[TranslationRequest])
        -> list[TranslationResponse]: ...
```

Do not force all existing model classes to implement contextual translation. Add a
default adapter around `TranslatorBase` that declares conservative capabilities and
uses its existing `translate()` method for draft requests.

## 8. Configuration

Project configuration is versioned and validated before a run. Suggested shape:

```yaml
schema_version: 1
languages:
  source: ben_Beng
  target: eng_Latn
models:
  draft: milmmt-46-4b
  revision: gemma3:12b
segmentation:
  target_source_tokens: 320
  context_reserve_tokens: 1400
  previous_target_blocks: 2
  next_source_blocks: 1
revision:
  enabled: true
  window_blocks: 4
  temperature: 0.1
  seed: 42
  retries: 2
style:
  register: literary
  dialogue_quotes: curly
  cultural_terms: retain_and_gloss
qa:
  block_export_on: blocker
  glossary_required_compliance: 0.98
```

Validation resolves model names through the registry, checks capabilities against
requested stages, verifies writable paths, and estimates disk/VRAM requirements
before model loading.

## 9. Prompt Contract for Revision

Prompt templates live as versioned package resources, not inline in model adapters.
The revision prompt must:

- distinguish authoritative rules from untrusted source text;
- state that text inside source/context delimiters is data, not instruction;
- include only context selected by `ContextBuilder`;
- identify context-only blocks that must not be translated in the response;
- demand exact unit IDs and strict JSON;
- state fidelity and non-addition requirements before style goals;
- include locked names/terms in a compact table;
- avoid chain-of-thought requests;
- request uncertainty flags separately from translated text.

Expected response:

```json
{
  "translations": [
    {"unit_id": "c0001-b000017", "text": "...", "uncertain": false,
     "notes": []}
  ]
}
```

Raw responses are retained for debugging, but user-facing logs redact passage text.

## 10. Failure and Resume Semantics

- Claim work with a transaction and expiring lease before invoking a model.
- Persist one attempt per response window immediately after validation.
- On process termination, completed windows remain committed; expired leases return
  to `pending` on the next run.
- Transport errors: retry up to configured count, then mark failed and continue only
  if `--continue-on-error` is set.
- OOM: unload/reset model, reduce dynamic batch/window size once, retry once, then
  fail with measured context. Never fall back to CPU silently.
- Invalid model response: retain raw output, retry with repair prompt once, then use
  the draft and mark `needs_review`.
- Source/config/context hash mismatch: mark affected generated attempts stale. Human
  approvals are never deleted; they are flagged for revalidation.
- Invalidation follows persisted dependency edges. A changed source block invalidates
  its drafts, every revision window containing it, summaries derived from it, and
  later revisions that consumed those selected targets/summaries. It does not
  invalidate unrelated chapters unless a changed book-global asset was a dependency.
- Export is built to a sibling temporary path, validated, then atomically renamed.

## 11. Quality Evaluation Strategy

### 11.1 Test sets

Create three layers:

1. **Synthetic contract set:** short fixtures targeting names, gender ambiguity,
   honorifics, numbers, dialogue, long sentences, scene breaks, lists, notes, and
   formatting. References can be hand-authored.
2. **Literary development set:** public-domain or properly licensed Bengali excerpts,
   sampled across dialogue, narration, dialect, poetry/prose, cultural references,
   and long-distance entity recurrence. At least 5,000 source words and two chapters.
3. **Blind holdout set:** at least 2,000 words not used for prompt tuning, glossary
   tuning, model selection, or fine-tuning.

No copyrighted book text may be committed without clear redistribution rights.

### 11.2 Metrics

Report, do not collapse, the following:

- sentence/corpus metrics: chrF++, BLEU, and COMET when its model/dependencies are
  available;
- structural coverage: translated block and sentence continuation coverage;
- semantic-risk proxies: number/entity retention and length-ratio outliers;
- consistency: canonical name/term compliance and alias entropy across chapters;
- editing effort: human changes per 1,000 target words and accepted-without-edit rate;
- performance: source words/sec, total stage time, peak VRAM, cache hit rate;
- human MQM-style errors per 1,000 source words by accuracy, terminology, fluency,
  style, locale, and document consistency.

### 11.3 Release quality gate

On the blind holdout, compared with direct MiLMMT-46-4B drafting:

- zero structural coverage errors;
- at least 98% locked-glossary compliance;
- no increase in critical accuracy errors;
- at least 20% reduction in character/term consistency errors;
- at least 15% reduction in total major+minor human error points, or a documented
  no-regression decision from two blinded reviewers;
- resumability and export tests pass with no repeated completed model calls.

BLEU alone cannot approve a release.

## 12. Security, Privacy, and Reproducibility

- Default all processing to localhost. Validate that configured Ollama URLs are local
  unless the user explicitly enables remote providers.
- Treat book content as untrusted prompt data. Delimit it and neutralize instruction
  injection in source text.
- Do not log full source/target passages by default.
- Pin model revision IDs in run manifests where the backend exposes them.
- Hash prompt templates, config, source, context assets, and executable package
  version/commit.
- Store license/provenance metadata for evaluation and fine-tuning corpora.

## 13. Compatibility and Migration

- Existing imports and `TranslationPipeline.translate()` remain valid.
- `bn-translate` remains a thin story/TXT command. Fix its ignored `--batch-size`
  without changing its default output.
- Keep `--ollama-polish` for one deprecation cycle, but document it as legacy and do
  not reuse it for the book revision implementation.
- Existing TXT files import as one synthetic chapter with stable paragraph blocks.
- New dependencies for DOCX/EPUB/review are optional extras (`book`, `epub`) so the
  lightweight story translator remains installable.

## 14. Acceptance Criteria

The book system is complete when all of the following are demonstrated:

1. Import a multi-chapter DOCX, preserve supported structure, and assign stable IDs.
2. Interrupt a run after several windows, resume it, and prove completed windows were
   not sent to a model again.
3. Edit one source block and prove only that block and declared context dependents
   become stale.
4. Translate and revise with models loaded sequentially within the measured 8 GB
   budget.
5. Enforce locked names such as one protagonist spelling across distant chapters.
6. Detect injected omission, number change, Bengali residue, malformed quotes,
   duplicate translation, and forbidden glossary form.
7. Round-trip DOCX structure and export TXT; EPUB is required for the subsequent
   format milestone, not the initial core release.
8. Produce reproducible manifests, aligned review output, and QA JSON/HTML.
9. Pass fast tests, strict source mypy, project lint, book integration tests, and the
   document-level blind quality gate.

## 15. Decisions Requiring Explicit Change Control

GPT-Luna must not change these without recording an ADR and obtaining user approval:

- local-first processing and no silent cloud use;
- stable block IDs as the alignment key;
- immutable source plus append-only generation attempts;
- bilingual source-grounded revision rather than English-only polishing;
- human approval precedence;
- sequential model loading for the 8 GB target;
- deterministic QA blocking export on unresolved blocker findings;
- preservation of the existing `bn-translate` API and CLI behavior.
