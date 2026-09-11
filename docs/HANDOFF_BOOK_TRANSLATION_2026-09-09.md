# Book Translation Implementation Handoff

**Date:** 2026-09-09  
**Branch:** `main`  
**Implementation model:** GPT-Luna  
**Review model:** GPT-Terra

## Current progress

The implementation plan is in `docs/superpowers/plans/2026-09-09-book-translation-implementation.md`,
with the design contract in the corresponding `specs` document.

### Complete or substantially implemented

- Phase 0.1: synthetic, redistributable book-quality fixtures, expected terms, and a
  reproducible baseline protocol. GPU/model scores are explicitly pending because
  this environment has no GPU/NVML access.
- Phase 0.2: CLI batch-size wiring (`ChunkConfig`), positive batch-size validation,
  and registry-driven model choices/help.
- Phase 0.3: default hermetic pytest filtering, explicit test-tier Make targets,
  NLLB explicit seq2seq APIs, IndicTrans2 compatibility handling, and startup
  compatibility reporting.
- Initial Phase 1 book package: schema, serialization, project/store scaffolding,
  TXT format support, `bn-book` entry point, and unit tests.
- Phase 1 narrow remediation: approval-pointer protection, graph-based stale
  invalidation with approved-output revalidation, CRLF/separator-preserving TXT
  round trips, and packaged SQL migration resources with schema markers and backup
  support. GPT-Terra approved this narrow remediation.
- Remaining Phase 1 acceptance coverage: deep immutable attributes, stable-ID
  re-import edits/insertions/ambiguity fail-closed behavior, and JSON/JSONL migration
  dispatch tests.
- Phase 2.1 initial DOCX reader: headings/chapters, paragraphs, quotes, lists,
  scene breaks, inline emphasis/links, metadata retention, and visible unsupported
  feature warnings.
- Phase 2.2 DOCX exporter: atomic output, private source-ID metadata, semantic
  round-trip validation, safe unstyled fallback with `inline_style_projection`
  findings, and direct-numbering reconstruction. GPT-Terra reviewed the exporter
  and GPT-Luna addressed its findings.
- Phase 2.3 EPUB reader/exporter: safe ZIP/package validation, spine-order import,
  asset retention, active-content and external-fetch blocking, atomic export, and
  semantic round-trip checks. Follow-up hardening preserves contiguous chapter
  ordinals when non-XHTML spine resources are skipped and proves asset retention
  byte-for-byte. Delivered on `main` in `0d12370` and `e9bd55c`. The declared
  optional dependencies were installed for verification; `epubcheck` is not
  installed, so that optional executable gate was not run.

## Verification

GPT-Luna reported:

- `make test`: **314 passed, 10 deselected**
- `make lint`: passed
- `make typecheck`: passed
- `git diff --check`: passed
- Focused compatibility and fixture tests: passed
- Local CPU NLLB-CT2 smoke translation: passed
- Phase 1 book tests: **19 passed** (GPT-Terra gate)
- Current focused book suite: **33 passed**
- Current focused book suite after DOCX export: **37 passed**
- EPUB focused suite: **7 passed** with `ebooklib` and BeautifulSoup installed
- Full book suite: **44 passed**; full Python suite: **343 passed**
- EPUB lint and strict type check: passed; `epubcheck`: unavailable locally

## Review findings still open

GPT-Terra returned `CHANGES_REQUESTED` for Phase 0 and the initial Phase 1 work.
The highest-priority items are:

1. Add `pytest-timeout` (or remove unsupported timeout flags) so explicit slow/GPU/
   e2e targets execute correctly.
2. Align the Transformers pin in `requirements.txt` with `pyproject.toml`.
3. Add registry-wide CLI acceptance coverage and direct tests for the IndicTrans2
   tokenizer compatibility shim.
4. Add adversarial DOCX fixtures and complete the DOCX semantic round-trip gate.
5. Obtain GPT-Terra’s independent gate review of the verified EPUB implementation;
   run `epubcheck` as an additional release check where it is available.
6. Add the missing Phase 1 acceptance tests for crash boundaries, leases/CAS,
   migrations, approval preservation, JSONL, and separator/source coverage.

## Delegation state

GPT-Luna implemented the narrow Phase 1 remediation. GPT-Terra approved that scope
after review. GPT-Luna implemented DOCX export and GPT-Terra’s DOCX findings were
addressed. EPUB dependency-backed tests, lint, type checking, and the full suite now
pass; Terra’s independent review remains the next gate.

## Repository safety notes

- Do not add or publish the untracked `inputs/` directory; it is user-owned and may
  contain private material.
- Do not overwrite `.claude/settings.local.json`; its changes predate this handoff.
- Do not commit model weights, generated projects, SQLite databases, or review
  exports.
