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

## Verification

GPT-Luna reported:

- `make test`: **314 passed, 10 deselected**
- `make lint`: passed
- `make typecheck`: passed
- `git diff --check`: passed
- Focused compatibility and fixture tests: passed
- Local CPU NLLB-CT2 smoke translation: passed
- Phase 1 book tests: **19 passed** (GPT-Terra gate)

## Review findings still open

GPT-Terra returned `CHANGES_REQUESTED` for Phase 0 and the initial Phase 1 work.
The highest-priority items are:

1. Add `pytest-timeout` (or remove unsupported timeout flags) so explicit slow/GPU/
   e2e targets execute correctly.
2. Align the Transformers pin in `requirements.txt` with `pyproject.toml`.
3. Add registry-wide CLI acceptance coverage and direct tests for the IndicTrans2
   tokenizer compatibility shim.
4. Complete the remaining Phase 1 stable-ID/immutable-schema acceptance coverage,
   including ambiguous re-import dry runs and full JSONL migration dispatch.
5. Add the missing Phase 1 acceptance tests for crash boundaries, leases/CAS,
   migrations, approval preservation, JSONL, and separator/source coverage.

## Delegation state

GPT-Luna implemented the narrow Phase 1 remediation. GPT-Terra approved that scope
after review. The next handoff should only advance to Phase 2 after the remaining
stable-ID/immutability coverage is completed and reviewed.

## Repository safety notes

- Do not add or publish the untracked `inputs/` directory; it is user-owned and may
  contain private material.
- Do not overwrite `.claude/settings.local.json`; its changes predate this handoff.
- Do not commit model weights, generated projects, SQLite databases, or review
  exports.
