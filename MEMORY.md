# Project Memory

Read this file with `CLAUDE.md` at the start of a new implementation, review, test,
or documentation session. It records verified continuity facts; the dated handoff is
the detailed delivery record.

## Current verified state — 2026-09-10

- EPUB book-format support is delivered on `main` in `0d12370` and `e9bd55c`.
- `EpubReader` imports safe, spine-ordered XHTML into the format-neutral book schema.
  `EpubWriter` preserves original package resources, performs atomic output, and
  validates semantic re-import coverage.
- EPUB imports reject unsafe ZIP paths/symlinks, active content, and external resource
  fetches. Unsupported table content is reported rather than silently imported.
- The optional EPUB extra is `bn-en-translate[epub]` (`ebooklib` and
  `beautifulsoup4`). `epubcheck` is optional and was unavailable in the verification
  environment; run it in a release environment when installed.
- EPUB regression coverage includes unchanged emphasis/link retention, changed-text
  fallback findings, notes and unsupported structures, external CSS blocking,
  byte-for-byte asset retention, and non-XHTML spine-item ordinal handling.
- Latest verified commands: full Python suite **343 passed**; book suite **44
  passed**; EPUB suite **7 passed**; EPUB lint and strict type check passed.

## Current follow-up work

- EPUB still needs an independent review and an `epubcheck` release run where that
  executable is available.
- The next planned implementation unit is Phase 3.1: translator capabilities and
  tokenizer-aware token counting. Do not invent model limits; ground them in adapter
  behavior and tests.
- Preserve the user-owned untracked `inputs/` directory and unrelated
  `.claude/settings.local.json` changes. Do not stage or commit either unless the
  user explicitly asks.

## Sources of truth

- Detailed handoff: `docs/HANDOFF_BOOK_TRANSLATION_2026-09-09.md`
- Design: `docs/superpowers/specs/2026-09-09-book-translation-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-09-book-translation-implementation.md`
