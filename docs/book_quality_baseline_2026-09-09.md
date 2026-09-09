# Book-quality baseline (2026-09-09)

This is a small, redistributable regression fixture for document-level Bengali →
English translation. It is deliberately synthetic so the baseline can be checked
into the repository and reproduced without the private `inputs/` directory. The
fixture exercises two chapters, recurring names and honorifics, dialogue and
pronouns, dates and numbers, the cultural term *panta bhat*, a long sentence,
scene breaks, and repeated terms across chapter boundaries.

## Fixture and provenance

| File | Purpose | Provenance |
| --- | --- | --- |
| `tests/fixtures/book/consistency_source.bn.txt` | Bengali source | Synthetic, written for this project; CC0-1.0 |
| `tests/fixtures/book/consistency_reference.en.txt` | Aligned English reference | Human-authored synthetic reference; CC0-1.0 |
| `tests/fixtures/book/expected_terms.json` | Stable term/occurrence expectations | Metadata for the two files; CC0-1.0 |

The files use UTF-8 and two newlines between blocks. The 12 blocks are numbered
one-based in file order. The source and reference have the same block count; the
scene-break blocks are intentionally retained. `expected_terms.json` records the
source block numbers where each recurring term must be considered by consistency
QA. This is an expectation for evaluation, not a claim that every model emits the
same spelling.

## Reproduction protocol

From the repository root:

```bash
source .venv/bin/activate
python -m bn_en_translate.cli \
  --input tests/fixtures/book/consistency_source.bn.txt \
  --output /tmp/consistency.nllb.en.txt \
  --model milmmt-46-4b --device cuda --batch-size 1
```

Repeat with `--model milmmt-46-1b` for the lower-VRAM comparison. Models are loaded
sequentially; do not run both in one process. The command is intentionally not run
as part of the test suite because model weights are local/optional and no network
download is permitted by the baseline gate.

For each run, retain the exact command, model revision or local snapshot hash,
backend, generation parameters, output path, and whether the output was served from
cache. Record package and hardware details with the result. A minimal capture is:

```bash
python - <<'PY'
import importlib.metadata as metadata
for package in ("bn-en-translate", "torch", "transformers", "ctranslate2", "sacrebleu"):
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "not installed")
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

At the time this protocol was recorded, GPU access was unavailable in the build
environment (`nvidia-smi` could not initialize NVML), and neither model result was
therefore produced. Results are **pending** rather than fabricated. When a model is
available, score only this aligned fixture with sacreBLEU/chrF and add one row below.

## Results

| Model | Revision | Backend | BLEU | chrF | Hardware | Generation parameters | Cache | Status |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `milmmt-46-4b` | pending | pending | — | — | GPU unavailable | `beam_size=1`, `max_decoding_length=512`, `batch_size=1` | unknown | pending |
| `milmmt-46-1b` | pending | pending | — | — | GPU unavailable | `beam_size=1`, `max_decoding_length=512`, `batch_size=1` | unknown | pending |

Automatic metrics are reported only when a reference-aligned output exists. They
are not a substitute for book-level review.

## Manual annotation categories

Review each output block against the source and reference and count findings using
these stable categories:

- `coverage.omission`: source meaning, clause, or block missing.
- `coverage.addition`: unsupported fact or explanation added.
- `consistency.name`: recurring name spelling changes (`রহিম মিয়া`, `গীতা`).
- `consistency.honorific`: honorific or form of address changes (`মাস্টার সাহেব`).
- `consistency.pronoun`: person, gender, number, or relationship changes.
- `consistency.term`: cultural/glossary term is mistranslated or inconsistent.
- `fidelity.number_date`: a number or date is altered or omitted.
- `structure.block`: block, chapter, dialogue, or scene-break boundary changes.
- `style.tone`: unsupported shift in tense, voice, register, or dialogue tone.
- `residue.bengali`: Bengali script remains in an English target without an allowlist entry.

Annotators should record the source block number, category, short evidence, and
severity (`blocker`, `high`, `medium`, or `low`). This makes the fixture useful as a
regression baseline even while model scores remain pending.
