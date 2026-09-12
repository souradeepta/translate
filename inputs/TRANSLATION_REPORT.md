# Translation Report — "যে আসে অন্ধকারে" (2026-07-09)

Source: `inputs/RAA -যে আসে অন্ধকারে- 1703.docx` (Bengali short story, ~1703 words,
90 paragraphs). Text extracted via `python-docx` to `inputs/story.bn.txt`
(10,049 characters, UTF-8).

Two top-quality models (by FLORES-200 BLEU) were run against the full text.
No reference English translation exists for this story, so per-run quality
below is the model's standing FLORES-200 benchmark score, not a score measured
on this specific text.

## Results

| Model | Load time | Translate time | Throughput | FLORES-200 BLEU | FLORES-200 chrF | Output file |
|-------|-----------|-----------------|------------|------------------|------------------|-------------|
| Seamless-medium (`facebook/seamless-m4t-v2-large`, HF float16) | 38.2 s | 88.2 s | 114 ch/s | 67.0 | 80.2 | `story.seamless.en.txt` |
| MiLMMT-46-1B (`xiaomi-research/MiLMMT-46-1B-v0.1`, HF bfloat16) | 22.9 s | 23.5 s | 428 ch/s | 65.2 | 79.6 | `story.milmmt.en.txt` |
| MiLMMT-46-4B (`xiaomi-research/MiLMMT-46-4B-v0.1`, 4-bit bnb) | 142.9 s | 53.0 s | 190 ch/s | **68.5** | **81.8** | `story.milmmt4b.en.txt` |

- Input: 10,049 chars / 90 paragraphs, all models.
- Hardware: RTX 5050 (sm_120), single run each, sequential (not concurrent) to avoid VRAM contention.
- **2026-07-10 update**: MiLMMT-46-4B (added after further evaluation — see `docs/MODELS.md`) is the best of the three on both the standard benchmark and this real document. The user judged MiLMMT-1B's output better than Seamless's despite the lower benchmark score (see memory); MiLMMT-4B visibly fixed real mistranslations the 1B model made on this same story (e.g. 1B's "the sounds of a lion" — nonsensical for rural Bengal — became 4B's correct "the howls of the jackals"). Caveat: 4B's load is slow (4-bit quantization overhead) and VRAM peaks at 8050/8151 MiB, very tight on this 8GB card — kept as an opt-in model (`--model milmmt-46-4b`), not the default.

## Method

Ad-hoc timing script (`run_story.py`, not part of the checked-in CLI): loads
the translator, times `load()` separately from `pipeline.translate()`, then
writes a metrics header into the output file plus a companion
`<output>.metrics.txt`. Quality figures are the models' existing FLORES-200
numbers from `docs/MODELS.md` / `CLAUDE.md` — not re-measured here, since this
story has no gold English reference to score against.

## Output files (`inputs/`)

- `story.bn.txt` — extracted Bengali source text
- `story.seamless.en.txt` / `.metrics.txt` — Seamless-medium translation + metrics
- `story.milmmt.en.txt` / `.metrics.txt` — MiLMMT-46-1B translation + metrics
- `story.milmmt4b.en.txt` / `.metrics.txt` — MiLMMT-46-4B translation + metrics (best quality found)
