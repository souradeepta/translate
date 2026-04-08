# Paper Revision History

This file tracks all changes to `paper/ieee_paper.tex`, `paper/survey_paper.tex`,
and the slide decks in `paper/slides/`. Before any edit to these files:
1. Copy the current version to `paper/archive/` with a `YYYY-MM-DD_vN` suffix.
2. Make the edits.
3. Generate a `.diff` file: `diff -u archive/old.tex current.tex > archive/old_to_new.diff`
4. Append an entry to this file.

---

## v1 — 2026-04-02 (initial published version)

### ieee_paper.tex
- Initial IEEE conference paper: 1,372 lines, 20 tables, 8 figures, 28 bibliography entries
- NLLB-200-distilled-600M as primary model, CTranslate2 float16, RTX 5050 sm_120
- BLEU 65.2 overall (90-sentence in-domain corpus), BLEU 0.17 post fine-tuning (Samanantar open-domain)
- LoRA fine-tuning results: 3 epochs, 7,863 pairs, bf16, 2.46 hours

### survey_paper.tex
- Initial survey: 1,512 lines, 9 tables, 2 pgfplots figures, 32 bibliography entries
- Coverage: 20+ systems, 2019–2025, FLORES-200 BLEU trend, Pareto frontier

Archived:
- `paper/archive/ieee_paper_2026-04-02_v1.tex`
- `paper/archive/survey_paper_2026-04-02_v1.tex`

---

## v2 — 2026-04-07 (model expansion: MADLAD-400, SeamlessM4T-v2, Gemma 3, Flash Attention 2)

### ieee_paper.tex
- Added MADLAD-400-3B and SeamlessM4T-v2 to Background/Related Work
- Added MADLAD/SeamlessM4T rows to model comparison table (FLORES BLEU from published results)
- Added Gemma 3 12B as default Ollama polish model note
- Added Flash Attention 2 note in inference section
- Added 4 bibliography entries: kudugunta2023madlad, seamlessm4t2023, gemma2025, dao2022flashattention
- New placeholder commands: \MADLADBLEU{36}, \SEAMLESSBLEU{39}, \MADLADPROJBLEU{TBD}, \SEAMLESSPROJBLEU{TBD}

Archived: paper/archive/ieee_paper_2026-04-07_v2_pre.tex
Diff: paper/archive/ieee_paper_v1_to_v2.diff

### survey_paper.tex
- Added MADLAD-400 subsection (36 FLORES BLEU, T5 architecture, target-language prefix)
- Added SeamlessM4T-v2 subsection (~38-40 FLORES BLEU, custom arch, no CT2)
- Added Gemma 3 paragraph in LLM-based translation section
- Added MADLAD-400 and SeamlessM4T-v2 rows to main BLEU comparison table
- 4 bibliography entries added (same as ieee_paper.tex)

Archived: paper/archive/survey_paper_2026-04-07_v2_pre.tex
Diff: paper/archive/survey_paper_v1_to_v2.diff

---

## v1-slides — 2026-04-08 (initial slide decks)

### Slides created
- `ieee_slides.tex` — Beamer (Madrid/seahorse), ~10 slides: motivation, pipeline, models, BLEU results, fine-tuning, hardware constraints, monitoring, conclusion
- `survey_slides.tex` — Beamer (Copenhagen/crane), ~12 slides: landscape, BLEU trend, comparison table, Pareto, domain variance, challenges, gaps, future
- `overview.md` — Marp, 15 slides: quick-start, pipeline, models table, results, fine-tuning, hardware, monitoring, architecture
- `survey_reveal.html` — Reveal.js (moon theme), ~9 slides: interactive Chart.js BLEU trend and Pareto frontier charts

Archived:
- `paper/archive/slides/ieee_slides_2026-04-07_v1.tex`
- `paper/archive/slides/survey_slides_2026-04-07_v1.tex`
- `paper/archive/slides/overview_2026-04-07_v1.md`
- `paper/archive/slides/survey_reveal_2026-04-07_v1.html`
