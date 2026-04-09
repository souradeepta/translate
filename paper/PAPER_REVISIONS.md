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

## v3 — 2026-04-08 (publishability overhaul + journal versions)

### ieee_paper.tex
- Anti-AI writing pass: removed vague adjectives; quantified all performance claims
- Abstract tightened to ≤150 words (from ~164): dropped redundant phrasing, tightened sentence structure
- GPU speedup paragraph in Discussion rewritten to be hardware-specific and quantitative
- Numbered contributions list in Introduction preserved; all BLEU numbers retained

### survey_paper.tex
- Abstract tightened to ≤150 words (from ~200+): five findings preserved in compact form
- "Novel Contributions" section header renamed to "Specific Contributions vs. Prior Work"
- "novel and reproducible contribution" rewritten as "reproducible, previously unpublished contribution"
- All BLEU claims in comparison table (Table tab:bleu) now carry inline \cite{} on every row:
  mBART-50 → \cite{liu2020mbart50}, mT5 → \cite{xue2021mt5}, M2M-100 → \cite{fan2021m2m100},
  NLLB family → \cite{nllb2022}, IndicTrans v1 → \cite{ramesh2022samanantar},
  IndicBART → \cite{dabre2022indicbart}, IndicTrans2 → \cite{gala2023indictrans2},
  commercial APIs → \cite{nllb2022}, domain-specific rows → respective author citations,
  "This Work" rows → \cite{biswas2025bnentr}
- n/r entries verified; no truly empty cells found

### New files
- `ieee_transactions_paper.tex` — IEEE journal version (`\documentclass[journal]{IEEEtran}`):
  targets TASLP/IEEE Access; includes \IEEEpeerreviewmaketitle, \IEEEraisesectionheading intro,
  full Related Work section (~600 words covering low-resource NMT, PEFT, quantized inference,
  Bengali resources), expanded Experimental Setup with hardware/software table, per-domain BLEU
  table (10 domains, 4.7–80.6 BLEU), throughput comparison table, full fine-tuning results table,
  new Limitations section, two-paragraph Conclusion; 200–250 word abstract; 30 bibliography entries
- `acm_paper.tex` — ACM TALLIP version (`\documentclass[sigconf]{acmart}`):
  CCS concepts block, ACM keywords, prose-heavy TALLIP structure across 9 sections,
  full inline bibliography in ACM format; all BLEU numbers, hardware specs, and
  experimental results identical to IEEE version; \begin{acks} acknowledgments block

Archived pre-v3 snapshots (created in prior session):
- `paper/archive/ieee_paper_2026-04-08_v3_pre.tex`
- `paper/archive/survey_paper_2026-04-08_v3_pre.tex`

Diffs (require Bash regeneration after this pass):
- `paper/archive/ieee_paper_v2_to_v3.diff` — update with: diff -u paper/archive/ieee_paper_2026-04-08_v3_pre.tex paper/ieee_paper.tex
- `paper/archive/survey_paper_v2_to_v3.diff` — update with: diff -u paper/archive/survey_paper_2026-04-08_v3_pre.tex paper/survey_paper.tex

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
