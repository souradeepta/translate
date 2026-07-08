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

---

## v4 — 2026-07-08 (July 2026 optimization-pass results + new efficiency paper)

Note: the paper tree was reorganized into per-venue subdirectories between v3
and v4 (`paper/ieee_paper.tex` → `paper/ieee_conference/ieee_conference.tex`,
`paper/survey_paper.tex` → `paper/survey/survey.tex`,
`paper/ieee_transactions_paper.tex` → `paper/ieee_transactions/ieee_transactions.tex`,
`paper/acm_paper.tex` → `paper/acm_tallip/acm_tallip.tex`). This entry uses the
current paths. `paper/ieee_conference/ieee_conference.tex` and
`paper/survey/survey.tex` did not previously mention MiLMMT-46-1B at all
(only NLLB-600M and Seamless-v2 were present); MiLMMT is introduced as a
baseline in both papers in this pass because the new-model rejections
(LMT-60, Hunyuan-MT Q4) are defined relative to it.

### ieee_conference/ieee_conference.tex
- New `\newcommand` placeholders for MiLMMT FLORES BLEU/chrF (65.2/79.6) and
  for July-2026 close-out throughput (load/translate split) and Phase-1
  cumulative before/after throughput, all traced to the 2026-07-08 measured
  run and `monitor/observations.md` — no invented numbers.
- Abstract, Introduction contributions list, Conclusion: added MiLMMT as a
  third evaluated model; added the three-optimization summary and BLEU-parity
  gate methodology; added the two rejected 2026 candidates and the MADLAD
  re-verification verdict.
- New Table `tab:throughput` (translate-only + load, 2026-07-08 close-out) and
  new Table `tab:optpass` (Phase-1 cumulative before/after); old `tab:benchmark`
  (97--100 chars/s, load+translate conflated) retained for provenance with an
  explicit non-comparability note.
- New subsection "July 2026 Inference Optimization Pass" (`sec:optpass`):
  the three optimizations (batching, SDPA + per-architecture fallback,
  length-sorted batching) with per-step measured deltas, the load-vs-translate
  measurement pitfall, and the single-run speed-noise caution.
- New subsection "Model Selection Under a Fixed VRAM Budget" (`sec:modelselect`):
  NiuTrans LMT-60-1.7B (rejected, incl. 73.2@5-sentence vs. 63.8@90-sentence
  smoke-test noise caution), Tencent Hunyuan-MT-7B Q4\_K\_M (rejected,
  quantization-loss finding), MADLAD-400 re-download verdict (still fails the
  new tied-embedding integrity guard), and the Ollama silent-CPU-fallback gotcha.
- New subsection "VRAM Budget Enforcement" under System Architecture
  (`sec:vrambudget`): the measured per-model VRAM table and
  `ensure_vram_available()` pre-flight pattern.
- `tab:comparison` gained MiLMMT, LMT-60, and Hunyuan-MT-7B Q4 rows.
- Discussion "GPU Utilisation and Batch Size" extended with the batching-vs-
  occupancy-tuning finding.
- No April BLEU/chrF numbers altered (NLLB 55.3/72.8 in-domain 65.2, Seamless
  67.0/80.2 all unchanged); MiLMMT is a new row, not an edit to an existing one.
- All `\cite`/`\bibitem` keys verified to match (no new bibliography entries
  needed — reused `dao2022flashattention`, `nvidia_blackwell`, etc.).

### survey/survey.tex
- `tab:bleu` "This Work" block: added MiLMMT-46-1B row; new "2026 Candidate
  Evaluation (measured, rejected)" block with LMT-60-1.7B and Hunyuan-MT-7B
  Q4\_K\_M, each footnoted as our measurement (Hunyuan explicitly flagged as
  a 4-bit quantization, not the model's full-precision capability).
- `tab:compute`: added MiLMMT-46-1B, LMT-60-1.7B (rejected), Hunyuan-MT-7B
  Q4\_K\_M (rejected) rows with VRAM figures.
- `tab:headtohead`: throughput row corrected from the load+translate-conflated
  97/32 chars/s to translate-only 2,346/372 chars/s, with an explicit
  non-comparability footnote.
- New subsection "July 2026 Inference Optimization and Model Selection" under
  `sec:ours`: three-optimization summary with per-model speedups, and the
  LMT-60 / Hunyuan-MT-7B Q4 rejection narrative including the small-sample
  BLEU noise caution.
- No new bibliography entries; all new content self-cites `biswas2025bnentr`
  or makes no citation (descriptive-only optimization narrative).

### ieee_transactions/ieee_transactions.tex (minimal edit, per scope)
- `tab:throughput`: added a translate-only 2026-07 row (2,346 chars/s) with a
  footnote explaining the load/translate conflation found in Run 1/2 timing;
  `tab:benchmark` (Run 1/2, 97/100 chars/s) left untouched as provenance.
- Abstract, Introduction contribution #1, Conclusion: "97--100 chars/s" now
  paired with the corrected 2,346 chars/s figure and a one-sentence
  methodology note. No new sections, no new-model content (out of scope per
  brief — this paper never discussed MiLMMT/LMT-60/Hunyuan).

### acm_tallip/acm_tallip.tex (minimal edit, per scope)
- Abstract, Introduction contribution (i), Conclusion: same "97--100 chars/s
  → 2,346 chars/s translate-only, methodology corrected" pattern as the
  transactions paper.
- Added one paragraph after `tab:benchmark` in "Baseline Inference
  Performance" noting the July 2026 load/translate correction; the table
  itself is unchanged (provenance).

### New paper: efficiency/efficiency.tex
- New IEEE-conference-format (IEEEtran) systems paper: "Batching, Kernels,
  and Quantization Trade-offs: A Measured Study of Bengali-English NMT
  Inference on an 8 GB Consumer GPU."
- Sections: Introduction (three generalizable findings) → Background
  (models/evaluation/inference engines) → Measurement Methodology (hardware,
  corpus, BLEU-parity gate, multi-run variance) → Three Inference
  Optimizations (batching, SDPA + per-architecture fallback, length-sorted
  batching, each with measured before/after deltas) → The Load-vs-Translate
  Measurement Pitfall → Model Selection Under a Fixed VRAM Budget (LMT-60,
  Hunyuan-MT-7B Q4, MADLAD re-verification, Ollama CPU-fallback gotcha) →
  VRAM Budget Enforcement as a Systems Pattern → Threats to Validity
  (single GPU, single language pair, single-run speed noise, one-model/one-
  quantization-level caveat) → Conclusion.
- 13 bibliography entries, all real and independently verifiable: NLLB
  (`nllb2022`), CTranslate2 (`ctranslate2_klein`), SeamlessM4T-v2
  (`seamlessm4t2023`), MADLAD-400 (`kudugunta2023madlad`), BLEU
  (`papineni2002bleu`), sacreBLEU (`post2018call`), chrF (`popovic2015chrf`),
  FlashAttention (`dao2022flashattention`), QLoRA (`dettmers2023qlora`,
  new — arXiv:2305.14314, cited only for the quantization threats-to-validity
  discussion), NVIDIA Blackwell (`nvidia_blackwell`), Ethnologue
  (`ethnologue2023`), Ollama (`ollama2024`, new — GitHub repo, no fabricated
  paper), and a self-citation to the companion system paper
  (`biswas2025bnentr`). No citation for NiuTrans LMT-60-1.7B: no
  independently verifiable publication was identified, so it is described
  only in prose (architecture, precision, prompt format — all facts from
  the measured run) with no `\cite`, per the no-fabricated-citation
  constraint.
- New figure `optimization_speedup.png`: (a) before/after translate-only
  throughput per model (log scale, grouped bars), (b) BLEU-vs-VRAM
  model-selection scatter (deployed vs. rejected, 8 GB budget line). Added
  as `fig_optimization_speedup()` in `scripts/gen_paper_figures.py`,
  hardcoded to the brief's measured numbers (does not read `monitor/runs.db`,
  since the 2026-07-08 close-out run may not be persisted there). Regenerated
  via `make figures` / `python scripts/gen_paper_figures.py`.

### Makefile
- Added `paper/efficiency/efficiency.tex` to `PAPER_SRCS`, a `paper-efficiency`
  target, and updated the `.PHONY` list and the `papers` comment (4 → 5 papers).

### Slides
- Not touched in this pass (brief scoped this revision to papers + Makefile
  `papers` target only; no `efficiency_slides.tex` was requested or created).

### Compile status (2026-07-08, `make papers`)
All 5 papers compiled with `tectonic`, zero errors (only pre-existing
underfull/overfull hbox warnings and Bengali-glyph-in-Latin-font warnings,
unrelated to this pass). Page counts: ieee_conference 13, survey 13,
ieee_transactions 6, acm_tallip 5, efficiency 6.

Archived:
- `paper/archive/ieee_conference_2026-07-08_v4_pre.tex`
- `paper/archive/survey_2026-07-08_v4_pre.tex`
- `paper/archive/ieee_transactions_2026-07-08_v4_pre.tex`
- `paper/archive/acm_tallip_2026-07-08_v4_pre.tex`

Diffs:
- `paper/archive/ieee_conference_v3_to_v4.diff`
- `paper/archive/survey_v3_to_v4.diff`
- `paper/archive/ieee_transactions_v3_to_v4.diff`
- `paper/archive/acm_tallip_v3_to_v4.diff`
