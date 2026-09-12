# Bengali test-story candidates (researched 2026-07-12)

Second-story candidates for the shakespeare pipeline (Bengali → MT via
translate → publishable English). All three fetched from Bengali Wikisource
(bn.wikisource.org), plain text extracted from the parse API, paragraph
structure preserved (blank-line separated), UTF-8.

## License / provenance

Rabindranath Tagore died in 1941 — all works are public domain worldwide
(India: life+60, expired 2001; US: pre-1930 publication). Wikisource
transcriptions of PD texts carry no additional restrictions. The pipeline's
output is therefore freely publishable.

## Candidates (all from গল্পগুচ্ছ প্রথম খণ্ড, Galpaguchchha vol. 1)

| File | Story | Words | Paras | Danda sents | Why |
|---|---|---|---|---|---|
| `postmaster.bn.txt` | পোস্ট্‌মাস্টার (The Postmaster) | 1,652 | 35 | 110 | **Recommended first**: closest in size to the validated test story (1,703 words), world-famous, quiet realist prose |
| `kankal.bn.txt` | কঙ্কাল (The Skeleton) | 2,198 | 68 | 174 | Ghost story — genre-matches the current test story; first-person frame narrative |
| `jibito-o-mrito.bn.txt` | জীবিত ও মৃত (The Living and the Dead) | 3,448 | 106 | 249 | Supernatural; the stress test — 2x current story length, famous last line |

## Known caveats

1. **Register**: all three are সাধু ভাষা (classical literary register:
   করিতাম/হইয়াছে verb forms), unlike the modern চলিত register of the
   validated test story. Expect milmmt4b MT quality to drop — this is a
   genuine stress test of the whole chain, and MT degeneration (repetition
   loops) may be more frequent; the shakespeare gate's arbitration will get
   exercised. Run the translate pipeline pilot-first (one paragraph batch)
   per measurement discipline.
2. **Furniture**: each file begins with a title/section paragraph (e.g.
   "কঙ্কাল", "প্রথম পরিচ্ছেদ") and কঙ্কাল ends with a date line
   ("ফাল্গুন ১২৯৮"). shakespeare's noise filter checks the first 4 / last 2
   paragraphs — verify it catches these or trim manually before running.
3. **Chapter headings**: জীবিত ও মৃত contains পরিচ্ছেদ (chapter) heading
   paragraphs mid-text; they will pass through as short paragraphs (the
   pipeline treats <3-sentence units leniently, but check the report).

## Other sources catalogued (not fetched)

- **Bengali Wikisource author pages**: শরৎচন্দ্র চট্টোপাধ্যায় (d. 1938, PD)
  and বিভূতিভূষণ বন্দ্যোপাধ্যায় (d. 1950, PD in India since 2011) both have
  author pages with transcribed works — the nearest source of PD stories in
  a more modern register if সাধু ভাষা proves too hard on the MT.
- **Vacaspati** (IIT Kanpur, 11M sentences of Bangla literature,
  bangla.iitk.ac.in / arXiv:2307.05083): large literary corpus, but
  distribution license is unclear and it is sentence/document oriented —
  useful for model evaluation, not as publishable story input.
- **Multilingual TinyStories (HF, CC-BY-4.0)**: synthetic children's
  narratives incl. Bengali — trivially easy prose, useful only as a smoke
  test, not a literary test.
- **BanglaNMT / Samanantar / FLORES** (already in `../corpus/`): sentence
  pairs, MT evaluation only — not story input.

## How to run (once a candidate is chosen)

1. translate pipeline: produce `<story>.milmmt4b.en.txt` from the `.bn.txt`
   (paragraph counts must be preserved — translate guarantees this).
2. shakespeare: `shakespeare --bengali <story>.bn.txt --english <story>.en.txt
   --output <story>.md` or the `/polish-story` skill (runs both reviewer
   agents automatically).
