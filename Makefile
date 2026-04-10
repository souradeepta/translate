.PHONY: install test test-fast test-slow test-e2e lint typecheck clean setup-cuda \
        papers slides figures \
        paper-ieee paper-survey paper-ieee-tr paper-acm \
        slides-ieee slides-survey slides-ieee-tr slides-acm

# ── Dev setup ──────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]" -r requirements-dev.txt

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest -m "not slow and not e2e" -v

test-unit:
	pytest tests/unit/ -v

test-slow:
	pytest -m "slow" -v --timeout=300

test-e2e:
	pytest -m "e2e" -v --timeout=600

test-all:
	pytest -v --timeout=600

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

typecheck:
	mypy src/bn_en_translate/

# ── CUDA setup ─────────────────────────────────────────────────────────────────
setup-cuda:
	bash scripts/setup_cuda.sh

# ── Model downloads ────────────────────────────────────────────────────────────
download-indicTrans2:
	python scripts/download_models.py --model indicTrans2-1B

download-nllb:
	python scripts/download_models.py --model nllb-600M

# ── Corpus ─────────────────────────────────────────────────────────────────────
get-corpus:
	python scripts/get_corpus.py

# ── Benchmarks ─────────────────────────────────────────────────────────────────
benchmark:
	python scripts/benchmark.py --models nllb-600M --sentences 50

benchmark-all:
	python scripts/benchmark.py --models nllb-600M nllb-1.3B indicTrans2-1B --sentences 100

# ── Paper figures ──────────────────────────────────────────────────────────────
# Generates paper/figures/*.png at 300 DPI (academic style, IEEE column widths)
figures:
	python scripts/gen_paper_figures.py

# ── Individual paper targets ───────────────────────────────────────────────────
TECTONIC     := ~/.local/bin/tectonic
PDF_OUT      := paper/pdf
PAPER_SRCS   := paper/ieee_conference/ieee_conference.tex \
                paper/survey/survey.tex \
                paper/ieee_transactions/ieee_transactions.tex \
                paper/acm_tallip/acm_tallip.tex
SLIDES_SRCS  := paper/ieee_conference/ieee_conference_slides.tex \
                paper/survey/survey_slides.tex \
                paper/ieee_transactions/ieee_transactions_slides.tex \
                paper/acm_tallip/acm_tallip_slides.tex

paper-ieee:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/ieee_conference/ieee_conference.tex -o $(PDF_OUT)/

paper-survey:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/survey/survey.tex -o $(PDF_OUT)/

paper-ieee-tr:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/ieee_transactions/ieee_transactions.tex -o $(PDF_OUT)/

paper-acm:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/acm_tallip/acm_tallip.tex -o $(PDF_OUT)/

slides-ieee:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/ieee_conference/ieee_conference_slides.tex -o $(PDF_OUT)/

slides-survey:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/survey/survey_slides.tex -o $(PDF_OUT)/

slides-ieee-tr:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/ieee_transactions/ieee_transactions_slides.tex -o $(PDF_OUT)/

slides-acm:
	mkdir -p $(PDF_OUT)
	$(TECTONIC) paper/acm_tallip/acm_tallip_slides.tex -o $(PDF_OUT)/

# ── Batch targets ──────────────────────────────────────────────────────────────
# Compile all 4 papers to PDF
papers: figures
	mkdir -p $(PDF_OUT)
	@for f in $(PAPER_SRCS); do \
		echo "  compiling $$f ..."; \
		$(TECTONIC) "$$f" -o $(PDF_OUT)/ 2>&1 \
		  | grep -E "^note:|^error:|^warning: .*error" || true; \
	done
	@echo "Papers in $(PDF_OUT)/:"
	@ls -lh $(PDF_OUT)/paper.pdf $(PDF_OUT)/survey_paper.pdf 2>/dev/null || ls -lh $(PDF_OUT)/*.pdf 2>/dev/null || true

# Compile all 4 slide decks to PDF
slides:
	mkdir -p $(PDF_OUT)
	@for f in $(SLIDES_SRCS); do \
		echo "  compiling $$f ..."; \
		$(TECTONIC) "$$f" -o $(PDF_OUT)/ 2>&1 \
		  | grep -E "^note:|^error:|^warning: .*error" || true; \
	done
	@echo "Slides in $(PDF_OUT)/:"
	@ls -lh $(PDF_OUT)/slides.pdf 2>/dev/null || ls -lh $(PDF_OUT)/*.pdf 2>/dev/null || true

# Compile everything: figures + all papers + all slides
all-papers: figures papers slides
	@echo "All PDFs:"
	@ls -lh $(PDF_OUT)/*.pdf 2>/dev/null

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache/ .mypy_cache/ dist/ build/
	find paper/ -name "*.aux" -o -name "*.log" -o -name "*.out" \
	  -o -name "*.toc" -o -name "*.bbl" -o -name "*.blg" \
	  -o -name "*.synctex.gz" 2>/dev/null | xargs rm -f 2>/dev/null; true
