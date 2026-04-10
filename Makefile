.PHONY: install test test-fast test-slow test-e2e lint typecheck clean setup-cuda papers figures

# Install dev dependencies
install:
	pip install -e ".[dev]" -r requirements-dev.txt

# Run only fast unit + integration tests (no model downloads)
test:
	pytest -m "not slow and not e2e" -v

# Run only unit tests
test-unit:
	pytest tests/unit/ -v

# Run slow tests (real model loading, ~1.2 GB download for NLLB)
test-slow:
	pytest -m "slow" -v --timeout=300

# Run full E2E tests (requires GPU + model downloads)
test-e2e:
	pytest -m "e2e" -v --timeout=600

# Run all tests
test-all:
	pytest -v --timeout=600

# Code quality
lint:
	ruff check src/ tests/

typecheck:
	mypy src/bn_en_translate/

# Setup CUDA environment
setup-cuda:
	bash scripts/setup_cuda.sh

# Download and convert primary model (IndicTrans2)
download-indicTrans2:
	python scripts/download_models.py --model indicTrans2-1B

# Download lightweight NLLB model
download-nllb:
	python scripts/download_models.py --model nllb-600M

# Download Bengali-English FLORES-200 corpus
get-corpus:
	python scripts/get_corpus.py

# Run benchmarks against available models (downloads corpus first if missing)
benchmark:
	python scripts/benchmark.py --models nllb-600M --sentences 50

benchmark-all:
	python scripts/benchmark.py --models nllb-600M nllb-1.3B indicTrans2-1B --sentences 100

# Regenerate paper figures (requires venv + monitor/runs.db)
figures:
	python scripts/gen_paper_figures.py

# Compile all LaTeX papers to PDF (requires tectonic: ~/.local/bin/tectonic)
papers: figures
	mkdir -p paper/pdf
	@for f in paper/ieee_paper.tex paper/survey_paper.tex paper/ieee_transactions_paper.tex paper/acm_paper.tex; do \
		echo "  compiling $$f ..."; \
		~/.local/bin/tectonic "$$f" -o paper/pdf/ 2>&1 | grep -E "^note:|^error:|^warning: .*error" || true; \
	done
	@echo "PDFs in paper/pdf/"
	@ls -lh paper/pdf/*.pdf

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache/ .mypy_cache/ dist/ build/
