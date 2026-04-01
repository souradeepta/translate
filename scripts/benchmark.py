"""Benchmark available models: BLEU score, speed, GPU utilization."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def _gpu_vram_mib() -> int:
    """Return current GPU memory usage in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def _gpu_util_pct() -> int:
    """Return GPU compute utilization % via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def benchmark_model(
    model_name: str,
    bengali_texts: list[str],
    references: list[str],
    device: str = "auto",
) -> dict:  # type: ignore[type-arg]
    import sacrebleu  # type: ignore[import-untyped]

    from bn_en_translate.config import ModelConfig, PipelineConfig
    from bn_en_translate.models.factory import get_translator
    from bn_en_translate.pipeline.pipeline import TranslationPipeline

    config = PipelineConfig(model=ModelConfig(model_name=model_name, device=device))

    try:
        translator = get_translator(config)
        pipeline = TranslationPipeline(translator, config)

        vram_before = _gpu_vram_mib()

        t0 = time.perf_counter()
        with translator:
            vram_loaded = _gpu_vram_mib()
            hypotheses = [pipeline.translate(t) for t in bengali_texts]
            vram_during = _gpu_vram_mib()
        elapsed = time.perf_counter() - t0

        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        vram_used = vram_loaded - vram_before

        return {
            "model": model_name,
            "backend": type(translator).__name__,
            "bleu": round(bleu.score, 2),
            "seconds": round(elapsed, 2),
            "chars_per_sec": round(sum(len(t) for t in bengali_texts) / elapsed),
            "vram_loaded_mib": vram_used,
            "vram_peak_mib": vram_during - vram_before,
            "output_preview": hypotheses[0][:80] if hypotheses else "",
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "backend": "N/A",
            "bleu": None,
            "seconds": None,
            "chars_per_sec": None,
            "vram_loaded_mib": None,
            "vram_peak_mib": None,
            "output_preview": None,
            "error": str(e),
        }


def load_corpus(n: int = 50) -> tuple[list[str], list[str]]:
    """Load up to n Bengali/English pairs from corpus, falling back to fixtures."""
    bn_corpus = CORPUS_DIR / "flores200_devtest.bn.txt"
    en_corpus = CORPUS_DIR / "flores200_devtest.en.txt"

    if bn_corpus.exists() and en_corpus.exists():
        bn_lines = bn_corpus.read_text(encoding="utf-8").strip().splitlines()[:n]
        en_lines = en_corpus.read_text(encoding="utf-8").strip().splitlines()[:n]
        print(f"Using FLORES-200 corpus: {len(bn_lines)} sentences")
        return bn_lines, en_lines

    # Fallback to fixture files
    bn = (FIXTURES_DIR / "sample_short.bn.txt").read_text(encoding="utf-8").strip()
    en = (FIXTURES_DIR / "expected_short.en.txt").read_text(encoding="utf-8").strip()
    print("Using fixture corpus (FLORES-200 not found; run scripts/get_corpus.py)")
    return [bn], [en]


def print_gpu_info() -> None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        )
        print(f"GPU: {out.strip()}")
    except Exception:
        print("GPU: not detected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Bengali translation models")
    parser.add_argument("--models", nargs="+", default=["nllb-600M"],
                        help="Models to benchmark (default: nllb-600M)")
    parser.add_argument("--device", default="auto", help="Device: cuda|cpu|auto")
    parser.add_argument("--sentences", type=int, default=50,
                        help="Number of sentences from corpus (default: 50)")
    args = parser.parse_args()

    print("=" * 72)
    print("Bengali → English Translation Benchmark")
    print("=" * 72)
    print_gpu_info()
    print()

    bn_texts, en_refs = load_corpus(n=args.sentences)

    print(f"\n{'Model':<22} {'Backend':<24} {'BLEU':>6} {'Time':>7} {'ch/s':>6} {'VRAM':>6}")
    print("-" * 72)

    for model_name in args.models:
        r = benchmark_model(model_name, bn_texts, en_refs, device=args.device)
        if r["error"]:
            print(f"{model_name:<22} {'ERROR':<24} {r['error'][:20]}")
        else:
            vram_str = f"{r['vram_loaded_mib']}M" if r["vram_loaded_mib"] else "N/A"
            print(
                f"{model_name:<22} {r['backend']:<24} "
                f"{r['bleu']:>6.1f} {r['seconds']:>6.1f}s "
                f"{r['chars_per_sec']:>6} {vram_str:>6}"
            )
            print(f"  Preview: {r['output_preview']}")

    print("=" * 72)


if __name__ == "__main__":
    main()
