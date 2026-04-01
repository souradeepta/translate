"""Benchmark all available models: BLEU score + speed."""

from __future__ import annotations

import time
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def benchmark_model(model_name: str, bengali_text: str, reference: str) -> dict:
    from bn_en_translate.config import ModelConfig, PipelineConfig
    from bn_en_translate.models.factory import get_translator
    from bn_en_translate.pipeline.pipeline import TranslationPipeline
    import sacrebleu

    config = PipelineConfig(model=ModelConfig(model_name=model_name, device="auto"))

    try:
        translator = get_translator(config)
        pipeline = TranslationPipeline(translator, config)

        t0 = time.perf_counter()
        with translator:
            hypothesis = pipeline.translate(bengali_text)
        elapsed = time.perf_counter() - t0

        bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])

        return {
            "model": model_name,
            "bleu": round(bleu.score, 2),
            "seconds": round(elapsed, 2),
            "output_preview": hypothesis[:100],
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "bleu": None,
            "seconds": None,
            "output_preview": None,
            "error": str(e),
        }


def main() -> None:
    bengali_text = (FIXTURES_DIR / "sample_short.bn.txt").read_text(encoding="utf-8")
    reference = (FIXTURES_DIR / "expected_short.en.txt").read_text(encoding="utf-8")

    models_to_test = ["nllb-600M", "nllb-1.3B", "indicTrans2-1B"]

    print(f"{'Model':<20} {'BLEU':>6} {'Time(s)':>8}  Preview")
    print("-" * 70)

    for model_name in models_to_test:
        result = benchmark_model(model_name, bengali_text, reference)
        if result["error"]:
            print(f"{model_name:<20} {'ERROR':>6}            {result['error'][:40]}")
        else:
            preview = (result["output_preview"] or "")[:40]
            print(
                f"{model_name:<20} {result['bleu']:>6.1f} {result['seconds']:>8.1f}s  {preview}"
            )


if __name__ == "__main__":
    main()
