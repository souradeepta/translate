"""Command-line interface for Bengali → English story translation."""

from __future__ import annotations

import click

from bn_en_translate.config import ModelConfig, PipelineConfig
from bn_en_translate.models.factory import get_translator
from bn_en_translate.pipeline.pipeline import TranslationPipeline
from bn_en_translate.utils.cuda_check import get_best_device


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to Bengali story file (.txt)")
@click.option("--output", "-o", "output_path", required=True, help="Path for English output file")
@click.option(
    "--model",
    "-m",
    default="nllb-600M",
    show_default=True,
    help=(
        "Translation model: nllb-600M | nllb-1.3B | indicTrans2-1B | "
        "madlad-3b | seamless-medium | ollama"
    ),
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Device: cuda | cpu | auto (auto picks cuda if available)",
)
@click.option("--batch-size", default=8, show_default=True, help="Translation batch size")
@click.option(
    "--beam-size",
    default=None,
    type=int,
    help="Beam search width (default: model-specific; NLLB=4, IndicTrans2/SeamlessM4T=5)",
)
@click.option("--ollama-polish", is_flag=True, default=False, help="Run Ollama polishing pass after translation")
@click.option(
    "--ollama-model",
    default="gemma3:12b",
    show_default=True,
    help="Ollama model tag for the polish pass (e.g. gemma3:4b, gemma3:12b, qwen2.5:7b-instruct-q4_K_M)",
)
def main(
    input_path: str,
    output_path: str,
    model: str,
    device: str,
    batch_size: int,
    beam_size: int | None,
    ollama_polish: bool,
    ollama_model: str,
) -> None:
    """Translate a Bengali story file to English using a local open-source model."""

    resolved_device = get_best_device() if device == "auto" else device
    click.echo(f"Device: {resolved_device} | Model: {model}")

    config = PipelineConfig(
        model=ModelConfig(
            model_name=model,
            device=resolved_device,
            beam_size=beam_size,
        ),
        ollama_polish=ollama_polish,
        ollama_model=ollama_model,
    )

    translator = get_translator(config)
    pipeline = TranslationPipeline(translator, config)

    click.echo(f"Loading model '{model}'...")
    with translator:
        click.echo(f"Translating '{input_path}'...")
        pipeline.translate_file(input_path, output_path)

    click.echo(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    main()
