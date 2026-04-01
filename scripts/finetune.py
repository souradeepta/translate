"""Fine-tune NLLB-600M on a Bengali-English corpus using LoRA.

Workflow:
  1. Load train/val/test splits from corpus/samanantar/
  2. Compute baseline BLEU with the current inference model
  3. Fine-tune with LoRA (PEFT) — saves adapter weights
  4. Evaluate BLEU after training
  5. Merge + export to CTranslate2 format (replaces or creates new CT2 model dir)

Usage:
    python scripts/finetune.py                           # defaults
    python scripts/finetune.py --epochs 5 --lr 1e-4
    python scripts/finetune.py --skip-baseline           # faster iteration
    python scripts/finetune.py --export-only             # re-export existing adapter
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune")

CORPUS_DIR = Path("corpus/samanantar")
BASELINE_MODEL_PATH = "models/nllb-600M-ct2"
HF_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEFAULT_OUTPUT_DIR = "models/nllb-600M-finetuned"
DEFAULT_CT2_OUTPUT = "models/nllb-600M-finetuned-ct2"


def _load_corpus_split(split: str) -> tuple[list[str], list[str]]:
    from bn_en_translate.training.corpus import load_corpus_files

    bn = CORPUS_DIR / f"{split}.bn.txt"
    en = CORPUS_DIR / f"{split}.en.txt"
    if not bn.exists():
        logger.error(
            "Corpus split '%s' not found at %s. Run: python scripts/download_corpus.py",
            split, bn,
        )
        sys.exit(1)
    src, tgt = load_corpus_files(bn, en)
    logger.info("Loaded %s split: %d pairs", split, len(src))
    return src, tgt


def _baseline_bleu(test_src: list[str], test_tgt: list[str]) -> float:
    """Compute BLEU using the existing CTranslate2 inference model."""
    from bn_en_translate.config import ModelConfig, PipelineConfig
    from bn_en_translate.models.factory import get_translator
    from bn_en_translate.training.trainer import compute_corpus_bleu

    logger.info("Computing baseline BLEU on %d test pairs …", len(test_src))

    if not Path(BASELINE_MODEL_PATH).exists():
        logger.warning(
            "Baseline CT2 model not found at %s — skipping baseline",
            BASELINE_MODEL_PATH,
        )
        return 0.0

    cfg = PipelineConfig(
        model=ModelConfig(
            model_name="nllb-600M",
            model_path=BASELINE_MODEL_PATH,
            device="auto",
        )
    )
    translator = get_translator(cfg)
    with translator:
        hypotheses = translator.translate(
            test_src, src_lang="ben_Beng", tgt_lang="eng_Latn"
        )

    bleu = compute_corpus_bleu(hypotheses, test_tgt)
    logger.info("Baseline BLEU: %.2f", bleu)
    return bleu


def _run_finetune(
    train_src: list[str],
    train_tgt: list[str],
    val_src: list[str],
    val_tgt: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    import torch

    from bn_en_translate.config import FineTuneConfig, ModelConfig
    from bn_en_translate.training.trainer import NLLBFineTuner

    model_cfg = ModelConfig(
        model_name="nllb-600M",
        model_path=HF_MODEL_ID,
        device="auto",
        compute_type="float16",
        src_lang="ben_Beng",
        tgt_lang="eng_Latn",
    )

    fp16 = torch.cuda.is_available()
    ft_cfg = FineTuneConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=max(50, len(train_src) // (args.train_batch_size * args.grad_accum) // 10),
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        output_dir=args.output_dir,
        fp16=fp16,
        save_steps=max(100, len(train_src) // (args.train_batch_size * args.grad_accum)),
        eval_steps=max(100, len(train_src) // (args.train_batch_size * args.grad_accum)),
        logging_steps=50,
    )

    logger.info(
        "Fine-tuning config: lr=%.0e  epochs=%d  batch=%d  lora_r=%d  fp16=%s",
        ft_cfg.learning_rate, ft_cfg.num_epochs, ft_cfg.train_batch_size,
        ft_cfg.lora_r, ft_cfg.fp16,
    )

    tuner = NLLBFineTuner(model_cfg, ft_cfg)
    tuner.load()

    metrics = tuner.train(train_src, train_tgt, val_src, val_tgt)

    return metrics, tuner  # type: ignore[return-value]


def _export_ct2(tuner: object, ct2_output: str) -> None:
    from bn_en_translate.training.trainer import NLLBFineTuner

    assert isinstance(tuner, NLLBFineTuner)
    out = Path(ct2_output)
    logger.info("Exporting fine-tuned model to CT2 at %s …", out)
    tuner.export_ct2(out, quantization="float16")
    logger.info("CT2 export complete: %s", out)


def _post_finetune_bleu(test_src: list[str], test_tgt: list[str], ct2_dir: str) -> float:
    """Compute BLEU using the newly exported CT2 model."""
    from bn_en_translate.config import ModelConfig, PipelineConfig
    from bn_en_translate.models.factory import get_translator
    from bn_en_translate.training.trainer import compute_corpus_bleu

    if not Path(ct2_dir).exists():
        logger.warning("Fine-tuned CT2 model not found at %s", ct2_dir)
        return 0.0

    logger.info("Computing post-fine-tune BLEU on %d test pairs …", len(test_src))
    cfg = PipelineConfig(
        model=ModelConfig(
            model_name="nllb-600M",
            model_path=ct2_dir,
            device="auto",
        )
    )
    translator = get_translator(cfg)
    with translator:
        hypotheses = translator.translate(
            test_src, src_lang="ben_Beng", tgt_lang="eng_Latn"
        )

    bleu = compute_corpus_bleu(hypotheses, test_tgt)
    logger.info("Post-fine-tune BLEU: %.2f", bleu)
    return bleu


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune NLLB-600M on Bengali-English corpus")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ct2-output", default=DEFAULT_CT2_OUTPUT)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline BLEU computation (faster)")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, just re-export existing adapter to CT2")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip CT2 export after training")
    parser.add_argument("--max-train-pairs", type=int, default=None,
                        help="Cap training corpus (e.g. 500 for a quick CPU demo)")
    args = parser.parse_args()

    # Load corpus splits
    train_src, train_tgt = _load_corpus_split("train")
    val_src, val_tgt = _load_corpus_split("val")
    test_src, test_tgt = _load_corpus_split("test")

    # Baseline BLEU
    baseline_bleu = 0.0
    if not args.skip_baseline and not args.export_only:
        baseline_bleu = _baseline_bleu(test_src, test_tgt)

    if args.export_only:
        # Just re-export existing adapter
        from bn_en_translate.config import FineTuneConfig, ModelConfig
        from bn_en_translate.training.trainer import NLLBFineTuner

        model_cfg = ModelConfig(model_path=HF_MODEL_ID, device="auto")
        ft_cfg = FineTuneConfig(output_dir=args.output_dir)
        tuner = NLLBFineTuner(model_cfg, ft_cfg)
        tuner.load()
        _export_ct2(tuner, args.ct2_output)
        tuner.unload()
        return

    # Optionally cap training size (useful for CPU runs / quick demos)
    if args.max_train_pairs and len(train_src) > args.max_train_pairs:
        logger.info("Capping training corpus to %d pairs (from %d)", args.max_train_pairs, len(train_src))
        train_src = train_src[: args.max_train_pairs]
        train_tgt = train_tgt[: args.max_train_pairs]

    # Fine-tune
    metrics, tuner = _run_finetune(train_src, train_tgt, val_src, val_tgt, args)

    # Export to CT2
    if not args.no_export:
        _export_ct2(tuner, args.ct2_output)

    tuner.unload()

    # Post fine-tune BLEU
    post_bleu = 0.0
    if not args.no_export:
        post_bleu = _post_finetune_bleu(test_src, test_tgt, args.ct2_output)

    # Summary
    print("\n" + "=" * 60)
    print("Fine-tuning complete")
    print(f"  Train loss:        {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Val BLEU (during): {metrics.get('eval_bleu', 0.0):.2f}")
    if not args.skip_baseline:
        print(f"  Baseline BLEU:     {baseline_bleu:.2f}")
    if not args.no_export:
        print(f"  Post FT BLEU:      {post_bleu:.2f}")
        if baseline_bleu > 0:
            delta = post_bleu - baseline_bleu
            print(f"  Delta:             {delta:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
