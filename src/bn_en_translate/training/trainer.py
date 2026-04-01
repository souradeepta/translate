"""LoRA fine-tuner for NLLB-200 seq2seq models.

Workflow:
    tuner = NLLBFineTuner(model_config, finetune_config)
    tuner.load()                              # loads HF model + wraps with LoRA
    tuner.train(train_src, train_tgt,
                val_src, val_tgt)             # fine-tunes, saves checkpoints
    tuner.export_ct2(output_dir)              # merges LoRA + converts to CT2
    tuner.unload()
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sacrebleu

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BLEU helper (no model needed)
# ---------------------------------------------------------------------------

def compute_corpus_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score (SacreBLEU).

    Args:
        hypotheses: list of model output strings
        references: list of reference translation strings (same length)

    Returns:
        BLEU score as a float in [0, 100]
    """
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return float(result.score)


# ---------------------------------------------------------------------------
# Fine-tuner
# ---------------------------------------------------------------------------

class NLLBFineTuner:
    """LoRA fine-tuner for NLLB-200 distilled models.

    Uses HuggingFace Seq2SeqTrainer + PEFT LoRA adapters.
    After training, merges adapters and exports to CTranslate2 float16.
    """

    def __init__(self, model_config: Any, finetune_config: Any) -> None:
        self.model_config = model_config
        self.finetune_config = finetune_config
        self._model: Any = None
        self._tokenizer: Any = None
        self._peft_model: Any = None
        self._use_cuda: bool = False  # set in load() after probe

    @property
    def is_loaded(self) -> bool:
        return self._peft_model is not None

    def load(self) -> None:
        """Load base model from HuggingFace and wrap with LoRA adapters."""
        import peft
        import torch
        import transformers

        model_id = self.model_config.model_path
        device = self.model_config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading tokenizer from %s", model_id)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, src_lang=self.model_config.src_lang
        )

        # Always load on CPU first — avoids "no kernel image" CUDA error on sm_120
        # when PyTorch tries to initialise weights on device before JIT compilation.
        logger.info("Loading base model from %s (cpu first, then → %s)", model_id, device)
        self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float32,  # load in float32 on CPU to avoid CUDA device issues
        )

        # Enable gradient checkpointing to reduce VRAM (encoder + decoder)
        self._model.gradient_checkpointing_enable()

        # Wrap with LoRA before moving to GPU so no CUDA ops happen on base weights
        lora_cfg = peft.LoraConfig(
            r=self.finetune_config.lora_r,
            lora_alpha=self.finetune_config.lora_alpha,
            lora_dropout=self.finetune_config.lora_dropout,
            target_modules=self.finetune_config.lora_target_modules,
            bias="none",
            task_type=peft.TaskType.SEQ_2_SEQ_LM,
        )
        self._peft_model = peft.get_peft_model(self._model, lora_cfg)

        # Detect whether CUDA is usable for PyTorch training.
        # Use a realistic probe (ne comparison, not just matmul) to catch
        # architectures where matmul works via PTX JIT but element-wise ops don't.
        use_cuda = False
        if device != "cpu" and torch.cuda.is_available():
            try:
                _probe = torch.tensor([1, -100, 3]).cuda().ne(-100).sum()
                use_cuda = True
            except RuntimeError as exc:
                logger.warning(
                    "CUDA probe failed (%s). "
                    "Training on CPU. Install PyTorch cu128 for GPU training on sm_120.",
                    exc,
                )

        self._use_cuda = use_cuda
        if use_cuda:
            if self.finetune_config.fp16:
                self._peft_model = self._peft_model.half()
            self._peft_model = self._peft_model.to("cuda")
            logger.info("Model moved to CUDA (fp16=%s)", self.finetune_config.fp16)
        else:
            logger.info("Training on CPU (PyTorch sm_120 not supported in cu124)")
        try:
            trainable, total = self._peft_model.get_nb_trainable_parameters()
            logger.info(
                "LoRA applied: %d trainable params / %d total (%.2f%%)",
                trainable, total, 100 * trainable / max(total, 1),
            )
        except (TypeError, ValueError):
            logger.info("LoRA applied (param count unavailable in this PEFT version)")

    def unload(self) -> None:
        """Free GPU memory."""
        self._peft_model = None
        self._model = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load() before training or exporting."
            )

    def train(
        self,
        train_src: list[str],
        train_tgt: list[str],
        val_src: list[str],
        val_tgt: list[str],
    ) -> dict[str, float]:
        """Fine-tune with LoRA on the given parallel corpus.

        Returns a dict with keys: train_loss, eval_bleu (after final epoch).
        """
        self._require_loaded()

        import torch
        import transformers

        from bn_en_translate.training.dataset import BengaliEnglishDataset

        ft = self.finetune_config
        tokenizer = self._tokenizer

        train_dataset = BengaliEnglishDataset(
            train_src, train_tgt, tokenizer,
            max_source_length=ft.max_source_length,
            max_target_length=ft.max_target_length,
        )
        val_dataset = BengaliEnglishDataset(
            val_src, val_tgt, tokenizer,
            max_source_length=ft.max_source_length,
            max_target_length=ft.max_target_length,
        )

        # Use fp16 only if model is actually on CUDA
        device_str = str(next(self._peft_model.parameters()).device)
        on_cuda = device_str.startswith("cuda")
        use_fp16 = ft.fp16 and on_cuda
        if not on_cuda:
            self._peft_model = self._peft_model.float()  # ensure float32 on CPU

        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir=ft.output_dir,
            num_train_epochs=ft.num_epochs,
            per_device_train_batch_size=ft.train_batch_size,
            per_device_eval_batch_size=ft.eval_batch_size,
            gradient_accumulation_steps=ft.gradient_accumulation_steps,
            warmup_steps=ft.warmup_steps,
            weight_decay=ft.weight_decay,
            max_grad_norm=ft.max_grad_norm,
            learning_rate=ft.learning_rate,
            fp16=use_fp16,
            eval_strategy="steps",
            eval_steps=ft.eval_steps,
            save_strategy="steps",
            save_steps=ft.save_steps,
            logging_steps=ft.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=0,  # avoid multiprocessing with CUDA
            use_cpu=not self._use_cuda,  # force CPU when sm_120 not supported by this PyTorch build
        )

        trainer = transformers.Trainer(
            model=self._peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        logger.info(
            "Starting training: %d train pairs, %d val pairs, %d epochs",
            len(train_src), len(val_src), ft.num_epochs,
        )
        train_result = trainer.train()
        metrics: dict[str, float] = {"train_loss": train_result.training_loss}

        # BLEU on validation set after training
        logger.info("Computing post-training BLEU on validation set …")
        val_bleu = self._eval_bleu(val_src, val_tgt)
        metrics["eval_bleu"] = val_bleu
        logger.info("Validation BLEU: %.2f", val_bleu)

        # Save adapter weights
        output_path = Path(ft.output_dir)
        self._peft_model.save_pretrained(str(output_path / "adapter"))
        tokenizer.save_pretrained(str(output_path / "tokenizer"))
        logger.info("Adapter weights saved to %s/adapter/", ft.output_dir)

        return metrics

    def _eval_bleu(self, src_texts: list[str], ref_texts: list[str]) -> float:
        """Generate translations for src_texts and compute BLEU against ref_texts."""
        import torch

        self._peft_model.eval()
        hypotheses = []
        batch_size = self.finetune_config.eval_batch_size

        for i in range(0, len(src_texts), batch_size):
            batch_src = src_texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch_src,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.finetune_config.max_source_length,
            )
            device = next(self._peft_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self._peft_model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(
                        self.model_config.tgt_lang
                    ),
                    max_new_tokens=self.finetune_config.max_target_length,
                    num_beams=4,
                )
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            hypotheses.extend(decoded)

        return compute_corpus_bleu(hypotheses, ref_texts)

    def export_ct2(self, output_dir: Path, quantization: str = "float16") -> None:
        """Merge LoRA adapters into base weights and convert to CTranslate2.

        Steps:
          1. Load adapter from <finetune_config.output_dir>/adapter/
          2. Merge LoRA into base weights (PEFT merge_and_unload)
          3. Save merged HF model to a temp directory
          4. Run ct2-transformers-converter
          5. Copy SPM tokenizer into CT2 output dir
        """
        self._require_loaded()

        import tempfile

        import peft

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Merging LoRA adapters into base model …")
        merged_model = self._peft_model.merge_and_unload()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            merged_model.save_pretrained(str(tmp_path))
            self._tokenizer.save_pretrained(str(tmp_path))
            logger.info("Merged model saved to temp dir %s", tmp_path)

            # Convert to CTranslate2
            logger.info("Converting to CTranslate2 (%s) …", quantization)
            cmd = [
                sys.executable, "-m", "ctranslate2.tools.transformers",
                "--model", str(tmp_path),
                "--output_dir", str(output_dir),
                "--quantization", quantization,
                "--force",
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                # Try CLI entry point
                cmd2 = [
                    "ct2-transformers-converter",
                    "--model", str(tmp_path),
                    "--output_dir", str(output_dir),
                    "--quantization", quantization,
                    "--force",
                ]
                result2 = subprocess.run(cmd2, check=False)
                if result2.returncode != 0:
                    raise RuntimeError(
                        f"CT2 conversion failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
                    )

            # Copy SPM tokenizer into CT2 dir
            spm_src = tmp_path / "sentencepiece.bpe.model"
            if spm_src.exists():
                shutil.copy(spm_src, output_dir / "sentencepiece.bpe.model")
                logger.info("Copied SPM tokenizer to %s", output_dir)
            else:
                logger.warning("SPM tokenizer not found in merged model dir — copy it manually")

        logger.info("CT2 model exported to %s", output_dir)
