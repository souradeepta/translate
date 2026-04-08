"""Download and convert models to CTranslate2 INT8 format."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODELS: dict[str, dict[str, str]] = {
    "nllb-600M": {
        "hf_id": "facebook/nllb-200-distilled-600M",
        "output_dir": "models/nllb-600M-ct2",
        "quantization": "float16",
        "type": "nllb",
    },
    "nllb-1.3B": {
        "hf_id": "facebook/nllb-200-distilled-1.3B",
        "output_dir": "models/nllb-1.3B-ct2",
        "quantization": "float16",
        "type": "nllb",
    },
    "indicTrans2-1B": {
        "hf_id": "ai4bharat/indictrans2-indic-en-1B",
        "output_dir": "models/indicTrans2-1B-ct2",
        "quantization": "float16",
        "type": "indictrans2",
    },
    "madlad-3b": {
        "hf_id": "google/madlad400-3b-mt",
        "output_dir": "models/madlad-3b-hf",
        "quantization": "float16",
        "type": "hf_only",
    },
    "seamless-medium": {
        "hf_id": "facebook/seamless-m4t-v2-large",
        "output_dir": "models/seamless-medium-hf",
        "quantization": "float16",
        "type": "hf_only",
    },
}


def download_and_convert(model_name: str, force: bool = False) -> None:
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    cfg = MODELS[model_name]
    output_dir = Path(cfg["output_dir"])

    if output_dir.exists() and not force:
        print(f"Model already exists at {output_dir}. Use --force to re-convert.")
        return

    hf_id = cfg["hf_id"]
    model_type = cfg.get("type", "nllb")

    if model_type == "hf_only":
        print(f"Pre-downloading {hf_id} to HF cache (no CT2 conversion)...")
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
            local = snapshot_download(hf_id, local_dir=str(output_dir))
            print(f"Done. Model saved to: {local}")
        except Exception as e:
            print(f"ERROR: download failed: {e}")
            sys.exit(1)
        return

    print(f"Converting {hf_id} → {output_dir} ({cfg['quantization']})...")
    print("This downloads the model from HuggingFace Hub first (1–3 GB).")

    cmd = [
        sys.executable, "-m", "ctranslate2.tools.transformers",
        "--model", hf_id,
        "--output_dir", str(output_dir),
        "--quantization", cfg["quantization"],
        "--force",
    ]

    # Try the module-based approach first (most reliable cross-version)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        # Fall back to the CLI entry point
        cmd2 = [
            "ct2-transformers-converter",
            "--model", hf_id,
            "--output_dir", str(output_dir),
            "--quantization", cfg["quantization"],
            "--force",
        ]
        result2 = subprocess.run(cmd2, check=False)
        if result2.returncode != 0:
            print("ERROR: conversion failed. Check that ctranslate2 is installed and the model ID is correct.")
            sys.exit(1)

    # Copy the SentencePiece tokenizer into the CT2 model dir if not already there
    spm_dest = output_dir / "sentencepiece.bpe.model"
    if not spm_dest.exists():
        _copy_spm_from_cache(hf_id, spm_dest)

    print(f"Done. Model saved to: {output_dir}")
    print(f"  Contents: {sorted(p.name for p in output_dir.iterdir())}")


def _copy_spm_from_cache(hf_id: str, dest: Path) -> None:
    """Find the SPM model in the HF cache and copy it to the CT2 output dir."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
        local = snapshot_download(hf_id, ignore_patterns=["*.bin", "*.safetensors", "*.pt"])
        candidates = list(Path(local).glob("*.model"))
        if not candidates:
            candidates = list(Path(local).glob("sentencepiece*"))
        if candidates:
            shutil.copy(candidates[0], dest)
            print(f"  Copied SPM tokenizer: {candidates[0].name} → {dest}")
        else:
            print(f"  Warning: SPM tokenizer not found in {local}")
    except Exception as e:
        print(f"  Warning: could not copy SPM tokenizer: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert translation models to CTranslate2 INT8")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--force", action="store_true", help="Re-convert even if output exists")
    args = parser.parse_args()
    download_and_convert(args.model, args.force)


if __name__ == "__main__":
    main()
