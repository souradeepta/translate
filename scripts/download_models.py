"""Download and convert models to CTranslate2 INT8 format."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "nllb-600M": {
        "hf_id": "facebook/nllb-200-distilled-600M",
        "output_dir": "models/nllb-600M-ct2",
        "quantization": "int8",
    },
    "nllb-1.3B": {
        "hf_id": "facebook/nllb-200-distilled-1.3B",
        "output_dir": "models/nllb-1.3B-ct2",
        "quantization": "int8",
    },
    "indicTrans2-1B": {
        "hf_id": "ai4bharat/indictrans2-indic-en-1B",
        "output_dir": "models/indicTrans2-1B-ct2",
        "quantization": "int8",
        # IndicTrans2 uses a custom conversion path
        "note": "Requires IndicTrans2 HuggingFace interface installed",
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

    print(f"Converting {cfg['hf_id']} → {output_dir} (INT8)...")
    print("Note: This will download the model from HuggingFace Hub first (~1-5 GB).")

    cmd = [
        "ct2-opus-mt-convert",
        "--model", cfg["hf_id"],
        "--output", str(output_dir),
        "--quantization", cfg["quantization"],
        "--force",
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nct2-opus-mt-convert failed. Trying generic converter...")
        cmd[0] = "ct2-transformers-converter"
        subprocess.run(cmd, check=True)

    print(f"Done. Model saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert translation models")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--force", action="store_true", help="Re-convert even if output exists")
    args = parser.parse_args()
    download_and_convert(args.model, args.force)


if __name__ == "__main__":
    main()
