"""Download Samanantar Bengali-English corpus and create train/val/test splits.

Source: ai4bharat/samanantar (HuggingFace datasets)
Language pair: Bengali (bn) ↔ English (en)

Usage:
    python scripts/download_corpus.py                    # 10 000 pairs
    python scripts/download_corpus.py --size 50000       # larger set
    python scripts/download_corpus.py --force            # re-download / re-split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to path so we can import bn_en_translate without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bn_en_translate.training.corpus import (
    filter_corpus,
    load_corpus_files,
    save_corpus_files,
    split_corpus,
)

CORPUS_DIR = Path("corpus")
SAMANANTAR_SPLIT_DIR = CORPUS_DIR / "samanantar"

# Output files per split
SPLITS = {
    "train": ("train.bn.txt", "train.en.txt"),
    "val": ("val.bn.txt", "val.en.txt"),
    "test": ("test.bn.txt", "test.en.txt"),
}


def _check_splits_exist() -> bool:
    return all(
        (SAMANANTAR_SPLIT_DIR / bn).exists() and (SAMANANTAR_SPLIT_DIR / en).exists()
        for bn, en in SPLITS.values()
    )


def download_samanantar(n: int = 10_000, force: bool = False, seed: int = 42) -> None:
    """Download n Bengali-English pairs from Samanantar and create splits."""
    if _check_splits_exist() and not force:
        print(f"Splits already exist at {SAMANANTAR_SPLIT_DIR}/. Use --force to re-download.")
        _print_split_stats()
        return

    print(f"Downloading up to {n:,} Bengali-English pairs from ai4bharat/samanantar …")
    print("(First run downloads ~500 MB; subsequent runs use HuggingFace cache)")

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install 'datasets>=3.0.0'")
        sys.exit(1)

    # Samanantar: config name = language code, split = "train"
    # Each row: {'idx': int, 'src': Bengali str, 'tgt': English str}
    dataset = load_dataset(
        "ai4bharat/samanantar",
        "bn",
        split="train",
        trust_remote_code=True,
    )

    total = len(dataset)
    print(f"  Dataset loaded: {total:,} pairs available")

    # Sample n rows (deterministic shuffle via seed)
    import random
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    selected = indices[:n]

    print(f"  Sampling {len(selected):,} pairs …")
    src_raw = [dataset[i]["src"] for i in selected]
    tgt_raw = [dataset[i]["tgt"] for i in selected]

    # Filter: remove very short or very long sentences
    src_filt, tgt_filt = filter_corpus(src_raw, tgt_raw, min_chars=10, max_chars=400)
    print(f"  After filtering: {len(src_filt):,} pairs (removed {len(src_raw) - len(src_filt):,})")

    # Create train/val/test splits (80/10/10)
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = split_corpus(
        src_filt, tgt_filt, train_ratio=0.8, val_ratio=0.1, seed=seed
    )

    SAMANANTAR_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, sizes, srcs, tgts in [
        ("train", len(train_src), train_src, train_tgt),
        ("val", len(val_src), val_src, val_tgt),
        ("test", len(test_src), test_src, test_tgt),
    ]:
        bn_file, en_file = SPLITS[split_name]
        save_corpus_files(
            srcs, tgts,
            SAMANANTAR_SPLIT_DIR / bn_file,
            SAMANANTAR_SPLIT_DIR / en_file,
        )
        print(f"  Saved {split_name}: {sizes:,} pairs → {SAMANANTAR_SPLIT_DIR / bn_file}")

    print(f"\nDone. Splits saved to {SAMANANTAR_SPLIT_DIR}/")
    _print_split_stats()


def _print_split_stats() -> None:
    for split_name, (bn_file, en_file) in SPLITS.items():
        bn_path = SAMANANTAR_SPLIT_DIR / bn_file
        if bn_path.exists():
            n = sum(1 for line in bn_path.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"  {split_name:6s}: {n:,} pairs  ({bn_path})")


def verify_splits() -> None:
    """Verify that all splits are aligned and non-empty."""
    ok = True
    for split_name, (bn_file, en_file) in SPLITS.items():
        bn_path = SAMANANTAR_SPLIT_DIR / bn_file
        en_path = SAMANANTAR_SPLIT_DIR / en_file
        try:
            src, tgt = load_corpus_files(bn_path, en_path)
            print(f"  {split_name:6s}: ✓  {len(src):,} aligned pairs")
        except (FileNotFoundError, ValueError) as e:
            print(f"  {split_name:6s}: ✗  {e}")
            ok = False
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Samanantar Bengali-English corpus and create train/val/test splits"
    )
    parser.add_argument(
        "--size", type=int, default=10_000,
        help="Number of sentence pairs to sample (default: 10 000)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and re-split even if splits already exist"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling and splitting"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing splits (alignment check) without downloading"
    )
    args = parser.parse_args()

    if args.verify:
        verify_splits()
    else:
        download_samanantar(n=args.size, force=args.force, seed=args.seed)


if __name__ == "__main__":
    main()
