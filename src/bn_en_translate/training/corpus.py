"""Corpus loading, filtering, splitting, and saving utilities."""

from __future__ import annotations

import random
from pathlib import Path


def load_corpus_files(src_path: Path, tgt_path: Path) -> tuple[list[str], list[str]]:
    """Load parallel corpus from two UTF-8 text files (one sentence per line).

    Empty lines are skipped from both files together.
    Raises ValueError if the non-empty line counts differ.
    """
    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Corpus files are misaligned: {src_path} has {len(src_lines)} lines "
            f"but {tgt_path} has {len(tgt_lines)} lines"
        )

    src_out, tgt_out = [], []
    for s, t in zip(src_lines, tgt_lines):
        s, t = s.strip(), t.strip()
        if s and t:
            src_out.append(s)
            tgt_out.append(t)

    return src_out, tgt_out


def save_corpus_files(
    src: list[str],
    tgt: list[str],
    src_path: Path,
    tgt_path: Path,
) -> None:
    """Save parallel corpus to two UTF-8 text files."""
    src_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text("\n".join(src) + "\n", encoding="utf-8")
    tgt_path.write_text("\n".join(tgt) + "\n", encoding="utf-8")


def filter_corpus(
    src: list[str],
    tgt: list[str],
    min_chars: int = 10,
    max_chars: int = 500,
) -> tuple[list[str], list[str]]:
    """Remove sentence pairs where either side is too short or too long."""
    filtered_src, filtered_tgt = [], []
    for s, t in zip(src, tgt):
        if min_chars <= len(s) <= max_chars and min_chars <= len(t) <= max_chars:
            filtered_src.append(s)
            filtered_tgt.append(t)
    return filtered_src, filtered_tgt


def split_corpus(
    src: list[str],
    tgt: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[
    tuple[list[str], list[str]],
    tuple[list[str], list[str]],
    tuple[list[str], list[str]],
]:
    """Randomly split a parallel corpus into train / val / test.

    Returns three (src, tgt) tuples: (train, val, test).
    The test split gets the remainder after train and val.
    """
    rng = random.Random(seed)
    indices = list(range(len(src)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _select(idx: list[int]) -> tuple[list[str], list[str]]:
        return [src[i] for i in idx], [tgt[i] for i in idx]

    return _select(train_idx), _select(val_idx), _select(test_idx)
