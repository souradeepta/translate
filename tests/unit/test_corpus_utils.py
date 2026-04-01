"""Tests for corpus loading, validation, and splitting utilities."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Tests for corpus split logic (pure functions — no downloads)
# ---------------------------------------------------------------------------

def test_split_ratios_sum_to_one() -> None:
    """Train/val/test ratios must sum to 1.0."""
    from bn_en_translate.training.corpus import split_corpus

    src = [f"Bengali sentence {i}" for i in range(100)]
    tgt = [f"English sentence {i}" for i in range(100)]
    train, val, test = split_corpus(src, tgt, train_ratio=0.8, val_ratio=0.1, seed=42)

    assert len(train[0]) + len(val[0]) + len(test[0]) == 100
    assert len(train[1]) + len(val[1]) + len(test[1]) == 100


def test_split_preserves_alignment() -> None:
    """Each split must keep src/tgt aligned (same index → same pair)."""
    from bn_en_translate.training.corpus import split_corpus

    src = [f"bn_{i}" for i in range(50)]
    tgt = [f"en_{i}" for i in range(50)]
    train, val, test = split_corpus(src, tgt, train_ratio=0.8, val_ratio=0.1, seed=99)

    for s, t in zip(train[0], train[1]):
        assert s.replace("bn_", "") == t.replace("en_", "")
    for s, t in zip(val[0], val[1]):
        assert s.replace("bn_", "") == t.replace("en_", "")
    for s, t in zip(test[0], test[1]):
        assert s.replace("bn_", "") == t.replace("en_", "")


def test_split_is_reproducible_with_same_seed() -> None:
    from bn_en_translate.training.corpus import split_corpus

    src = [str(i) for i in range(200)]
    tgt = [str(i) for i in range(200)]
    a, _, _ = split_corpus(src, tgt, seed=7)
    b, _, _ = split_corpus(src, tgt, seed=7)
    assert a[0] == b[0]


def test_split_differs_with_different_seeds() -> None:
    from bn_en_translate.training.corpus import split_corpus

    src = [str(i) for i in range(200)]
    tgt = [str(i) for i in range(200)]
    a, _, _ = split_corpus(src, tgt, seed=1)
    b, _, _ = split_corpus(src, tgt, seed=2)
    assert a[0] != b[0]


def test_split_proportions_are_approximately_correct() -> None:
    from bn_en_translate.training.corpus import split_corpus

    src = list(range(1000))
    tgt = list(range(1000))
    train, val, test = split_corpus(src, tgt, train_ratio=0.8, val_ratio=0.1, seed=0)  # type: ignore[arg-type]

    assert 750 <= len(train[0]) <= 850
    assert 80 <= len(val[0]) <= 120
    assert 80 <= len(test[0]) <= 120


# ---------------------------------------------------------------------------
# Tests for corpus file loading
# ---------------------------------------------------------------------------

def test_load_corpus_from_files(tmp_path: Path) -> None:
    from bn_en_translate.training.corpus import load_corpus_files

    bn = tmp_path / "test.bn.txt"
    en = tmp_path / "test.en.txt"
    bn.write_text("বাক্য এক\nবাক্য দুই\n", encoding="utf-8")
    en.write_text("Sentence one\nSentence two\n", encoding="utf-8")

    src, tgt = load_corpus_files(bn, en)
    assert src == ["বাক্য এক", "বাক্য দুই"]
    assert tgt == ["Sentence one", "Sentence two"]


def test_load_corpus_strips_whitespace(tmp_path: Path) -> None:
    from bn_en_translate.training.corpus import load_corpus_files

    bn = tmp_path / "test.bn.txt"
    en = tmp_path / "test.en.txt"
    bn.write_text("  বাক্য  \n", encoding="utf-8")
    en.write_text("  Sentence  \n", encoding="utf-8")

    src, tgt = load_corpus_files(bn, en)
    assert src == ["বাক্য"]
    assert tgt == ["Sentence"]


def test_load_corpus_rejects_misaligned_files(tmp_path: Path) -> None:
    from bn_en_translate.training.corpus import load_corpus_files

    bn = tmp_path / "a.bn.txt"
    en = tmp_path / "a.en.txt"
    bn.write_text("line1\nline2\n", encoding="utf-8")
    en.write_text("line1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="misaligned"):
        load_corpus_files(bn, en)


def test_load_corpus_skips_empty_lines(tmp_path: Path) -> None:
    from bn_en_translate.training.corpus import load_corpus_files

    bn = tmp_path / "b.bn.txt"
    en = tmp_path / "b.en.txt"
    bn.write_text("line1\n\nline3\n", encoding="utf-8")
    en.write_text("sent1\n\nsent3\n", encoding="utf-8")

    src, tgt = load_corpus_files(bn, en)
    assert src == ["line1", "line3"]
    assert tgt == ["sent1", "sent3"]


# ---------------------------------------------------------------------------
# Tests for corpus filtering
# ---------------------------------------------------------------------------

def test_filter_corpus_removes_too_short(tmp_path: Path) -> None:
    from bn_en_translate.training.corpus import filter_corpus

    src = ["hi", "Hello world how are you", "ok"]
    tgt = ["হাই", "হ্যালো বিশ্ব আপনি কেমন আছেন", "ঠিক আছে"]
    fsrc, ftgt = filter_corpus(src, tgt, min_chars=5)

    assert "Hello world how are you" in fsrc
    assert "হ্যালো বিশ্ব আপনি কেমন আছেন" in ftgt
    assert "hi" not in fsrc
    assert "ok" not in fsrc


def test_filter_corpus_removes_too_long() -> None:
    from bn_en_translate.training.corpus import filter_corpus

    src = ["short", "x" * 600]
    tgt = ["ছোট", "y" * 600]
    fsrc, ftgt = filter_corpus(src, tgt, min_chars=1, max_chars=500)

    assert fsrc == ["short"]
    assert ftgt == ["ছোট"]


def test_filter_corpus_preserves_alignment() -> None:
    from bn_en_translate.training.corpus import filter_corpus

    src = ["a" * 10, "b" * 10, "c" * 3]
    tgt = ["x" * 10, "y" * 10, "z" * 3]
    fsrc, ftgt = filter_corpus(src, tgt, min_chars=5)

    assert len(fsrc) == len(ftgt)
    for s, t in zip(fsrc, ftgt):
        assert s[0] != "c"
        assert t[0] != "z"


# ---------------------------------------------------------------------------
# Tests for corpus saving
# ---------------------------------------------------------------------------

def test_save_corpus_files_round_trip(tmp_path: Path) -> None:
    from bn_en_translate.training.corpus import load_corpus_files, save_corpus_files

    src = ["বাক্য এক", "বাক্য দুই"]
    tgt = ["Sentence one", "Sentence two"]
    bn_path = tmp_path / "out.bn.txt"
    en_path = tmp_path / "out.en.txt"

    save_corpus_files(src, tgt, bn_path, en_path)
    loaded_src, loaded_tgt = load_corpus_files(bn_path, en_path)

    assert loaded_src == src
    assert loaded_tgt == tgt
