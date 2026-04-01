"""Tests for BengaliEnglishDataset — training data pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal mock tokenizer (no GPU / no model download)
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(vocab_size: int = 256200) -> MagicMock:
    """Return a mock that behaves like NllbTokenizer for dataset tests."""
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.pad_token_id = 1
    tok.eos_token_id = 2
    tok.bos_token_id = 0

    def _call(texts: list[str] | str, **kwargs: object) -> dict[str, list[list[int]]]:
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        max_len = int(kwargs.get("max_length", 64))
        ids = [[10, 20, 30, tok.eos_token_id, tok.pad_token_id] + [tok.pad_token_id] * (max_len - 5)] * n
        mask = [[1, 1, 1, 1, 1] + [0] * (max_len - 5)] * n
        return {"input_ids": ids, "attention_mask": mask}

    tok.side_effect = _call
    tok.__call__ = _call
    return tok


# ---------------------------------------------------------------------------
# Dataset construction and length
# ---------------------------------------------------------------------------

def test_dataset_length_matches_input() -> None:
    from bn_en_translate.training.dataset import BengaliEnglishDataset

    src = ["বাক্য এক", "বাক্য দুই", "বাক্য তিন"]
    tgt = ["Sentence one", "Sentence two", "Sentence three"]
    tok = _make_mock_tokenizer()

    ds = BengaliEnglishDataset(src, tgt, tok)
    assert len(ds) == 3


def test_dataset_rejects_misaligned_inputs() -> None:
    from bn_en_translate.training.dataset import BengaliEnglishDataset

    tok = _make_mock_tokenizer()
    with pytest.raises(ValueError, match="must have the same length"):
        BengaliEnglishDataset(["a", "b"], ["x"], tok)


def test_dataset_getitem_returns_expected_keys() -> None:
    from bn_en_translate.training.dataset import BengaliEnglishDataset

    src = ["বাক্য এক"]
    tgt = ["Sentence one"]
    tok = _make_mock_tokenizer()

    ds = BengaliEnglishDataset(src, tgt, tok)
    item = ds[0]

    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


def test_dataset_labels_replace_pad_with_ignore_index() -> None:
    """Padding tokens in labels must be -100 so loss ignores them."""
    from bn_en_translate.training.dataset import BengaliEnglishDataset

    src = ["বাক্য এক"]
    tgt = ["Sentence one"]
    tok = _make_mock_tokenizer()

    ds = BengaliEnglishDataset(src, tgt, tok)
    labels = ds[0]["labels"]

    # -100 must appear (mock tokenizer generates padding)
    assert -100 in labels


def test_dataset_getitem_returns_lists_or_tensors() -> None:
    """__getitem__ must return values that can be collated by DataLoader."""
    from bn_en_translate.training.dataset import BengaliEnglishDataset

    src = ["বাক্য এক", "বাক্য দুই"]
    tgt = ["Sentence one", "Sentence two"]
    tok = _make_mock_tokenizer()

    ds = BengaliEnglishDataset(src, tgt, tok)
    item = ds[0]

    # Must be lists or torch Tensors — both work with DataLoader default collate
    for key, val in item.items():
        assert hasattr(val, "__len__"), f"{key} must be a sequence"


def test_dataset_respects_max_source_length() -> None:
    from bn_en_translate.training.dataset import BengaliEnglishDataset

    src = ["a" * 1000]
    tgt = ["b" * 100]
    tok = _make_mock_tokenizer()

    ds = BengaliEnglishDataset(src, tgt, tok, max_source_length=64, max_target_length=64)
    item = ds[0]

    assert len(item["input_ids"]) == 64


def test_dataset_respects_max_target_length() -> None:
    from bn_en_translate.training.dataset import BengaliEnglishDataset

    src = ["a" * 10]
    tgt = ["b" * 1000]
    tok = _make_mock_tokenizer()

    ds = BengaliEnglishDataset(src, tgt, tok, max_source_length=64, max_target_length=32)
    item = ds[0]

    assert len(item["labels"]) == 32


# ---------------------------------------------------------------------------
# Collation helper (used by DataLoader)
# ---------------------------------------------------------------------------

def test_collate_fn_pads_to_same_length() -> None:
    from bn_en_translate.training.dataset import collate_fn

    pad_id = 1
    batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 11, 12]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [20, -100]},
    ]
    result = collate_fn(batch, pad_token_id=pad_id)

    # All sequences in the same key must have the same length
    assert len(result["input_ids"][0]) == len(result["input_ids"][1])
    assert len(result["labels"][0]) == len(result["labels"][1])


def test_collate_fn_pads_labels_with_ignore_index() -> None:
    from bn_en_translate.training.dataset import collate_fn

    batch = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": [10, 11, 12, 13]},
        {"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [20, 21]},
    ]
    result = collate_fn(batch, pad_token_id=0)

    # Shorter labels sequence should be padded with -100
    shorter = result["labels"][1]
    assert -100 in shorter
