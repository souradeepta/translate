"""PyTorch Dataset for Bengali-English fine-tuning."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class BengaliEnglishDataset(Dataset):  # type: ignore[type-arg]
    """Parallel Bengali-English dataset suitable for seq2seq fine-tuning.

    Each item contains:
        input_ids      — tokenised source (Bengali), padded to max_source_length
        attention_mask — 1 for real tokens, 0 for padding
        labels         — tokenised target (English), padded to max_target_length
                         padding positions are replaced with -100 so the loss
                         ignores them
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        tokenizer: Any,
        src_lang: str = "ben_Beng",
        tgt_lang: str = "eng_Latn",
        max_source_length: int = 256,
        max_target_length: int = 256,
    ) -> None:
        if len(src_texts) != len(tgt_texts):
            raise ValueError(
                f"src_texts and tgt_texts must have the same length, "
                f"got {len(src_texts)} and {len(tgt_texts)}"
            )
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]

        # Tokenise source
        model_inputs = self.tokenizer(
            src,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenise target — newer transformers (≥4.x) use text_target kwarg;
        # older versions used a context manager API.
        if hasattr(self.tokenizer, "as_target_tokenizer"):
            with self.tokenizer.as_target_tokenizer():
                target_enc = self.tokenizer(
                    tgt,
                    max_length=self.max_target_length,
                    padding="max_length",
                    truncation=True,
                )
        else:
            target_enc = self.tokenizer(
                text_target=tgt,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
            )

        labels: list[int] = target_enc["input_ids"]
        if isinstance(labels[0], list):
            labels = labels[0]  # unwrap batch dim if tokenizer returned batched output

        # Replace padding token IDs in labels with -100 (ignored by cross-entropy loss)
        pad_id = self.tokenizer.pad_token_id
        labels = [t if t != pad_id else self.IGNORE_INDEX for t in labels]

        input_ids: list[int] = model_inputs["input_ids"]
        attention_mask: list[int] = model_inputs["attention_mask"]
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(
    batch: list[dict[str, list[int]]],
    pad_token_id: int = 1,
) -> dict[str, list[list[int]]]:
    """Pad a batch of variable-length examples to the same length.

    Labels are padded with -100 (not pad_token_id) so the loss ignores them.
    """
    keys = list(batch[0].keys())
    result: dict[str, list[list[int]]] = {k: [] for k in keys}

    for key in keys:
        seqs = [item[key] for item in batch]
        max_len = max(len(s) for s in seqs)

        pad_val = BengaliEnglishDataset.IGNORE_INDEX if key == "labels" else pad_token_id
        padded = [s + [pad_val] * (max_len - len(s)) for s in seqs]
        result[key] = padded

    return result
