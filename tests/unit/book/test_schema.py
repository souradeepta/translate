from __future__ import annotations

import pytest

from bn_en_translate.book.schema import (
    BlockKind,
    BookBlock,
    BookDocument,
    BookMetadata,
    Chapter,
    InlineRun,
    make_block_id,
)


def _document() -> BookDocument:
    title = BookBlock.create(
        block_id=make_block_id(1, 1),
        chapter_id="c0001",
        ordinal=1,
        kind=BlockKind.TITLE,
        source_text="একটি গল্প",
    )
    body = BookBlock.create(
        block_id=make_block_id(1, 2),
        chapter_id="c0001",
        ordinal=2,
        kind=BlockKind.PARAGRAPH,
        source_text="রহিম বলল।",
        runs=(InlineRun("রহিম ", italic=True), InlineRun("বলল।")),
    )
    return BookDocument(
        document_id="fixture",
        metadata=BookMetadata(title="Fixture"),
        chapters=(Chapter("c0001", 1, None, (title.block_id, body.block_id)),),
        blocks=(title, body),
    )


def test_document_validates_unicode_and_inline_runs() -> None:
    document = _document()
    document.validate()
    assert document.blocks[1].source_text == "রহিম বলল।"


def test_tampered_source_hash_is_rejected() -> None:
    document = _document()
    block = document.blocks[0]
    tampered = BookBlock(
        block_id=block.block_id,
        chapter_id=block.chapter_id,
        ordinal=block.ordinal,
        kind=block.kind,
        source_text=block.source_text,
        source_hash="bad",
    )
    with pytest.raises(ValueError, match="source hash"):
        tampered.validate()


def test_document_rejects_missing_or_reordered_block_membership() -> None:
    document = _document()
    invalid = BookDocument(
        document_id=document.document_id,
        metadata=document.metadata,
        chapters=(Chapter("c0001", 1, None, (document.blocks[1].block_id,)),),
        blocks=document.blocks,
    )
    with pytest.raises(ValueError, match="chapter block IDs"):
        invalid.validate()
