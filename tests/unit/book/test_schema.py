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


def test_block_source_attributes_are_deeply_immutable() -> None:
    attrs = {"format": {"flags": ["keep", "order"]}}
    block = BookBlock.create(
        block_id="c0001-b000001",
        chapter_id="c0001",
        ordinal=1,
        kind=BlockKind.PARAGRAPH,
        source_text="পাঠ।",
        attrs=attrs,
    )

    # Freezing must detach nested source metadata from the caller as well as
    # prevent mutation through the block itself.
    attrs["format"]["flags"].append("caller-change")
    assert block.attrs["format"]["flags"] == ("keep", "order")
    with pytest.raises(TypeError):
        block.attrs["format"]["new"] = True  # type: ignore[index]
    with pytest.raises(AttributeError):
        block.attrs["format"]["flags"].append("block-change")  # type: ignore[attr-defined]


def test_block_runs_and_chapter_membership_are_detached_as_tuples() -> None:
    runs = [InlineRun("পাঠ।")]
    block = BookBlock.create(
        block_id="c0001-b000001",
        chapter_id="c0001",
        ordinal=1,
        kind=BlockKind.PARAGRAPH,
        source_text="পাঠ।",
        runs=runs,  # type: ignore[arg-type]
    )
    runs.append(InlineRun("পরের পাঠ।"))
    chapter = Chapter("c0001", 1, None, [block.block_id])  # type: ignore[arg-type]

    assert block.runs == (InlineRun("পাঠ।"),)
    assert chapter.block_ids == (block.block_id,)
