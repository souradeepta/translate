from __future__ import annotations

import pytest

from bn_en_translate.book.project import BookProject
from bn_en_translate.book.schema import BlockKind, BookBlock, BookDocument, BookMetadata, Chapter


def _document() -> BookDocument:
    block = BookBlock.create(
        block_id="c0001-b000001",
        chapter_id="c0001",
        ordinal=1,
        kind=BlockKind.PARAGRAPH,
        source_text="পাঠ।",
    )
    return BookDocument(
        document_id="doc",
        metadata=BookMetadata(),
        chapters=(Chapter("c0001", 1, None, (block.block_id,)),),
        blocks=(block,),
    )


def test_project_creates_and_reopens_document_and_state(tmp_path) -> None:
    root = tmp_path / "project"
    created = BookProject.create(root, _document())
    reopened = BookProject.open(root)
    assert reopened.document() == created.document()
    with reopened.store() as store:
        assert store.get_unit("c0001-b000001") is not None


def test_project_refuses_nonempty_directory(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        BookProject.create(root, _document())
