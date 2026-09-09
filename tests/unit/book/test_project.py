from __future__ import annotations

import pytest

from bn_en_translate.book.project import BookProject, ReconciliationError
from bn_en_translate.book.schema import (
    BlockKind,
    BookBlock,
    BookDocument,
    BookMetadata,
    Chapter,
    make_block_id,
)


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


def _document_with_blocks(
    source_texts: list[str], *, locators: list[str | None] | None = None
) -> BookDocument:
    blocks = tuple(
        BookBlock.create(
            block_id=make_block_id(1, ordinal),
            chapter_id="c0001",
            ordinal=ordinal,
            kind=BlockKind.PARAGRAPH,
            source_text=source_text,
            attrs=(
                {"source_locator": locators[ordinal - 1]}
                if locators is not None and locators[ordinal - 1] is not None
                else {}
            ),
        )
        for ordinal, source_text in enumerate(source_texts, start=1)
    )
    return BookDocument(
        document_id="doc",
        metadata=BookMetadata(),
        chapters=(Chapter("c0001", 1, None, tuple(block.block_id for block in blocks)),),
        blocks=blocks,
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


def test_reimport_edit_preserves_id_and_marks_only_changed_unit_stale(tmp_path) -> None:
    root = tmp_path / "project"
    BookProject.create(root, _document_with_blocks(["প্রথম।", "দ্বিতীয়।"]))
    project = BookProject.open(root)
    incoming = _document_with_blocks(["প্রথম সম্পাদিত।", "দ্বিতীয়।"])

    before_source = project.source_path.read_bytes()
    report = project.reimport(incoming, dry_run=True)
    assert report.matched == {
        "c0001-b000001": "c0001-b000001",
        "c0001-b000002": "c0001-b000002",
    }
    assert report.inserted == ()
    assert project.source_path.read_bytes() == before_source

    project.reimport(incoming)
    assert [block.block_id for block in project.document().blocks] == [
        "c0001-b000001",
        "c0001-b000002",
    ]
    assert project.document().blocks[0].source_text == "প্রথম সম্পাদিত।"
    with project.store() as store:
        assert store.get_unit("c0001-b000001")["status"] == "stale"
        assert store.get_unit("c0001-b000002")["status"] == "pending"


def test_reimport_insertion_gets_monotonic_id_without_renumbering_later_blocks(tmp_path) -> None:
    root = tmp_path / "project"
    BookProject.create(root, _document_with_blocks(["প্রথম।", "শেষ।"]))
    project = BookProject.open(root)
    incoming = _document_with_blocks(["প্রথম।", "মাঝের নতুন অনুচ্ছেদ।", "শেষ।"])

    report = project.reimport(incoming, dry_run=True)
    assert report.inserted == ("c0001-b000003",)
    assert report.matched["c0001-b000001"] == "c0001-b000001"
    assert report.matched["c0001-b000003"] == "c0001-b000002"

    project.reimport(incoming)
    document = project.document()
    assert [block.block_id for block in document.blocks] == [
        "c0001-b000001",
        "c0001-b000003",
        "c0001-b000002",
    ]
    assert document.blocks[2].source_text == "শেষ।"


def test_ambiguous_reimport_dry_run_reports_without_mutating_project(tmp_path) -> None:
    root = tmp_path / "project"
    BookProject.create(
        root,
        _document_with_blocks(["এক।", "দুই।"], locators=["paragraph-1", "paragraph-1"]),
    )
    project = BookProject.open(root)
    incoming = _document_with_blocks(["পরিবর্তিত।"], locators=["paragraph-1"])
    before_source = project.source_path.read_bytes()
    before_structure = project.structure_path.read_bytes()
    before_state = project.state_path.read_bytes()

    report = project.reimport(incoming, dry_run=True)
    assert report.ambiguous
    assert project.source_path.read_bytes() == before_source
    assert project.structure_path.read_bytes() == before_structure
    assert project.state_path.read_bytes() == before_state

    with pytest.raises(ReconciliationError):
        project.reimport(incoming)
    assert project.source_path.read_bytes() == before_source
    assert project.structure_path.read_bytes() == before_structure
    assert project.state_path.read_bytes() == before_state
