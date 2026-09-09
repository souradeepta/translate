from __future__ import annotations

from bn_en_translate.book.schema import BlockKind, BookBlock, BookDocument, BookMetadata, Chapter
from bn_en_translate.book.store import BookStore


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


def test_store_claims_and_selects_an_append_only_attempt(tmp_path) -> None:
    document = _document()
    with BookStore(tmp_path / "state.sqlite3") as store:
        store.register_document(document)
        unit = store.claim_pending("worker")
        assert unit is not None
        attempt_id = store.add_attempt(
            block_id=unit["block_id"],
            stage="draft",
            source_hash=unit["source_hash"],
            config_hash="config",
            target_text="Translation.",
            status="ok",
            dependencies={("source", unit["block_id"]): unit["source_hash"]},
        )
        store.select_attempt(unit["block_id"], attempt_id, "drafted")
        selected = store.get_unit(unit["block_id"])
        assert selected is not None
        assert selected["status"] == "drafted"
        assert selected["selected_attempt_id"] == attempt_id


def test_changed_dependency_marks_generated_selection_stale(tmp_path) -> None:
    document = _document()
    with BookStore(tmp_path / "state.sqlite3") as store:
        store.register_document(document)
        unit = store.claim_pending("worker")
        assert unit is not None
        attempt_id = store.add_attempt(
            block_id=unit["block_id"],
            stage="draft",
            source_hash=unit["source_hash"],
            config_hash="config",
            target_text="Translation.",
            status="ok",
            dependencies={("glossary", "book"): "old"},
        )
        store.select_attempt(unit["block_id"], attempt_id, "drafted")
        assert store.mark_stale_for_dependency("glossary", "book", "new") == 1
        assert store.get_unit(unit["block_id"])["status"] == "stale"
