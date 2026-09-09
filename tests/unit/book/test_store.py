from __future__ import annotations

import sqlite3

import pytest

from bn_en_translate.book.schema import BlockKind, BookBlock, BookDocument, BookMetadata, Chapter
from bn_en_translate.book.store import STORE_SCHEMA_VERSION, BookStore


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


def _two_block_document() -> BookDocument:
    first = BookBlock.create(
        block_id="c0001-b000001",
        chapter_id="c0001",
        ordinal=1,
        kind=BlockKind.PARAGRAPH,
        source_text="প্রথম।",
    )
    second = BookBlock.create(
        block_id="c0001-b000002",
        chapter_id="c0001",
        ordinal=2,
        kind=BlockKind.PARAGRAPH,
        source_text="দ্বিতীয়।",
    )
    return BookDocument(
        document_id="doc",
        metadata=BookMetadata(),
        chapters=(Chapter("c0001", 1, None, (first.block_id, second.block_id)),),
        blocks=(first, second),
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


def test_invalidation_is_transitive_and_preserves_approved_selection(tmp_path) -> None:
    document = _two_block_document()
    with BookStore(tmp_path / "state.sqlite3") as store:
        store.register_document(document)
        first = store.claim_pending("worker")
        assert first is not None
        first_attempt = store.add_attempt(
            block_id=first["block_id"],
            stage="draft",
            source_hash=first["source_hash"],
            config_hash="config",
            target_text="First.",
            status="ok",
            dependencies={("glossary", "book"): "old"},
        )
        store.select_attempt(first["block_id"], first_attempt, "drafted")
        second = store.claim_pending("worker")
        assert second is not None
        second_attempt = store.add_attempt(
            block_id=second["block_id"],
            stage="draft",
            source_hash=second["source_hash"],
            config_hash="config",
            target_text="Second.",
            status="ok",
            dependencies={("target", first["block_id"]): "first-target"},
        )
        store.select_attempt(second["block_id"], second_attempt, "drafted")
        store.approve_attempt(second["block_id"], second_attempt)

        assert store.mark_stale_for_dependency("glossary", "book", "new") == 2
        assert store.get_unit(first["block_id"])["status"] == "stale"
        approved = store.get_unit(second["block_id"])
        assert approved is not None
        assert approved["status"] == "approved"
        assert approved["selected_attempt_id"] == second_attempt
        assert approved["approved_attempt_id"] == second_attempt
        assert approved["approval_needs_revalidation"] == 1

        replacement = store.add_attempt(
            block_id=second["block_id"],
            stage="draft",
            source_hash=second["source_hash"],
            config_hash="config",
            target_text="Replacement.",
            status="ok",
        )
        with pytest.raises(ValueError, match="immutable"):
            store.select_attempt(second["block_id"], replacement, "drafted")


def test_store_migrates_legacy_database_with_schema_marker_and_backup(tmp_path) -> None:
    state = tmp_path / "state.sqlite3"
    legacy = sqlite3.connect(state)
    legacy.executescript(
        """
        CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE units (
            block_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', selected_attempt_id INTEGER,
            approved_attempt_id INTEGER, lease_owner TEXT, lease_expires_at TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, block_id TEXT NOT NULL REFERENCES units(block_id),
            stage TEXT NOT NULL, source_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
            target_text TEXT NOT NULL, status TEXT NOT NULL, error_message TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    legacy.commit()
    legacy.close()

    with BookStore(state) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == STORE_SCHEMA_VERSION
        marker = store.connection.execute(
            "SELECT value FROM project_meta WHERE key = 'schema_version'"
        ).fetchone()
        assert marker is not None and marker["value"] == str(STORE_SCHEMA_VERSION)
        columns = {row[1] for row in store.connection.execute("PRAGMA table_info(units)")}
        assert "approval_needs_revalidation" in columns
    assert state.with_suffix(".sqlite3.bak").is_file()
