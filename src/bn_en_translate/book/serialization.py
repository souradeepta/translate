"""Deterministic JSON serialization for :mod:`bn_en_translate.book` types."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bn_en_translate.book.schema import (
    SCHEMA_VERSION,
    BlockKind,
    BookBlock,
    BookDocument,
    BookMetadata,
    Chapter,
    InlineRun,
    jsonable,
)

SOURCE_JSONL_VERSION = 1


def _identity_migration(value: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a document already in the current representation."""
    return dict(value)


def _identity_jsonl_migration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of JSONL records already in the current representation."""
    return list(rows)


# Keep migrations as an explicit dispatch table even while v1 is the only
# released schema.  Adding a new representation must add a deliberate function
# here; accepting an unknown version by accident would make immutable source
# records impossible to audit.
_DOCUMENT_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    SCHEMA_VERSION: _identity_migration,
}
_SOURCE_JSONL_MIGRATIONS: dict[
    int, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
] = {
    SOURCE_JSONL_VERSION: _identity_jsonl_migration,
}


def migrate_document_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Migrate a serialized document to the current schema.

    Keeping this dispatch point explicit makes an upgrade auditable.  Version one
    is the first released representation, so it currently needs no transformation.
    Future versions must add a step here instead of silently accepting newer data.
    """
    version = value.get("schema_version")
    if not isinstance(version, int) or version > SCHEMA_VERSION or version < 1:
        raise ValueError(f"unsupported schema version: {version}")
    migration = _DOCUMENT_MIGRATIONS.get(version)
    if migration is None:
        raise ValueError(f"no migration registered for schema version {version}")
    return migration(value)


def migrate_source_jsonl_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dispatch versioned JSONL records through an explicit migration table."""
    if not rows or rows[0].get("record_type") != "header":
        raise ValueError("source JSONL must begin with a header record")
    version = rows[0].get("schema_version")
    if not isinstance(version, int) or version > SOURCE_JSONL_VERSION or version < 1:
        raise ValueError(f"unsupported source JSONL version: {version}")
    migration = _SOURCE_JSONL_MIGRATIONS.get(version)
    if migration is None:
        raise ValueError(f"no migration registered for source JSONL version {version}")
    return migration(rows)


def document_to_dict(document: BookDocument) -> dict[str, Any]:
    document.validate()
    return {
        "blocks": [
            {
                "attrs": jsonable(block.attrs),
                "block_id": block.block_id,
                "chapter_id": block.chapter_id,
                "kind": block.kind.value,
                "ordinal": block.ordinal,
                "runs": [run.__dict__ for run in block.runs],
                "source_hash": block.source_hash,
                "source_text": block.source_text,
            }
            for block in document.blocks
        ],
        "chapters": [
            {
                "block_ids": list(chapter.block_ids),
                "chapter_id": chapter.chapter_id,
                "ordinal": chapter.ordinal,
                "title": chapter.title,
            }
            for chapter in document.chapters
        ],
        "document_id": document.document_id,
        "metadata": jsonable(document.metadata.__dict__),
        "schema_version": document.schema_version,
    }


def document_from_dict(value: dict[str, Any]) -> BookDocument:
    value = migrate_document_dict(value)
    metadata = BookMetadata(**value["metadata"])
    blocks = tuple(
        BookBlock(
            block_id=item["block_id"],
            chapter_id=item["chapter_id"],
            ordinal=item["ordinal"],
            kind=BlockKind(item["kind"]),
            source_text=item["source_text"],
            source_hash=item["source_hash"],
            runs=tuple(InlineRun(**run) for run in item.get("runs", [])),
            attrs=dict(item.get("attrs", {})),
        )
        for item in value["blocks"]
    )
    chapters = tuple(
        Chapter(
            chapter_id=item["chapter_id"],
            ordinal=item["ordinal"],
            title=item.get("title"),
            block_ids=tuple(item["block_ids"]),
        )
        for item in value["chapters"]
    )
    document = BookDocument(
        document_id=value["document_id"], metadata=metadata, chapters=chapters, blocks=blocks
    )
    document.validate()
    return document


def dumps(document: BookDocument) -> str:
    return (
        json.dumps(document_to_dict(document), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def loads(value: str) -> BookDocument:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("book document JSON must be an object")
    return document_from_dict(parsed)


def write_document(document: BookDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(document), encoding="utf-8")
    temporary.replace(path)


def read_document(path: Path) -> BookDocument:
    return loads(path.read_text(encoding="utf-8"))


def document_to_source_jsonl(document: BookDocument) -> str:
    """Serialize immutable source blocks as deterministic versioned JSONL."""
    document.validate()
    header = {
        "chapters": [
            {
                "block_ids": list(chapter.block_ids),
                "chapter_id": chapter.chapter_id,
                "ordinal": chapter.ordinal,
                "title": chapter.title,
            }
            for chapter in document.chapters
        ],
        "document_id": document.document_id,
        "metadata": jsonable(document.metadata.__dict__),
        "record_type": "header",
        "schema_version": SOURCE_JSONL_VERSION,
    }
    records: list[dict[str, Any]] = [header]
    for block in document.blocks:
        records.append(
            {
                "attrs": jsonable(block.attrs),
                "block_id": block.block_id,
                "chapter_id": block.chapter_id,
                "kind": block.kind.value,
                "ordinal": block.ordinal,
                "record_type": "block",
                "runs": [run.__dict__ for run in block.runs],
                "source_hash": block.source_hash,
                "source_text": block.source_text,
            }
        )
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def document_from_source_jsonl(value: str) -> BookDocument:
    """Parse versioned source JSONL and validate all immutable source records."""
    parsed = [json.loads(line) for line in value.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in parsed):
        raise ValueError("source JSONL records must be objects")
    rows = migrate_source_jsonl_rows(parsed)
    header = rows[0]
    if any(row.get("record_type") != "block" for row in rows[1:]):
        raise ValueError("source JSONL contains an unknown record type")
    blocks = tuple(
        BookBlock(
            block_id=item["block_id"],
            chapter_id=item["chapter_id"],
            ordinal=item["ordinal"],
            kind=BlockKind(item["kind"]),
            source_text=item["source_text"],
            source_hash=item["source_hash"],
            runs=tuple(InlineRun(**run) for run in item.get("runs", [])),
            attrs=dict(item.get("attrs", {})),
        )
        for item in rows[1:]
    )
    chapters = tuple(
        Chapter(
            chapter_id=item["chapter_id"],
            ordinal=item["ordinal"],
            title=item.get("title"),
            block_ids=tuple(item["block_ids"]),
        )
        for item in header["chapters"]
    )
    document = BookDocument(
        document_id=header["document_id"],
        metadata=BookMetadata(**header["metadata"]),
        chapters=chapters,
        blocks=blocks,
    )
    document.validate()
    return document


def write_source_jsonl(document: BookDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(document_to_source_jsonl(document), encoding="utf-8", newline="\n")
    temporary.replace(path)


def read_source_jsonl(path: Path) -> BookDocument:
    return document_from_source_jsonl(path.read_text(encoding="utf-8"))
