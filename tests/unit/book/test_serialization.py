from __future__ import annotations

import pytest

from bn_en_translate.book.schema import BlockKind, BookBlock, BookDocument, BookMetadata, Chapter
from bn_en_translate.book.serialization import (
    document_from_source_jsonl,
    document_to_source_jsonl,
    dumps,
    loads,
)


def _document() -> BookDocument:
    block = BookBlock.create(
        block_id="c0001-b000001",
        chapter_id="c0001",
        ordinal=1,
        kind=BlockKind.PARAGRAPH,
        source_text="বাংলা পাঠ।",
    )
    return BookDocument(
        document_id="document",
        metadata=BookMetadata(),
        chapters=(Chapter("c0001", 1, None, (block.block_id,)),),
        blocks=(block,),
    )


def test_serialization_is_deterministic_and_round_trips() -> None:
    encoded = dumps(_document())
    assert dumps(loads(encoded)) == encoded


def test_unknown_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported schema"):
        loads('{"schema_version": 999}')


def test_source_jsonl_round_trip_is_byte_stable() -> None:
    document = _document()
    encoded = document_to_source_jsonl(document)

    assert document_from_source_jsonl(encoded) == document
    assert document_to_source_jsonl(document_from_source_jsonl(encoded)) == encoded


def test_source_jsonl_unknown_future_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported source JSONL version"):
        document_from_source_jsonl(
            '{"document_id":"document","record_type":"header",'
            '"schema_version":999}\n'
        )


def test_source_jsonl_unknown_record_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown record type"):
        document_from_source_jsonl(
            '{"chapters":[],"document_id":"document","metadata":{},'
            '"record_type":"header","schema_version":1}\n'
            '{"record_type":"future"}\n'
        )


def test_json_migration_dispatch_rejects_unknown_past_version() -> None:
    with pytest.raises(ValueError, match="unsupported schema"):
        loads('{"schema_version": 0}')
