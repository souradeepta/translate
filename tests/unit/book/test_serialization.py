from __future__ import annotations

import pytest

from bn_en_translate.book.schema import BlockKind, BookBlock, BookDocument, BookMetadata, Chapter
from bn_en_translate.book.serialization import dumps, loads


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
