"""Immutable, format-neutral document types for book translation."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = 1


class BlockKind(StrEnum):
    """Semantic block types preserved by readers and writers."""

    TITLE = "title"
    CHAPTER_HEADING = "chapter_heading"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BLOCK_QUOTE = "block_quote"
    LIST_ITEM = "list_item"
    SCENE_BREAK = "scene_break"
    FOOTNOTE = "footnote"
    ENDNOTE = "endnote"
    BLANK = "blank"


@dataclass(frozen=True)
class InlineRun:
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    href: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("inline run text must be a string")


@dataclass(frozen=True)
class BookMetadata:
    title: str | None = None
    author: str | None = None
    source_language: str = "ben_Beng"
    target_language: str = "eng_Latn"
    source_format: str = "txt"

    def __post_init__(self) -> None:
        for name in ("source_language", "target_language", "source_format"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")


def canonical_text(text: str) -> str:
    """Normalize text used for content identity without destroying layout elsewhere."""
    return unicodedata.normalize("NFC", text)


def source_hash(text: str, kind: BlockKind, attrs: Mapping[str, Any] | None = None) -> str:
    """Return a stable hash for a source block's translatable identity."""
    payload = {"attrs": _jsonable(attrs or {}), "kind": kind.value, "text": canonical_text(text)}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert immutable mapping/sequence wrappers back to JSON values."""
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def jsonable(value: Any) -> Any:
    """Return a JSON-serializable copy of an attribute value."""
    return _jsonable(value)


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-like attributes so source records stay immutable."""
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def make_block_id(chapter_ordinal: int, block_ordinal: int) -> str:
    if chapter_ordinal < 1 or block_ordinal < 1:
        raise ValueError("chapter_ordinal and block_ordinal must be positive")
    return f"c{chapter_ordinal:04d}-b{block_ordinal:06d}"


@dataclass(frozen=True)
class BookBlock:
    block_id: str
    chapter_id: str
    ordinal: int
    kind: BlockKind
    source_text: str
    source_hash: str
    runs: tuple[InlineRun, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` protects attribute replacement but not a nested dict;
        # freeze imported source metadata as well to prevent accidental mutation.
        object.__setattr__(self, "attrs", _freeze(dict(self.attrs)))
        object.__setattr__(self, "runs", tuple(self.runs))

    @classmethod
    def create(
        cls,
        *,
        block_id: str,
        chapter_id: str,
        ordinal: int,
        kind: BlockKind,
        source_text: str,
        runs: tuple[InlineRun, ...] = (),
        attrs: Mapping[str, Any] | None = None,
    ) -> BookBlock:
        normalized_attrs = dict(attrs or {})
        normalized_text = canonical_text(source_text)
        return cls(
            block_id=block_id,
            chapter_id=chapter_id,
            ordinal=ordinal,
            kind=kind,
            source_text=normalized_text,
            source_hash=source_hash(normalized_text, kind, normalized_attrs),
            runs=runs,
            attrs=normalized_attrs,
        )

    def validate(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must not be empty")
        if not self.chapter_id:
            raise ValueError("chapter_id must not be empty")
        if self.ordinal < 1:
            raise ValueError("block ordinal must be positive")
        expected_hash = source_hash(self.source_text, self.kind, self.attrs)
        if self.source_hash != expected_hash:
            raise ValueError(f"source hash mismatch for block {self.block_id}")
        if self.runs and "".join(run.text for run in self.runs) != self.source_text:
            raise ValueError(f"inline runs do not reconstruct source text for {self.block_id}")


@dataclass(frozen=True)
class Chapter:
    chapter_id: str
    ordinal: int
    title: str | None
    block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_ids", tuple(self.block_ids))


@dataclass(frozen=True)
class BookDocument:
    document_id: str
    metadata: BookMetadata
    chapters: tuple[Chapter, ...]
    blocks: tuple[BookBlock, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "chapters", tuple(self.chapters))
        object.__setattr__(self, "blocks", tuple(self.blocks))

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        chapter_ids = [chapter.chapter_id for chapter in self.chapters]
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("chapter IDs must be unique")
        if [chapter.ordinal for chapter in self.chapters] != list(range(1, len(self.chapters) + 1)):
            raise ValueError("chapter ordinals must be contiguous and start at one")
        blocks_by_id = {block.block_id: block for block in self.blocks}
        if len(blocks_by_id) != len(self.blocks):
            raise ValueError("block IDs must be unique")
        expected_block_ids = [block.block_id for block in self.blocks]
        if [block.ordinal for block in self.blocks] != list(range(1, len(self.blocks) + 1)):
            raise ValueError("block ordinals must be contiguous and start at one")
        listed_block_ids = [block_id for chapter in self.chapters for block_id in chapter.block_ids]
        if listed_block_ids != expected_block_ids:
            raise ValueError(
                "chapter block IDs must list every block in document order exactly once"
            )
        for chapter in self.chapters:
            for block_id in chapter.block_ids:
                block = blocks_by_id.get(block_id)
                if block is None or block.chapter_id != chapter.chapter_id:
                    raise ValueError(
                        f"block {block_id} does not belong to chapter {chapter.chapter_id}"
                    )
        for block in self.blocks:
            block.validate()

    @property
    def blocks_by_id(self) -> dict[str, BookBlock]:
        return {block.block_id: block for block in self.blocks}
