"""Loss-aware TXT import and export for book projects."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

from bn_en_translate.book.schema import (
    BlockKind,
    BookBlock,
    BookDocument,
    BookMetadata,
    Chapter,
    make_block_id,
)

_SCENE_BREAK = re.compile(r"^[*#—-]{3,}$")
_HEADING = re.compile(r"^(?:chapter|অধ্যায়)\s+.+$", re.IGNORECASE)


def _document_id(path: Path, text: str) -> str:
    payload = f"{path.name}\0{text}".encode()
    return f"txt-{hashlib.sha256(payload).hexdigest()[:16]}"


class TextReader:
    """Import a UTF-8 text file as a single or heading-delimited chapter sequence."""

    def read(self, path: Path) -> BookDocument:
        text = path.read_text(encoding="utf-8")
        newline = "\r\n" if "\r\n" in text else "\n"
        paragraphs = re.split(r"(?:\r?\n){2,}", text)
        blocks: list[BookBlock] = []
        chapters: list[Chapter] = []
        chapter_ordinal = 1
        chapter_id = f"c{chapter_ordinal:04d}"
        chapter_block_ids: list[str] = []
        chapter_title: str | None = None

        def finish_chapter() -> None:
            if chapter_block_ids or not chapters:
                chapters.append(
                    Chapter(
                        chapter_id=chapter_id,
                        ordinal=chapter_ordinal,
                        title=chapter_title,
                        block_ids=tuple(chapter_block_ids),
                    )
                )

        for raw_paragraph in paragraphs:
            text_value = raw_paragraph.strip()
            if _HEADING.match(text_value):
                if chapter_block_ids:
                    finish_chapter()
                    chapter_ordinal += 1
                    chapter_id = f"c{chapter_ordinal:04d}"
                    chapter_block_ids = []
                chapter_title = text_value
                kind = BlockKind.CHAPTER_HEADING
            elif not text_value:
                kind = BlockKind.BLANK
            elif _SCENE_BREAK.match(text_value):
                kind = BlockKind.SCENE_BREAK
            else:
                kind = BlockKind.PARAGRAPH
            ordinal = len(blocks) + 1
            block = BookBlock.create(
                block_id=make_block_id(chapter_ordinal, ordinal),
                chapter_id=chapter_id,
                ordinal=ordinal,
                kind=kind,
                source_text=text_value,
                attrs={"newline": newline},
            )
            blocks.append(block)
            chapter_block_ids.append(block.block_id)
        finish_chapter()
        document = BookDocument(
            document_id=_document_id(path, text),
            metadata=BookMetadata(title=path.stem, source_format="txt"),
            chapters=tuple(chapters),
            blocks=tuple(blocks),
        )
        document.validate()
        return document


class TextWriter:
    """Write translated text blocks atomically, retaining source only for non-text blocks."""

    def write(self, document: BookDocument, translations: Mapping[str, str], path: Path) -> None:
        document.validate()
        output: list[str] = []
        newline = "\n"
        for block in document.blocks:
            newline = str(block.attrs.get("newline", newline))
            if block.kind in {BlockKind.BLANK, BlockKind.SCENE_BREAK}:
                output.append(block.source_text)
                continue
            try:
                output.append(translations[block.block_id])
            except KeyError as exc:
                raise ValueError(f"missing translation for {block.block_id}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text((newline * 2).join(output), encoding="utf-8")
        temporary.replace(path)
