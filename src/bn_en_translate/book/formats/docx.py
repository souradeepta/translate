"""Structure-aware DOCX import for book translation projects.

``python-docx`` deliberately exposes only part of Word's document model. This
reader keeps the data it can represent in the format-neutral schema and issues a
visible warning for note bodies, which the library does not expose reliably.
"""

from __future__ import annotations

import hashlib
import re
import warnings
import zipfile
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from bn_en_translate.book.schema import (
    BlockKind,
    BookBlock,
    BookDocument,
    BookMetadata,
    Chapter,
    InlineRun,
    make_block_id,
)

try:  # Optional: importing the book package must not require DOCX support.
    from docx import Document as _open_document
    from docx.oxml.ns import qn
    from docx.text.run import Run
except ImportError as exc:  # pragma: no cover - exercised by monkeypatch in tests.
    _DOCX_IMPORT_ERROR: ImportError | None = exc
    globals()["_open_document"] = None
    globals()["qn"] = None
    globals()["Run"] = None
else:
    _DOCX_IMPORT_ERROR = None


_ASTERISM = re.compile(r"^\s*\*{3,}\s*$")
_CORE_PROPERTY_NAMES = (
    "author",
    "category",
    "comments",
    "content_status",
    "created",
    "identifier",
    "keywords",
    "language",
    "last_modified_by",
    "last_printed",
    "modified",
    "revision",
    "subject",
    "title",
    "version",
)


class DocxImportError(ValueError):
    """Raised when a DOCX cannot be opened as a safe, supported document."""


class DocxImportWarning(UserWarning):
    """A source feature was retained as metadata but cannot yet be fully imported."""


def _document_id(path: Path) -> str:
    """Use the original package bytes so importing the same file is stable."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"docx-{digest}"


def _require_docx() -> None:
    if _open_document is None:
        raise DocxImportError(
            "DOCX import requires the optional 'book' dependency; install bn-en-translate[book]"
        ) from _DOCX_IMPORT_ERROR


def _text_value(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _core_properties(document: Any) -> dict[str, str]:
    properties: dict[str, str] = {}
    for name in _CORE_PROPERTY_NAMES:
        value = getattr(document.core_properties, name, None)
        if value is not None and value != "":
            properties[name] = _text_value(value)
    return properties


def _paragraph_style_name(paragraph: Any) -> str:
    style = getattr(paragraph, "style", None)
    return str(getattr(style, "name", "") or "")


def _numbering_properties(paragraph: Any) -> dict[str, int | str] | None:
    """Read direct or style-provided numbering without depending on private APIs."""
    candidates: list[Any] = []
    paragraph_properties = getattr(paragraph._p, "pPr", None)
    if paragraph_properties is not None:
        candidates.append(paragraph_properties)
    style_element = getattr(getattr(paragraph, "style", None), "element", None)
    style_properties = getattr(style_element, "pPr", None)
    if style_properties is not None:
        candidates.append(style_properties)
    for properties in candidates:
        num_properties = properties.find(qn("w:numPr"))
        if num_properties is None:
            continue
        level = num_properties.find(qn("w:ilvl"))
        number_id = num_properties.find(qn("w:numId"))
        metadata: dict[str, int | str] = {
            "level": int(level.get(qn("w:val"), "0")) if level is not None else 0
        }
        if number_id is not None and number_id.get(qn("w:val")) is not None:
            metadata["num_id"] = number_id.get(qn("w:val"))
        return metadata
    return None


def _href(paragraph: Any, hyperlink: Any) -> str | None:
    relationship_id = hyperlink.get(qn("r:id"))
    if relationship_id:
        relationship = paragraph.part.rels.get(relationship_id)
        if relationship is not None:
            return str(relationship.target_ref)
    anchor = hyperlink.get(qn("w:anchor"))
    return f"#{anchor}" if anchor else None


def _inline_runs(paragraph: Any) -> tuple[InlineRun, ...]:
    """Return runs in XML order, including runs nested in Word hyperlinks."""
    result: list[InlineRun] = []
    for child in paragraph._p:
        if child.tag == qn("w:r"):
            candidate_runs: Iterable[Any] = (child,)
            href = None
        elif child.tag == qn("w:hyperlink"):
            candidate_runs = child.findall(qn("w:r"))
            href = _href(paragraph, child)
        else:
            continue
        for element in candidate_runs:
            run = Run(element, paragraph)
            if run.text:
                result.append(
                    InlineRun(
                        text=run.text,
                        bold=bool(run.bold),
                        italic=bool(run.italic),
                        underline=bool(run.underline),
                        href=href,
                    )
                )
    # A field or other unsupported inline element can contribute to ``.text``.
    # Retain its text rather than emitting invalid run data that fails validation.
    return tuple(result) if "".join(run.text for run in result) == paragraph.text else ()


def _note_references(paragraph: Any) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for element in paragraph._p.iter():
        if element.tag == qn("w:footnoteReference"):
            reference_id = element.get(qn("w:id"))
            if reference_id is not None:
                references.append({"kind": "footnote", "id": str(reference_id)})
        elif element.tag == qn("w:endnoteReference"):
            reference_id = element.get(qn("w:id"))
            if reference_id is not None:
                references.append({"kind": "endnote", "id": str(reference_id)})
    return references


def _contains_note_part(document: Any) -> bool:
    return any(
        reltype.endswith("/footnotes") or reltype.endswith("/endnotes")
        for reltype in (relationship.reltype for relationship in document.part.rels.values())
    )


def _block_kind(style_name: str, text: str, numbering: dict[str, int | str] | None) -> BlockKind:
    normalized_style = style_name.casefold()
    if _ASTERISM.fullmatch(text):
        return BlockKind.SCENE_BREAK
    if not text:
        return BlockKind.BLANK
    if normalized_style == "title":
        return BlockKind.TITLE
    if normalized_style.startswith("heading 1"):
        return BlockKind.CHAPTER_HEADING
    if normalized_style.startswith("heading"):
        return BlockKind.HEADING
    if "quote" in normalized_style:
        return BlockKind.BLOCK_QUOTE
    if numbering is not None or "list" in normalized_style:
        return BlockKind.LIST_ITEM
    return BlockKind.PARAGRAPH


class DocxReader:
    """Import a DOCX while retaining the semantic structure needed for later export."""

    def read(self, path: Path) -> BookDocument:
        _require_docx()
        if not path.is_file():
            raise FileNotFoundError(path)
        # Encrypted OOXML is an OLE compound file rather than a DOCX ZIP package.
        # This also rejects truncated/corrupt input before a project can be created.
        if not zipfile.is_zipfile(path):
            raise DocxImportError(
                f"cannot import DOCX {path}: file is password-protected, corrupt, "
                "or not a DOCX package"
            )
        try:
            document = _open_document(str(path))
        except Exception as exc:
            raise DocxImportError(f"cannot import DOCX {path}: {exc}") from exc

        if document.tables:
            warnings.warn(
                "DOCX contains tables; table contents are not imported in this release",
                DocxImportWarning,
                stacklevel=2,
            )

        blocks: list[BookBlock] = []
        chapters: list[Chapter] = []
        chapter_ordinal = 1
        chapter_id = f"c{chapter_ordinal:04d}"
        chapter_title: str | None = None
        chapter_block_ids: list[str] = []
        saw_notes = _contains_note_part(document)

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

        for index, paragraph in enumerate(document.paragraphs, start=1):
            source_text = paragraph.text
            style_name = _paragraph_style_name(paragraph)
            numbering = _numbering_properties(paragraph)
            kind = _block_kind(style_name, source_text, numbering)
            if kind is BlockKind.CHAPTER_HEADING and chapter_block_ids:
                finish_chapter()
                chapter_ordinal += 1
                chapter_id = f"c{chapter_ordinal:04d}"
                chapter_block_ids = []
                chapter_title = None
            if kind is BlockKind.CHAPTER_HEADING:
                chapter_title = source_text

            attrs: dict[str, Any] = {
                "paragraph_style": style_name,
                "source_locator": f"word/document.xml:p[{index}]",
            }
            if numbering is not None:
                attrs["list"] = numbering
            note_references = _note_references(paragraph)
            if note_references:
                attrs["note_references"] = note_references
                saw_notes = True

            ordinal = len(blocks) + 1
            block = BookBlock.create(
                block_id=make_block_id(chapter_ordinal, ordinal),
                chapter_id=chapter_id,
                ordinal=ordinal,
                kind=kind,
                source_text=source_text,
                runs=_inline_runs(paragraph),
                attrs=attrs,
            )
            blocks.append(block)
            chapter_block_ids.append(block.block_id)
        finish_chapter()

        if saw_notes:
            warnings.warn(
                "DOCX footnote/endnote bodies are not exposed reliably by python-docx; "
                "references were retained in block metadata but note bodies were not imported",
                DocxImportWarning,
                stacklevel=2,
            )

        core_properties = _core_properties(document)
        document_value = BookDocument(
            document_id=_document_id(path),
            metadata=BookMetadata(
                title=core_properties.get("title") or path.stem,
                author=core_properties.get("author"),
                source_format="docx",
                core_properties=core_properties,
            ),
            chapters=tuple(chapters),
            blocks=tuple(blocks),
        )
        document_value.validate()
        return document_value
