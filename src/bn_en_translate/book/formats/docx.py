"""Structure-aware DOCX import for book translation projects.

``python-docx`` deliberately exposes only part of Word's document model. This
reader keeps the data it can represent in the format-neutral schema and issues a
visible warning for note bodies, which the library does not expose reliably.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.text.run import Run
except ImportError as exc:  # pragma: no cover - exercised by monkeypatch in tests.
    _DOCX_IMPORT_ERROR: ImportError | None = exc
    globals()["_open_document"] = None
    globals()["OxmlElement"] = None
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


class DocxExportError(ValueError):
    """Raised when a DOCX cannot be safely generated or validated."""


class DocxExportWarning(UserWarning):
    """A DOCX export retained block semantics but could not project inline style."""


_EMBEDDED_METADATA_PATH = "customXml/bn_book_blocks.json"
_EMBEDDED_METADATA_VERSION = 1


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


def _embedded_block_metadata(path: Path) -> dict[int, dict[str, Any]]:
    """Read our private package metadata, if present.

    The metadata is deliberately kept in a non-rendered OOXML package part.  It
    lets an exported document retain the source IDs used by a project without
    leaking implementation markers into the translated prose.
    """
    try:
        with zipfile.ZipFile(path) as package:
            raw = package.read(_EMBEDDED_METADATA_PATH)
    except (KeyError, OSError, zipfile.BadZipFile):
        return {}
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict) or value.get("version") != _EMBEDDED_METADATA_VERSION:
        return {}
    rows = value.get("blocks")
    if not isinstance(rows, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("block_id"), str) and isinstance(row.get("kind"), str):
            result[index] = row
    return result


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
        embedded_blocks = _embedded_block_metadata(path)
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
            embedded = embedded_blocks.get(index)
            try:
                embedded_kind = BlockKind(embedded["kind"]) if embedded is not None else None
            except ValueError:
                embedded_kind = None
            kind = embedded_kind or _block_kind(style_name, source_text, numbering)
            if kind is BlockKind.CHAPTER_HEADING and chapter_block_ids:
                finish_chapter()
                chapter_ordinal += 1
                chapter_id = f"c{chapter_ordinal:04d}"
                chapter_block_ids = []
                chapter_title = None
            if embedded is not None and isinstance(embedded.get("chapter_id"), str):
                # Export metadata is trusted only for the private IDs it carries;
                # chapter transitions above still come from the paragraph stream.
                chapter_id = embedded["chapter_id"]
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
                block_id=(
                    str(embedded["block_id"])
                    if embedded is not None and isinstance(embedded.get("block_id"), str)
                    else make_block_id(chapter_ordinal, ordinal)
                ),
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


@dataclass(frozen=True)
class DocxRoundTripReport:
    """Structural comparison between a source document and exported DOCX."""

    expected_block_ids: tuple[str, ...]
    actual_block_ids: tuple[str, ...]
    expected_kinds: tuple[BlockKind, ...]
    actual_kinds: tuple[BlockKind, ...]
    expected_chapters: tuple[tuple[str, ...], ...]
    actual_chapters: tuple[tuple[str, ...], ...]

    @property
    def missing_block_ids(self) -> tuple[str, ...]:
        return tuple(item for item in self.expected_block_ids if item not in self.actual_block_ids)

    @property
    def unexpected_block_ids(self) -> tuple[str, ...]:
        return tuple(item for item in self.actual_block_ids if item not in self.expected_block_ids)

    @property
    def reordered(self) -> bool:
        return (
            not self.missing_block_ids
            and not self.unexpected_block_ids
            and self.expected_block_ids != self.actual_block_ids
        )

    @property
    def ok(self) -> bool:
        return (
            self.expected_block_ids == self.actual_block_ids
            and self.expected_kinds == self.actual_kinds
            and self.expected_chapters == self.actual_chapters
        )


def validate_docx_round_trip(
    source: BookDocument, exported_path: Path
) -> DocxRoundTripReport:
    """Re-import *exported_path* and verify semantic block coverage and order."""
    exported = DocxReader().read(exported_path)
    report = DocxRoundTripReport(
        expected_block_ids=tuple(block.block_id for block in source.blocks),
        actual_block_ids=tuple(block.block_id for block in exported.blocks),
        expected_kinds=tuple(block.kind for block in source.blocks),
        actual_kinds=tuple(block.kind for block in exported.blocks),
        expected_chapters=tuple(chapter.block_ids for chapter in source.chapters),
        actual_chapters=tuple(chapter.block_ids for chapter in exported.chapters),
    )
    return report


def _fallback_style(kind: BlockKind) -> str:
    return {
        BlockKind.TITLE: "Title",
        BlockKind.CHAPTER_HEADING: "Heading 1",
        BlockKind.HEADING: "Heading 2",
        BlockKind.BLOCK_QUOTE: "Quote",
        BlockKind.LIST_ITEM: "List Bullet",
    }.get(kind, "Normal")


def _set_paragraph_style(paragraph: Any, block: BookBlock) -> None:
    style_name = str(block.attrs.get("paragraph_style", "") or "")
    try:
        paragraph.style = style_name or _fallback_style(block.kind)
    except (KeyError, ValueError):
        paragraph.style = _fallback_style(block.kind)


def _project_inline_runs(
    block: BookBlock, target_text: str
) -> tuple[tuple[InlineRun, ...], bool]:
    """Project source formatting only when the target is unchanged.

    Equal character counts do not imply equal word/character alignment between
    languages, so changed translations intentionally fall back to one unstyled
    run rather than applying formatting to the wrong target span.
    """
    if not block.runs:
        return (InlineRun(target_text),), True
    if target_text == block.source_text:
        return block.runs, True
    return (InlineRun(target_text),), False


def _append_hyperlink(paragraph: Any, run: InlineRun) -> None:
    """Append a hyperlink run using the OOXML form python-docx supports."""
    hyperlink = OxmlElement("w:hyperlink")
    href = run.href or ""
    if href.startswith("#"):
        hyperlink.set(qn("w:anchor"), href[1:])
    else:
        from docx.opc.constants import RELATIONSHIP_TYPE as _RT

        relationship_id = paragraph.part.relate_to(href, _RT.HYPERLINK, is_external=True)
        hyperlink.set(qn("r:id"), relationship_id)
    xml_run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    if run.bold:
        properties.append(OxmlElement("w:b"))
    if run.italic:
        properties.append(OxmlElement("w:i"))
    if run.underline:
        underline = OxmlElement("w:u")
        underline.set(qn("w:val"), "single")
        properties.append(underline)
    if len(properties):
        xml_run.append(properties)
    text = OxmlElement("w:t")
    text.text = run.text
    xml_run.append(text)
    hyperlink.append(xml_run)
    paragraph._p.append(hyperlink)


def _append_run(paragraph: Any, run: InlineRun) -> None:
    if run.href:
        _append_hyperlink(paragraph, run)
        return
    output = paragraph.add_run(run.text)
    output.bold = run.bold
    output.italic = run.italic
    output.underline = run.underline


def _apply_numbering(paragraph: Any, block: BookBlock) -> None:
    """Reconstruct direct numbering retained by the DOCX reader.

    A source document may use direct numbering while its paragraph style is
    ``Normal``.  Applying ``numPr`` keeps that list visible after rebuilding a
    fresh package; the style fallback supplies a usable default when no style
    was retained.
    """
    metadata = block.attrs.get("list")
    if not isinstance(metadata, Mapping):
        return
    properties = paragraph._p.get_or_add_pPr()
    numbering = properties.find(qn("w:numPr"))
    if numbering is None:
        numbering = OxmlElement("w:numPr")
        properties.append(numbering)
    level = numbering.find(qn("w:ilvl"))
    if level is None:
        level = OxmlElement("w:ilvl")
        numbering.append(level)
    level.set(qn("w:val"), str(metadata.get("level", 0)))
    number_id = numbering.find(qn("w:numId"))
    if number_id is None:
        number_id = OxmlElement("w:numId")
        numbering.append(number_id)
    # Built-in numbering definitions in a new python-docx document use small
    # IDs.  Unknown source IDs cannot resolve in the rebuilt package, so use the
    # default bullet definition while retaining the source level.
    raw_num_id = str(metadata.get("num_id", "1"))
    number_id.set(qn("w:val"), raw_num_id if raw_num_id.isdigit() else "1")


def _set_core_properties(document: Any, metadata: BookMetadata) -> None:
    properties = dict(metadata.core_properties)
    if metadata.title:
        properties.setdefault("title", metadata.title)
    if metadata.author:
        properties.setdefault("author", metadata.author)
    date_fields = {"created", "last_printed", "modified"}
    for name, value in properties.items():
        if not hasattr(document.core_properties, name):
            continue
        converted: object = value
        if name in date_fields:
            try:
                converted = datetime.fromisoformat(value)
            except ValueError:
                continue
        try:
            setattr(document.core_properties, name, converted)
        except (TypeError, ValueError):
            continue


def _embedded_metadata(document: BookDocument, findings: list[dict[str, Any]]) -> bytes:
    payload = {
        "blocks": [
            {
                "block_id": block.block_id,
                "chapter_id": block.chapter_id,
                "ordinal": block.ordinal,
                "kind": block.kind.value,
                "source_hash": block.source_hash,
            }
            for block in document.blocks
        ],
        "document_id": document.document_id,
        "findings": findings,
        "version": _EMBEDDED_METADATA_VERSION,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _package_metadata(base: Path, packaged: Path, metadata: bytes) -> None:
    """Copy a DOCX and add a private metadata part without altering prose."""
    with zipfile.ZipFile(base, "r") as source, zipfile.ZipFile(
        packaged, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for info in source.infolist():
            if info.filename == _EMBEDDED_METADATA_PATH:
                continue
            contents = source.read(info.filename)
            if info.filename == "[Content_Types].xml":
                marker = b'<Default Extension="json"'
                if marker not in contents:
                    contents = contents.replace(
                        b"</Types>",
                        b'<Default Extension="json" ContentType="application/json"/></Types>',
                    )
            destination.writestr(info, contents)
        destination.writestr(_EMBEDDED_METADATA_PATH, metadata)


class DocxWriter:
    """Export translated blocks to a structure-preserving DOCX atomically."""

    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []

    def write(
        self, document: BookDocument, translations: Mapping[str, str], path: Path
    ) -> None:
        _require_docx()
        document.validate()
        self.findings = []
        output_blocks: list[tuple[BookBlock, str, tuple[InlineRun, ...]]] = []
        for block in document.blocks:
            if block.kind in {BlockKind.BLANK, BlockKind.SCENE_BREAK}:
                target_text = block.source_text
            else:
                try:
                    target_text = str(translations[block.block_id])
                except KeyError as exc:
                    raise DocxExportError(f"missing translation for {block.block_id}") from exc
                if not target_text.strip():
                    raise DocxExportError(f"empty translation for {block.block_id}")
            runs, aligned = _project_inline_runs(block, target_text)
            if not aligned:
                finding = {
                    "rule": "inline_style_projection",
                    "severity": "warning",
                    "block_ids": [block.block_id],
                    "evidence": "target text length differs from source; block style retained",
                }
                self.findings.append(finding)
                warnings.warn(
                    f"could not project inline styles for {block.block_id}; retained block style",
                    DocxExportWarning,
                    stacklevel=2,
                )
            if block.attrs.get("note_references"):
                self.findings.append(
                    {
                        "rule": "unsupported_note",
                        "severity": "warning",
                        "block_ids": [block.block_id],
                        "evidence": (
                            "note references retained as metadata; note bodies are not exported"
                        ),
                    }
                )
            output_blocks.append((block, target_text, runs))

        output = _open_document()
        # A newly created document contains one empty paragraph.  Remove it so
        # every output paragraph corresponds exactly to one source block.
        body = output._element.body
        for paragraph in list(output.paragraphs):
            body.remove(paragraph._p)
        _set_core_properties(output, document.metadata)
        for block, _target_text, runs in output_blocks:
            paragraph = output.add_paragraph()
            _set_paragraph_style(paragraph, block)
            _apply_numbering(paragraph, block)
            for run in runs:
                _append_run(paragraph, run)

        path.parent.mkdir(parents=True, exist_ok=True)
        base = path.with_suffix(path.suffix + ".base.tmp")
        packaged = path.with_suffix(path.suffix + ".packaged.tmp")
        try:
            output.save(str(base))
            _package_metadata(base, packaged, _embedded_metadata(document, self.findings))
            report = validate_docx_round_trip(document, packaged)
            if not report.ok:
                raise DocxExportError(
                    "DOCX export failed semantic round-trip: "
                    f"missing={report.missing_block_ids}, "
                    f"unexpected={report.unexpected_block_ids}, "
                    f"reordered={report.reordered}"
                )
            packaged.replace(path)
        except DocxExportError:
            raise
        except Exception as exc:
            raise DocxExportError(f"cannot export DOCX {path}: {exc}") from exc
        finally:
            base.unlink(missing_ok=True)
            packaged.unlink(missing_ok=True)
