"""Safe, structure-aware EPUB import and export.

The EPUB reader intentionally treats the package as an untrusted ZIP archive.
It resolves the OPF/spine itself (rather than relying on a renderer), never
fetches URLs, and keeps a private copy of package entries in the document
metadata so an export can preserve images and other non-text resources exactly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
import warnings
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from bn_en_translate.book.schema import (
    BlockKind,
    BookBlock,
    BookDocument,
    BookMetadata,
    Chapter,
    InlineRun,
    make_block_id,
)

try:  # EPUB support is optional and must not affect the base book package.
    import ebooklib  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    _EBOOKLIB_IMPORT_ERROR: ImportError | None = exc
else:
    _EBOOKLIB_IMPORT_ERROR = None

BeautifulSoup: Any
NavigableString: Any
Tag: Any
try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    from bs4.element import NavigableString as _NavigableString
    from bs4.element import Tag as _Tag
except ImportError as exc:  # pragma: no cover - environment dependent
    _BS4_IMPORT_ERROR: ImportError | None = exc
    # The names are deliberately dynamic in the optional-dependency path.
    # ``_require_epub`` prevents them from being used before BeautifulSoup is
    # available, while this keeps static type checking independent of extras.
    BeautifulSoup = None
    NavigableString = None
    Tag = None
else:
    _BS4_IMPORT_ERROR = None
    BeautifulSoup = _BeautifulSoup
    NavigableString = _NavigableString
    Tag = _Tag


class EpubImportError(ValueError):
    """Raised when an EPUB is unavailable, unsafe, corrupt, or unsupported."""


class EpubExportError(ValueError):
    """Raised when an EPUB cannot be safely generated or validated."""


class EpubImportWarning(UserWarning):
    """A source feature was retained or skipped with an explicit finding."""


class EpubExportWarning(UserWarning):
    """A supported block had to fall back to plain target text."""


_EPUB_PACKAGE_KEY = "epub_package_entries_v1"
_BLOCK_ID_ATTR = "data-bn-block-id"
_BLOCK_KIND_ATTR = "data-bn-block-kind"
_XHTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html"}
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "li", "aside"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_LIST_TAGS = {"ul", "ol", "dl"}
_NOTE_TYPES = {"footnote", "endnote", "rearnote", "footnotes", "endnotes"}
_EXTERNAL_SCHEMES = {"http", "https", "ftp", "file", "javascript", "data"}


def _require_epub() -> None:
    missing: list[str] = []
    if _EBOOKLIB_IMPORT_ERROR is not None:
        missing.append("ebooklib")
    if _BS4_IMPORT_ERROR is not None:
        missing.append("beautifulsoup4")
    if missing:
        raise EpubImportError(
            "EPUB support requires the optional 'epub' dependencies: "
            + ", ".join(missing)
            + "; install bn-en-translate[epub]"
        ) from (_EBOOKLIB_IMPORT_ERROR or _BS4_IMPORT_ERROR)


def _document_id(path: Path) -> str:
    return f"epub-{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _safe_member_names(package: zipfile.ZipFile) -> list[str]:
    names: list[str] = []
    for info in package.infolist():
        name = info.filename
        parsed = PurePosixPath(name)
        if not name or parsed.is_absolute() or ".." in parsed.parts:
            raise EpubImportError(f"EPUB contains unsafe package path: {name!r}")
        # Reject UNIX symlinks.  Reading a symlink target from an archive could
        # otherwise make resource handling depend on the host filesystem.
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode and (mode & 0o170000) == 0o120000:
            raise EpubImportError(f"EPUB contains unsupported symlink: {name!r}")
        if name in names:
            raise EpubImportError(f"EPUB contains duplicate package path: {name!r}")
        names.append(name)
    return names


def _xml_root(package: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        raw = package.read(name)
        return ET.fromstring(raw)
    except (KeyError, ET.ParseError, UnicodeDecodeError) as exc:
        raise EpubImportError(f"invalid EPUB XML part: {name}") from exc


def _resolve_href(base: str, href: str) -> tuple[str, str | None]:
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        raise EpubImportError(f"EPUB references external resource: {href}")
    path = posixpath.normpath(posixpath.join(posixpath.dirname(base), parts.path))
    if path.startswith("../") or path == "..":
        raise EpubImportError(f"EPUB resource escapes package root: {href}")
    return path, parts.fragment or None


def _package_entries(package: zipfile.ZipFile) -> dict[str, str]:
    return {
        info.filename: base64.b64encode(package.read(info.filename)).decode("ascii")
        for info in package.infolist()
    }


def _normalise_runs(runs: list[InlineRun]) -> tuple[InlineRun, ...]:
    """Collapse pretty-print whitespace while retaining inline boundaries."""
    if not runs:
        return ()
    cleaned: list[InlineRun] = []
    for run in runs:
        # XHTML indentation is presentation whitespace; preserve meaningful
        # spaces at either edge of inline runs.
        text = re.sub(r"\s+", " ", run.text)
        if text:
            cleaned.append(
                InlineRun(
                    text, bold=run.bold, italic=run.italic,
                    underline=run.underline, href=run.href,
                )
            )
    if not cleaned:
        return ()
    joined = "".join(run.text for run in cleaned)
    left = len(joined) - len(joined.lstrip())
    right = len(joined) - len(joined.rstrip())
    if left:
        first = cleaned[0]
        cleaned[0] = InlineRun(
            first.text[left:], bold=first.bold, italic=first.italic,
            underline=first.underline, href=first.href,
        )
    if right:
        last = cleaned[-1]
        cleaned[-1] = InlineRun(
            last.text[:-right] if right < len(last.text) else "",
            bold=last.bold, italic=last.italic, underline=last.underline, href=last.href,
        )
    return tuple(run for run in cleaned if run.text)


def _is_external_fetch(tag: Any, attr: str, value: str) -> bool:
    if tag.name == "a" and attr == "href" and urlsplit(value.strip()).scheme.casefold() not in {
        "javascript", "data"
    }:
        return False  # Links are retained; the importer never follows them.
    parsed = urlsplit(value.strip())
    return bool(parsed.scheme.casefold() in _EXTERNAL_SCHEMES or parsed.netloc)


def _check_active_content(soup: Any, href: str) -> None:
    if soup.find("script") is not None:
        raise EpubImportError(f"active script content is not allowed in EPUB item {href}")
    for element in soup.find_all(True):
        for attr, value in element.attrs.items():
            attr_name = str(attr).casefold()
            if attr_name.startswith("on"):
                raise EpubImportError(f"active event handler {attr!r} in EPUB item {href}")
            if attr_name in {"src", "srcset", "data", "poster", "action", "href"}:
                values = str(value).split(",") if attr_name == "srcset" else [str(value)]
                candidates = [item.split()[0] for item in values if item.split()]
                if any(_is_external_fetch(element, attr_name, item) for item in candidates):
                    raise EpubImportError(f"external fetch blocked in EPUB item {href}: {value}")
            if attr_name == "href" and element.name == "link" and isinstance(value, str):
                if _is_external_fetch(element, attr_name, value):
                    raise EpubImportError(f"external stylesheet blocked in EPUB item {href}")
        if element.name in {"iframe", "object", "embed", "frame"}:
            raise EpubImportError(f"active embedded content is not allowed in EPUB item {href}")
    for style in soup.find_all(style=True):
        if re.search(r"url\s*\(\s*(?:[a-z][a-z0-9+.-]*:|//)", str(style.get("style")), re.I):
            raise EpubImportError(f"external CSS fetch blocked in EPUB item {href}")
    for style in soup.find_all("style"):
        if re.search(r"(?:url\s*\(\s*(?:[a-z][a-z0-9+.-]*:|//)|@import\s+url)",
                     style.get_text(), re.I):
            raise EpubImportError(f"external CSS fetch blocked in EPUB item {href}")


def _scan_package_entries(entries: Mapping[str, bytes]) -> None:
    """Reject active content and network fetches in every text package part.

    Only markup and stylesheet-like entries are decoded. Binary resources are
    preserved verbatim and are never handed to an HTML parser.
    """
    text_suffixes = {
        ".css", ".html", ".htm", ".xhtml", ".svg", ".xml", ".opf", ".ncx", ".js"
    }
    for name, content in entries.items():
        suffix = PurePosixPath(name).suffix.casefold()
        if suffix not in text_suffixes and b"\x00" in content[:4096]:
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if suffix == ".css":
            if re.search(
                r"(?:@import\s+(?:(?:url\s*\(\s*)?[\"']?)?|url\s*\(\s*)"
                r"(?:[a-z][a-z0-9+.-]*:|//)",
                text,
                re.I,
            ):
                raise EpubImportError(f"external CSS fetch blocked in EPUB item {name}")
            continue
        if re.search(r"<\s*script\b", text, re.I):
            raise EpubImportError(f"active script content is not allowed in EPUB item {name}")
        if re.search(r"\bon[a-z][a-z0-9_-]*\s*=", text, re.I):
            raise EpubImportError(
                f"active event handler content is not allowed in EPUB item {name}"
            )
        if re.search(
            r"(?:url\s*\(\s*|(?:src|srcset|data|poster|action|href|xlink:href)\s*=\s*[\"']?)"
            r"(?:[a-z][a-z0-9+.-]*:|//)",
            text,
            re.I,
        ):
            # External anchors are safe to retain, but all other markup
            # references must remain package-local.
            assert BeautifulSoup is not None
            soup = _package_soup(text, suffix)
            _check_active_content(soup, name)
        elif suffix in {".html", ".htm", ".xhtml", ".svg", ".xml", ".opf", ".ncx"}:
            assert BeautifulSoup is not None
            soup = _package_soup(text, suffix)
            _check_active_content(soup, name)


def _package_soup(text: str, suffix: str) -> Any:
    """Parse XML package metadata as XML and XHTML content as HTML."""
    assert BeautifulSoup is not None
    parser = "xml" if suffix in {".svg", ".xml", ".opf", ".ncx"} else "html.parser"
    return BeautifulSoup(text, parser)


def _ancestor(element: Any, names: set[str]) -> Any | None:
    parent = element.parent
    while parent is not None and getattr(parent, "name", None) != "[document]":
        if getattr(parent, "name", "").casefold() in names:
            return parent
        parent = parent.parent
    return None


def _runs_for(element: Any, *, skip_nested_blocks: bool = True) -> tuple[InlineRun, ...]:
    output: list[InlineRun] = []

    def visit(node: Any, bold: bool = False, italic: bool = False, underline: bool = False,
              href: str | None = None) -> None:
        if isinstance(node, NavigableString):
            output.append(
                InlineRun(
                    str(node), bold=bold, italic=italic,
                    underline=underline, href=href,
                )
            )
            return
        if not isinstance(node, Tag):
            return
        name = node.name.casefold()
        if skip_nested_blocks and node is not element and name in _BLOCK_TAGS | _LIST_TAGS:
            return
        if name == "br":
            output.append(InlineRun("\n", bold=bold, italic=italic, underline=underline, href=href))
            return
        anchor = href
        if name == "a":
            raw_href = node.get("href")
            anchor = str(raw_href) if raw_href is not None else href
        for child in node.children:
            visit(
                child,
                bold=bold or name in {"b", "strong"},
                italic=italic or name in {"i", "em", "cite"},
                underline=underline or name == "u",
                href=anchor,
            )

    visit(element)
    return _normalise_runs(output)


def _semantic_elements(body: Any) -> list[Any]:
    elements: list[Any] = []
    for element in body.find_all(True):
        name = element.name.casefold()
        is_div_only = name == "div" and not element.find(list(_BLOCK_TAGS | _LIST_TAGS))
        if name not in _BLOCK_TAGS and not is_div_only:
            continue
        # A table is retained as an opaque unsupported structure. Importing
        # paragraphs nested inside it would make export lose the table while
        # falsely claiming semantic coverage.
        if _ancestor(element, {"table"}) is not None:
            continue
        # A blockquote wrapping paragraphs is represented by the paragraphs,
        # with the quote kind inherited. This avoids duplicate source text.
        if name == "blockquote" and element.find(list(_BLOCK_TAGS - {"blockquote"})) is not None:
            continue
        # An aside note is a container. Prefer its paragraph/list children so
        # each note body is represented once and retains its inline markup.
        if name == "aside" and element.find(list(_BLOCK_TAGS - {"aside"})) is not None:
            continue
        if name != "li" and _ancestor(element, _BLOCK_TAGS | _LIST_TAGS) is not None:
            parent = _ancestor(element, _BLOCK_TAGS | _LIST_TAGS)
            if parent is not None and parent.name.casefold() not in {"blockquote", "aside"}:
                continue
        elements.append(element)
    return elements


def _kind_for(element: Any) -> BlockKind:
    name = element.name.casefold()
    text = "".join(run.text for run in _runs_for(element))
    if name in _HEADING_TAGS:
        return BlockKind.CHAPTER_HEADING if name == "h1" else BlockKind.HEADING
    if name == "blockquote" or _ancestor(element, {"blockquote"}) is not None:
        return BlockKind.BLOCK_QUOTE
    if name == "li":
        return BlockKind.LIST_ITEM
    if name == "aside":
        note_type = str(element.get("epub:type", element.get("role", ""))).casefold()
        if any(note in note_type for note in _NOTE_TYPES):
            return BlockKind.FOOTNOTE if "foot" in note_type else BlockKind.ENDNOTE
    note_ancestor = _ancestor(element, {"aside"})
    if note_ancestor is not None:
        note_type = str(
            note_ancestor.get("epub:type", note_ancestor.get("role", ""))
        ).casefold()
        if any(note in note_type for note in _NOTE_TYPES):
            return BlockKind.FOOTNOTE if "foot" in note_type else BlockKind.ENDNOTE
    if not text.strip():
        return BlockKind.BLANK
    if re.fullmatch(r"\s*(?:\*{3,}|#\s*#\s*#|—{3,})\s*", text):
        return BlockKind.SCENE_BREAK
    return BlockKind.PARAGRAPH


def _item_blocks(
    content: bytes, item_href: str, chapter_ordinal: int, start_ordinal: int
) -> list[BookBlock]:
    assert BeautifulSoup is not None
    try:
        soup = BeautifulSoup(content, "html.parser")
    except Exception as exc:
        raise EpubImportError(f"cannot parse EPUB XHTML item {item_href}: {exc}") from exc
    _check_active_content(soup, item_href)
    if soup.find("table") is not None:
        warnings.warn(
            f"EPUB item {item_href} contains tables; table contents are not imported",
            EpubImportWarning,
            stacklevel=2,
        )
    if any(
        element.name.casefold() == "div"
        and not element.find(list(_BLOCK_TAGS | _LIST_TAGS))
        and element.get_text(strip=True)
        for element in soup.find_all("div")
    ):
        warnings.warn(
            f"EPUB item {item_href} contains div-only content; imported as paragraphs",
            EpubImportWarning,
            stacklevel=2,
        )
    body = soup.find("body") or soup
    result: list[BookBlock] = []
    for item_index, element in enumerate(_semantic_elements(body), start=1):
        runs = _runs_for(element)
        source_text = "".join(run.text for run in runs)
        kind = _kind_for(element)
        if kind is BlockKind.SCENE_BREAK and not source_text.strip():
            source_text = "***"
            runs = (InlineRun(source_text),)
        attrs: dict[str, Any] = {
            "source_locator": f"{item_href}::block[{item_index}]",
            "epub_href": item_href,
            "epub_block_index": item_index,
            "epub_tag": element.name.casefold(),
        }
        if element.get("id"):
            attrs["epub_element_id"] = str(element.get("id"))
        if element.name.casefold() == "li":
            parent = element.find_parent(_LIST_TAGS)
            attrs["list"] = {
                "tag": parent.name.casefold() if parent is not None else "ul",
                "level": len(element.find_parents(_LIST_TAGS)),
            }
        note_type = str(element.get("epub:type", element.get("role", "")))
        note_ancestor = _ancestor(element, {"aside"})
        if not note_type and note_ancestor is not None:
            note_type = str(
                note_ancestor.get("epub:type", note_ancestor.get("role", ""))
            )
        if note_type:
            attrs["note_type"] = note_type
        block = BookBlock.create(
            block_id=(
                str(element.get(_BLOCK_ID_ATTR))
                if element.get(_BLOCK_ID_ATTR)
                else make_block_id(chapter_ordinal, start_ordinal + len(result))
            ),
            chapter_id=f"c{chapter_ordinal:04d}",
            ordinal=start_ordinal + len(result),
            kind=kind,
            source_text=source_text,
            runs=runs,
            attrs=attrs,
        )
        result.append(block)
    return result


def _metadata(opf: ET.Element, path: Path) -> BookMetadata:
    values: dict[str, str] = {}
    for node in opf.iter():
        name = _local_name(node.tag)
        value = (node.text or "").strip()
        if not value:
            continue
        if name == "title" and "title" not in values:
            values["title"] = value
        elif name in {"creator", "author"} and "author" not in values:
            values["author"] = value
        elif name == "language" and "language" not in values:
            values["language"] = value
    core = {f"epub.{key}": value for key, value in values.items()}
    return BookMetadata(
        title=values.get("title") or path.stem,
        author=values.get("author"),
        source_language=values.get("language") or "ben_Beng",
        source_format="epub",
        core_properties=core,
    )


class EpubReader:
    """Read spine-ordered XHTML into the format-neutral book schema."""

    def read(self, path: Path) -> BookDocument:
        _require_epub()
        if not path.is_file() or not zipfile.is_zipfile(path):
            raise EpubImportError(f"cannot import EPUB {path}: corrupt or non-ZIP package")
        try:
            with zipfile.ZipFile(path) as package:
                names = _safe_member_names(package)
                if "mimetype" not in names or package.read("mimetype") != b"application/epub+zip":
                    raise EpubImportError("EPUB package has invalid mimetype")
                _scan_package_entries({name: package.read(name) for name in names})
                container = _xml_root(package, "META-INF/container.xml")
                rootfile = next(
                    (
                        node.get("full-path")
                        for node in container.iter()
                        if _local_name(node.tag) == "rootfile"
                    ),
                    None,
                )
                if not rootfile:
                    raise EpubImportError("EPUB container.xml has no rootfile")
                rootfile = str(rootfile)
                if rootfile not in names:
                    raise EpubImportError(f"EPUB OPF is missing: {rootfile}")
                opf = _xml_root(package, rootfile)
                manifest: dict[str, tuple[str, str, str]] = {}
                for node in opf.iter():
                    if _local_name(node.tag) != "item":
                        continue
                    item_id = node.get("id")
                    href = node.get("href")
                    media = node.get("media-type", "")
                    if not item_id or not href:
                        continue
                    resolved, _ = _resolve_href(rootfile, href)
                    if resolved not in names:
                        raise EpubImportError(f"EPUB manifest item is missing: {resolved}")
                    manifest[item_id] = (resolved, media, node.get("properties", ""))
                spine_ids = [
                    node.get("idref") for node in opf.iter()
                    if _local_name(node.tag) == "itemref" and node.get("linear", "yes") != "no"
                ]
                blocks: list[BookBlock] = []
                chapters: list[Chapter] = []
                # Spine entries can include non-linear or non-XHTML resources.
                # Chapter ordinals describe imported chapters, so they must not
                # inherit gaps from skipped spine entries.
                for item_id in spine_ids:
                    if not item_id or item_id not in manifest:
                        raise EpubImportError(f"EPUB spine references unknown item: {item_id}")
                    item_href, media_type, _properties = manifest[item_id]
                    if media_type not in _XHTML_MEDIA_TYPES:
                        warnings.warn(
                            f"skipping non-XHTML spine item {item_href}",
                            EpubImportWarning,
                            stacklevel=2,
                        )
                        continue
                    chapter_ordinal = len(chapters) + 1
                    chapter_blocks = _item_blocks(
                        package.read(item_href), item_href, chapter_ordinal, len(blocks) + 1
                    )
                    blocks.extend(chapter_blocks)
                    title = next(
                        (
                            block.source_text
                            for block in chapter_blocks
                            if block.kind is BlockKind.CHAPTER_HEADING
                        ),
                        Path(item_href).stem,
                    )
                    chapters.append(
                        Chapter(
                            chapter_id=f"c{chapter_ordinal:04d}",
                            ordinal=chapter_ordinal,
                            title=title,
                            block_ids=tuple(block.block_id for block in chapter_blocks),
                        )
                    )
                metadata = _metadata(opf, path)
                core = dict(metadata.core_properties)
                core[_EPUB_PACKAGE_KEY] = json.dumps(
                    _package_entries(package), sort_keys=True, separators=(",", ":")
                )
                core["epub.opf_path"] = rootfile
                metadata = BookMetadata(
                    title=metadata.title,
                    author=metadata.author,
                    source_language=metadata.source_language,
                    target_language=metadata.target_language,
                    source_format="epub",
                    core_properties=core,
                )
        except EpubImportError:
            raise
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise EpubImportError(f"cannot read EPUB {path}: {exc}") from exc
        document = BookDocument(
            document_id=_document_id(path), metadata=metadata,
            chapters=tuple(chapters), blocks=tuple(blocks),
        )
        document.validate()
        return document


def _decode_entries(document: BookDocument) -> dict[str, bytes]:
    raw = document.metadata.core_properties.get(_EPUB_PACKAGE_KEY)
    if not isinstance(raw, str):
        raise EpubExportError(
            "EPUB source package is unavailable; import the source EPUB before exporting"
        )
    try:
        encoded = json.loads(raw)
        return {str(name): base64.b64decode(value) for name, value in encoded.items()}
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise EpubExportError("EPUB source package metadata is corrupt") from exc


def _replace_element_text(element: Any, target: str) -> None:
    # Preserve nested lists inside list items. Their independent blocks are
    # translated separately and should not disappear when the parent changes.
    nested_lists = [child for child in element.find_all(_LIST_TAGS) if child.parent is element]
    for child in list(element.contents):
        if child not in nested_lists:
            child.extract()
    if nested_lists:
        element.insert(0, NavigableString(target))
    else:
        element.append(NavigableString(target))


class EpubRoundTripReport:
    """Semantic coverage result for an EPUB export/re-import."""

    def __init__(self, source: BookDocument, actual: BookDocument) -> None:
        self.expected_block_ids = tuple(block.block_id for block in source.blocks)
        self.actual_block_ids = tuple(block.block_id for block in actual.blocks)
        self.expected_kinds = tuple(block.kind for block in source.blocks)
        self.actual_kinds = tuple(block.kind for block in actual.blocks)
        self.expected_chapters = tuple(chapter.block_ids for chapter in source.chapters)
        self.actual_chapters = tuple(chapter.block_ids for chapter in actual.chapters)

    @property
    def missing_block_ids(self) -> tuple[str, ...]:
        return tuple(item for item in self.expected_block_ids if item not in self.actual_block_ids)

    @property
    def unexpected_block_ids(self) -> tuple[str, ...]:
        return tuple(item for item in self.actual_block_ids if item not in self.expected_block_ids)

    @property
    def reordered(self) -> bool:
        return (
            not self.missing_block_ids and not self.unexpected_block_ids
            and self.expected_block_ids != self.actual_block_ids
        )

    @property
    def ok(self) -> bool:
        return (
            self.expected_block_ids == self.actual_block_ids
            and self.expected_kinds == self.actual_kinds
            and self.expected_chapters == self.actual_chapters
        )


def validate_epub_round_trip(source: BookDocument, exported_path: Path) -> EpubRoundTripReport:
    actual = EpubReader().read(exported_path)
    return EpubRoundTripReport(source, actual)


def _run_epubcheck(path: Path) -> None:
    """Run the optional epubcheck executable when it is available."""
    checker = shutil.which("epubcheck")
    if checker is None:
        return
    try:
        result = subprocess.run(
            [checker, str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EpubExportError(f"epubcheck could not validate {path}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or "epubcheck reported errors").strip()
        raise EpubExportError(f"epubcheck rejected EPUB {path}: {detail[-2000:]}")


class EpubWriter:
    """Write target text into the original EPUB package atomically."""

    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []

    def write(self, document: BookDocument, translations: Mapping[str, str], path: Path) -> None:
        _require_epub()
        document.validate()
        self.findings = []
        entries = _decode_entries(document)
        opf_path = str(document.metadata.core_properties.get("epub.opf_path", ""))
        if not opf_path or opf_path not in entries:
            raise EpubExportError("EPUB source package metadata has no valid OPF path")
        assert BeautifulSoup is not None
        blocks_by_href: dict[str, list[BookBlock]] = defaultdict(list)
        for block in document.blocks:
            href = block.attrs.get("epub_href")
            if isinstance(href, str):
                blocks_by_href[href].append(block)
        rendered: dict[str, bytes] = {}
        for href, source_blocks in blocks_by_href.items():
            if href not in entries:
                raise EpubExportError(f"EPUB source XHTML item is missing: {href}")
            soup = BeautifulSoup(entries[href], "html.parser")
            _check_active_content(soup, href)
            elements = _semantic_elements(soup.find("body") or soup)
            if len(elements) != len(source_blocks):
                raise EpubExportError(
                    f"EPUB item {href} changed structure: expected {len(source_blocks)} blocks, "
                    f"found {len(elements)}"
                )
            for element, block in zip(elements, source_blocks, strict=True):
                element[_BLOCK_ID_ATTR] = block.block_id
                element[_BLOCK_KIND_ATTR] = block.kind.value
                if block.kind in {BlockKind.BLANK, BlockKind.SCENE_BREAK}:
                    target = block.source_text
                else:
                    try:
                        target = str(translations[block.block_id])
                    except KeyError as exc:
                        raise EpubExportError(f"missing translation for {block.block_id}") from exc
                    if not target.strip():
                        raise EpubExportError(f"empty translation for {block.block_id}")
                if target != block.source_text:
                    _replace_element_text(element, target)
                    if block.runs:
                        self.findings.append({
                            "rule": "inline_style_projection",
                            "severity": "warning",
                            "block_ids": [block.block_id],
                            "evidence": "target differs from source; block semantics retained",
                        })
                        warnings.warn(
                            f"could not project inline styles for {block.block_id}; "
                            "retained block style",
                            EpubExportWarning, stacklevel=2,
                        )
            rendered[href] = str(soup).encode("utf-8")
        output = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(output, "w") as package:
                # EPUB requires mimetype to be the first, uncompressed member.
                package.writestr("mimetype", entries["mimetype"], compress_type=zipfile.ZIP_STORED)
                for name, content in entries.items():
                    if name == "mimetype":
                        continue
                    package.writestr(
                        name, rendered.get(name, content), compress_type=zipfile.ZIP_DEFLATED
                    )
            report = validate_epub_round_trip(document, output)
            if not report.ok:
                raise EpubExportError(
                    "EPUB export failed semantic round-trip: "
                    f"missing={report.missing_block_ids}, "
                    f"unexpected={report.unexpected_block_ids}, "
                    f"reordered={report.reordered}"
                )
            _run_epubcheck(output)
            output.replace(path)
        except EpubExportError:
            raise
        except Exception as exc:
            raise EpubExportError(f"cannot export EPUB {path}: {exc}") from exc
        finally:
            output.unlink(missing_ok=True)


__all__ = [
    "EpubExportError", "EpubExportWarning", "EpubImportError", "EpubImportWarning",
    "EpubReader", "EpubRoundTripReport", "EpubWriter", "validate_epub_round_trip",
]
