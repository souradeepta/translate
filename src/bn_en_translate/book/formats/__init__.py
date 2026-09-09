"""Document reader and writer implementations and extension registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bn_en_translate.book.formats.base import DocumentReader, DocumentWriter
from bn_en_translate.book.formats.text import TextReader, TextWriter

ReaderFactory = Callable[[], DocumentReader]
WriterFactory = Callable[[], DocumentWriter]

_READERS: dict[str, ReaderFactory] = {".txt": TextReader}
_WRITERS: dict[str, WriterFactory] = {".txt": TextWriter}


def register_format(extension: str, *, reader: ReaderFactory, writer: WriterFactory) -> None:
    """Register a format by extension, rejecting ambiguous registrations."""
    normalized = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    if not normalized or normalized == ".":
        raise ValueError("format extension must not be empty")
    _READERS[normalized] = reader
    _WRITERS[normalized] = writer


def reader_for(path: Path | str) -> DocumentReader:
    extension = Path(path).suffix.lower()
    try:
        return _READERS[extension]()
    except KeyError as exc:
        raise ValueError(f"no book reader registered for {extension or '<none>'}") from exc


def writer_for(path: Path | str) -> DocumentWriter:
    extension = Path(path).suffix.lower()
    try:
        return _WRITERS[extension]()
    except KeyError as exc:
        raise ValueError(f"no book writer registered for {extension or '<none>'}") from exc


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(set(_READERS) & set(_WRITERS)))


__all__ = [
    "TextReader",
    "TextWriter",
    "reader_for",
    "register_format",
    "supported_extensions",
    "writer_for",
]
