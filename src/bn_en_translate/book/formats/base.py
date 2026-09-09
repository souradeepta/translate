"""Protocols shared by structure-preserving document formats."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from bn_en_translate.book.schema import BookDocument


class DocumentReader(Protocol):
    """Parse a source document into its stable semantic representation."""

    def read(self, path: Path) -> BookDocument: ...


class DocumentWriter(Protocol):
    """Write selected translations while preserving supported structure."""

    def write(
        self, document: BookDocument, translations: Mapping[str, str], path: Path
    ) -> None: ...
