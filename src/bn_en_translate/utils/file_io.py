"""File I/O helpers for Bengali story translation."""

from __future__ import annotations

from pathlib import Path


def read_story(path: str | Path) -> str:
    """Read a Bengali story file, returning the content as a UTF-8 string."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Story file not found: {path}")
    return p.read_text(encoding="utf-8")


def write_translation(text: str, path: str | Path) -> None:
    """Write translated English text to a UTF-8 file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def is_valid_utf8_file(path: str | Path) -> bool:
    """Return True if the file exists and is valid UTF-8."""
    try:
        Path(path).read_text(encoding="utf-8")
        return True
    except (FileNotFoundError, UnicodeDecodeError):
        return False
