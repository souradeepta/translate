"""TDD tests for file I/O utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from bn_en_translate.utils.file_io import is_valid_utf8_file, read_story, write_translation


def test_read_story_returns_string(fixtures_dir: Path) -> None:
    text = read_story(fixtures_dir / "sample_short.bn.txt")
    assert isinstance(text, str)
    assert len(text) > 0


def test_read_story_contains_bengali(fixtures_dir: Path) -> None:
    text = read_story(fixtures_dir / "sample_short.bn.txt")
    # Bengali characters have Unicode range U+0980–U+09FF
    assert any("\u0980" <= c <= "\u09ff" for c in text)


def test_read_story_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_story("/nonexistent/path/story.txt")


def test_write_translation_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "output.txt"
    write_translation("Hello world.", str(out))
    assert out.exists()


def test_write_translation_content_correct(tmp_path: Path) -> None:
    out = tmp_path / "output.txt"
    content = "The quick brown fox."
    write_translation(content, str(out))
    assert out.read_text(encoding="utf-8") == content


def test_write_translation_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "subdir" / "deep" / "output.txt"
    write_translation("text", str(out))
    assert out.exists()


def test_is_valid_utf8_file_true_for_valid(fixtures_dir: Path) -> None:
    assert is_valid_utf8_file(fixtures_dir / "sample_short.bn.txt") is True


def test_is_valid_utf8_file_false_for_missing() -> None:
    assert is_valid_utf8_file("/nonexistent/file.txt") is False


def test_roundtrip_write_then_read(tmp_path: Path) -> None:
    original = "বাংলা গল্প — A Bengali Story"
    path = tmp_path / "story.txt"
    write_translation(original, str(path))
    recovered = read_story(str(path))
    assert recovered == original
