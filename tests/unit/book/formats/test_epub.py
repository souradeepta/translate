"""EPUB behavior tests; the optional dependency gate keeps the base suite hermetic."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("ebooklib")
pytest.importorskip("bs4")

import bn_en_translate.book.formats.epub as epub
from bn_en_translate.book.schema import BlockKind


def _write_epub(path: Path, chapter: str, *, extra: dict[str, bytes] | None = None) -> None:
    entries = {
        "mimetype": b"application/epub+zip",
        "META-INF/container.xml": (
            b'<?xml version="1.0"?><container '
            b'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            b'<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>'
        ),
        "OEBPS/content.opf": (
            b'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            b'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Fixture</dc:title>'
            b"</metadata><manifest><item id=\"chapter\" href=\"chapter.xhtml\" "
            b'media-type="application/xhtml+xml"/></manifest><spine><itemref idref="chapter"/>'
            b"</spine></package>"
        ),
        "OEBPS/chapter.xhtml": chapter.encode("utf-8"),
    }
    entries.update(extra or {})
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("mimetype", entries.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
        for name, content in entries.items():
            package.writestr(name, content)


def test_unchanged_blocks_keep_links_and_emphasis(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    target = tmp_path / "target.epub"
    _write_epub(
        source,
        "<html><body><h1>Chapter</h1><p><strong>Bold</strong> "
        '<a href="https://example.test">link</a></p></body></html>',
    )
    document = epub.EpubReader().read(source)
    epub.EpubWriter().write(
        document, {block.block_id: block.source_text for block in document.blocks}, target
    )
    exported = epub.EpubReader().read(target)
    assert [(run.text, run.bold, run.href) for run in exported.blocks[1].runs] == [
        ("Bold", True, None),
        (" ", False, None),
        ("link", False, "https://example.test"),
    ]


def test_changed_block_drops_stale_inline_markup(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    target = tmp_path / "target.epub"
    _write_epub(
        source,
        "<html><body><h1>Chapter</h1><p><strong>Old</strong> "
        '<a href="https://example.test">link</a></p></body></html>',
    )
    document = epub.EpubReader().read(source)
    translations = {block.block_id: ("New target" if block.ordinal == 2 else block.source_text)
                    for block in document.blocks}
    epub.EpubWriter().write(document, translations, target)
    with zipfile.ZipFile(target) as package:
        markup = package.read("OEBPS/chapter.xhtml").decode("utf-8")
    assert "New target" in markup
    assert "<strong>" not in markup and "href=\"https://example.test\"" not in markup


def test_aside_note_is_not_imported_twice_and_div_content_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    _write_epub(
        source,
        "<html><body><h1>Chapter</h1><aside epub:type=\"footnote\"><p>Note</p></aside>"
        "<div>Div paragraph</div><table><tr><td>Table text</td></tr></table></body></html>",
    )
    with pytest.warns(epub.EpubImportWarning, match="tables"):
        with pytest.warns(epub.EpubImportWarning, match="div-only"):
            document = epub.EpubReader().read(source)
    assert [block.kind for block in document.blocks] == [
        BlockKind.CHAPTER_HEADING,
        BlockKind.FOOTNOTE,
        BlockKind.PARAGRAPH,
    ]
    assert [block.source_text for block in document.blocks] == [
        "Chapter", "Note", "Div paragraph"
    ]


def test_non_spine_stylesheet_external_fetch_is_blocked(tmp_path: Path) -> None:
    source = tmp_path / "source.epub"
    _write_epub(source, "<html><body><h1>Chapter</h1></body></html>", extra={
        "OEBPS/extra.css": b".cover { background: url(https://example.test/a.png) }",
    })
    with pytest.raises(epub.EpubImportError, match="external CSS fetch"):
        epub.EpubReader().read(source)


def test_epubcheck_is_invoked_when_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[list[str]] = []
    monkeypatch.setattr(epub.shutil, "which", lambda name: "/usr/bin/epubcheck")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        checked.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(epub.subprocess, "run", fake_run)
    output = tmp_path / "output.epub"
    epub._run_epubcheck(output)
    assert checked == [["/usr/bin/epubcheck", str(output)]]
