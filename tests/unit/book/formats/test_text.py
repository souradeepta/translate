from __future__ import annotations

from bn_en_translate.book.formats.text import TextReader, TextWriter
from bn_en_translate.book.schema import BlockKind


def test_text_import_preserves_scene_break_and_chapter_order(tmp_path) -> None:
    source = tmp_path / "book.bn.txt"
    source.write_text("Chapter 1\n\nপ্রথম।\n\n***\n\nChapter 2\n\nদ্বিতীয়।", encoding="utf-8")

    document = TextReader().read(source)

    assert len(document.chapters) == 2
    assert [block.kind for block in document.blocks] == [
        BlockKind.CHAPTER_HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.SCENE_BREAK,
        BlockKind.CHAPTER_HEADING,
        BlockKind.PARAGRAPH,
    ]


def test_text_export_is_atomic_and_uses_selected_translations(tmp_path) -> None:
    source = tmp_path / "book.bn.txt"
    source.write_text("প্রথম।\n\nদ্বিতীয়।", encoding="utf-8")
    document = TextReader().read(source)
    translations = {block.block_id: f"EN {block.ordinal}" for block in document.blocks}
    destination = tmp_path / "out.en.txt"

    TextWriter().write(document, translations, destination)

    assert destination.read_text(encoding="utf-8") == "EN 1\n\nEN 2"
    assert not destination.with_suffix(".txt.tmp").exists()
