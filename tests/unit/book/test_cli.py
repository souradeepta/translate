from __future__ import annotations

from click.testing import CliRunner

from bn_en_translate.book_cli import main


def test_init_and_inspect_text_project() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("source.txt", "w", encoding="utf-8") as source:
            source.write("প্রথম।\n\nদ্বিতীয়।")
        result = runner.invoke(main, ["init", "source.txt", "--project", "project"])
        assert result.exit_code == 0, result.output
        assert "milmmt-46-4b" in result.output

        result = runner.invoke(main, ["inspect", "project"])
        assert result.exit_code == 0, result.output
        assert "Blocks: 2" in result.output
        assert "pending: 2" in result.output


def test_init_rejects_non_text_source() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("source.md", "w", encoding="utf-8") as source:
            source.write("# source")
        result = runner.invoke(main, ["init", "source.md", "--project", "project"])
    assert result.exit_code != 0
    assert "only UTF-8 .txt" in result.output
