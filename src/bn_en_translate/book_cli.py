"""CLI for persistent, structure-aware book translation projects."""

from __future__ import annotations

from pathlib import Path

import click

from bn_en_translate.book.formats.text import TextReader
from bn_en_translate.book.project import BookProject
from bn_en_translate.book.schema import BookDocument


def _read_source(path: Path) -> BookDocument:
    if path.suffix.lower() != ".txt":
        raise click.UsageError("only UTF-8 .txt import is available in this release")
    return TextReader().read(path)


@click.group()
def main() -> None:
    """Create and inspect resumable book-translation projects."""


@main.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--project", "project_path", required=True, type=click.Path(path_type=Path))
def init(source: Path, project_path: Path) -> None:
    """Import SOURCE into a new PROJECT directory without translating it."""
    document = _read_source(source)
    project = BookProject.create(project_path, document)
    click.echo(f"Created project: {project.root}")
    click.echo(f"Chapters: {len(document.chapters)} | Blocks: {len(document.blocks)}")
    click.echo("Draft model: milmmt-46-4b (fallback: milmmt-46-1b)")
    click.echo("Revision model: gemma3:12b (loaded only after draft model unloads)")


@main.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path))
def inspect(project_path: Path) -> None:
    """Show structure and pending-work state for a PROJECT."""
    project = BookProject.open(project_path)
    document = project.document()
    with project.store() as store:
        status_rows = store.connection.execute(
            "SELECT status, COUNT(*) AS count FROM units GROUP BY status ORDER BY status"
        ).fetchall()
    click.echo(f"Project: {project.root}")
    click.echo(f"Document: {document.metadata.title or document.document_id}")
    click.echo(f"Chapters: {len(document.chapters)} | Blocks: {len(document.blocks)}")
    for row in status_rows:
        click.echo(f"{row['status']}: {row['count']}")


if __name__ == "__main__":
    main()
