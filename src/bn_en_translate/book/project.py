"""On-disk book project creation and opening."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from bn_en_translate.book.schema import BookBlock, BookDocument, Chapter, make_block_id
from bn_en_translate.book.serialization import (
    read_source_jsonl,
    write_document,
    write_source_jsonl,
)
from bn_en_translate.book.store import BookStore


class BookProject:
    """A self-contained, resumable translation project directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def structure_path(self) -> Path:
        return self.root / "structure.json"

    @property
    def config_path(self) -> Path:
        return self.root / "project.yaml"

    @property
    def state_path(self) -> Path:
        return self.root / "state.sqlite3"

    @classmethod
    def create(cls, root: Path, document: BookDocument) -> BookProject:
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"book project directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        for name in ("exports", "manifests", "reports"):
            (root / name).mkdir(exist_ok=True)
        project = cls(root)
        write_document(document, project.structure_path)
        write_source_jsonl(document, project.source_path)
        project._atomic_write_text(
            project.config_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "source": "ben_Beng",
                    "target": "eng_Latn",
                    "models": {
                        "draft": "milmmt-46-4b",
                        "fallback_draft": "milmmt-46-1b",
                        "revision": "gemma3:12b",
                    },
                },
                indent=2,
            )
            + "\n",
        )
        with BookStore(project.state_path) as store:
            store.register_document(document)
        return project

    @classmethod
    def open(cls, root: Path) -> BookProject:
        project = cls(root)
        if (
            not project.structure_path.is_file()
            or not project.source_path.is_file()
            or not project.state_path.is_file()
        ):
            raise FileNotFoundError(f"not a book project: {root}")
        return project

    def document(self) -> BookDocument:
        return read_source_jsonl(self.source_path)

    def store(self) -> BookStore:
        return BookStore(self.state_path)

    @property
    def source_path(self) -> Path:
        return self.root / "source.jsonl"

    def config(self) -> dict[str, Any]:
        """Load the human-editable JSON-compatible project YAML."""
        return cast(dict[str, Any], json.loads(self.config_path.read_text(encoding="utf-8")))

    @staticmethod
    def _atomic_write_text(path: Path, value: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8", newline="\n")
        temporary.replace(path)

    def reimport(self, incoming: BookDocument, *, dry_run: bool = False) -> ReconciliationReport:
        """Reconcile an explicit re-import; ambiguity never mutates project files."""
        report, reconciled = reconcile_documents(self.document(), incoming)
        if report.ambiguous:
            raise ReconciliationError(report)
        if not dry_run:
            write_source_jsonl(reconciled, self.source_path)
            write_document(reconciled, self.structure_path)
            with self.store() as store:
                store.register_document(reconciled)
        return report


@dataclass(frozen=True)
class ReconciliationReport:
    matched: dict[str, str]
    inserted: tuple[str, ...]
    ambiguous: tuple[dict[str, Any], ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.inserted or self.ambiguous or any(k != v for k, v in self.matched.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": dict(self.matched),
            "inserted": list(self.inserted),
            "ambiguous": list(self.ambiguous),
        }


class ReconciliationError(ValueError):
    def __init__(self, report: ReconciliationReport) -> None:
        super().__init__("ambiguous source reconciliation; project was not modified")
        self.report = report


def reconcile_documents(
    existing: BookDocument, incoming: BookDocument
) -> tuple[ReconciliationReport, BookDocument]:
    """Match incoming blocks to project IDs using locators, hashes, then anchors."""
    existing_by_chapter: dict[str, list[BookBlock]] = {
        chapter.chapter_id: [existing.blocks_by_id[block_id] for block_id in chapter.block_ids]
        for chapter in existing.chapters
    }
    incoming_by_chapter = {
        chapter.chapter_id: [incoming.blocks_by_id[block_id] for block_id in chapter.block_ids]
        for chapter in incoming.chapters
    }
    # Chapter IDs from the importer are ordinal identities; preserve project chapter IDs.
    existing_chapters_by_ordinal = {chapter.ordinal: chapter for chapter in existing.chapters}
    matched: dict[str, str] = {}
    inserted: list[str] = []
    ambiguous: list[dict[str, Any]] = []
    used_existing: set[str] = set()

    for incoming_chapter in incoming.chapters:
        old_chapter = existing_chapters_by_ordinal.get(incoming_chapter.ordinal)
        old_blocks = existing_by_chapter.get(old_chapter.chapter_id, []) if old_chapter else []
        new_blocks = incoming_by_chapter[incoming_chapter.chapter_id]
        old_by_locator: dict[str, list[BookBlock]] = {}
        for block in old_blocks:
            locator = block.attrs.get("source_locator")
            if locator is not None:
                old_by_locator.setdefault(str(locator), []).append(block)
        new_by_locator: dict[str, list[BookBlock]] = {}
        for block in new_blocks:
            locator = block.attrs.get("source_locator")
            if locator is not None:
                new_by_locator.setdefault(str(locator), []).append(block)
        for locator, candidates in new_by_locator.items():
            old_candidates = old_by_locator.get(locator, [])
            if len(candidates) == 1 and len(old_candidates) == 1:
                matched[candidates[0].block_id] = old_candidates[0].block_id
                used_existing.add(old_candidates[0].block_id)
            elif candidates and old_candidates:
                ambiguous.append(
                    {
                        "reason": "duplicate_source_locator",
                        "locator": locator,
                        "incoming": [item.block_id for item in candidates],
                        "candidates": [item.block_id for item in old_candidates],
                    }
                )

        for block in new_blocks:
            if block.block_id in matched:
                continue
            candidates = [
                old
                for old in old_blocks
                if old.block_id not in used_existing and old.source_hash == block.source_hash
            ]
            if len(candidates) == 1:
                matched[block.block_id] = candidates[0].block_id
                used_existing.add(candidates[0].block_id)

        old_indices = {block.block_id: i for i, block in enumerate(old_blocks)}
        new_indices = {block.block_id: i for i, block in enumerate(new_blocks)}
        for block in new_blocks:
            if block.block_id in matched:
                continue
            candidates = [old for old in old_blocks if old.block_id not in used_existing]
            prior = [
                old_indices[matched[new_blocks[i].block_id]]
                for i in range(new_indices[block.block_id])
                if new_blocks[i].block_id in matched
            ]
            following = [
                old_indices[matched[new_blocks[i].block_id]]
                for i in range(new_indices[block.block_id] + 1, len(new_blocks))
                if new_blocks[i].block_id in matched
            ]
            bounded = [
                old
                for old in candidates
                if (not prior or old_indices[old.block_id] > max(prior))
                and (not following or old_indices[old.block_id] < min(following))
            ]
            if len(bounded) == 1:
                matched[block.block_id] = bounded[0].block_id
                used_existing.add(bounded[0].block_id)
            elif len(bounded) > 1:
                ambiguous.append(
                    {
                        "reason": "multiple_sequence_candidates",
                        "incoming": block.block_id,
                        "candidates": [item.block_id for item in bounded],
                    }
                )

    # IDs for inserted records are monotonic project identities.  Ordinals remain
    # positional fields and are rebuilt from the incoming document so validation
    # can retain contiguous order without renumbering any existing IDs.
    used_numeric_ids = [
        int(item.block_id.rsplit("-b", 1)[1])
        for item in existing.blocks
        if "-b" in item.block_id and item.block_id.rsplit("-b", 1)[1].isdigit()
    ]
    max_ordinal = max(used_numeric_ids, default=0)
    position = 0
    new_block_records: list[BookBlock] = []
    id_map: dict[str, str] = {}
    for chapter in incoming.chapters:
        old_chapter = existing_chapters_by_ordinal.get(chapter.ordinal)
        chapter_id = old_chapter.chapter_id if old_chapter else chapter.chapter_id
        for block_id in chapter.block_ids:
            block = incoming.blocks_by_id[block_id]
            position += 1
            output_id = matched.get(block_id)
            if output_id is None:
                max_ordinal += 1
                output_id = make_block_id(chapter.ordinal, max_ordinal)
                inserted.append(output_id)
            id_map[block_id] = output_id
            new_block_records.append(
                BookBlock.create(
                    block_id=output_id,
                    chapter_id=chapter_id,
                    ordinal=position,
                    kind=block.kind,
                    source_text=block.source_text,
                    runs=block.runs,
                    attrs=dict(block.attrs),
                )
            )
    chapters = tuple(
        Chapter(
            chapter_id=existing_chapters_by_ordinal.get(chapter.ordinal, chapter).chapter_id,
            ordinal=chapter.ordinal,
            title=chapter.title,
            block_ids=tuple(id_map[block_id] for block_id in chapter.block_ids),
        )
        for chapter in incoming.chapters
    )
    reconciled = BookDocument(
        existing.document_id, incoming.metadata, chapters, tuple(new_block_records)
    )
    reconciled.validate()
    return ReconciliationReport(id_map, tuple(inserted), tuple(ambiguous)), reconciled
