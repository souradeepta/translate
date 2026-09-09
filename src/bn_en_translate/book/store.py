"""Transactional state store for resumable book translation stages."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from bn_en_translate.book.schema import BookDocument

_TERMINAL_STATUSES = {"drafted", "revised", "needs_review", "approved", "failed", "stale"}
_TRANSITIONS = {
    "pending": {"running"},
    "running": _TERMINAL_STATUSES | {"pending"},
    "drafted": {"running", "revised", "needs_review", "approved", "stale"},
    "revised": {"running", "needs_review", "approved", "stale"},
    "needs_review": {"running", "approved", "stale"},
    "approved": {"stale"},
    "failed": {"pending", "running"},
    "stale": {"pending", "running"},
}


class BookStore:
    """SQLite-backed, append-only attempt store with expiring leases."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> BookStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS project_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS units (
                block_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                selected_attempt_id INTEGER,
                approved_attempt_id INTEGER,
                lease_owner TEXT,
                lease_expires_at TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL REFERENCES units(block_id),
                stage TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                target_text TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempt_dependencies (
                attempt_id INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
                dependency_kind TEXT NOT NULL,
                dependency_key TEXT NOT NULL,
                dependency_hash TEXT NOT NULL,
                PRIMARY KEY (attempt_id, dependency_kind, dependency_key)
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def register_document(self, document: BookDocument) -> None:
        document.validate()
        with self.transaction() as connection:
            for block in document.blocks:
                existing = connection.execute(
                    "SELECT source_hash FROM units WHERE block_id = ?", (block.block_id,)
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO units(block_id, source_hash, updated_at) VALUES (?, ?, ?)",
                        (block.block_id, block.source_hash, self._now()),
                    )
                elif existing["source_hash"] != block.source_hash:
                    connection.execute(
                        "UPDATE units SET source_hash = ?, status = 'stale', updated_at = ? "
                        "WHERE block_id = ? AND approved_attempt_id IS NULL",
                        (block.source_hash, self._now(), block.block_id),
                    )

    def claim_pending(self, owner: str, lease_seconds: int = 300) -> sqlite3.Row | None:
        if not owner:
            raise ValueError("lease owner must not be empty")
        now = datetime.now(UTC)
        expiry = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT block_id FROM units WHERE status IN ('pending', 'stale') "
                "OR (status = 'running' AND lease_expires_at < ?) "
                "ORDER BY block_id LIMIT 1",
                (now.isoformat(),),
            ).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                "UPDATE units SET status = 'running', lease_owner = ?, lease_expires_at = ?, "
                "updated_at = ? WHERE block_id = ? AND (status IN ('pending', 'stale') "
                "OR (status = 'running' AND lease_expires_at < ?))",
                (owner, expiry, now.isoformat(), row["block_id"], now.isoformat()),
            )
            if updated.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM units WHERE block_id = ?", (row["block_id"],)
            ).fetchone()
            return cast(sqlite3.Row | None, claimed)

    def add_attempt(
        self,
        *,
        block_id: str,
        stage: str,
        source_hash: str,
        config_hash: str,
        target_text: str,
        status: str,
        dependencies: dict[tuple[str, str], str] | None = None,
    ) -> int:
        if not target_text.strip():
            raise ValueError("attempt target_text must not be empty")
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO attempts("
                "block_id, stage, source_hash, config_hash, target_text, status, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (block_id, stage, source_hash, config_hash, target_text, status, self._now()),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an attempt ID")
            attempt_id = cursor.lastrowid
            for (kind, key), dependency_hash in (dependencies or {}).items():
                connection.execute(
                    "INSERT INTO attempt_dependencies VALUES (?, ?, ?, ?)",
                    (attempt_id, kind, key, dependency_hash),
                )
            return attempt_id

    def select_attempt(self, block_id: str, attempt_id: int, status: str) -> None:
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"selected status must be terminal, got {status}")
        with self.transaction() as connection:
            unit = connection.execute(
                "SELECT status FROM units WHERE block_id = ?", (block_id,)
            ).fetchone()
            if unit is None:
                raise KeyError(block_id)
            if status not in _TRANSITIONS[unit["status"]]:
                raise ValueError(f"illegal status transition {unit['status']} -> {status}")
            exists = connection.execute(
                "SELECT 1 FROM attempts WHERE id = ? AND block_id = ?", (attempt_id, block_id)
            ).fetchone()
            if exists is None:
                raise ValueError("attempt does not belong to unit")
            connection.execute(
                "UPDATE units SET status = ?, selected_attempt_id = ?, lease_owner = NULL, "
                "lease_expires_at = NULL, updated_at = ? WHERE block_id = ?",
                (status, attempt_id, self._now(), block_id),
            )

    def mark_stale_for_dependency(self, kind: str, key: str, dependency_hash: str) -> int:
        """Mark non-approved selected output stale when a recorded dependency changes."""
        with self.transaction() as connection:
            result = connection.execute(
                "UPDATE units SET status = 'stale', updated_at = ? "
                "WHERE approved_attempt_id IS NULL "
                "AND selected_attempt_id IN (SELECT attempt_id FROM attempt_dependencies "
                "WHERE dependency_kind = ? AND dependency_key = ? AND dependency_hash != ?)",
                (self._now(), kind, key, dependency_hash),
            )
            return result.rowcount

    def get_unit(self, block_id: str) -> sqlite3.Row | None:
        unit = self.connection.execute(
            "SELECT * FROM units WHERE block_id = ?", (block_id,)
        ).fetchone()
        return cast(sqlite3.Row | None, unit)
