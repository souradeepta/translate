"""Transactional state store for resumable book translation stages."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from importlib.resources import files
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

STORE_SCHEMA_VERSION = 1
_MIGRATIONS = ((1, "0001_initial.sql"),)


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
        """Apply each released migration exactly once, with a recoverable backup."""
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > STORE_SCHEMA_VERSION:
            raise ValueError(f"state database schema {version} is newer than this application")
        if version == STORE_SCHEMA_VERSION:
            return
        if version < STORE_SCHEMA_VERSION and self.path.exists() and self.path.stat().st_size:
            # WAL content must be folded into the main file before the backup is
            # copied, otherwise restoring it could silently lose committed rows.
            self.connection.execute("PRAGMA wal_checkpoint(FULL)")
            backup = self.path.with_suffix(self.path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(self.path, backup)
        with self.transaction() as connection:
            for migration_version, filename in _MIGRATIONS:
                if migration_version <= version:
                    continue
                sql = files("bn_en_translate.book.migrations").joinpath(filename).read_text(
                    encoding="utf-8"
                )
                # The project migrations intentionally contain plain DDL only;
                # execute statements under our explicit transaction rather than
                # ``executescript``, which commits pending work implicitly.
                for statement in sql.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                self._add_legacy_columns(connection)
                connection.execute(f"PRAGMA user_version = {migration_version}")
                connection.execute(
                    "INSERT INTO project_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(migration_version),),
                )

    @staticmethod
    def _add_legacy_columns(connection: sqlite3.Connection) -> None:
        """Bring the unversioned prototype schema up to the v1 contract."""
        columns: dict[str, dict[str, str]] = {
            "units": {
                "approval_needs_revalidation": "INTEGER NOT NULL DEFAULT 0",
            },
            "attempts": {
                "context_hash": "TEXT NOT NULL DEFAULT ''",
                "model": "TEXT",
                "model_revision": "TEXT",
                "prompt_version": "TEXT",
                "raw_response": "TEXT",
                "error_type": "TEXT",
                "started_at": "TEXT",
                "finished_at": "TEXT",
            },
        }
        for table, additions in columns.items():
            existing_columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for name, definition in additions.items():
                if name not in existing_columns:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def rollback_migration(self) -> None:
        """Restore the pre-migration backup, if one was created."""
        backup = self.path.with_suffix(self.path.suffix + ".bak")
        if not backup.is_file():
            raise FileNotFoundError(f"migration backup does not exist: {backup}")
        self.close()
        shutil.copy2(backup, self.path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")

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
                        "UPDATE units SET source_hash = ?, "
                        "status = CASE WHEN approved_attempt_id IS NULL THEN 'stale' "
                        "ELSE status END, "
                        "approval_needs_revalidation = CASE WHEN approved_attempt_id IS NULL "
                        "THEN approval_needs_revalidation ELSE 1 END, updated_at = ? "
                        "WHERE block_id = ?",
                        (block.source_hash, self._now(), block.block_id),
                    )
            # Mark changed source blocks and their graph dependents in one atomic
            # transaction while preserving approved pointers.
            for block in document.blocks:
                self._mark_stale_dependents(connection, "source", block.block_id, block.source_hash)

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
        model: str | None = None,
        model_revision: str | None = None,
        prompt_version: str | None = None,
        context_hash: str = "",
        raw_response: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> int:
        if not target_text.strip():
            raise ValueError("attempt target_text must not be empty")
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO attempts("
                "block_id, stage, source_hash, config_hash, context_hash, model, "
                "model_revision, prompt_version, target_text, raw_response, status, "
                "error_type, error_message, started_at, finished_at, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    block_id,
                    stage,
                    source_hash,
                    config_hash,
                    context_hash,
                    model,
                    model_revision,
                    prompt_version,
                    target_text,
                    raw_response,
                    status,
                    error_type,
                    error_message,
                    self._now(),
                    self._now(),
                    self._now(),
                ),
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
            if unit["status"] == "approved":
                raise ValueError("approved units are immutable; explicitly unlock before replacing")
            if status not in _TRANSITIONS[unit["status"]]:
                raise ValueError(f"illegal status transition {unit['status']} -> {status}")
            exists = connection.execute(
                "SELECT 1 FROM attempts WHERE id = ? AND block_id = ?", (attempt_id, block_id)
            ).fetchone()
            if exists is None:
                raise ValueError("attempt does not belong to unit")
            if status == "approved":
                connection.execute(
                    "UPDATE units SET status = ?, selected_attempt_id = ?, "
                    "approved_attempt_id = ?, lease_owner = NULL, lease_expires_at = NULL, "
                    "updated_at = ? WHERE block_id = ?",
                    (status, attempt_id, attempt_id, self._now(), block_id),
                )
            else:
                connection.execute(
                    "UPDATE units SET status = ?, selected_attempt_id = ?, lease_owner = NULL, "
                    "lease_expires_at = NULL, updated_at = ? WHERE block_id = ?",
                    (status, attempt_id, self._now(), block_id),
                )

    def approve_attempt(self, block_id: str, attempt_id: int) -> None:
        """Select an attempt as the human-approved immutable candidate."""
        with self.transaction() as connection:
            unit = connection.execute(
                "SELECT status, approved_attempt_id FROM units WHERE block_id = ?", (block_id,)
            ).fetchone()
            if unit is None:
                raise KeyError(block_id)
            if (
                unit["approved_attempt_id"] is not None
                and unit["approved_attempt_id"] != attempt_id
            ):
                raise ValueError("unit already has an approved attempt")
            exists = connection.execute(
                "SELECT 1 FROM attempts WHERE id = ? AND block_id = ?", (attempt_id, block_id)
            ).fetchone()
            if exists is None:
                raise ValueError("attempt does not belong to unit")
            connection.execute(
                "UPDATE units SET status='approved', selected_attempt_id=?, "
                "approved_attempt_id=?, lease_owner=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE block_id=?",
                (attempt_id, attempt_id, self._now(), block_id),
            )

    def transition_status(
        self,
        block_id: str,
        from_status: str,
        to_status: str,
        *,
        attempt_id: int | None = None,
        owner: str | None = None,
    ) -> bool:
        """Compare-and-set a unit status, optionally checking lease ownership."""
        if to_status not in _TRANSITIONS.get(from_status, set()):
            raise ValueError(f"illegal status transition {from_status} -> {to_status}")
        with self.transaction() as connection:
            clauses = ["block_id = ?", "status = ?"]
            params: list[object] = [block_id, from_status]
            if owner is not None:
                clauses.extend(["lease_owner = ?", "lease_expires_at >= ?"])
                params.extend([owner, self._now()])
            if attempt_id is not None:
                clauses.append("selected_attempt_id = ?")
                params.append(attempt_id)
            result = connection.execute(
                "UPDATE units SET status=?, updated_at=? WHERE " + " AND ".join(clauses),
                [to_status, self._now(), *params],
            )
            return result.rowcount == 1

    def release_lease(self, block_id: str, owner: str) -> bool:
        with self.transaction() as connection:
            result = connection.execute(
                "UPDATE units SET status='pending', lease_owner=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE block_id=? AND status='running' AND lease_owner=?",
                (self._now(), block_id, owner),
            )
            return result.rowcount == 1

    def renew_lease(self, block_id: str, owner: str, lease_seconds: int = 300) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        with self.transaction() as connection:
            result = connection.execute(
                "UPDATE units SET lease_expires_at=?, updated_at=? WHERE block_id=? "
                "AND status='running' AND lease_owner=? AND lease_expires_at >= ?",
                (
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    now.isoformat(),
                    block_id,
                    owner,
                    now.isoformat(),
                ),
            )
            return result.rowcount == 1

    def mark_stale_for_dependency(self, kind: str, key: str, dependency_hash: str) -> int:
        """Invalidate every selected output dependent on a changed input.

        An approved translation remains selected and approved, but carries an
        explicit revalidation flag.  It must never be silently replaced merely
        because its source/context changed.
        """
        with self.transaction() as connection:
            return self._mark_stale_dependents(connection, kind, key, dependency_hash)

    def _mark_stale_dependents(
        self, connection: sqlite3.Connection, kind: str, key: str, dependency_hash: str
    ) -> int:
        """Propagate invalidation over dependency keys, preserving approvals.

        The first hop compares the recorded dependency hash with its new value.
        Every later hop is a selected unit whose output changed validity; attempts
        that record that unit as a context/target dependency are invalidated even
        though their stored hash is intentionally opaque to this repository.
        """
        affected = 0
        queue: list[tuple[str, str, str | None]] = [(kind, key, dependency_hash)]
        visited_keys: set[str] = set()
        while queue:
            current_kind, current_key, current_hash = queue.pop(0)
            if current_key in visited_keys:
                continue
            visited_keys.add(current_key)
            predicate = "d.dependency_key=?"
            params: tuple[str, ...] = (current_key,)
            if current_hash is not None:
                predicate = "d.dependency_kind=? AND d.dependency_key=? AND d.dependency_hash != ?"
                params = (current_kind, current_key, current_hash)
            rows = connection.execute(
                "SELECT DISTINCT a.block_id, a.id FROM attempts a "
                "JOIN attempt_dependencies d ON d.attempt_id=a.id "
                "JOIN units u ON u.selected_attempt_id=a.id "
                "WHERE " + predicate,
                params,
            ).fetchall()
            for row in rows:
                result = connection.execute(
                    "UPDATE units SET status=CASE WHEN approved_attempt_id IS NULL THEN 'stale' "
                    "ELSE status END, approval_needs_revalidation=CASE "
                    "WHEN approved_attempt_id IS NULL THEN approval_needs_revalidation ELSE 1 END, "
                    "updated_at=? WHERE block_id=? AND ("
                    "(approved_attempt_id IS NULL AND status != 'stale') OR "
                    "(approved_attempt_id IS NOT NULL AND approval_needs_revalidation = 0))",
                    (self._now(), row["block_id"]),
                )
                affected += result.rowcount
                # A dependency on a unit's selected target may be labeled
                # ``target``, ``context``, or another pipeline-specific kind;
                # its stable common identity is always the dependency key.
                queue.append(("unit", row["block_id"], None))
        return affected

    def put_context_asset(
        self,
        kind: str,
        asset_key: str,
        value: object,
        *,
        locked: bool = False,
        source: str = "machine",
    ) -> None:
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT locked FROM context_assets WHERE kind=? AND asset_key=?", (kind, asset_key)
            ).fetchone()
            if existing is not None and existing["locked"]:
                raise ValueError("locked context asset cannot be overwritten")
            connection.execute(
                "INSERT INTO context_assets(kind, asset_key, value_json, locked, source, "
                "updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(kind, asset_key) DO UPDATE SET "
                "value_json=excluded.value_json, locked=excluded.locked, source=excluded.source, "
                "updated_at=excluded.updated_at",
                (
                    kind,
                    asset_key,
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    int(locked),
                    source,
                    self._now(),
                ),
            )

    def get_context_asset(self, kind: str, asset_key: str) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            self.connection.execute(
                "SELECT * FROM context_assets WHERE kind=? AND asset_key=?", (kind, asset_key)
            ).fetchone(),
        )

    def create_run(
        self, run_id: str, stage: str, config_hash: str, *, status: str = "running"
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, stage, config_hash, status, started_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, stage, config_hash, status, self._now()),
            )

    def finish_run(self, run_id: str, *, status: str, summary: object | None = None) -> None:
        with self.transaction() as connection:
            if (
                connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
                is None
            ):
                raise KeyError(run_id)
            connection.execute(
                "UPDATE runs SET status=?, finished_at=?, summary_json=? WHERE run_id=?",
                (
                    status,
                    self._now(),
                    json.dumps(summary or {}, ensure_ascii=False, sort_keys=True),
                    run_id,
                ),
            )

    def add_qa_finding(
        self,
        run_id: str | None,
        rule: str,
        severity: str,
        block_ids: object,
        evidence: object,
        *,
        status: str = "open",
    ) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO qa_findings(run_id, rule, severity, block_ids_json, "
                "evidence_json, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    rule,
                    severity,
                    json.dumps(block_ids),
                    json.dumps(evidence),
                    status,
                    self._now(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a finding ID")
            return cursor.lastrowid

    def get_unit(self, block_id: str) -> sqlite3.Row | None:
        unit = self.connection.execute(
            "SELECT * FROM units WHERE block_id = ?", (block_id,)
        ).fetchone()
        return cast(sqlite3.Row | None, unit)
