"""SQLite-backed database for recording resource usage per run.

Each call to ``save_run()`` upserts one row into the ``runs`` table.
The DB is created automatically on first use at ``monitor/runs.db``.

Usage::

    from bn_en_translate.utils.run_db import RunDatabase

    with RunDatabase() as db:
        db.save_run(run_id=monitor.run_id, ..., summary=monitor.summary)
        rows = db.list_runs(run_type="benchmark", limit=10)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from bn_en_translate.utils.monitor import ResourceSummary

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL UNIQUE,
    run_type            TEXT    NOT NULL,
    model_name          TEXT    NOT NULL,
    started_at          TEXT    NOT NULL,
    finished_at         TEXT,
    duration_s          REAL,
    status              TEXT    NOT NULL DEFAULT 'ok',
    error_msg           TEXT,

    -- Translation / benchmark metadata
    input_chars         INTEGER,
    bleu_score          REAL,
    chars_per_sec       REAL,

    -- CPU
    cpu_peak_pct        REAL,
    cpu_avg_pct         REAL,

    -- RAM (MiB)
    ram_peak_mib        REAL,
    ram_avg_mib         REAL,

    -- Swap (MiB)
    swap_peak_mib       REAL,
    swap_avg_mib        REAL,

    -- Disk I/O (MB delta during the run)
    disk_read_mb        REAL,
    disk_write_mb       REAL,

    -- GPU
    gpu_vram_peak_mib   REAL,
    gpu_vram_avg_mib    REAL,
    gpu_util_peak_pct   REAL,
    gpu_util_avg_pct    REAL,

    -- Sampling metadata
    sample_count        INTEGER,
    sample_interval_s   REAL
);

CREATE INDEX IF NOT EXISTS idx_runs_run_type  ON runs(run_type);
CREATE INDEX IF NOT EXISTS idx_runs_model     ON runs(model_name);
CREATE INDEX IF NOT EXISTS idx_runs_started   ON runs(started_at);
"""


class RunDatabase:
    """SQLite store for per-run resource and quality statistics.

    Thread safety: not designed for concurrent writes. All writes happen
    from the main thread after the ResourceMonitor thread has joined.
    """

    def __init__(self, db_path: Path | str = "monitor/runs.db") -> None:
        self._db_path = db_path
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_run(
        self,
        *,
        run_id: str,
        run_type: str,
        model_name: str,
        started_at: datetime,
        finished_at: datetime,
        status: str = "ok",
        error_msg: str | None = None,
        summary: ResourceSummary,
        input_chars: int | None = None,
        bleu_score: float | None = None,
        chars_per_sec: float | None = None,
        sample_interval_s: float = 2.0,
    ) -> None:
        """Upsert a run record.  ``run_id`` is the natural key."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, run_type, model_name, started_at, finished_at,
                duration_s, status, error_msg,
                input_chars, bleu_score, chars_per_sec,
                cpu_peak_pct, cpu_avg_pct,
                ram_peak_mib, ram_avg_mib,
                swap_peak_mib, swap_avg_mib,
                disk_read_mb, disk_write_mb,
                gpu_vram_peak_mib, gpu_vram_avg_mib,
                gpu_util_peak_pct, gpu_util_avg_pct,
                sample_count, sample_interval_s
            ) VALUES (
                :run_id, :run_type, :model_name, :started_at, :finished_at,
                :duration_s, :status, :error_msg,
                :input_chars, :bleu_score, :chars_per_sec,
                :cpu_peak_pct, :cpu_avg_pct,
                :ram_peak_mib, :ram_avg_mib,
                :swap_peak_mib, :swap_avg_mib,
                :disk_read_mb, :disk_write_mb,
                :gpu_vram_peak_mib, :gpu_vram_avg_mib,
                :gpu_util_peak_pct, :gpu_util_avg_pct,
                :sample_count, :sample_interval_s
            )
            """,
            {
                "run_id": run_id,
                "run_type": run_type,
                "model_name": model_name,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_s": summary.duration_s,
                "status": status,
                "error_msg": error_msg,
                "input_chars": input_chars,
                "bleu_score": bleu_score,
                "chars_per_sec": chars_per_sec,
                "cpu_peak_pct": summary.cpu_peak_pct,
                "cpu_avg_pct": summary.cpu_avg_pct,
                "ram_peak_mib": summary.ram_peak_mib,
                "ram_avg_mib": summary.ram_avg_mib,
                "swap_peak_mib": summary.swap_peak_mib,
                "swap_avg_mib": summary.swap_avg_mib,
                "disk_read_mb": summary.disk_read_mb,
                "disk_write_mb": summary.disk_write_mb,
                "gpu_vram_peak_mib": summary.gpu_vram_peak_mib,
                "gpu_vram_avg_mib": summary.gpu_vram_avg_mib,
                "gpu_util_peak_pct": summary.gpu_util_peak_pct,
                "gpu_util_avg_pct": summary.gpu_util_avg_pct,
                "sample_count": summary.sample_count,
                "sample_interval_s": sample_interval_s,
            },
        )
        self._conn.commit()
        logger.debug("Saved run %s (%s) to DB", run_id, run_type)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a single run as a dict, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(
        self,
        run_type: str | None = None,
        model_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return runs ordered by started_at DESC, optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        if run_type is not None:
            clauses.append("run_type = ?")
            params.append(run_type)
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT ?",  # noqa: S608
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trend(
        self,
        metric: str,
        run_type: str | None = None,
        model_name: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Return [(started_at_iso, metric_value), ...] ordered oldest-first.

        Only rows where the metric is non-NULL are returned.
        ``metric`` must be a valid column name in the ``runs`` table.
        """
        # Validate metric name against known columns to prevent SQL injection
        valid_columns = {
            "bleu_score", "chars_per_sec", "duration_s",
            "cpu_peak_pct", "cpu_avg_pct",
            "ram_peak_mib", "ram_avg_mib",
            "swap_peak_mib", "swap_avg_mib",
            "disk_read_mb", "disk_write_mb",
            "gpu_vram_peak_mib", "gpu_vram_avg_mib",
            "gpu_util_peak_pct", "gpu_util_avg_pct",
        }
        if metric not in valid_columns:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of: {sorted(valid_columns)}"
            )

        clauses = [f"{metric} IS NOT NULL"]
        params: list[Any] = []
        if run_type is not None:
            clauses.append("run_type = ?")
            params.append(run_type)
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)

        where = "WHERE " + " AND ".join(clauses)
        params.append(limit)

        # Subquery: take latest `limit` rows, then order oldest-first for trend view
        rows = self._conn.execute(
            f"""
            SELECT started_at, {metric}
            FROM (
                SELECT started_at, {metric}
                FROM runs
                {where}
                ORDER BY started_at DESC
                LIMIT ?
            )
            ORDER BY started_at ASC
            """,  # noqa: S608
            params,
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def count_runs(
        self,
        run_type: str | None = None,
        model_name: str | None = None,
    ) -> int:
        """Return total number of runs matching the optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if run_type is not None:
            clauses.append("run_type = ?")
            params.append(run_type)
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM runs {where}",  # noqa: S608
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RunDatabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
