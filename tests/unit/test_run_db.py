"""Unit tests for RunDatabase — uses in-memory SQLite, no file I/O."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bn_en_translate.utils.monitor import ResourceSummary
from bn_en_translate.utils.run_db import RunDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_s: int = 0) -> datetime:
    """Return a fixed UTC datetime, optionally offset by seconds."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta
    return base + timedelta(seconds=offset_s)


def _summary(**kwargs: float) -> ResourceSummary:
    defaults: dict[str, object] = dict(
        sample_count=10, duration_s=5.0,
        cpu_peak_pct=42.0, cpu_avg_pct=30.0,
        ram_peak_mib=2048.0, ram_avg_mib=1800.0,
        swap_peak_mib=0.0, swap_avg_mib=0.0,
        disk_read_mb=10.0, disk_write_mb=2.0,
        gpu_vram_peak_mib=1942.0, gpu_vram_avg_mib=1800.0,
        gpu_util_peak_pct=95.0, gpu_util_avg_pct=80.0,
    )
    defaults.update(kwargs)
    return ResourceSummary(**defaults)  # type: ignore[arg-type]


def _save(db: RunDatabase, run_id: str = "run-001", run_type: str = "benchmark",
          model_name: str = "nllb-600M", bleu: float | None = 65.0,
          started_offset: int = 0) -> None:
    db.save_run(
        run_id=run_id,
        run_type=run_type,
        model_name=model_name,
        started_at=_ts(started_offset),
        finished_at=_ts(started_offset + 30),
        status="ok",
        summary=_summary(),
        bleu_score=bleu,
        input_chars=5000,
        chars_per_sec=100.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> RunDatabase:
    return RunDatabase(db_path=":memory:")


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_run_db_creates_tables_on_init(db: RunDatabase) -> None:
    cur = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    )
    assert cur.fetchone() is not None


# ---------------------------------------------------------------------------
# save_run / get_run
# ---------------------------------------------------------------------------

def test_run_db_save_run_inserts_record(db: RunDatabase) -> None:
    _save(db)
    row = db.get_run("run-001")
    assert row is not None
    assert row["run_type"] == "benchmark"
    assert row["model_name"] == "nllb-600M"
    assert row["bleu_score"] == pytest.approx(65.0)


def test_run_db_save_run_stores_summary_fields(db: RunDatabase) -> None:
    _save(db)
    row = db.get_run("run-001")
    assert row is not None
    assert row["cpu_peak_pct"] == pytest.approx(42.0)
    assert row["ram_peak_mib"] == pytest.approx(2048.0)
    assert row["gpu_vram_peak_mib"] == pytest.approx(1942.0)
    assert row["disk_read_mb"] == pytest.approx(10.0)


def test_run_db_save_run_upserts_on_same_run_id(db: RunDatabase) -> None:
    _save(db, run_id="run-001", bleu=60.0)
    _save(db, run_id="run-001", bleu=70.0)  # should overwrite
    row = db.get_run("run-001")
    assert row is not None
    assert row["bleu_score"] == pytest.approx(70.0)
    assert db.count_runs() == 1  # still one row


def test_run_db_get_run_returns_none_for_missing(db: RunDatabase) -> None:
    assert db.get_run("nonexistent") is None


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

def test_run_db_list_runs_returns_all(db: RunDatabase) -> None:
    _save(db, run_id="r1", started_offset=0)
    _save(db, run_id="r2", started_offset=60)
    rows = db.list_runs()
    assert len(rows) == 2


def test_run_db_list_runs_ordered_by_started_at_desc(db: RunDatabase) -> None:
    _save(db, run_id="r1", started_offset=0)
    _save(db, run_id="r2", started_offset=120)
    _save(db, run_id="r3", started_offset=60)
    rows = db.list_runs()
    run_ids = [r["run_id"] for r in rows]
    assert run_ids == ["r2", "r3", "r1"]


def test_run_db_list_runs_filters_by_run_type(db: RunDatabase) -> None:
    _save(db, run_id="b1", run_type="benchmark")
    _save(db, run_id="f1", run_type="finetune")
    rows = db.list_runs(run_type="benchmark")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "b1"


def test_run_db_list_runs_filters_by_model_name(db: RunDatabase) -> None:
    _save(db, run_id="r1", model_name="nllb-600M")
    _save(db, run_id="r2", model_name="nllb-1.3B")
    rows = db.list_runs(model_name="nllb-1.3B")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r2"


def test_run_db_list_runs_respects_limit(db: RunDatabase) -> None:
    for i in range(10):
        _save(db, run_id=f"r{i}", started_offset=i * 10)
    rows = db.list_runs(limit=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# get_trend
# ---------------------------------------------------------------------------

def test_run_db_get_trend_returns_metric_values(db: RunDatabase) -> None:
    _save(db, run_id="r1", bleu=60.0, started_offset=0)
    _save(db, run_id="r2", bleu=63.0, started_offset=60)
    _save(db, run_id="r3", bleu=65.0, started_offset=120)
    trend = db.get_trend("bleu_score")
    assert len(trend) == 3
    values = [v for _, v in trend]
    assert values == pytest.approx([60.0, 63.0, 65.0])


def test_run_db_get_trend_oldest_first(db: RunDatabase) -> None:
    _save(db, run_id="r1", bleu=60.0, started_offset=0)
    _save(db, run_id="r2", bleu=65.0, started_offset=60)
    trend = db.get_trend("bleu_score")
    assert trend[0][1] < trend[-1][1]  # older score < newer score


def test_run_db_get_trend_filters_by_run_type(db: RunDatabase) -> None:
    _save(db, run_id="b1", run_type="benchmark", bleu=65.0)
    _save(db, run_id="f1", run_type="finetune", bleu=55.0)
    trend = db.get_trend("bleu_score", run_type="benchmark")
    assert len(trend) == 1
    assert trend[0][1] == pytest.approx(65.0)


def test_run_db_get_trend_skips_null_metric(db: RunDatabase) -> None:
    _save(db, run_id="r1", bleu=None)   # bleu_score is NULL
    _save(db, run_id="r2", bleu=65.0)
    trend = db.get_trend("bleu_score")
    assert len(trend) == 1


def test_run_db_get_trend_invalid_metric_raises(db: RunDatabase) -> None:
    with pytest.raises(ValueError, match="Invalid metric"):
        db.get_trend("DROP TABLE runs")


# ---------------------------------------------------------------------------
# count_runs
# ---------------------------------------------------------------------------

def test_run_db_count_runs_empty(db: RunDatabase) -> None:
    assert db.count_runs() == 0


def test_run_db_count_runs_with_filter(db: RunDatabase) -> None:
    _save(db, run_id="b1", run_type="benchmark")
    _save(db, run_id="b2", run_type="benchmark")
    _save(db, run_id="f1", run_type="finetune")
    assert db.count_runs(run_type="benchmark") == 2
    assert db.count_runs(run_type="finetune") == 1


# ---------------------------------------------------------------------------
# Context manager and file creation
# ---------------------------------------------------------------------------

def test_run_db_context_manager_closes_connection() -> None:
    with RunDatabase(db_path=":memory:") as db:
        _save(db)
    # Connection is closed — further queries should raise
    with pytest.raises(Exception):
        db._conn.execute("SELECT 1")


def test_run_db_creates_parent_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "subdir" / "nested" / "runs.db"
    assert not db_path.parent.exists()
    db = RunDatabase(db_path=db_path)
    db.close()
    assert db_path.parent.exists()
    assert db_path.exists()
