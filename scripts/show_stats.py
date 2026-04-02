#!/usr/bin/env python3
"""CLI for querying run history and detecting regressions.

Commands:
    list         Show recent runs
    show         Show detail for one run
    trend        Show a metric trend over time
    compare      Compare two runs side by side
    regressions  Report runs where a metric degraded vs the prior N runs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path("monitor/runs.db")

_METRICS = [
    "bleu_score", "chars_per_sec", "duration_s",
    "cpu_peak_pct", "cpu_avg_pct",
    "ram_peak_mib", "ram_avg_mib",
    "swap_peak_mib", "swap_avg_mib",
    "disk_read_mb", "disk_write_mb",
    "gpu_vram_peak_mib", "gpu_vram_avg_mib",
    "gpu_util_peak_pct", "gpu_util_avg_pct",
]


def _open_db(db_path: Path) -> object:
    if not db_path.exists():
        click.echo(f"[error] Database not found: {db_path}", err=True)
        click.echo("Run a benchmark or fine-tune first to create it.", err=True)
        sys.exit(1)
    from bn_en_translate.utils.run_db import RunDatabase
    return RunDatabase(db_path=db_path)


def _fmt_row(row: dict[str, Any], keys: list[str], width: int = 14) -> str:
    return "  ".join(
        str(row.get(k, "—") or "—").ljust(width)[:width]
        for k in keys
    )


def _fmt_float(v: Any, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{decimals}f}"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--db", "db_path",
    default=str(_DEFAULT_DB),
    show_default=True,
    help="Path to the SQLite runs database.",
)
@click.pass_context
def cli(ctx: click.Context, db_path: str) -> None:
    """Resource usage and quality statistics for bn-en-translate runs."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db_path)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.option("--run-type", default=None, help="Filter by run type (benchmark/finetune/translate).")
@click.option("--model", default=None, help="Filter by model name.")
@click.option("--limit", default=20, show_default=True, help="Max rows to show.")
@click.pass_context
def list_runs(ctx: click.Context, run_type: str | None, model: str | None, limit: int) -> None:
    """Show recent runs in a summary table."""
    db = _open_db(ctx.obj["db_path"])
    rows = db.list_runs(run_type=run_type, model_name=model, limit=limit)  # type: ignore[union-attr]
    db.close()  # type: ignore[union-attr]

    if not rows:
        click.echo("No runs found.")
        return

    header_keys = ["run_id", "run_type", "model_name", "started_at", "duration_s",
                   "bleu_score", "gpu_vram_peak_mib", "ram_peak_mib", "status"]
    click.echo("  ".join(k.ljust(14)[:14] for k in header_keys))
    click.echo("-" * (16 * len(header_keys)))
    for row in rows:
        vals = {
            **row,
            "run_id": row["run_id"][:12],
            "started_at": (row.get("started_at") or "")[:16],
            "duration_s": _fmt_float(row.get("duration_s")),
            "bleu_score": _fmt_float(row.get("bleu_score"), decimals=1),
            "gpu_vram_peak_mib": _fmt_float(row.get("gpu_vram_peak_mib"), decimals=0),
            "ram_peak_mib": _fmt_float(row.get("ram_peak_mib"), decimals=0),
        }
        click.echo(_fmt_row(vals, header_keys))


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("run_id")
@click.pass_context
def show(ctx: click.Context, run_id: str) -> None:
    """Show full detail for a single run by RUN_ID (or prefix)."""
    db = _open_db(ctx.obj["db_path"])
    row = db.get_run(run_id)  # type: ignore[union-attr]

    # Fallback: search by prefix
    if row is None:
        rows = db.list_runs(limit=1000)  # type: ignore[union-attr]
        matches = [r for r in rows if r["run_id"].startswith(run_id)]
        row = matches[0] if matches else None

    db.close()  # type: ignore[union-attr]

    if row is None:
        click.echo(f"[error] Run '{run_id}' not found.", err=True)
        sys.exit(1)

    click.echo(f"\n{'─' * 60}")
    click.echo(f"  Run ID       : {row['run_id']}")
    click.echo(f"  Type         : {row['run_type']}")
    click.echo(f"  Model        : {row['model_name']}")
    click.echo(f"  Status       : {row['status']}")
    click.echo(f"  Started      : {row.get('started_at', '—')}")
    click.echo(f"  Finished     : {row.get('finished_at', '—')}")
    click.echo(f"  Duration     : {_fmt_float(row.get('duration_s'))} s")
    click.echo(f"\n  Quality")
    click.echo(f"    BLEU       : {_fmt_float(row.get('bleu_score'), 2)}")
    click.echo(f"    chars/s    : {_fmt_float(row.get('chars_per_sec'), 1)}")
    click.echo(f"\n  CPU")
    click.echo(f"    peak       : {_fmt_float(row.get('cpu_peak_pct'))} %")
    click.echo(f"    avg        : {_fmt_float(row.get('cpu_avg_pct'))} %")
    click.echo(f"\n  RAM")
    click.echo(f"    peak       : {_fmt_float(row.get('ram_peak_mib'), 0)} MiB")
    click.echo(f"    avg        : {_fmt_float(row.get('ram_avg_mib'), 0)} MiB")
    click.echo(f"  Swap peak    : {_fmt_float(row.get('swap_peak_mib'), 0)} MiB")
    click.echo(f"\n  GPU")
    click.echo(f"    VRAM peak  : {_fmt_float(row.get('gpu_vram_peak_mib'), 0)} MiB")
    click.echo(f"    VRAM avg   : {_fmt_float(row.get('gpu_vram_avg_mib'), 0)} MiB")
    click.echo(f"    util peak  : {_fmt_float(row.get('gpu_util_peak_pct'), 0)} %")
    click.echo(f"    util avg   : {_fmt_float(row.get('gpu_util_avg_pct'), 0)} %")
    click.echo(f"\n  Disk I/O")
    click.echo(f"    read       : {_fmt_float(row.get('disk_read_mb'))} MB")
    click.echo(f"    write      : {_fmt_float(row.get('disk_write_mb'))} MB")
    click.echo(f"\n  Samples      : {row.get('sample_count', '—')}"
               f" @ {_fmt_float(row.get('sample_interval_s'))} s interval")
    click.echo(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("metric", type=click.Choice(_METRICS))
@click.option("--run-type", default=None)
@click.option("--model", default=None)
@click.option("--limit", default=20, show_default=True)
@click.pass_context
def trend(
    ctx: click.Context,
    metric: str,
    run_type: str | None,
    model: str | None,
    limit: int,
) -> None:
    """Show a metric trend over time (oldest → newest)."""
    db = _open_db(ctx.obj["db_path"])
    points = db.get_trend(metric, run_type=run_type, model_name=model, limit=limit)  # type: ignore[union-attr]
    db.close()  # type: ignore[union-attr]

    if not points:
        click.echo(f"No data for metric '{metric}'.")
        return

    vals = [v for _, v in points]
    min_v, max_v = min(vals), max(vals)
    bar_width = 40

    click.echo(f"\n  {metric}  (n={len(points)}, min={min_v:.2f}, max={max_v:.2f})\n")
    for ts, v in points:
        frac = (v - min_v) / (max_v - min_v) if max_v != min_v else 1.0
        bar = "█" * int(frac * bar_width)
        click.echo(f"  {ts[:16]}  {bar:<{bar_width}}  {v:.2f}")
    click.echo()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("run_id_a")
@click.argument("run_id_b")
@click.pass_context
def compare(ctx: click.Context, run_id_a: str, run_id_b: str) -> None:
    """Compare two runs side by side."""
    db = _open_db(ctx.obj["db_path"])
    a = db.get_run(run_id_a)  # type: ignore[union-attr]
    b = db.get_run(run_id_b)  # type: ignore[union-attr]
    db.close()  # type: ignore[union-attr]

    for rid, row in [(run_id_a, a), (run_id_b, b)]:
        if row is None:
            click.echo(f"[error] Run '{rid}' not found.", err=True)
            sys.exit(1)

    fields = [
        ("bleu_score", "BLEU", 2),
        ("duration_s", "Duration (s)", 1),
        ("chars_per_sec", "chars/s", 1),
        ("cpu_peak_pct", "CPU peak %", 1),
        ("ram_peak_mib", "RAM peak MiB", 0),
        ("gpu_vram_peak_mib", "VRAM peak MiB", 0),
        ("gpu_util_peak_pct", "GPU util peak %", 1),
        ("swap_peak_mib", "Swap peak MiB", 0),
        ("disk_read_mb", "Disk read MB", 1),
    ]

    label_w = 20
    click.echo(f"\n  {'Metric':<{label_w}}  {'A':>12}  {'B':>12}  {'Δ':>10}")
    click.echo(f"  {'-' * label_w}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

    for key, label, dec in fields:
        av = a.get(key)  # type: ignore[union-attr]
        bv = b.get(key)  # type: ignore[union-attr]
        av_s = _fmt_float(av, dec)
        bv_s = _fmt_float(bv, dec)
        delta_s = "—"
        if av is not None and bv is not None:
            delta = float(bv) - float(av)
            sign = "+" if delta >= 0 else ""
            delta_s = f"{sign}{delta:.{dec}f}"
        click.echo(f"  {label:<{label_w}}  {av_s:>12}  {bv_s:>12}  {delta_s:>10}")
    click.echo()


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------

_REGRESSION_RULES: list[tuple[str, float, bool, str]] = [
    # (metric, threshold, higher_is_better, severity)
    ("bleu_score",        1.0,  True,  "WARNING"),
    ("bleu_score",        3.0,  True,  "CRITICAL"),
    ("duration_s",        0.20, False, "WARNING"),   # 20% slower
    ("gpu_vram_peak_mib", 200,  False, "WARNING"),   # 200 MiB more VRAM
    ("ram_peak_mib",      500,  False, "WARNING"),
    ("chars_per_sec",     0.15, True,  "WARNING"),   # 15% slower
]


@cli.command()
@click.option("--run-type", default=None)
@click.option("--model", default=None)
@click.option("--lookback", default=5, show_default=True, help="Prior runs to compare against.")
@click.pass_context
def regressions(
    ctx: click.Context,
    run_type: str | None,
    model: str | None,
    lookback: int,
) -> None:
    """Report regressions vs rolling average of prior N runs.

    Exits with code 1 if any CRITICAL regression is found.
    """
    db = _open_db(ctx.obj["db_path"])
    found_critical = False

    for metric, threshold, higher_is_better, severity in _REGRESSION_RULES:
        points = db.get_trend(metric, run_type=run_type, model_name=model, limit=lookback + 1)  # type: ignore[union-attr]
        if len(points) < lookback + 1:
            continue  # not enough history

        prior = [v for _, v in points[:-1]]
        latest_ts, latest_v = points[-1]
        avg_prior = sum(prior) / len(prior)

        if higher_is_better:
            drop = avg_prior - latest_v
            pct = drop / avg_prior if avg_prior else 0.0
            regressed = drop >= threshold or (metric in ("chars_per_sec",) and pct >= threshold)
        else:
            rise = latest_v - avg_prior
            pct = rise / avg_prior if avg_prior else 0.0
            regressed = rise >= threshold or (metric == "duration_s" and pct >= threshold)

        if regressed:
            click.echo(
                f"[{severity}] {metric}: latest={latest_v:.2f}, "
                f"prior_avg={avg_prior:.2f}  ({latest_ts[:16]})"
            )
            if severity == "CRITICAL":
                found_critical = True

    db.close()  # type: ignore[union-attr]

    if found_critical:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
