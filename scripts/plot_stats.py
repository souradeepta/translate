#!/usr/bin/env python3
"""Generate performance charts from monitor/runs.db.

Produces PNG plots saved to monitor/plots/:
  - bleu_over_runs.png          BLEU score per run (benchmark)
  - resource_usage.png          CPU / RAM / VRAM / GPU-util per run
  - duration_vs_input.png       Translation speed scatter (duration vs input chars)
  - finetune_loss.png           Training loss over finetune runs
  - radar_latest.png            Radar / spider chart of latest run's resource profile

Usage:
    python scripts/plot_stats.py                      # all plots
    python scripts/plot_stats.py --run-type benchmark # filter by run type
    python scripts/plot_stats.py --limit 20           # last N runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

PLOTS_DIR = ROOT / "monitor" / "plots"
DB_PATH = ROOT / "monitor" / "runs.db"


def _open_db() -> object:
    if not DB_PATH.exists():
        print(f"[error] Database not found: {DB_PATH}", file=sys.stderr)
        print("Run a benchmark first: python scripts/benchmark.py --models nllb-600M --sentences 5",
              file=sys.stderr)
        sys.exit(1)
    from bn_en_translate.utils.run_db import RunDatabase
    return RunDatabase(db_path=DB_PATH)


def _short_id(run_id: str) -> str:
    return run_id[:8]


# ---------------------------------------------------------------------------
# BLEU over runs
# ---------------------------------------------------------------------------

def plot_bleu(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    benchmark_runs = [r for r in runs if r.get("bleu_score") is not None
                      and r.get("run_type") == "benchmark"]

    fig, ax = plt.subplots(figsize=(10, 4))

    if benchmark_runs:
        labels = [_short_id(r["run_id"]) for r in benchmark_runs]
        scores = [r["bleu_score"] for r in benchmark_runs]
        colors = ["#2196F3" if s >= 60 else "#FF9800" if s >= 40 else "#F44336"
                  for s in scores]
        bars = ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{score:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.axhline(y=25, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7,
                   label="BLEU≥25 threshold")
        ax.set_ylim(0, max(scores) * 1.2 + 5)
    else:
        ax.text(0.5, 0.5, "No benchmark runs with BLEU data yet.\nRun: python scripts/benchmark.py",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="grey")

    ax.set_title("BLEU Score per Benchmark Run", fontsize=13, fontweight="bold")
    ax.set_xlabel("Run ID (first 8 chars)")
    ax.set_ylabel("BLEU Score")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "bleu_over_runs.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Resource usage (CPU / RAM / VRAM / GPU util)
# ---------------------------------------------------------------------------

def plot_resource_usage(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = [_short_id(r["run_id"]) for r in runs]
    x = range(len(runs))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Resource Usage per Run", fontsize=13, fontweight="bold")

    def _vals(key: str) -> list[float]:
        return [float(r.get(key) or 0) for r in runs]

    # CPU peak
    ax = axes[0, 0]
    ax.plot(x, _vals("cpu_peak_pct"), "o-", color="#2196F3", label="Peak", linewidth=1.5)
    ax.plot(x, _vals("cpu_avg_pct"), "s--", color="#90CAF9", label="Avg", linewidth=1)
    ax.axhline(80, color="#F44336", linestyle=":", linewidth=1, alpha=0.7, label="80% threshold")
    ax.set_title("CPU Utilisation (%)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # RAM
    ax = axes[0, 1]
    ax.bar(x, _vals("ram_peak_mib"), color="#4CAF50", alpha=0.7, label="RAM peak MiB")
    ax.bar(x, _vals("swap_peak_mib"), bottom=_vals("ram_peak_mib"),
           color="#FF9800", alpha=0.8, label="Swap peak MiB")
    ax.axhline(7 * 1024, color="#F44336", linestyle=":", linewidth=1, alpha=0.7,
               label="7 GB limit")
    ax.set_title("Memory Usage (MiB)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # VRAM
    ax = axes[1, 0]
    ax.plot(x, _vals("gpu_vram_peak_mib"), "o-", color="#9C27B0", label="VRAM peak MiB", linewidth=1.5)
    ax.plot(x, _vals("gpu_vram_avg_mib"), "s--", color="#CE93D8", label="VRAM avg MiB", linewidth=1)
    ax.axhline(8 * 1024, color="#F44336", linestyle=":", linewidth=1, alpha=0.7,
               label="8 GB VRAM total")
    ax.set_title("GPU VRAM Usage (MiB)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # GPU util
    ax = axes[1, 1]
    util_vals = _vals("gpu_util_peak_pct")
    colors = ["#4CAF50" if v >= 50 else "#FF9800" if v > 0 else "#9E9E9E" for v in util_vals]
    ax.bar(x, util_vals, color=colors, alpha=0.8)
    ax.axhline(50, color="#FF9800", linestyle=":", linewidth=1, alpha=0.7, label="50% target")
    ax.set_title("GPU Utilisation Peak (%)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "resource_usage.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Duration vs input chars scatter
# ---------------------------------------------------------------------------

def plot_duration_vs_input(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in runs if r.get("duration_s") and r.get("input_chars")]
    fig, ax = plt.subplots(figsize=(8, 5))

    if valid:
        x = [r["input_chars"] for r in valid]
        y = [r["duration_s"] for r in valid]
        bleu = [r.get("bleu_score") or 0 for r in valid]
        sc = ax.scatter(x, y, c=bleu, cmap="RdYlGn", vmin=0, vmax=100,
                        s=60, edgecolors="grey", linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="BLEU Score")

        # Linear fit
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(min(x), max(x), 100)
            ax.plot(xs, m * xs + b, "--", color="#2196F3", linewidth=1,
                    label=f"Linear fit (slope={m*1000:.2f} ms/char)")
            ax.legend(fontsize=8)

        ax.set_xlabel("Input Characters")
        ax.set_ylabel("Duration (s)")
        ax.set_title("Translation Duration vs Input Size\n(colour = BLEU score)", fontsize=11)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No runs with duration+input_chars data yet.",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Duration vs Input Size")

    fig.tight_layout()
    out_path = out_dir / "duration_vs_input.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Fine-tuning loss
# ---------------------------------------------------------------------------

def plot_finetune_loss(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    ft_runs = [r for r in runs
               if r.get("run_type") == "finetune" and r.get("bleu_score") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Fine-tuning Runs", fontsize=12, fontweight="bold")

    ax = axes[0]
    if ft_runs:
        labels = [_short_id(r["run_id"]) for r in ft_runs]
        bleus = [r["bleu_score"] for r in ft_runs]
        ax.bar(labels, bleus, color="#9C27B0", alpha=0.8)
        for i, (label, v) in enumerate(zip(labels, bleus)):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title("Post-FT BLEU")
        ax.set_ylabel("BLEU Score")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No finetune runs yet.\nRun: python scripts/finetune.py",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Post-FT BLEU")

    ax = axes[1]
    if ft_runs:
        labels = [_short_id(r["run_id"]) for r in ft_runs]
        durations_h = [float(r.get("duration_s") or 0) / 3600 for r in ft_runs]
        ax.barh(labels, durations_h, color="#FF9800", alpha=0.8)
        ax.set_xlabel("Duration (hours)")
        ax.set_title("Fine-tuning Duration")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No finetune runs yet.",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Fine-tuning Duration")

    fig.tight_layout()
    out_path = out_dir / "finetune_runs.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Radar chart of latest run
# ---------------------------------------------------------------------------

def plot_radar_latest(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    if not runs:
        ax.set_title("No runs yet")
        out_path = out_dir / "radar_latest.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    r = runs[0]  # most recent

    # Normalise each metric to 0–1 (higher = better efficiency)
    metrics = [
        ("BLEU",         min(float(r.get("bleu_score") or 0) / 70.0, 1.0)),
        ("GPU util",     min(float(r.get("gpu_util_peak_pct") or 0) / 100.0, 1.0)),
        ("Speed\n(ch/s)", min(float(r.get("chars_per_sec") or 0) / 120.0, 1.0)),
        ("VRAM\neffic",  1.0 - min(float(r.get("gpu_vram_peak_mib") or 0) / (8 * 1024), 1.0)),
        ("RAM\neffic",   1.0 - min(float(r.get("ram_peak_mib") or 0) / (11 * 1024), 1.0)),
        ("CPU\neffic",   1.0 - min(float(r.get("cpu_avg_pct") or 0) / 100.0, 1.0)),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    n = len(labels)

    angles = [i * 2 * np.pi / n for i in range(n)] + [0]
    values_plot = values + [values[0]]

    ax.plot(angles, values_plot, "o-", linewidth=2, color="#2196F3")
    ax.fill(angles, values_plot, alpha=0.25, color="#2196F3")
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Latest Run Profile\n{_short_id(r['run_id'])}… ({r.get('run_type', '?')} / {r.get('model_name', '?')})",
        pad=15, fontsize=10,
    )

    fig.tight_layout()
    out_path = out_dir / "radar_latest.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance plots from monitor/runs.db")
    parser.add_argument("--run-type", default=None, help="Filter: benchmark|finetune|translate")
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument("--limit", type=int, default=50, help="Max runs to include")
    parser.add_argument("--out-dir", default=str(PLOTS_DIR),
                        help=f"Output directory (default: {PLOTS_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = _open_db()
    runs = db.list_runs(run_type=args.run_type, model_name=args.model, limit=args.limit)  # type: ignore[union-attr]
    db.close()  # type: ignore[union-attr]

    if not runs:
        print("No runs found. Run a benchmark first:")
        print("  python scripts/benchmark.py --models nllb-600M --sentences 5")
        sys.exit(0)

    # Reverse for chronological order in most plots
    chron = list(reversed(runs))

    print(f"\nGenerating plots for {len(runs)} runs → {out_dir}/\n")
    plot_bleu(chron, out_dir)
    plot_resource_usage(chron, out_dir)
    plot_duration_vs_input(chron, out_dir)
    plot_finetune_loss(chron, out_dir)
    plot_radar_latest(runs, out_dir)  # runs[0] is most recent
    print(f"\nDone. Open {out_dir}/ to view the plots.")


if __name__ == "__main__":
    main()
