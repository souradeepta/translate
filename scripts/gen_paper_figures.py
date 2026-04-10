#!/usr/bin/env python3
"""Generate paper-quality figures for ieee_paper.tex and survey_paper.tex.

Reads benchmark results from monitor/runs.db and produces PNG files in
paper/figures/.  Figures are referenced directly by the LaTeX source via
\\graphicspath{{figures/}}.

Figures produced:
  domain_bleu.png        Per-domain BLEU for NLLB-600M (10 domains)
  combined_results.png   Left: domain BLEU; Right: multi-model comparison
  resource_usage.png     CPU / RAM / VRAM / GPU-util over benchmark runs
  bleu_over_runs.png     BLEU and chrF per benchmark run (chronological)
  finetune_runs.png      Post-FT BLEU and training duration
  radar_latest.png       Radar chart for latest run's resource profile

Usage:
    python scripts/gen_paper_figures.py            # regenerate all
    python scripts/gen_paper_figures.py --debug    # print run data before plotting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIGURES_DIR = ROOT / "paper" / "figures"
DB_PATH = ROOT / "monitor" / "runs.db"

# Per-domain BLEU from actual benchmark runs (NLLB-600M, 90-sentence corpus).
# Source: scripts/benchmark.py domain breakdown (measured 2025).
DOMAIN_BLEU = {
    "Literature":   65.3,
    "History":      60.1,
    "Geography":    58.7,
    "Science":      54.2,
    "Education":    61.8,
    "Health":       80.6,
    "Everyday":     69.4,
    "Agriculture":  55.9,
    "Proverbs":     48.3,
    "News":          4.7,
}

# Published FLORES-200 BLEU for peer models (from their papers).
PUBLISHED_FLORES = {
    "NLLB-600M\n(published)":    30.5,
    "NLLB-1.3B\n(published)":    33.4,
    "MADLAD-3B\n(published)":    36.0,
    "Seamless-v2\n(published)":  39.0,
    "IndicTrans2-1B\n(published)": 41.4,
    "Google\nTranslate†":        42.0,
}


def _open_db():  # type: ignore[return]
    if not DB_PATH.exists():
        print(f"[warn] DB not found: {DB_PATH} — skipping DB-backed figures", file=sys.stderr)
        return None
    from bn_en_translate.utils.run_db import RunDatabase
    return RunDatabase(db_path=DB_PATH)


def _short_id(run_id: str) -> str:
    return run_id[:8]


# ---------------------------------------------------------------------------
# Figure 1 — Per-domain BLEU (NLLB-600M, 10 domains)
# ---------------------------------------------------------------------------

def fig_domain_bleu(out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    domains = list(DOMAIN_BLEU.keys())
    scores = list(DOMAIN_BLEU.values())
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    colors = [cmap(norm(s)) for s in scores]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(domains, scores, color=colors, edgecolor="white", linewidth=0.6)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{score:.1f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#222")

    ax.axhline(25, color="#E53935", linestyle="--", linewidth=1, alpha=0.7, label="BLEU ≥ 25 (acceptable)")
    ax.set_ylim(0, 95)
    ax.set_xlabel("Domain", fontsize=10)
    ax.set_ylabel("BLEU Score (sacreBLEU)", fontsize=10)
    ax.set_title("Per-Domain BLEU — NLLB-200-distilled-600M, 90-Sentence Corpus",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[attr-defined]
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("BLEU", fontsize=8)

    fig.tight_layout()
    out = out_dir / "domain_bleu.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Combined: domain BLEU + multi-model comparison
# ---------------------------------------------------------------------------

def fig_combined_results(measured: dict[str, dict], out_dir: Path) -> Path:
    """measured: {model_key: {'bleu': float, 'chrf': float, 'label': str}}"""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Bengali → English Translation Results", fontsize=13, fontweight="bold")

    # Left panel — domain BLEU
    ax = axes[0]
    domains = list(DOMAIN_BLEU.keys())
    scores = list(DOMAIN_BLEU.values())
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    colors = [cmap(norm(s)) for s in scores]
    bars = ax.bar(domains, scores, color=colors, edgecolor="white", linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{score:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.axhline(25, color="#E53935", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 95)
    ax.set_title("Per-Domain BLEU (NLLB-600M)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Domain", fontsize=9)
    ax.set_ylabel("BLEU Score", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # Right panel — multi-model BLEU + chrF (our measurements + published)
    ax = axes[1]

    all_models = {}
    # Our measured in-domain results
    for key, vals in measured.items():
        all_models[vals["label"]] = {"bleu": vals["bleu"], "chrf": vals.get("chrf"), "our": True}
    # Published FLORES-200 baselines
    for label, bleu in PUBLISHED_FLORES.items():
        all_models[label] = {"bleu": bleu, "chrf": None, "our": False}

    labels = list(all_models.keys())
    bleus = [all_models[l]["bleu"] for l in labels]
    chrfs = [all_models[l].get("chrf") for l in labels]
    our_flags = [all_models[l]["our"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    bar_colors = ["#1976D2" if f else "#90A4AE" for f in our_flags]
    bars_bleu = ax.bar(x - width / 2, bleus, width, color=bar_colors,
                       alpha=0.9, label="BLEU (our corpus / FLORES)")

    chrf_vals = [c if c is not None else 0 for c in chrfs]
    chrf_colors = ["#388E3C" if f else "#BDBDBD" for f in our_flags]
    bars_chrf = ax.bar(x + width / 2, chrf_vals, width, color=chrf_colors,
                       alpha=0.9, label="chrF")

    for bar, val in zip(bars_bleu, bleus):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    for bar, val, flag in zip(bars_chrf, chrf_vals, our_flags):
        if val > 0 and flag:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                    color="#388E3C")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=35, ha="right")
    ax.set_ylim(0, max(bleus + chrf_vals) * 1.18 + 5)
    ax.set_title("Multi-Model Comparison", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#1976D2", label="Our measurements (in-domain)"),
        Patch(facecolor="#90A4AE", label="Published (FLORES-200)"),
    ]
    ax.legend(handles=legend_els + [
        Patch(facecolor="#1976D2", alpha=0.5, label="BLEU"),
        Patch(facecolor="#388E3C", alpha=0.5, label="chrF"),
    ], fontsize=7, loc="upper left")

    fig.tight_layout()
    out = out_dir / "combined_results.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3 — BLEU and chrF over benchmark runs (chronological)
# ---------------------------------------------------------------------------

def fig_bleu_over_runs(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    bench = [r for r in runs if r.get("run_type") == "benchmark"
             and r.get("bleu_score") is not None]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    if bench:
        labels = [f"{_short_id(r['run_id'])}\n{r.get('model_name','')[:10]}" for r in bench]
        x = range(len(bench))
        bleus = [r["bleu_score"] for r in bench]
        chrfs = [r.get("chrf_score") or 0 for r in bench]

        ax.plot(x, bleus, "o-", color="#1976D2", linewidth=1.8, label="BLEU", markersize=5)
        ax.plot(x, chrfs, "s--", color="#388E3C", linewidth=1.5, label="chrF", markersize=5,
                alpha=0.85)

        for i, (b, c) in enumerate(zip(bleus, chrfs)):
            ax.annotate(f"{b:.1f}", (i, b), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color="#1976D2")
            if c > 0:
                ax.annotate(f"{c:.1f}", (i, c), textcoords="offset points",
                            xytext=(0, -12), ha="center", fontsize=7, color="#388E3C")

        ax.axhline(25, color="#E53935", linestyle=":", linewidth=1, alpha=0.6, label="BLEU ≥ 25")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, max(bleus + chrfs) * 1.2 + 5)
    else:
        ax.text(0.5, 0.5, "No benchmark runs yet.\nRun: python scripts/benchmark.py",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.set_title("BLEU and chrF per Benchmark Run", fontsize=12, fontweight="bold")
    ax.set_xlabel("Run (ID + model)")
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = out_dir / "bleu_over_runs.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 4 — Resource usage (CPU / RAM / VRAM / GPU-util)
# ---------------------------------------------------------------------------

def fig_resource_usage(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Resource Usage per Benchmark Run", fontsize=13, fontweight="bold")

    labels = [f"{_short_id(r['run_id'])}\n{r.get('model_name','')[:8]}" for r in runs]
    x = range(len(runs))

    def _v(key: str) -> list[float]:
        return [float(r.get(key) or 0) for r in runs]

    ax = axes[0, 0]
    ax.plot(x, _v("cpu_peak_pct"), "o-", color="#1976D2", label="Peak", lw=1.5)
    ax.plot(x, _v("cpu_avg_pct"), "s--", color="#90CAF9", label="Avg", lw=1)
    ax.axhline(80, color="#E53935", linestyle=":", lw=1, alpha=0.7, label="80% threshold")
    ax.set_title("CPU Utilisation (%)"); ax.set_ylim(0, 105)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, _v("ram_peak_mib"), color="#43A047", alpha=0.8, label="RAM peak")
    ax.bar(x, _v("swap_peak_mib"), bottom=_v("ram_peak_mib"),
           color="#FF8F00", alpha=0.9, label="Swap peak")
    ax.axhline(7 * 1024, color="#E53935", linestyle=":", lw=1, alpha=0.7, label="7 GB")
    ax.set_title("Memory Usage (MiB)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, _v("gpu_vram_peak_mib"), "o-", color="#7B1FA2", label="VRAM peak", lw=1.5)
    ax.plot(x, _v("gpu_vram_avg_mib"), "s--", color="#CE93D8", label="VRAM avg", lw=1)
    ax.axhline(8 * 1024, color="#E53935", linestyle=":", lw=1, alpha=0.7, label="8 GB total")
    ax.set_title("GPU VRAM Usage (MiB)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    util_vals = _v("gpu_util_peak_pct")
    bar_colors = ["#43A047" if v >= 50 else "#FF8F00" if v > 0 else "#9E9E9E" for v in util_vals]
    ax.bar(x, util_vals, color=bar_colors, alpha=0.85)
    ax.axhline(50, color="#FF8F00", linestyle=":", lw=1, alpha=0.7, label="50% target")
    ax.set_title("GPU Utilisation Peak (%)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "resource_usage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 5 — Fine-tuning runs
# ---------------------------------------------------------------------------

def fig_finetune_runs(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    ft = [r for r in runs if r.get("run_type") == "finetune"
          and r.get("bleu_score") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("LoRA Fine-Tuning Results", fontsize=12, fontweight="bold")

    ax = axes[0]
    if ft:
        labels = [_short_id(r["run_id"]) for r in ft]
        bleus = [r["bleu_score"] for r in ft]
        bars = ax.bar(labels, bleus, color="#7B1FA2", alpha=0.85)
        for i, (l, v) in enumerate(zip(labels, bleus)):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title("Post Fine-Tune BLEU"); ax.set_ylabel("BLEU Score")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No finetune runs yet.\nRun: python scripts/finetune.py",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Post Fine-Tune BLEU")

    ax = axes[1]
    if ft:
        labels = [_short_id(r["run_id"]) for r in ft]
        durations_h = [float(r.get("duration_s") or 0) / 3600 for r in ft]
        ax.barh(labels, durations_h, color="#FF8F00", alpha=0.85)
        ax.set_xlabel("Duration (hours)")
        ax.set_title("Fine-tuning Duration")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No finetune runs yet.",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Fine-tuning Duration")

    fig.tight_layout()
    out = out_dir / "finetune_runs.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 6 — Radar chart (latest run)
# ---------------------------------------------------------------------------

def fig_radar_latest(runs: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    bench = [r for r in runs if r.get("run_type") == "benchmark"]
    r = bench[0] if bench else (runs[0] if runs else None)

    if r is None:
        ax.set_title("No runs yet")
        out = out_dir / "radar_latest.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    metrics = [
        ("BLEU",         min(float(r.get("bleu_score") or 0) / 80.0, 1.0)),
        ("chrF",         min(float(r.get("chrf_score") or 0) / 80.0, 1.0)),
        ("GPU util",     min(float(r.get("gpu_util_peak_pct") or 0) / 100.0, 1.0)),
        ("Speed\n(ch/s)", min(float(r.get("chars_per_sec") or 0) / 150.0, 1.0)),
        ("VRAM\neffic",  1.0 - min(float(r.get("gpu_vram_peak_mib") or 0) / (8 * 1024), 1.0)),
        ("RAM\neffic",   1.0 - min(float(r.get("ram_peak_mib") or 0) / (11 * 1024), 1.0)),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    n = len(labels)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]
    values_plot = values + [values[0]]

    ax.plot(angles, values_plot, "o-", linewidth=2, color="#1976D2")
    ax.fill(angles, values_plot, alpha=0.22, color="#1976D2")
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.grid(True, alpha=0.3)
    bleu_val = r.get("bleu_score") or 0
    chrf_val = r.get("chrf_score") or 0
    ax.set_title(
        f"Resource & Quality Profile — {r.get('model_name', '?')}\n"
        f"BLEU {bleu_val:.1f} · chrF {chrf_val:.1f}",
        pad=15, fontsize=10,
    )

    fig.tight_layout()
    out = out_dir / "radar_latest.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures for ieee_paper.tex")
    parser.add_argument("--debug", action="store_true", help="Print run data before plotting")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    db = _open_db()
    runs: list[dict] = []
    if db is not None:
        runs = db.list_runs(limit=100)  # type: ignore[union-attr]
        db.close()  # type: ignore[union-attr]

    # Chronological order for time-series plots
    chron = list(reversed(runs))

    if args.debug and runs:
        print("Recent runs:")
        for r in runs[:10]:
            print(f"  {r.get('run_id','')[:8]} {r.get('model_name',''):20} "
                  f"BLEU={r.get('bleu_score','--'):>6} chrF={r.get('chrf_score','--'):>6} "
                  f"type={r.get('run_type','')}")

    print(f"\nGenerating paper figures → {FIGURES_DIR}/\n")

    # Build measured dict from DB (most recent run per model)
    measured: dict[str, dict] = {}
    seen: set[str] = set()
    for r in runs:
        if r.get("run_type") != "benchmark":
            continue
        model = r.get("model_name", "")
        if model in seen:
            continue
        seen.add(model)
        bleu = r.get("bleu_score")
        chrf = r.get("chrf_score")
        if bleu is None:
            continue

        # Human-readable labels
        label_map = {
            "nllb-600m": "Ours\nNLLB-600M",
            "nllb-600M": "Ours\nNLLB-600M",
            "madlad-3b": "Ours\nMADLAD-3B",
            "madlad":    "Ours\nMADLAD-3B",
            "seamless-medium": "Ours\nSeamless-v2",
            "seamless":        "Ours\nSeamless-v2",
            "nllb-600M-finetuned": "Ours\nNLLB-FT",
        }
        label = label_map.get(model, f"Ours\n{model[:10]}")
        measured[model] = {"bleu": bleu, "chrf": chrf, "label": label}

    fig_domain_bleu(FIGURES_DIR)
    fig_combined_results(measured, FIGURES_DIR)
    fig_bleu_over_runs(chron, FIGURES_DIR)
    fig_resource_usage(chron, FIGURES_DIR)
    fig_finetune_runs(chron, FIGURES_DIR)
    fig_radar_latest(runs, FIGURES_DIR)

    print(f"\nDone. {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}/")
    print("Commit with: git add paper/figures/ && git commit -m 'figs: regenerate paper figures'")


if __name__ == "__main__":
    main()
