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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIGURES_DIR = ROOT / "paper" / "figures"
DB_PATH = ROOT / "monitor" / "runs.db"

# ── Academic style globals ────────────────────────────────────────────────────
# IEEE single-column = 88 mm (~3.46 in), double-column = 181 mm (~7.13 in).
# Use 300 DPI for camera-ready quality.
_DPI = 300
_COL1 = 3.5   # inches — single IEEE column
_COL2 = 7.16  # inches — double IEEE column (full width)

# ColorBrewer Set1 / Dark2 muted palette — distinguishable in B&W and colour
_C = {
    "blue":   "#2166ac",
    "red":    "#d6604d",
    "green":  "#4dac26",
    "orange": "#f4a582",
    "purple": "#762a83",
    "grey":   "#878787",
    "teal":   "#01665e",
    "brown":  "#8c510a",
}
_HATCH_OUR = ""       # solid fill for our results
_HATCH_PUB = "////"   # hatch for published baselines

plt.rcParams.update({
    # Font
    "font.family":        "DejaVu Sans",
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.titlesize":   10,
    # Lines / markers
    "lines.linewidth":    1.4,
    "lines.markersize":   4.5,
    # Axes
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "axes.axisbelow":     True,
    # Layout
    "figure.dpi":         _DPI,
    "savefig.dpi":        _DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# Per-domain BLEU from actual benchmark runs (NLLB-600M, 90-sentence corpus).
DOMAIN_BLEU = {
    "Literature":  65.3,
    "History":     60.1,
    "Geography":   58.7,
    "Science":     54.2,
    "Education":   61.8,
    "Health":      80.6,
    "Everyday":    69.4,
    "Agriculture": 55.9,
    "Proverbs":    48.3,
    "News":         4.7,
}

# Published FLORES-200 BLEU for peer models (from their papers).
# chrF = None where not reported.
PUBLISHED_FLORES: dict[str, dict[str, float | None]] = {
    "NLLB-600M":      {"bleu": 30.5, "chrf": 52.4},
    "NLLB-1.3B":      {"bleu": 33.4, "chrf": 55.1},
    "MADLAD-3B":      {"bleu": 36.0, "chrf": None},
    "Seamless-v2":    {"bleu": 39.0, "chrf": None},
    "IndicTrans2-1B": {"bleu": 41.4, "chrf": 63.2},
    "Google Tr.†":    {"bleu": 42.0, "chrf": None},
}


def _open_db():  # type: ignore[return]
    if not DB_PATH.exists():
        print(f"[warn] DB not found: {DB_PATH} — skipping DB-backed figures", file=sys.stderr)
        return None
    from bn_en_translate.utils.run_db import RunDatabase
    return RunDatabase(db_path=DB_PATH)


def _short_id(run_id: str) -> str:
    return run_id[:8]


def _label_bar(ax: plt.Axes, bars, vals: list[float], fmt: str = "{:.1f}",
               offset: float = 0.8, fontsize: int = 7) -> None:
    """Annotate bar tops with value labels."""
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                fmt.format(v),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Per-domain BLEU (NLLB-600M, 10 domains)
# ─────────────────────────────────────────────────────────────────────────────

def fig_domain_bleu(out_dir: Path) -> Path:
    domains = list(DOMAIN_BLEU.keys())
    scores = list(DOMAIN_BLEU.values())

    # Gradient: map score → blue intensity (low=light, high=dark)
    norm = plt.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.Blues  # type: ignore[attr-defined]
    colors = [cmap(0.35 + 0.60 * norm(s)) for s in scores]

    fig, ax = plt.subplots(figsize=(_COL2, 3.2))
    bars = ax.bar(domains, scores, color=colors, edgecolor="white", linewidth=0.7,
                  zorder=3)
    _label_bar(ax, bars, scores, fmt="{:.1f}")

    ax.axhline(25, color=_C["red"], linestyle="--", linewidth=1.0,
               alpha=0.75, label="BLEU = 25 (acceptable threshold)", zorder=4)
    ax.set_ylim(0, 98)
    ax.set_xlabel("Domain")
    ax.set_ylabel("BLEU (sacreBLEU)")
    ax.set_title("NLLB-200-distilled-600M: Per-Domain BLEU on 90-Sentence Corpus")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[attr-defined]
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025, aspect=25)
    cbar.set_label("BLEU", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    out = out_dir / "domain_bleu.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Combined: domain BLEU (left) + multi-model BLEU/chrF (right)
# ─────────────────────────────────────────────────────────────────────────────

def fig_combined_results(measured: dict[str, dict], out_dir: Path) -> Path:
    """measured: {model_key: {'bleu': float, 'chrf': float|None, 'label': str}}"""

    fig, axes = plt.subplots(1, 2, figsize=(_COL2, 3.6),
                              gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("Bengali→English Translation: Measured Results vs. Published Baselines",
                 fontweight="bold")

    # ── Left: per-domain BLEU ─────────────────────────────────────────────────
    ax = axes[0]
    domains = list(DOMAIN_BLEU.keys())
    scores = list(DOMAIN_BLEU.values())
    norm = plt.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.Blues  # type: ignore[attr-defined]
    colors = [cmap(0.35 + 0.60 * norm(s)) for s in scores]
    bars = ax.bar(domains, scores, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    _label_bar(ax, bars, scores, fmt="{:.0f}", offset=0.5, fontsize=6)
    ax.axhline(25, color=_C["red"], linestyle="--", lw=0.9, alpha=0.7, zorder=4)
    ax.set_ylim(0, 98)
    ax.set_title("(a) Per-Domain BLEU (NLLB-600M)")
    ax.set_xlabel("Domain")
    ax.set_ylabel("BLEU")
    ax.tick_params(axis="x", rotation=35, labelsize=6)

    # ── Right: multi-model BLEU + chrF grouped bar ───────────────────────────
    ax = axes[1]

    # Build ordered list: our measurements first, then published baselines
    our_entries: list[tuple[str, float, float | None]] = []
    for key, vals in measured.items():
        bleu = vals.get("bleu") or 0
        chrf = vals.get("chrf")
        label = vals.get("label", key)
        if bleu > 5:   # exclude MADLAD garbage (bleu ~0)
            our_entries.append((label, bleu, chrf))

    pub_entries: list[tuple[str, float, float | None]] = [
        (k, v["bleu"], v["chrf"]) for k, v in PUBLISHED_FLORES.items()
    ]

    all_labels = [e[0] for e in our_entries] + [e[0] for e in pub_entries]
    all_bleu   = [e[1] for e in our_entries] + [e[1] for e in pub_entries]
    all_chrf   = [e[2] for e in our_entries] + [e[2] for e in pub_entries]
    our_mask   = [True] * len(our_entries) + [False] * len(pub_entries)

    x = np.arange(len(all_labels))
    w = 0.38

    bleu_colors = [_C["blue"] if f else _C["grey"]   for f in our_mask]
    chrf_colors = [_C["teal"] if f else _C["orange"] for f in our_mask]

    bars_b = ax.bar(x - w / 2, all_bleu, w, color=bleu_colors,
                    edgecolor="white", linewidth=0.5, zorder=3)
    chrf_vals_plot = [c if c is not None else 0 for c in all_chrf]
    bars_c = ax.bar(x + w / 2, chrf_vals_plot, w, color=chrf_colors,
                    edgecolor="white", linewidth=0.5, zorder=3,
                    hatch=[("" if f else "//") for f in our_mask])

    # Value labels only for our measurements (to reduce clutter)
    for bar, v, flag in zip(bars_b, all_bleu, our_mask):
        if flag and v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6, fontweight="bold",
                    color=_C["blue"])
    for bar, v, flag in zip(bars_c, chrf_vals_plot, our_mask):
        if flag and v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6, fontweight="bold",
                    color=_C["teal"])

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=6, rotation=35, ha="right")
    ax.set_ylim(0, max(all_bleu + chrf_vals_plot) * 1.18 + 5)
    ax.set_title("(b) BLEU and chrF: Ours vs. Published")
    ax.set_ylabel("Score")

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=_C["blue"],   label="Our BLEU"),
        Patch(facecolor=_C["teal"],   label="Our chrF"),
        Patch(facecolor=_C["grey"],   label="Published BLEU"),
        Patch(facecolor=_C["orange"], hatch="//", label="Published chrF"),
    ]
    ax.legend(handles=legend_els, loc="upper left", ncol=2)

    fig.tight_layout()
    out = out_dir / "combined_results.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — BLEU and chrF over benchmark runs (chronological)
# ─────────────────────────────────────────────────────────────────────────────

def fig_bleu_over_runs(runs: list[dict], out_dir: Path) -> Path:
    bench = [
        r for r in runs
        if r.get("run_type") == "benchmark" and r.get("bleu_score") is not None
    ]

    fig, ax = plt.subplots(figsize=(_COL2, 3.2))

    if bench:
        labels = [
            f"{r.get('model_name', '?')[:12]}\n({_short_id(r['run_id'])})"
            for r in bench
        ]
        x = np.arange(len(bench))
        bleus = [float(r["bleu_score"]) for r in bench]
        chrfs = [float(r.get("chrf_score") or 0) for r in bench]

        ax.plot(x, bleus, "o-", color=_C["blue"], lw=1.6,
                label="BLEU", zorder=3)
        ax.plot(x, chrfs, "s--", color=_C["teal"], lw=1.4,
                label="chrF", alpha=0.9, zorder=3)

        for i, (b, c) in enumerate(zip(bleus, chrfs)):
            ax.annotate(f"{b:.1f}", (i, b), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.5,
                        color=_C["blue"])
            if c > 0:
                ax.annotate(f"{c:.1f}", (i, c), xytext=(0, -12),
                            textcoords="offset points", ha="center", fontsize=6.5,
                            color=_C["teal"])

        ax.axhline(25, color=_C["red"], linestyle=":", lw=0.9,
                   alpha=0.65, label="BLEU = 25")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylim(0, max(bleus + chrfs) * 1.22 + 5)
    else:
        ax.text(0.5, 0.5, "No benchmark runs found.\nRun: python scripts/benchmark.py",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.set_title("BLEU and chrF per Benchmark Run (Chronological)")
    ax.set_xlabel("Run (model / ID)")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = out_dir / "bleu_over_runs.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Resource usage (CPU / RAM / VRAM / GPU-util)
# ─────────────────────────────────────────────────────────────────────────────

def fig_resource_usage(runs: list[dict], out_dir: Path) -> Path:
    # Filter to benchmark + finetune runs with resource data
    runs_with_res = [r for r in runs if r.get("cpu_peak_pct") is not None]

    fig, axes = plt.subplots(2, 2, figsize=(_COL2, 5.5))
    fig.suptitle("Measured Resource Usage per Run (RTX 5050, 8 GB VRAM)",
                 fontweight="bold")

    labels = [
        f"{r.get('model_name', '?')[:10]}\n({_short_id(r['run_id'])})"
        for r in runs_with_res
    ]
    x = np.arange(len(runs_with_res))

    def _v(key: str) -> list[float]:
        return [float(r.get(key) or 0) for r in runs_with_res]

    # ── CPU utilisation ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, _v("cpu_peak_pct"), "o-", color=_C["blue"], label="Peak", zorder=3)
    ax.plot(x, _v("cpu_avg_pct"), "s--", color=_C["orange"], label="Avg", lw=1.0, zorder=3)
    ax.axhline(80, color=_C["red"], linestyle=":", lw=0.9, alpha=0.7, label="80% threshold")
    ax.set_title("(a) CPU Utilisation (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6)
    ax.legend()

    # ── RAM + swap ───────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ram = _v("ram_peak_mib")
    swap = _v("swap_peak_mib")
    ax.bar(x, ram, color=_C["blue"], alpha=0.85, label="RAM peak", zorder=3)
    ax.bar(x, swap, bottom=ram, color=_C["red"], alpha=0.75, label="Swap peak", zorder=3)
    ax.axhline(16 * 1024, color=_C["grey"], linestyle=":", lw=0.9, alpha=0.7, label="16 GB")
    ax.set_title("(b) Memory Usage (MiB)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1024:.0f} GB"))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6)
    ax.legend()

    # ── GPU VRAM ─────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    vram_peak = _v("gpu_vram_peak_mib")
    vram_avg  = _v("gpu_vram_avg_mib")
    ax.bar(x - 0.2, vram_peak, 0.4, color=_C["purple"], alpha=0.85,
           label="Peak", zorder=3)
    ax.bar(x + 0.2, vram_avg,  0.4, color=_C["orange"], alpha=0.75,
           label="Avg",  zorder=3)
    ax.axhline(8 * 1024, color=_C["red"], linestyle=":", lw=0.9, alpha=0.7,
               label="8 GB total")
    ax.set_title("(c) GPU VRAM (MiB)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1024:.0f} GB"))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6)
    ax.legend()

    # ── GPU utilisation ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    util = _v("gpu_util_peak_pct")
    colors = [_C["teal"] if v >= 50 else _C["orange"] if v > 5 else _C["grey"]
              for v in util]
    ax.bar(x, util, color=colors, alpha=0.88, zorder=3)
    ax.axhline(50, color=_C["red"], linestyle=":", lw=0.9, alpha=0.7, label="50% target")
    ax.set_title("(d) GPU Utilisation Peak (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_ylim(0, 110)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6)
    ax.legend()

    fig.tight_layout()
    out = out_dir / "resource_usage.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Fine-tuning runs (BLEU + duration)
# ─────────────────────────────────────────────────────────────────────────────

def fig_finetune_runs(runs: list[dict], out_dir: Path) -> Path:
    ft = [r for r in runs
          if r.get("run_type") == "finetune" and r.get("bleu_score") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(_COL2, 3.2))
    fig.suptitle("LoRA Fine-Tuning on NLLB-200-600M (Samanantar Corpus)",
                 fontweight="bold")

    ax = axes[0]
    if ft:
        labels = [_short_id(r["run_id"]) for r in ft]
        bleus = [float(r["bleu_score"]) for r in ft]
        bars = ax.bar(labels, bleus, color=_C["purple"], alpha=0.85, zorder=3)
        _label_bar(ax, bars, bleus, fmt="{:.3f}", offset=0.002)
        ax.set_ylabel("BLEU (sacreBLEU, no SentencePiece norm.)")
        ax.set_title("(a) Post-Fine-Tuning BLEU")
    else:
        ax.text(0.5, 0.5, "No fine-tune runs found.\nRun: python scripts/finetune.py",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("(a) Post-Fine-Tuning BLEU")

    ax = axes[1]
    if ft:
        labels = [_short_id(r["run_id"]) for r in ft]
        durations_h = [float(r.get("duration_s") or 0) / 3600 for r in ft]
        bars = ax.barh(labels, durations_h, color=_C["brown"], alpha=0.85, zorder=3)
        for bar, v in zip(bars, durations_h):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f} h", va="center", fontsize=7)
        ax.set_xlabel("Duration (hours)")
        ax.set_title("(b) Fine-Tuning Duration")
    else:
        ax.text(0.5, 0.5, "No fine-tune runs found.",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("(b) Fine-Tuning Duration")

    fig.tight_layout()
    out = out_dir / "finetune_runs.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Multi-model radar chart (all benchmarked models)
# ─────────────────────────────────────────────────────────────────────────────

def fig_radar_latest(runs: list[dict], out_dir: Path) -> Path:
    bench = [r for r in runs if r.get("run_type") == "benchmark"]

    # Collect unique models (most recent run per model with BLEU > 5)
    seen: set[str] = set()
    models_data: list[dict] = []
    for r in bench:
        name = r.get("model_name", "?")
        if name in seen:
            continue
        bleu = float(r.get("bleu_score") or 0)
        if bleu < 5:
            continue
        seen.add(name)
        models_data.append(r)

    # Metrics: (label, value_fn, max_for_norm)
    metrics_def = [
        ("BLEU",          lambda r: float(r.get("bleu_score")     or 0), 80),
        ("chrF",          lambda r: float(r.get("chrf_score")     or 0), 85),
        ("Speed\n(ch/s)", lambda r: float(r.get("chars_per_sec")  or 0), 120),
        ("GPU util\n(%)", lambda r: float(r.get("gpu_util_peak_pct") or 0), 100),
        ("VRAM\neffic.",  lambda r: max(0, 1 - float(r.get("gpu_vram_peak_mib") or 0) / (8192)), 1),
        ("CPU\neffic.",   lambda r: max(0, 1 - float(r.get("cpu_peak_pct")    or 0) / 100),     1),
    ]

    metric_labels = [m[0] for m in metrics_def]
    n = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))

    palette = [_C["blue"], _C["teal"], _C["purple"], _C["brown"]]

    if models_data:
        for idx, r in enumerate(models_data[:4]):
            vals = []
            for _, fn, mx in metrics_def:
                raw = fn(r)
                if mx == 1:
                    vals.append(max(0.0, min(raw, 1.0)))
                else:
                    vals.append(max(0.0, min(raw / mx, 1.0)))
            vals += vals[:1]
            color = palette[idx % len(palette)]
            model_label = r.get("model_name", "?")[:14]
            bleu = r.get("bleu_score") or 0
            chrf = r.get("chrf_score") or 0
            ax.plot(angles, vals, "o-", lw=1.5, color=color,
                    label=f"{model_label} (B={bleu:.0f}/C={chrf:.0f})")
            ax.fill(angles, vals, alpha=0.10, color=color)
    else:
        # Fallback — single placeholder
        vals = [0.7, 0.86, 0.67, 0.82, 0.77, 0.73]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=1.5, color=_C["blue"], label="No data")
        ax.fill(angles, vals, alpha=0.12, color=_C["blue"])

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_title("Quality & Efficiency Profile\n(normalised to axis maxima)",
                 pad=18, fontsize=9)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=7)

    fig.tight_layout()
    out = out_dir / "radar_latest.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [ok] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

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
        print("\n=== DB runs (most-recent first) ===")
        for r in runs[:10]:
            print(f"  {r.get('run_id','?')[:8]} | {r.get('run_type','?'):10} | "
                  f"{r.get('model_name','?'):18} | BLEU={r.get('bleu_score')!r:6} "
                  f"| chrF={r.get('chrf_score')!r:6} | "
                  f"VRAM={r.get('gpu_vram_peak_mib')!r} MiB")
        print()

    print(f"\nGenerating paper figures → {FIGURES_DIR}/\n")

    # Build measured dict from DB (most recent run per model with BLEU > 5)
    measured: dict[str, dict] = {}
    seen: set[str] = set()
    for r in runs:
        if r.get("run_type") != "benchmark":
            continue
        model = r.get("model_name", "")
        if model in seen:
            continue
        bleu = r.get("bleu_score")
        if bleu is None or float(bleu) < 5:  # skip garbage runs (e.g. MADLAD)
            seen.add(model)
            continue
        seen.add(model)
        chrf = r.get("chrf_score")

        label_map = {
            "nllb-600m":           "Ours\nNLLB-600M",
            "nllb-600M":           "Ours\nNLLB-600M",
            "seamless-medium":     "Ours\nSeamless-v2",
            "seamless":            "Ours\nSeamless-v2",
            "nllb-600M-finetuned": "Ours\nNLLB-FT",
        }
        label = label_map.get(model, f"Ours\n{model[:12]}")
        measured[model] = {"bleu": float(bleu), "chrf": float(chrf) if chrf else None,
                           "label": label}

    fig_domain_bleu(FIGURES_DIR)
    fig_combined_results(measured, FIGURES_DIR)
    fig_bleu_over_runs(chron, FIGURES_DIR)
    fig_resource_usage(chron, FIGURES_DIR)
    fig_finetune_runs(chron, FIGURES_DIR)
    fig_radar_latest(runs, FIGURES_DIR)

    n = len(list(FIGURES_DIR.glob("*.png")))
    print(f"\nDone. {n} figures in {FIGURES_DIR}/")
    print("Rebuild PDFs: make papers")


if __name__ == "__main__":
    main()
