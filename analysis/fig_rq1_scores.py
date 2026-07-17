"""RQ1 figure: per-language ChrF++ — official baseline vs generic vs cultural-VQA.

Grouped bar chart, one group per language, three bars per group. This is the
paper's headline comparison (RQ1) rendered as a figure — the prelim currently has
only tables. Reads numbers from ``analysis.results_data`` (single source of truth)
and writes ``figures/rq1_chrf_comparison.{pdf,png}``.

Design (per the dataviz method):
  * 3 categorical hues in fixed slot order (blue / aqua / yellow), colorblind-safe
    (validated: worst adjacent CVD ΔE 21.6). Because aqua & yellow are sub-3:1 on a
    light surface, the *relief rule* applies — every bar carries a direct value
    label, so identity/magnitude never rely on color alone.
  * Recessive grid, no top/right spines, legend present (3 series).
  * Maya has no MT baseline: its baseline bar is drawn as an empty hatched slot
    labelled "no MT baseline" rather than a zero (a zero would read as a real score).

Run:  python -m analysis.fig_rq1_scores
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from .results_data import NOISE_BAND, mean_delta, ordered_results

# --- dataviz palette (light mode, categorical slots 1-3) ---------------------
C_BASELINE = "#2a78d6"  # slot 1 blue   — official baseline (reference)
C_GENERIC = "#1baf7a"   # slot 2 aqua   — generic control
C_CULTURAL = "#eda100"  # slot 3 yellow — cultural-VQA treatment
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def build_figure():
    rows = ordered_results()
    langs = [r.language for r in rows]
    x = np.arange(len(langs))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 3.9))

    # --- baseline bars (None -> hatched empty slot, not a zero) --------------
    base_vals = [r.official_baseline for r in rows]
    for xi, v in zip(x - width, base_vals):
        if v is None:
            ax.bar(xi, 25, width, facecolor="none", edgecolor=MUTED,
                   hatch="////", linewidth=0.6, alpha=0.35, zorder=2)
            ax.text(xi, 1.0, "no MT\nbaseline", ha="center", va="bottom",
                    fontsize=6.5, color=MUTED, rotation=0)
        else:
            ax.bar(xi, v, width, color=C_BASELINE, zorder=3,
                   label="_" if xi != x[0] - width else None)

    generic = ax.bar(x, [r.generic for r in rows], width, color=C_GENERIC, zorder=3)
    cultural = ax.bar(x + width, [r.cultural_vqa for r in rows], width,
                      color=C_CULTURAL, zorder=3)

    # A clean proxy handle for the baseline series (its first real bar handle is
    # awkward to capture with the None-slot loop above).
    baseline_proxy = plt.Rectangle((0, 0), 1, 1, color=C_BASELINE)

    # --- direct value labels (relief rule) -----------------------------------
    def label(bar_x: float, val: float | None) -> None:
        if val is None:
            return
        ax.text(bar_x, val + 0.35, f"{val:.1f}", ha="center", va="bottom",
                fontsize=6.8, color=INK)

    for xi, v in zip(x - width, base_vals):
        label(xi, v)
    for rect, r in zip(generic, rows):
        label(rect.get_x() + rect.get_width() / 2, r.generic)
    for rect, r in zip(cultural, rows):
        label(rect.get_x() + rect.get_width() / 2, r.cultural_vqa)

    # --- delta annotations under each language (RQ1 read) --------------------
    for xi, r in zip(x, rows):
        noise = abs(r.delta) < NOISE_BAND
        txt = f"Δ {r.delta:+.2f}" + ("*" if noise else "")
        ax.annotate(txt, (xi + width / 2, -2.3), ha="center", va="top",
                    fontsize=6.8, color=(MUTED if noise else INK),
                    annotation_clip=False)

    # --- axes chrome ---------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=8.5, color=INK)
    ax.set_ylabel("ChrF++ (dev, k=5)", fontsize=9, color=INK)
    ax.set_ylim(0, 25)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_title("Per-language ChrF++: cultural-VQA vs. controls",
                 fontsize=10.5, color=INK, pad=26, loc="left")

    ax.legend([baseline_proxy, generic, cultural],
              ["Official baseline", "Generic (control)", "Cultural-VQA"],
              loc="upper right", frameon=False, fontsize=8, ncol=3,
              bbox_to_anchor=(1.0, 1.10), handlelength=1.1, columnspacing=1.3)

    ax.text(x[-1] + 0.55, -4.4,
            f"Δ = cultural − generic;  * within ±{NOISE_BAND} ChrF++ (noise at n≤50)."
            f"  Mean Δ = {mean_delta():+.2f}.",
            ha="right", va="top", fontsize=6.5, color=MUTED, clip_on=False)

    fig.subplots_adjust(bottom=0.22, top=0.86, left=0.08, right=0.98)
    return fig


def main() -> None:
    _style()
    FIG_DIR.mkdir(exist_ok=True)
    fig = build_figure()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"rq1_chrf_comparison.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
