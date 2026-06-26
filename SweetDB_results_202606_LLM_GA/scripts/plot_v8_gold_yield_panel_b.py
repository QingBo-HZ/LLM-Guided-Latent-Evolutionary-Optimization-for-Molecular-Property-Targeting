#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA")
REAL_DIR = ROOT / "sweet_ga_results_0622_v8_hard_metrics" / "real_vina_panels_final175"
OUT_DIR = REAL_DIR / "clean_main_panels"

METHOD_ORDER = ["A", "B", "C", "D"]
LABELS = {
    "A": "Random-\nSeeded",
    "B": "SweetDB-\nSeeded",
    "C": "LLM-\nInitialized",
    "D": "Iterative\nLLM-Guided",
}
COLORS = {
    "A": "#7B8292",
    "B": "#4F7EC7",
    "C": "#E8753A",
    "D": "#2F9B63",
    "ink": "#1F2430",
    "muted": "#667085",
    "grid": "#E7EAF1",
    "axis": "#D7DBE7",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "Segoe UI", "sans-serif"],
            "axes.unicode_minus": True,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "font.size": 16,
            "axes.titlesize": 24,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        }
    )


def polish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], lw=1.0, alpha=0.95)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REAL_DIR / "v8_final_real_vina_by_seed.csv")
    df["gold_real_vina"] = pd.to_numeric(df["gold_real_vina"], errors="coerce")

    values = [df.loc[df["method"] == m, "gold_real_vina"].to_numpy() for m in METHOD_ORDER]
    means = np.array([v.mean() for v in values])
    sds = np.array([v.std(ddof=1) if len(v) > 1 else 0 for v in values])

    setup_style()
    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    x = np.arange(len(METHOD_ORDER))
    ax.bar(
        x,
        means,
        yerr=sds,
        color=[COLORS[m] for m in METHOD_ORDER],
        edgecolor=COLORS["ink"],
        linewidth=1.05,
        capsize=4,
        width=0.62,
        alpha=0.88,
        zorder=2,
    )
    for i, vals in enumerate(values):
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=42,
            facecolor="#2A2F3A",
            edgecolor="white",
            linewidth=0.65,
            alpha=0.82,
            zorder=4,
        )
        ax.text(
            i,
            means[i] + sds[i] + 0.32,
            f"{means[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=COLORS["ink"],
        )

    ax.set_title("Gold-standard candidate yield", loc="left", fontweight="bold", color=COLORS["ink"], pad=8)
    ax.set_ylabel("Candidates per seed")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER])
    ax.set_ylim(0, max(10.5, float((means + sds).max()) + 1.1))
    polish(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.88)
    fig.savefig(OUT_DIR / "panel_b_gold_standard_candidate_yield.png", bbox_inches="tight", dpi=600)
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "method": METHOD_ORDER,
            "label": [LABELS[m].replace("\n", " ") for m in METHOD_ORDER],
            "mean": means,
            "sd": sds,
            "seed_values": [";".join(map(lambda z: f"{z:.0f}", vals)) for vals in values],
        }
    )
    summary.to_csv(OUT_DIR / "panel_b_gold_standard_candidate_yield_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {OUT_DIR / 'panel_b_gold_standard_candidate_yield.png'}")


if __name__ == "__main__":
    main()
