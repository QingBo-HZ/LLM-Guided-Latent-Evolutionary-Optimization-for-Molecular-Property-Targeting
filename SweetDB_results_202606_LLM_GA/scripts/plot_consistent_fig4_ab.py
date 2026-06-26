#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FITNESS_ORDER = ["Sweet-only", "Docking-only", "Gate+Sweet", "Gate+Docking"]
FITNESS_COLORS = {
    "Sweet-only": "#5AA9E6",
    "Docking-only": "#F4A261",
    "Gate+Sweet": "#2F9B63",
    "Gate+Docking": "#8E5CB8",
}
FITNESS_MARKERS = {
    "Sweet-only": "o",
    "Docking-only": "s",
    "Gate+Sweet": "D",
    "Gate+Docking": "^",
}

METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "Random-\nSeeded",
    "B": "SweetDB-\nSeeded",
    "C": "LLM-\nInitialized",
    "D": "Iterative\nLLM-Guided",
}
METHOD_COLORS = {
    "A": "#7B8292",
    "B": "#4F7EC7",
    "C": "#E8753A",
    "D": "#2F9B63",
}

TOKENS = {
    "ink": "#1F2430",
    "muted": "#667085",
    "grid": "#E7EAF1",
    "axis": "#D7DBE7",
    "pass": "#EEF7F0",
}


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "Segoe UI", "sans-serif"],
            "axes.unicode_minus": True,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "font.size": 15,
            "axes.titlesize": 22,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 10.5,
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
        }
    )


def polish(ax: plt.Axes, xgrid: bool = True) -> None:
    ax.grid(axis="y", color=TOKENS["grid"], lw=0.9, alpha=0.95)
    ax.grid(axis="x", color=TOKENS["grid"], lw=0.75, alpha=0.58 if xgrid else 0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def plot_panel_a(summary_csv: Path, out_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_csv)
    summary["dual_pass_count"] = pd.to_numeric(summary["dual_pass_count"], errors="coerce").fillna(0)
    summary["dual_pass_per_seed"] = pd.to_numeric(summary["dual_pass_per_seed"], errors="coerce").fillna(0)
    counts = summary.set_index("method")["dual_pass_count"].reindex(FITNESS_ORDER).fillna(0).astype(int)
    per_seed = summary.set_index("method")["dual_pass_per_seed"].reindex(FITNESS_ORDER).fillna(0)
    p_thr = float(summary["p_threshold"].dropna().iloc[0]) if "p_threshold" in summary else 0.80
    x_thr = float(summary["logsw_threshold"].dropna().iloc[0]) if "logsw_threshold" in summary else 2.60
    y_thr = float(summary["vina_threshold"].dropna().iloc[0]) if "vina_threshold" in summary else -6.80

    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    x = np.arange(len(FITNESS_ORDER))
    bars = ax.bar(
        x,
        counts.to_numpy(),
        color=[FITNESS_COLORS[m] for m in FITNESS_ORDER],
        edgecolor=TOKENS["ink"],
        linewidth=1.0,
        width=0.62,
        alpha=0.88,
        zorder=2,
    )
    for i, bar in enumerate(bars):
        val = int(counts.iloc[i])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.22,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=TOKENS["ink"],
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.22, bar.get_height() * 0.22),
            f"{per_seed.iloc[i]:.1f}/seed",
            ha="center",
            va="center",
            fontsize=9.5,
            color="white",
            fontweight="bold",
        )

    ax.set_title("Gold-standard fitness ablation", loc="left", fontweight="bold", color=TOKENS["ink"], pad=8)
    ax.text(
        0.02,
        0.965,
        f"P(sweet)>={p_thr:.2f}; logSw>={x_thr:.2f}; real Vina<={y_thr:.1f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    ax.set_ylabel("Gold-standard candidates")
    ax.set_xticks(x)
    ax.set_xticklabels(["Sweet-\nonly", "Docking-\nonly", "Gate+\nSweet", "Gate+\nDocking"])
    ax.set_ylim(0, max(8.5, float(counts.max()) + 1.3))
    polish(ax, xgrid=False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.88)
    fig.savefig(out_dir / "panel_a_gold_standard_fitness_ablation_consistent.png", bbox_inches="tight", dpi=600)
    plt.close(fig)

    summary = counts.reset_index()
    summary.columns = ["method", "gold_standard_candidates"]
    summary["gold_standard_candidates_per_seed"] = per_seed.to_numpy()
    summary.to_csv(out_dir / "panel_a_gold_standard_fitness_ablation_counts.csv", index=False)
    return summary


def plot_panel_b(by_seed_csv: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(by_seed_csv)
    df["gold_real_vina"] = pd.to_numeric(df["gold_real_vina"], errors="coerce")
    values = [df.loc[df["method"] == m, "gold_real_vina"].to_numpy() for m in METHOD_ORDER]
    means = np.array([v.mean() for v in values])
    sds = np.array([v.std(ddof=1) if len(v) > 1 else 0 for v in values])

    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    x = np.arange(len(METHOD_ORDER))
    ax.bar(
        x,
        means,
        yerr=sds,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor=TOKENS["ink"],
        linewidth=1.0,
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
            s=40,
            facecolor="#2A2F3A",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.82,
            zorder=4,
        )
        ax.text(
            i,
            means[i] + sds[i] + 0.32,
            f"{means[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=TOKENS["ink"],
        )

    ax.set_title("Gold-standard candidate yield", loc="left", fontweight="bold", color=TOKENS["ink"], pad=8)
    ax.set_ylabel("Candidates per seed")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylim(0, max(10.5, float((means + sds).max()) + 1.1))
    polish(ax, xgrid=False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.88)
    fig.savefig(out_dir / "panel_b_gold_standard_candidate_yield_consistent.png", bbox_inches="tight", dpi=600)
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "method": METHOD_ORDER,
            "mean": means,
            "sd": sds,
            "seed_values": [";".join([f"{v:.0f}" for v in vals]) for vals in values],
        }
    )
    summary.to_csv(out_dir / "panel_b_gold_standard_candidate_yield_consistent_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitness_summary_csv", required=True, type=Path)
    parser.add_argument("--formal_by_seed_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    a = plot_panel_a(args.fitness_summary_csv, args.out_dir)
    b = plot_panel_b(args.formal_by_seed_csv, args.out_dir)
    print("Panel A")
    print(a.to_string(index=False))
    print("Panel B")
    print(b.to_string(index=False))
    print(f"Wrote consistent panels to {args.out_dir}")


if __name__ == "__main__":
    main()
