#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None


ROOT = Path("/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA")
RESULT_ROOT = ROOT / "sweet_ga_results_0622_v8_hard_metrics"
SOURCE_DATA = RESULT_ROOT / "nature_style_panels" / "data"
OUT_DIR = RESULT_ROOT / "nature_style_panels" / "main_3panels"

METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "Random-Seeded Latent GA",
    "B": "SweetDB-Seeded Latent GA",
    "C": "LLM-Initialized Latent GA",
    "D": "Iterative LLM-Guided Latent GA",
}
COLORS = {
    "A": "#6B7280",
    "B": "#3B73B9",
    "C": "#E9783A",
    "D": "#2C9B63",
}
MARKERS = {"A": "s", "B": "o", "C": "^", "D": "D"}

TOKENS = {
    "bg": "#FFFFFF",
    "ink": "#1E2430",
    "muted": "#667085",
    "grid": "#E7EAF1",
    "axis": "#D3D8E5",
    "llm": "#9B59B6",
}


def setup_style():
    if sns is not None:
        sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "Segoe UI", "sans-serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": TOKENS["bg"],
            "axes.facecolor": TOKENS["bg"],
            "savefig.facecolor": TOKENS["bg"],
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 15,
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
        }
    )


def polish(ax):
    ax.grid(axis="y", color=TOKENS["grid"], lw=1.1, alpha=0.95)
    ax.grid(axis="x", color=TOKENS["grid"], lw=0.85, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.spines["left"].set_linewidth(1.35)
    ax.spines["bottom"].set_linewidth(1.35)


def add_llm_marks(ax, y_text, label=True):
    for gen in [3, 6, 9]:
        ax.axvline(gen, color=TOKENS["llm"], lw=1.6, ls=(0, (4, 4)), alpha=0.68, zorder=0)
        if label:
            ax.text(
                gen + 0.05,
                y_text,
                "LLM\nfeedback",
                ha="left",
                va="top",
                fontsize=12.5,
                color=TOKENS["llm"],
                linespacing=0.9,
            )


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)


def summarize_cumulative(per_seed):
    frame = per_seed.copy()
    frame = frame.sort_values(["method", "seed", "generation"])
    frame["cum_hard_pass"] = frame.groupby(["method", "seed"])["hard_pass"].cumsum()
    return (
        frame.groupby(["method", "generation"])
        .agg(
            mean=("cum_hard_pass", "mean"),
            sd=("cum_hard_pass", "std"),
            n=("seed", "nunique"),
        )
        .reset_index()
    )


def plot_cumulative_hard_pass(per_seed):
    summary = summarize_cumulative(per_seed)
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("generation")
        x = sub["generation"].to_numpy()
        y = sub["mean"].to_numpy()
        sd = sub["sd"].fillna(0).to_numpy()
        ax.step(
            x,
            y,
            where="post",
            color=COLORS[method],
            lw=3.0 if method == "D" else 2.45,
            alpha=1.0 if method in {"C", "D"} else 0.86,
            label=METHOD_LABELS[method],
        )
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=MARKERS[method],
            markersize=7.2,
            markerfacecolor="white",
            markeredgewidth=1.8,
            color=COLORS[method],
        )
        ax.fill_between(x, y - sd, y + sd, step="post", color=COLORS[method], alpha=0.08, linewidth=0)
    add_llm_marks(ax, y_text=59.0)
    ax.set_xlim(1, 12.15)
    ax.set_ylim(0, 61)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Generation (population size = 30)")
    ax.set_ylabel("Cumulative hard-pass candidates")
    ax.set_title("Hard-pass candidate accumulation", loc="left", fontweight="bold", color=TOKENS["ink"])
    polish(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    save(fig, "main_v8_cumulative_hard_pass_evolution_llm_marked")


def plot_top5_logsw(per_seed):
    summary = (
        per_seed.groupby(["method", "generation"])
        .agg(mean=("topk_logsw", "mean"), sd=("topk_logsw", "std"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("generation")
        x = sub["generation"].to_numpy()
        y = sub["mean"].to_numpy()
        sd = sub["sd"].fillna(0).to_numpy()
        ax.plot(
            x,
            y,
            color=COLORS[method],
            lw=3.0 if method == "D" else 2.45,
            alpha=1.0 if method in {"C", "D"} else 0.82,
            label=METHOD_LABELS[method],
        )
        ax.scatter(
            x,
            y,
            marker=MARKERS[method],
            s=62,
            facecolors="white",
            edgecolors=COLORS[method],
            linewidths=1.8,
            zorder=3,
        )
        ax.fill_between(x, y - sd, y + sd, color=COLORS[method], alpha=0.08, linewidth=0)
    add_llm_marks(ax, y_text=3.73)
    ax.set_xlim(1, 12.15)
    ax.set_ylim(2.48, 3.82)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Generation (population size = 30)")
    ax.set_ylabel("Top-5 mean predicted logSw")
    ax.set_title("Sweet-potency evolution", loc="left", fontweight="bold", color=TOKENS["ink"])
    polish(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    save(fig, "main_v8_top5_logsw_evolution_llm_marked")


def plot_surrogate_docking_scatter():
    frame = pd.read_csv(OUT_DIR / "v8_final_top10_with_raw_docking_surrogate.csv")
    for col in ["pred_logsw_reencoded", "p_sweet_reencoded", "pred_vina_kcal_mol_raw_surrogate"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for method in METHOD_ORDER:
        sub = frame[frame["method"] == method].copy()
        hard = sub["pre_docking_hard_pass"].astype(bool)
        ax.scatter(
            sub.loc[~hard, "pred_logsw_reencoded"],
            sub.loc[~hard, "pred_vina_kcal_mol_raw_surrogate"],
            s=58,
            marker=MARKERS[method],
            facecolors="white",
            edgecolors=COLORS[method],
            linewidths=1.35,
            alpha=0.62,
        )
        ax.scatter(
            sub.loc[hard, "pred_logsw_reencoded"],
            sub.loc[hard, "pred_vina_kcal_mol_raw_surrogate"],
            s=82,
            marker=MARKERS[method],
            facecolors=COLORS[method],
            edgecolors="#FFFFFF",
            linewidths=0.75,
            alpha=0.88,
            label=METHOD_LABELS[method],
        )
    ax.axvline(2.60, color=TOKENS["ink"], lw=1.55, ls="--", alpha=0.62)
    ax.axhline(-6.80, color=TOKENS["ink"], lw=1.55, ls="--", alpha=0.62)
    ax.text(2.615, -7.02, "logSw threshold", ha="left", va="top", fontsize=12.5, color=TOKENS["muted"])
    ax.text(1.82, -6.87, "docking threshold", ha="left", va="top", fontsize=12.5, color=TOKENS["muted"])
    ax.set_xlabel("Predicted logSw after re-encoding")
    ax.set_ylabel("Predicted Vina affinity (kcal/mol)")
    ax.set_title("Final sweet-docking candidate landscape", loc="left", fontweight="bold", color=TOKENS["ink"])
    ax.set_xlim(1.72, max(4.12, frame["pred_logsw_reencoded"].max() + 0.08))
    low = np.floor(frame["pred_vina_kcal_mol_raw_surrogate"].min() * 2) / 2 - 0.1
    high = np.ceil(frame["pred_vina_kcal_mol_raw_surrogate"].max() * 2) / 2 + 0.1
    ax.set_ylim(low, high)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    polish(ax)
    ax.legend(loc="lower left", frameon=False, handlelength=1.4)
    save(fig, "main_v8_final_logsw_vs_predicted_vina_surrogate")


def write_readme():
    readme = OUT_DIR / "main_3panels_readme.md"
    readme.write_text(
        """# v8 main three-panel figure set

Generated panels:

- `main_v8_cumulative_hard_pass_evolution_llm_marked.png`
- `main_v8_top5_logsw_evolution_llm_marked.png`
- `main_v8_final_logsw_vs_predicted_vina_surrogate.png`

The first two panels use the v8 generation-level external metrics and show all four groups.
Purple dashed vertical lines mark the iterative LLM feedback/injection checks in D at generations 3, 6, and 9.

The third panel uses the raw-Vina latent docking surrogate to place the final ranked population on a sweet-docking landscape.
Its y-axis is `predicted Vina affinity (kcal/mol)`, with more negative values indicating better predicted binding.
This is not a replacement for the real-Vina backfilled panel. Once the 775 unique SMILES docking job returns, use
`merge_v8_real_vina_and_plot.py` to generate the real-Vina version with the same y-axis concept.
""",
        encoding="utf-8",
    )


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = pd.read_csv(SOURCE_DATA / "v8_generation_per_seed_metrics.csv")
    for col in ["generation", "hard_pass", "topk_logsw"]:
        per_seed[col] = pd.to_numeric(per_seed[col], errors="coerce")
    plot_cumulative_hard_pass(per_seed)
    plot_top5_logsw(per_seed)
    plot_surrogate_docking_scatter()
    write_readme()
    print(f"Wrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
