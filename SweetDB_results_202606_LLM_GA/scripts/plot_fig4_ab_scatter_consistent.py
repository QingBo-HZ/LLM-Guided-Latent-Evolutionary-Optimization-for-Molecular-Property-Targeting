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
    "A": "Random-Seeded Latent GA",
    "B": "SweetDB-Seeded Latent GA",
    "C": "LLM-Initialized Latent GA",
    "D": "Iterative LLM-Guided Latent GA",
}
METHOD_COLORS = {
    "A": "#7B8292",
    "B": "#4F7EC7",
    "C": "#E8753A",
    "D": "#2F9B63",
}
METHOD_MARKERS = {"A": "s", "B": "o", "C": "^", "D": "D"}

TOKENS = {
    "ink": "#1F2430",
    "muted": "#667085",
    "grid": "#E7EAF1",
    "axis": "#D7DBE7",
    "pass": "#EEF7F0",
}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "Segoe UI", "sans-serif"],
            "axes.unicode_minus": True,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "font.size": 15,
            "axes.titlesize": 18,
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


def polish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=TOKENS["grid"], lw=1.0, alpha=0.95)
    ax.grid(axis="x", color=TOKENS["grid"], lw=0.85, alpha=0.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def draw_scatter(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    markers: dict[str, str],
    count_col: str,
    title: str,
    out_png: Path,
    counts_override: pd.Series | None = None,
    threshold_note: str | None = None,
    x_thr: float = 2.60,
    y_thr: float = -6.80,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    for col in ["pred_logsw_reencoded", "vina_kcal_mol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[count_col] = as_bool(df[count_col])
    df = df.dropna(subset=["pred_logsw_reencoded", "vina_kcal_mol"])

    if counts_override is None:
        counts = df.groupby(group_col)[count_col].sum().reindex(groups).fillna(0).astype(int)
    else:
        counts = counts_override.reindex(groups).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if xlim is None:
        xlim = (max(1.30, float(df["pred_logsw_reencoded"].min()) - 0.12), float(df["pred_logsw_reencoded"].max()) + 0.12)
    if ylim is None:
        ymin = np.floor(float(df["vina_kcal_mol"].min()) * 2) / 2 - 0.15
        ymax = np.ceil(float(df["vina_kcal_mol"].max()) * 2) / 2 + 0.15
        ylim = (ymax, ymin)

    ax.axvspan(x_thr, xlim[1], color=TOKENS["pass"], alpha=0.86, zorder=0)
    ax.axhspan(ylim[1], y_thr, color=TOKENS["pass"], alpha=0.86, zorder=0)
    ax.axvline(x_thr, color=TOKENS["ink"], lw=1.25, ls="--", alpha=0.60)
    ax.axhline(y_thr, color=TOKENS["ink"], lw=1.25, ls="--", alpha=0.60)

    for group in groups:
        sub = df[df[group_col] == group]
        if sub.empty:
            continue
        ok = sub[count_col].astype(bool)
        ax.scatter(
            sub.loc[~ok, "pred_logsw_reencoded"],
            sub.loc[~ok, "vina_kcal_mol"],
            s=36,
            marker=markers[group],
            facecolors="none",
            edgecolors=colors[group],
            linewidths=1.2,
            alpha=0.42,
        )
        ax.scatter(
            sub.loc[ok, "pred_logsw_reencoded"],
            sub.loc[ok, "vina_kcal_mol"],
            s=78,
            marker=markers[group],
            facecolors=colors[group],
            edgecolors=TOKENS["ink"],
            linewidths=0.65,
            alpha=0.94,
            label=f"{labels[group]} (n={counts[group]})",
        )

    ax.set_title(title, loc="left", fontweight="bold", color=TOKENS["ink"], pad=8)
    if threshold_note:
        ax.text(
            0.03,
            0.055,
            threshold_note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.2,
            color=TOKENS["muted"],
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": TOKENS["axis"], "alpha": 0.82},
            clip_on=False,
        )
    ax.set_xlabel("Predicted logSw after re-encoding")
    ax.set_ylabel("Real Vina score (kcal/mol)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    polish(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.38, borderaxespad=0.25)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.88)
    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)

    return counts.reset_index().rename(columns={group_col: "group", count_col: "gold_standard_candidates", 0: "gold_standard_candidates"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitness_per_mol_csv", required=True, type=Path)
    parser.add_argument("--fitness_summary_csv", required=True, type=Path)
    parser.add_argument("--formal_scored_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    fitness = pd.read_csv(args.fitness_per_mol_csv)
    summary = pd.read_csv(args.fitness_summary_csv)
    fitness["gold_standard"] = (
        (pd.to_numeric(fitness["p_sweet_reencoded"], errors="coerce") >= 0.80)
        & (pd.to_numeric(fitness["pred_logsw_reencoded"], errors="coerce") >= 2.60)
        & (pd.to_numeric(fitness["vina_kcal_mol"], errors="coerce") <= -6.80)
    )
    count_override = summary.set_index("method")["dual_pass_count"]
    panel_a_counts = draw_scatter(
        fitness,
        "method",
        FITNESS_ORDER,
        {m: m for m in FITNESS_ORDER},
        FITNESS_COLORS,
        FITNESS_MARKERS,
        "gold_standard",
        "Gold-standard fitness ablation",
        args.out_dir / "panel_a_gold_standard_fitness_ablation_scatter.png",
        counts_override=count_override,
        threshold_note="Gold standard: P(sweet)>=0.80 and predicted logSw>=2.60\nReal Vina score<=-6.8 kcal/mol",
        xlim=(1.35, 3.55),
    )

    formal = pd.read_csv(args.formal_scored_csv)
    formal["gold_real_vina"] = as_bool(formal["gold_real_vina"])
    panel_b_counts = draw_scatter(
        formal,
        "method",
        METHOD_ORDER,
        METHOD_LABELS,
        METHOD_COLORS,
        METHOD_MARKERS,
        "gold_real_vina",
        "Gold-standard candidate yield",
        args.out_dir / "panel_b_gold_standard_candidate_yield_scatter.png",
        xlim=(1.45, 4.20),
    )

    panel_a_counts.to_csv(args.out_dir / "panel_a_gold_standard_fitness_ablation_scatter_counts.csv", index=False)
    panel_b_counts.to_csv(args.out_dir / "panel_b_gold_standard_candidate_yield_scatter_counts.csv", index=False)
    print("Panel A counts")
    print(panel_a_counts.to_string(index=False))
    print("Panel B counts")
    print(panel_b_counts.to_string(index=False))
    print(f"Wrote scatter panels to {args.out_dir}")


if __name__ == "__main__":
    main()
