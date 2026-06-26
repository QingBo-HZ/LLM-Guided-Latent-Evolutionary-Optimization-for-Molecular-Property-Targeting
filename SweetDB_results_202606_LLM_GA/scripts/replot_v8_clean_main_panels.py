#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA")
RESULT_ROOT = ROOT / "sweet_ga_results_0622_v8_hard_metrics"
DATA_DIR = RESULT_ROOT / "nature_style_panels" / "data"
REAL_DIR = RESULT_ROOT / "real_vina_panels_final175"
OUT_DIR = RESULT_ROOT / "real_vina_panels_final175" / "clean_main_panels"

METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "Random-Seeded Latent GA",
    "B": "SweetDB-Seeded Latent GA",
    "C": "LLM-Initialized Latent GA",
    "D": "Iterative LLM-Guided Latent GA",
}
SHORT_LABELS = {
    "A": "Random",
    "B": "SweetDB-seeded",
    "C": "LLM-initialized",
    "D": "Iterative LLM-guided",
}
COLORS = {
    "A": "#7B8292",
    "B": "#4F7EC7",
    "C": "#E8753A",
    "D": "#2F9B63",
    "purple": "#9B59B6",
    "ink": "#1F2430",
    "muted": "#667085",
    "grid": "#E6E9F0",
    "axis": "#D7DBE7",
}
MARKERS = {"A": "s", "B": "o", "C": "^", "D": "D"}


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
            "font.size": 18,
            "axes.titlesize": 28,
            "axes.labelsize": 23,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 13.5,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        }
    )


def polish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], lw=1.05, alpha=0.92)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8, alpha=0.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])


def plot_feedback() -> None:
    summary = pd.read_csv(DATA_DIR / "v8_d_group_injection_summary.csv")
    gain = pd.read_csv(DATA_DIR / "v8_d_vs_c_top5_logsw_gain_summary.csv")
    for col in ["generation", "generated_basic_gate_count", "strict_bpe_injected_count"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    for col in ["generation", "mean_gain", "sd_gain"]:
        gain[col] = pd.to_numeric(gain[col], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    width = 0.28
    x = summary["generation"].to_numpy()
    ax.bar(
        x - width / 2,
        summary["generated_basic_gate_count"],
        width=width,
        color=COLORS["purple"],
        alpha=0.24,
        edgecolor=COLORS["purple"],
        linewidth=1.5,
        label="LLM-proposed molecules",
    )
    ax.bar(
        x + width / 2,
        summary["strict_bpe_injected_count"],
        width=width,
        color=COLORS["purple"],
        alpha=0.82,
        edgecolor=COLORS["purple"],
        linewidth=1.5,
        label="Accepted LLM injections",
    )
    for row in summary.itertuples():
        ax.text(
            float(row.generation) + width / 2,
            float(row.strict_bpe_injected_count) + 0.25,
            f"{int(row.strict_bpe_injected_count)}",
            ha="center",
            va="bottom",
            fontsize=12.5,
            color=COLORS["purple"],
            fontweight="bold",
        )
    ax.set_xlim(1, 12.25)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Generation")
    ax.set_ylabel("LLM feedback molecules")
    ax.set_ylim(0, 18)
    polish(ax)

    ax2 = ax.twinx()
    gx = gain["generation"].to_numpy()
    gy = gain["mean_gain"].to_numpy()
    gsd = gain["sd_gain"].to_numpy()
    ax2.axhline(0, color=COLORS["ink"], lw=1.1, ls="--", alpha=0.42)
    ax2.plot(
        gx,
        gy,
        color=COLORS["D"],
        lw=3.0,
        marker="D",
        markersize=7.0,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="Iterative LLM-Guided GA over LLM-Initialized GA",
    )
    ax2.fill_between(gx, gy - gsd, gy + gsd, color=COLORS["D"], alpha=0.12, linewidth=0)
    ax2.set_ylabel("Top-5 logSw improvement")
    ax2.tick_params(axis="y", colors=COLORS["muted"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["axis"])
    for gen in [3, 6, 9]:
        ax.axvline(gen, color=COLORS["purple"], lw=1.35, ls=(0, (4, 4)), alpha=0.50, zorder=0)

    ax.set_title("LLM feedback intervention", loc="left", color=COLORS["ink"], fontweight="bold", pad=10)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, handlelength=2.0)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.88)
    fig.savefig(OUT_DIR / "main_v8_llm_feedback_intervention_effect_clean.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_real_vina_scatter() -> None:
    df = pd.read_csv(REAL_DIR / "v8_final_real_vina_scored.csv")
    for col in ["pred_logsw_reencoded", "vina_kcal_mol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["gold_real_vina"]:
        df[col] = as_bool(df[col])
    df = df.dropna(subset=["pred_logsw_reencoded", "vina_kcal_mol"]).copy()

    x_thr = 2.60
    y_thr = -6.80
    counts = df.groupby("method")["gold_real_vina"].sum().reindex(METHOD_ORDER).fillna(0).astype(int)
    count_text = "Gold-standard counts: " + "   ".join([f"{SHORT_LABELS[m]} {counts[m]}" for m in METHOD_ORDER])

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        ok = sub["gold_real_vina"].astype(bool)
        ax.scatter(
            sub.loc[~ok, "pred_logsw_reencoded"],
            sub.loc[~ok, "vina_kcal_mol"],
            s=58,
            marker=MARKERS[method],
            facecolors="white",
            edgecolors=COLORS[method],
            linewidths=1.55,
            alpha=0.70,
        )
        ax.scatter(
            sub.loc[ok, "pred_logsw_reencoded"],
            sub.loc[ok, "vina_kcal_mol"],
            s=80,
            marker=MARKERS[method],
            facecolors=COLORS[method],
            edgecolors="#FFFFFF",
            linewidths=0.9,
            alpha=0.92,
            label=METHOD_LABELS[method],
        )
    ax.axhline(y_thr, color=COLORS["ink"], lw=1.35, ls="--", alpha=0.62)
    ax.axvline(x_thr, color=COLORS["ink"], lw=1.35, ls="--", alpha=0.62)
    ax.text(
        0.985,
        0.965,
        count_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": COLORS["axis"], "alpha": 0.90},
    )
    ax.set_title("Final candidates under real docking", loc="left", color=COLORS["ink"], fontweight="bold", pad=10)
    ax.set_xlabel("Predicted logSw after re-encoding")
    ax.set_ylabel("Real Vina score (kcal/mol)")
    ymin = np.floor(df["vina_kcal_mol"].min() * 2) / 2 - 0.2
    ymax = np.ceil(df["vina_kcal_mol"].max() * 2) / 2 + 0.2
    ax.set_ylim(ymax, ymin)
    ax.set_xlim(1.45, 4.20)
    polish(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.3, borderaxespad=0.4)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.88)
    fig.savefig(OUT_DIR / "v8_real_vina_final_scatter_clean.png", bbox_inches="tight", dpi=600)
    plt.close(fig)

    counts.to_csv(OUT_DIR / "v8_real_vina_gold_standard_counts_for_scatter.csv", header=["count"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_feedback()
    plot_real_vina_scatter()
    print(f"Wrote clean panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
