#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


ROOT = Path("/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA")
RESULT_ROOT = ROOT / "sweet_ga_results_0622_v8_hard_metrics"
AUDIT_DIR = RESULT_ROOT / "docking_audit"
OUT_DIR = RESULT_ROOT / "nature_style_panels"
DATA_DIR = OUT_DIR / "data"

METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "Random-Seeded Latent GA",
    "B": "SweetDB-Seeded Latent GA",
    "C": "LLM-Initialized Latent GA",
    "D": "Iterative LLM-Guided Latent GA",
}
COLORS = {
    "A": "#6F768A",
    "B": "#4F7EC7",
    "C": "#E8753A",
    "D": "#2F9B63",
}
MARKERS = {"A": "s", "B": "o", "C": "^", "D": "D"}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "Segoe UI", "sans-serif"],
            "font.size": 18,
            "axes.titlesize": 25,
            "axes.labelsize": 21,
            "axes.linewidth": 1.5,
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 15.5,
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "savefig.facecolor": TOKENS["surface"],
            "savefig.dpi": 600,
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.965, title, ha="left", va="top", fontsize=25, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.07, 0.915, subtitle, ha="left", va="top", fontsize=15.5, color=TOKENS["muted"])


def polish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=1.15, alpha=0.9)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.9, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def summarize_generation(gen: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        gen.groupby(["method", "seed", "generation"])
        .agg(
            hard_pass=("pre_docking_goldlike", "sum"),
            n=("ID", "count"),
            topk_logsw=("pred_logsw", "mean"),
            topk_p_sweet=("p_sweet", "mean"),
            best_logsw=("pred_logsw", "max"),
            mean_ood=("d_ood", "mean"),
        )
        .reset_index()
    )
    summary = (
        per_seed.groupby(["method", "generation"])
        .agg(
            hard_pass_mean=("hard_pass", "mean"),
            hard_pass_sd=("hard_pass", "std"),
            topk_logsw_mean=("topk_logsw", "mean"),
            topk_logsw_sd=("topk_logsw", "std"),
            best_logsw_mean=("best_logsw", "mean"),
            mean_ood=("mean_ood", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return per_seed, summary


def plot_hard_pass_evolution(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    add_header(
        fig,
        "Hard-pass candidate evolution",
        "Mean top-5 candidates per seed satisfying P(sweet) >= 0.80, predicted logSw >= 2.60, and OOD <= 7.225",
    )
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("generation")
        x = sub["generation"].to_numpy()
        y = sub["hard_pass_mean"].to_numpy()
        sd = sub["hard_pass_sd"].fillna(0).to_numpy()
        ax.step(x, y, where="mid", lw=2.8, color=COLORS[method], label=METHOD_LABELS[method])
        ax.plot(x, y, linestyle="none", marker=MARKERS[method], markersize=7.5, markerfacecolor="white", markeredgewidth=1.8, color=COLORS[method])
        ax.fill_between(x, np.clip(y - sd, 0, 5), np.clip(y + sd, 0, 5), step="mid", color=COLORS[method], alpha=0.10, linewidth=0)
    for x in [3, 6, 9]:
        ax.axvline(x, color=TOKENS["axis"], lw=1.2, ls=":", zorder=0)
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 5.25)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hard-pass candidates in top-5")
    polish(ax)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=2.0, fontsize=14.5)
    fig.subplots_adjust(top=0.84, left=0.13, right=0.97, bottom=0.14)
    save(fig, "v8_hard_pass_candidate_evolution")


def plot_llm_feedback_gain(gen_per_seed: pd.DataFrame) -> None:
    pivot = gen_per_seed.pivot_table(
        index=["seed", "generation"],
        columns="method",
        values="topk_logsw",
        aggfunc="first",
    ).reset_index()
    pivot["D_minus_C"] = pivot["D"] - pivot["C"]
    gain = (
        pivot.groupby("generation")
        .agg(mean_gain=("D_minus_C", "mean"), sd_gain=("D_minus_C", "std"))
        .reset_index()
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(DATA_DIR / "v8_d_vs_c_top5_logsw_gain_per_seed.csv", index=False)
    gain.to_csv(DATA_DIR / "v8_d_vs_c_top5_logsw_gain_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    add_header(
        fig,
        "LLM feedback gain over initialization",
        "Difference in top-5 mean predicted logSw: Iterative LLM-guided GA minus LLM-initialized GA",
    )
    x = gain["generation"].to_numpy()
    y = gain["mean_gain"].to_numpy()
    sd = gain["sd_gain"].fillna(0).to_numpy()
    ax.axhline(0, color=TOKENS["ink"], lw=1.35, ls="-", alpha=0.55)
    ax.plot(x, y, lw=3.3, color=COLORS["D"], label="D minus C")
    ax.scatter(x, y, s=76, marker=MARKERS["D"], facecolors="white", edgecolors=COLORS["D"], linewidths=2.0, zorder=3)
    ax.fill_between(x, y - sd, y + sd, color=COLORS["D"], alpha=0.14, linewidth=0)
    for x in [3, 6, 9]:
        ax.axvline(x, color=TOKENS["axis"], lw=1.25, ls=":", zorder=0)
    ax.set_xlim(1, 12)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Top-5 logSw gain over C")
    polish(ax)
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=2.2)
    fig.subplots_adjust(top=0.84, left=0.13, right=0.97, bottom=0.14)
    save(fig, "v8_d_vs_c_llm_feedback_gain")


def plot_final_scatter(final: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    add_header(
        fig,
        "Final external candidate pool",
        "Top-10 decoded and re-encoded molecules per seed; dashed guides show P(sweet)=0.80 and predicted logSw=2.60",
    )
    for method in METHOD_ORDER:
        sub = final[final["method"] == method].copy()
        ok = sub["pre_docking_goldlike"].astype(bool)
        ax.scatter(
            sub.loc[~ok, "pred_logsw_reencoded"],
            sub.loc[~ok, "p_sweet_reencoded"],
            s=64,
            marker=MARKERS[method],
            facecolors="white",
            edgecolors=COLORS[method],
            linewidths=1.55,
            alpha=0.72,
        )
        ax.scatter(
            sub.loc[ok, "pred_logsw_reencoded"],
            sub.loc[ok, "p_sweet_reencoded"],
            s=92,
            marker=MARKERS[method],
            facecolors=COLORS[method],
            edgecolors="#FFFFFF",
            linewidths=0.9,
            alpha=0.88,
            label=METHOD_LABELS[method],
        )
    ax.axvline(2.60, color=TOKENS["ink"], lw=1.5, ls="--", alpha=0.65)
    ax.axhline(0.80, color=TOKENS["ink"], lw=1.5, ls="--", alpha=0.65)
    ax.set_xlabel("Predicted logSw after re-encoding")
    ax.set_ylabel("P(sweet) after re-encoding")
    ax.set_xlim(1.85, max(4.25, final["pred_logsw_reencoded"].max() + 0.08))
    ax.set_ylim(-0.02, 1.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    polish(ax)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        frameon=False,
        ncol=1,
        handlelength=1.4,
        fontsize=14.5,
    )
    fig.subplots_adjust(top=0.84, left=0.13, right=0.76, bottom=0.14)
    save(fig, "v8_final_candidate_pool_sweetness_scatter")


def write_tables(final: pd.DataFrame, gen_per_seed: pd.DataFrame, gen_summary: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_summary = (
        final.groupby("method")
        .agg(
            evaluated=("ID", "count"),
            seeds=("seed", "nunique"),
            hard_pass=("pre_docking_goldlike", "sum"),
            hard_pass_rate=("pre_docking_goldlike", "mean"),
            mean_pred_logsw=("pred_logsw_reencoded", "mean"),
            max_pred_logsw=("pred_logsw_reencoded", "max"),
            mean_p_sweet=("p_sweet_reencoded", "mean"),
            mean_ood=("d_ood_reencoded", "mean"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    final_summary["method_label"] = final_summary["method"].map(METHOD_LABELS)
    final_summary.to_csv(DATA_DIR / "v8_final_external_pool_summary.csv", index=False)
    gen_per_seed.to_csv(DATA_DIR / "v8_generation_per_seed_metrics.csv", index=False)
    gen_summary.to_csv(DATA_DIR / "v8_generation_summary_metrics.csv", index=False)

    injection_rows = []
    for path in RESULT_ROOT.glob("group_d_llm_iterative_*/llm_injection_summary.csv"):
        df = pd.read_csv(path)
        seed = int(path.parent.name.split("seed")[-1])
        df["seed"] = seed
        injection_rows.append(df)
    if injection_rows:
        inj = pd.concat(injection_rows, ignore_index=True)
        inj_summary = (
            inj.groupby("generation")
            .agg(
                generated_basic_gate_count=("generated_basic_gate_count", "sum"),
                strict_bpe_injected_count=("strict_bpe_injected_count", "sum"),
                events=("seed", "nunique"),
            )
            .reset_index()
        )
        inj.to_csv(DATA_DIR / "v8_d_group_injection_events.csv", index=False)
        inj_summary.to_csv(DATA_DIR / "v8_d_group_injection_summary.csv", index=False)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    gen = pd.read_csv(AUDIT_DIR / "generation_docking_candidate_pool.csv")
    final = pd.read_csv(AUDIT_DIR / "final_docking_candidate_pool.csv")
    gen_per_seed, gen_summary = summarize_generation(gen)
    plot_hard_pass_evolution(gen_summary)
    plot_llm_feedback_gain(gen_per_seed)
    plot_final_scatter(final)
    write_tables(final, gen_per_seed, gen_summary)

    manifest = json.loads((AUDIT_DIR / "audit_manifest.json").read_text())
    (DATA_DIR / "v8_plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
