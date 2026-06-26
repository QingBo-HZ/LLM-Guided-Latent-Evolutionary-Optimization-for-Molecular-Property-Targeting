#!/usr/bin/env python3
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "Sweet predictor only",
    "Docking predictor only",
    "Gate + sweet predictor",
    "Gate + docking predictor",
]

SHORT = {
    "Sweet predictor only": "Sweet\nonly",
    "Docking predictor only": "Docking\nonly",
    "Gate + sweet predictor": "Gate +\nSweet",
    "Gate + docking predictor": "Gate +\nDocking",
}

COLORS = {
    "Sweet predictor only": "#4C78A8",
    "Docking predictor only": "#F58518",
    "Gate + sweet predictor": "#54A24B",
    "Gate + docking predictor": "#B279A2",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "axes.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "legend.fontsize": 18,
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.dpi": 600,
    })


def save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}_highres.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ordered_stats(df, value):
    stats = df.groupby("method")[value].agg(["mean", "std"]).reindex(METHODS)
    return stats["mean"].to_numpy(), stats["std"].fillna(0).to_numpy()


def plot_final_yield(run_df, out_dir):
    means, stds = ordered_stats(run_df, "unique_reliable_count")
    fig, ax = plt.subplots(figsize=(12.8, 9.2))
    x = np.arange(len(METHODS))
    ax.bar(
        x, means, yerr=stds, width=0.66,
        color=[COLORS[m] for m in METHODS],
        capsize=7,
        error_kw={"elinewidth": 2.8, "capthick": 2.8},
    )
    for xi, mean, sd in zip(x, means, stds):
        ax.text(xi, mean + sd + 0.25, f"{mean:.1f}", ha="center", va="bottom", fontsize=21)
    ax.set_xticks(x, [SHORT[m] for m in METHODS])
    ax.set_ylabel("Unique reliable molecule count", labelpad=14)
    ax.set_title("Final reliable yield under a shared evaluator", pad=18)
    ax.set_ylim(0, max(means + stds) + 1.6)
    ax.grid(axis="y", alpha=0.22, linewidth=1.1)
    fig.tight_layout()
    save(fig, out_dir, "A_final_unique_reliable_yield")


def plot_near_target(progress_df, out_dir):
    fig, ax = plt.subplots(figsize=(14.5, 9.2))
    for method in METHODS:
        sub = progress_df[progress_df["method"] == method]
        stats = sub.groupby("generation")["success_count"].agg(["mean", "std"]).sort_index()
        x = stats.index.to_numpy()
        y = stats["mean"].to_numpy()
        sd = stats["std"].fillna(0).to_numpy()
        ax.plot(x, y, lw=4.2, color=COLORS[method], label=method)
        ax.fill_between(
            x,
            np.clip(y - sd, 0, None),
            np.clip(y + sd, 0, 30),
            color=COLORS[method],
            alpha=0.13,
            lw=0,
        )
    ax.set_xlabel("Generation", labelpad=12)
    ax.set_ylabel("Near-target molecules in population", labelpad=14)
    ax.set_title("Near-target population evolution", pad=18)
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 31)
    ax.set_xticks(np.arange(1, 13, 1))
    ax.grid(alpha=0.22, linewidth=1.1)
    ax.legend(loc="lower right", fontsize=17, handlelength=2.4)
    fig.tight_layout()
    save(fig, out_dir, "B_near_target_population_evolution")


def plot_uniqueness_diversity(run_df, out_dir):
    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    x = np.arange(len(METHODS))
    width = 0.34
    unique_mean, unique_std = ordered_stats(run_df, "unique_smiles_ratio")
    div_mean, div_std = ordered_stats(run_df, "internal_diversity")
    ax.bar(
        x - width / 2, unique_mean, yerr=unique_std, width=width,
        color=[COLORS[m] for m in METHODS], alpha=0.82, capsize=6,
        label="Unique SMILES ratio",
        error_kw={"elinewidth": 2.2, "capthick": 2.2},
    )
    ax.bar(
        x + width / 2, div_mean, yerr=div_std, width=width,
        color=[COLORS[m] for m in METHODS], alpha=0.42, capsize=6,
        label="Internal diversity",
        error_kw={"elinewidth": 2.2, "capthick": 2.2},
    )
    ax.set_xticks(x, [SHORT[m] for m in METHODS])
    ax.set_ylabel("Ratio / diversity score", labelpad=14)
    ax.set_title("Final uniqueness and diversity", pad=18)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.22, linewidth=1.1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, out_dir, "C_final_uniqueness_and_diversity")


def plot_reliable_quality(run_df, out_dir):
    fig, ax = plt.subplots(figsize=(14.5, 9.2))
    x = np.arange(len(METHODS))
    logsw_mean, logsw_std = ordered_stats(run_df, "mean_logsw_reliable")
    ax.bar(
        x, logsw_mean, yerr=logsw_std, width=0.66,
        color=[COLORS[m] for m in METHODS],
        capsize=7,
        error_kw={"elinewidth": 2.8, "capthick": 2.8},
    )
    ax.axhline(2.0, color="#222222", lw=2.8, ls="--", label="Target logSw = 2.0")
    ax.set_xticks(x, [SHORT[m] for m in METHODS])
    ax.set_ylabel("Mean predicted logSw of reliable candidates", labelpad=14)
    ax.set_title("Sweetness potency of reliable candidates", pad=18)
    ax.set_ylim(1.9, max(logsw_mean + logsw_std) + 0.25)
    ax.grid(axis="y", alpha=0.22, linewidth=1.1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "D_reliable_candidate_predicted_logsw")


def plot_docking_quality(run_df, out_dir):
    means, stds = ordered_stats(run_df, "mean_docking_reliable")
    fig, ax = plt.subplots(figsize=(14.5, 9.2))
    x = np.arange(len(METHODS))
    ax.bar(
        x, means, yerr=stds, width=0.66,
        color=[COLORS[m] for m in METHODS],
        capsize=7,
        error_kw={"elinewidth": 2.8, "capthick": 2.8},
    )
    ax.set_xticks(x, [SHORT[m] for m in METHODS])
    ax.set_ylabel("Mean predicted -Vina affinity", labelpad=14)
    ax.set_title("Docking-surrogate score of reliable candidates", pad=18)
    ax.grid(axis="y", alpha=0.22, linewidth=1.1)
    fig.tight_layout()
    save(fig, out_dir, "E_reliable_candidate_docking_score")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    setup_style()
    run_df = pd.read_csv(result_dir / "ablation_run_metrics.csv")
    progress_df = pd.read_csv(result_dir / "ablation_progress_all.csv")
    run_df.to_csv(out_dir / "four_fitness_run_metrics.csv", index=False)
    progress_df.to_csv(out_dir / "four_fitness_progress_all.csv", index=False)
    plot_final_yield(run_df, out_dir)
    plot_near_target(progress_df, out_dir)
    plot_uniqueness_diversity(run_df, out_dir)
    plot_reliable_quality(run_df, out_dir)
    plot_docking_quality(run_df, out_dir)


if __name__ == "__main__":
    main()
