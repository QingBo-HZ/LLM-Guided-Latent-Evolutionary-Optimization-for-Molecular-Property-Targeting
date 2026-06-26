#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
FIG_DIR = ROOT / "Main_results_202604_LLM_GA" / "figures"
PROJ_CSV = FIG_DIR / "seed42_space_distribution_projection.csv"

METHODS = [
    ("random", "Random Latent Search", "Random", "C0"),
    ("brics", "BRICS-based SMILES GA", "BRICS", "C2"),
    ("latent_ga", "Latent GA", "Latent GA", "C1"),
    ("llm_init", "LLM-Initialized Latent GA", "LLM-Init", "C3"),
    ("ours", "Iterative LLM-Guided Latent GA", "Iterative LLM", "C4"),
]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "axes.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 19,
        "figure.dpi": 180,
        "savefig.dpi": 600,
    })


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb * (1 - amount) + np.ones(3) * amount)


def gradient_line(ax, x: np.ndarray, y: np.ndarray, color: str, *, lw: float, zorder: int) -> None:
    if len(x) < 2:
        return
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    base = np.array(mpl.colors.to_rgb(color))
    colors = []
    for i in range(len(segments)):
        t = i / max(1, len(segments) - 1)
        amount = 0.70 * (1 - t) + 0.03 * t
        c = base * (1 - amount) + np.ones(3) * amount
        colors.append((*c, 0.35 + 0.60 * t))
    lc = LineCollection(segments, colors=colors, linewidths=lw, capstyle="round", zorder=zorder)
    ax.add_collection(lc)


def add_direction_arrows(ax, x: np.ndarray, y: np.ndarray, color: str, n_arrows: int = 3) -> None:
    if len(x) < 3:
        return
    candidates = np.linspace(1, len(x) - 1, min(n_arrows, len(x) - 1), dtype=int)
    for idx in candidates:
        x0, y0 = x[idx - 1], y[idx - 1]
        x1, y1 = x[idx], y[idx]
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.7,
                "color": color,
                "alpha": 0.72,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=8,
        )


def main() -> None:
    if not PROJ_CSV.exists():
        raise FileNotFoundError(f"Projection CSV not found: {PROJ_CSV}")

    setup_style()
    df = pd.read_csv(PROJ_CSV)

    train = df[df["stage"] == "training_background"]
    fig, ax = plt.subplots(figsize=(14.5, 10.4))

    ax.scatter(
        train["umap_x"],
        train["umap_y"],
        s=9,
        c="#bfc4cc",
        alpha=0.16,
        linewidths=0,
        rasterized=True,
        label="QM9 training background",
        zorder=1,
    )

    handles = []
    for method_key, method_name, short_name, color in METHODS:
        sub = df[df["method_key"] == method_key]
        final = sub[sub["stage"] == "final"]
        path = sub[sub["stage"] == "encoded_evolution_path"].copy()
        path = path.sort_values(["generation_min", "generation_max"])

        if len(final):
            ax.scatter(
                final["umap_x"],
                final["umap_y"],
                s=34 if method_key != "brics" else 78,
                c=color,
                alpha=0.20,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

        if len(path):
            x = path["umap_x"].to_numpy(dtype=float)
            y = path["umap_y"].to_numpy(dtype=float)
            gradient_line(ax, x, y, color, lw=3.1, zorder=5)
            add_direction_arrows(ax, x, y, color, n_arrows=3)

            sizes = 60 + 17 * np.sqrt(path["n_generations"].fillna(1).clip(lower=1).to_numpy(dtype=float))
            progress = np.linspace(0, 1, len(path))
            point_colors = [
                (*lighten(color, 0.68 * (1 - t) + 0.04 * t), 0.58 + 0.34 * t)
                for t in progress
            ]
            ax.scatter(
                x,
                y,
                s=sizes,
                c=point_colors,
                edgecolors=color,
                linewidths=1.0,
                rasterized=True,
                zorder=7,
            )

            ax.scatter(
                [x[0]],
                [y[0]],
                s=190,
                facecolors="white",
                edgecolors=color,
                linewidths=2.2,
                marker="o",
                zorder=9,
            )
            ax.scatter(
                [x[-1]],
                [y[-1]],
                s=260,
                facecolors=color,
                edgecolors="white",
                linewidths=1.5,
                marker="*",
                zorder=10,
            )

        handles.append(Line2D([0], [0], color=color, lw=4.0, marker="o", markersize=9, label=short_name))

    start_handle = Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", lw=0, markersize=11, label="Path start")
    end_handle = Line2D([0], [0], marker="*", color="#555555", markerfacecolor="#555555", lw=0, markersize=15, label="Path end")
    final_handle = Line2D([0], [0], marker="o", color="#777777", alpha=0.35, lw=0, markersize=9, label="Final population")

    ax.legend(
        handles=handles + [start_handle, end_handle, final_handle],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=False,
        borderaxespad=0,
        title="Method / marker",
        title_fontsize=20,
    )

    ax.text(
        1.01,
        0.36,
        "Lighter path color: early stage\nDarker path color: late stage\nLarger node: longer persistence",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        color="#464C55",
        linespacing=1.28,
    )

    ax.set_title("Integrated UMAP trajectories of seed-42 molecular optimization")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.20, linewidth=1.0)
    ax.tick_params(width=1.5, length=5)

    all_points = df[df["stage"].isin(["final", "encoded_evolution_path"])]
    xmin, xmax = all_points["umap_x"].min(), all_points["umap_x"].max()
    ymin, ymax = all_points["umap_y"].min(), all_points["umap_y"].max()
    xr, yr = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.08 * xr, xmax + 0.08 * xr)
    ax.set_ylim(ymin - 0.10 * yr, ymax + 0.10 * yr)

    fig.subplots_adjust(left=0.08, right=0.76, top=0.91, bottom=0.11)

    out_png = FIG_DIR / "fig2d_integrated_umap_trajectory_seed42_highres.png"
    out_pdf = FIG_DIR / "fig2d_integrated_umap_trajectory_seed42_highres.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")

    make_inset_version(df)
    make_milestone_version(df)


def draw_integrated_paths(ax, df: pd.DataFrame, *, compact: bool = False) -> None:
    train = df[df["stage"] == "training_background"]
    ax.scatter(
        train["umap_x"],
        train["umap_y"],
        s=6 if compact else 9,
        c="#bfc4cc",
        alpha=0.11 if compact else 0.16,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    for method_key, _, _, color in METHODS:
        sub = df[df["method_key"] == method_key]
        final = sub[sub["stage"] == "final"]
        path = sub[sub["stage"] == "encoded_evolution_path"].copy()
        path = path.sort_values(["generation_min", "generation_max"])

        if len(final):
            ax.scatter(
                final["umap_x"],
                final["umap_y"],
                s=22 if compact else (34 if method_key != "brics" else 78),
                c=color,
                alpha=0.18 if compact else 0.20,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
        if len(path):
            x = path["umap_x"].to_numpy(dtype=float)
            y = path["umap_y"].to_numpy(dtype=float)
            gradient_line(ax, x, y, color, lw=2.1 if compact else 3.1, zorder=5)
            if not compact:
                add_direction_arrows(ax, x, y, color, n_arrows=3)
            sizes = 35 + 11 * np.sqrt(path["n_generations"].fillna(1).clip(lower=1).to_numpy(dtype=float))
            if not compact:
                sizes = 60 + 17 * np.sqrt(path["n_generations"].fillna(1).clip(lower=1).to_numpy(dtype=float))
            progress = np.linspace(0, 1, len(path))
            point_colors = [
                (*lighten(color, 0.68 * (1 - t) + 0.04 * t), 0.58 + 0.34 * t)
                for t in progress
            ]
            ax.scatter(
                x,
                y,
                s=sizes,
                c=point_colors,
                edgecolors=color,
                linewidths=0.8 if compact else 1.0,
                rasterized=True,
                zorder=7,
            )
            if not compact:
                ax.scatter([x[0]], [y[0]], s=190, facecolors="white", edgecolors=color, linewidths=2.2, marker="o", zorder=9)
                ax.scatter([x[-1]], [y[-1]], s=260, facecolors=color, edgecolors="white", linewidths=1.5, marker="*", zorder=10)


def make_inset_version(df: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(14.5, 10.4))
    draw_integrated_paths(ax, df, compact=False)

    ax.set_title("Integrated UMAP trajectories of seed-42 molecular optimization")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.20, linewidth=1.0)
    ax.tick_params(width=1.5, length=5)

    # Focus the main panel on the convergence region; the inset keeps the full landscape.
    ax.set_xlim(22.6, 30.3)
    ax.set_ylim(-2.45, 3.35)

    inset = fig.add_axes([0.108, 0.655, 0.285, 0.235])
    draw_integrated_paths(inset, df, compact=True)
    all_points = df[df["stage"].isin(["final", "encoded_evolution_path"])]
    xmin, xmax = all_points["umap_x"].min(), all_points["umap_x"].max()
    ymin, ymax = all_points["umap_y"].min(), all_points["umap_y"].max()
    xr, yr = xmax - xmin, ymax - ymin
    inset.set_xlim(xmin - 0.08 * xr, xmax + 0.08 * xr)
    inset.set_ylim(ymin - 0.10 * yr, ymax + 0.10 * yr)
    inset.set_title("Global overview", fontsize=16, pad=4)
    inset.tick_params(labelsize=10, width=0.8, length=3)
    inset.grid(alpha=0.18, linewidth=0.7)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#464C55")

    handles = [
        Line2D([0], [0], color=color, lw=4.0, marker="o", markersize=9, label=short)
        for _, _, short, color in METHODS
    ]
    handles.extend([
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", lw=0, markersize=11, label="Path start"),
        Line2D([0], [0], marker="*", color="#555555", markerfacecolor="#555555", lw=0, markersize=15, label="Path end"),
        Line2D([0], [0], marker="o", color="#777777", alpha=0.35, lw=0, markersize=9, label="Final population"),
    ])
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=False,
        borderaxespad=0,
        title="Method / marker",
        title_fontsize=20,
    )
    ax.text(
        1.01,
        0.36,
        "Lighter path color: early stage\nDarker path color: late stage\nLarger node: longer persistence",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        color="#464C55",
        linespacing=1.28,
    )

    fig.subplots_adjust(left=0.08, right=0.76, top=0.91, bottom=0.11)
    out_png = FIG_DIR / "fig2d_integrated_umap_trajectory_seed42_inset_highres.png"
    out_pdf = FIG_DIR / "fig2d_integrated_umap_trajectory_seed42_inset_highres.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


def path_milestones(path: pd.DataFrame, max_nodes: int = 18) -> pd.DataFrame:
    path = path.sort_values(["generation_min", "generation_max"]).copy()
    if len(path) <= max_nodes:
        return path
    chunks = np.array_split(path, max_nodes)
    rows = []
    for chunk in chunks:
        weights = chunk["n_generations"].fillna(1).clip(lower=1).to_numpy(dtype=float)
        rows.append({
            "umap_x": np.average(chunk["umap_x"].to_numpy(dtype=float), weights=weights),
            "umap_y": np.average(chunk["umap_y"].to_numpy(dtype=float), weights=weights),
            "generation_min": float(chunk["generation_min"].min()),
            "generation_max": float(chunk["generation_max"].max()),
            "n_generations": float(weights.sum()),
        })
    return pd.DataFrame(rows)


def draw_milestones(ax, df: pd.DataFrame, *, compact: bool = False) -> None:
    train = df[df["stage"] == "training_background"]
    ax.scatter(
        train["umap_x"],
        train["umap_y"],
        s=5 if compact else 8,
        c="#bfc4cc",
        alpha=0.10 if compact else 0.14,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    for method_key, _, _, color in METHODS:
        sub = df[df["method_key"] == method_key]
        final = sub[sub["stage"] == "final"]
        path = path_milestones(sub[sub["stage"] == "encoded_evolution_path"], max_nodes=16)
        if len(final):
            ax.scatter(
                final["umap_x"],
                final["umap_y"],
                s=18 if compact else 26,
                c=color,
                alpha=0.12 if compact else 0.16,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
        if len(path):
            x = path["umap_x"].to_numpy(dtype=float)
            y = path["umap_y"].to_numpy(dtype=float)
            gradient_line(ax, x, y, color, lw=2.2 if compact else 4.0, zorder=5)
            if not compact:
                add_direction_arrows(ax, x, y, color, n_arrows=3)
            progress = np.linspace(0, 1, len(path))
            sizes = 70 + 15 * np.sqrt(path["n_generations"].fillna(1).clip(lower=1).to_numpy(dtype=float))
            if compact:
                sizes = 28 + 7 * np.sqrt(path["n_generations"].fillna(1).clip(lower=1).to_numpy(dtype=float))
            colors = [
                (*lighten(color, 0.72 * (1 - t) + 0.03 * t), 0.62 + 0.32 * t)
                for t in progress
            ]
            ax.scatter(
                x,
                y,
                s=sizes,
                c=colors,
                edgecolors=color,
                linewidths=1.2 if not compact else 0.7,
                rasterized=True,
                zorder=7,
            )
            if not compact:
                ax.scatter([x[0]], [y[0]], s=220, facecolors="white", edgecolors=color, linewidths=2.4, marker="o", zorder=9)
                ax.scatter([x[-1]], [y[-1]], s=310, facecolors=color, edgecolors="white", linewidths=1.7, marker="*", zorder=10)


def make_milestone_version(df: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(14.5, 10.4))
    draw_milestones(ax, df, compact=False)
    ax.set_title("Integrated UMAP milestone trajectories of seed-42 optimization")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.20, linewidth=1.0)
    ax.tick_params(width=1.5, length=5)
    ax.set_xlim(22.6, 30.3)
    ax.set_ylim(-2.45, 3.35)

    inset = fig.add_axes([0.108, 0.655, 0.285, 0.235])
    draw_milestones(inset, df, compact=True)
    all_points = df[df["stage"].isin(["final", "encoded_evolution_path"])]
    xmin, xmax = all_points["umap_x"].min(), all_points["umap_x"].max()
    ymin, ymax = all_points["umap_y"].min(), all_points["umap_y"].max()
    xr, yr = xmax - xmin, ymax - ymin
    inset.set_xlim(xmin - 0.08 * xr, xmax + 0.08 * xr)
    inset.set_ylim(ymin - 0.10 * yr, ymax + 0.10 * yr)
    inset.set_title("Global overview", fontsize=16, pad=4)
    inset.tick_params(labelsize=10, width=0.8, length=3)
    inset.grid(alpha=0.18, linewidth=0.7)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#464C55")

    handles = [
        Line2D([0], [0], color=color, lw=4.0, marker="o", markersize=9, label=short)
        for _, _, short, color in METHODS
    ]
    handles.extend([
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", lw=0, markersize=11, label="Path start"),
        Line2D([0], [0], marker="*", color="#555555", markerfacecolor="#555555", lw=0, markersize=15, label="Path end"),
        Line2D([0], [0], marker="o", color="#777777", alpha=0.35, lw=0, markersize=9, label="Final population"),
    ])
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=False,
        borderaxespad=0,
        title="Method / marker",
        title_fontsize=20,
    )
    ax.text(
        1.01,
        0.36,
        "Lighter path color: early stage\nDarker path color: late stage\nEach node summarizes one path stage",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        color="#464C55",
        linespacing=1.28,
    )
    fig.subplots_adjust(left=0.08, right=0.76, top=0.91, bottom=0.11)
    out_png = FIG_DIR / "fig2d_integrated_umap_milestone_trajectory_seed42_highres.png"
    out_pdf = FIG_DIR / "fig2d_integrated_umap_milestone_trajectory_seed42_highres.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
