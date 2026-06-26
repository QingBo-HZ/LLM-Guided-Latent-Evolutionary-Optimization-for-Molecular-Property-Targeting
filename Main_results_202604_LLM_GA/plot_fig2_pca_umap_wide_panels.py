#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "64")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
EXP = ROOT / "Main_results_202604_LLM_GA"
OUT = EXP / "figures"
PROJECTION_CSV = OUT / "seed42_space_distribution_projection.csv"

METHODS = [
    {"key": "random", "name": "Random Latent Search", "short": "Random", "color": "C0"},
    {"key": "brics", "name": "BRICS-based SMILES GA", "short": "BRICS", "color": "C2"},
    {"key": "latent_ga", "name": "Latent GA", "short": "Latent GA", "color": "C1"},
    {"key": "llm_init", "name": "LLM-Initialized Latent GA", "short": "LLM-init", "color": "C3"},
    {"key": "ours", "name": "Iterative LLM-Guided Latent GA", "short": "Iterative LLM", "color": "C4"},
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "train": "#BFC4CC",
    "dark": "#464C55",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 600,
    })


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb * (1 - amount) + np.ones(3) * amount)


def set_limits(ax, xs, ys, pad: float = 0.13) -> None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) == 0:
        return
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xr = xmax - xmin if xmax > xmin else 1.0
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_xlim(xmin - xr * pad, xmax + xr * pad)
    ax.set_ylim(ymin - yr * pad, ymax + yr * pad)



def nearest_train_distance(path: pd.DataFrame, train: pd.DataFrame, xcol: str, ycol: str) -> pd.Series:
    path_xy = path[[xcol, ycol]].to_numpy(dtype=float)
    train_xy = train[[xcol, ycol]].dropna().to_numpy(dtype=float)
    if len(path_xy) == 0 or len(train_xy) == 0:
        return pd.Series(np.zeros(len(path)), index=path.index)
    # Small arrays here, so a dense distance matrix is simpler and avoids extra dependencies.
    d2 = ((path_xy[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=2)
    return pd.Series(np.sqrt(d2.min(axis=1)), index=path.index)


def key_step_rows(path: pd.DataFrame, train: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    if path.empty:
        return path
    chosen = []
    # Start, the most off-manifold/island-like state, and the final/late state.
    start_idx = path["generation_min"].astype(float).idxmin()
    dist = nearest_train_distance(path, train, xcol, ycol)
    island_idx = dist.idxmax()
    end_idx = path["generation_max"].fillna(path["generation_min"]).astype(float).idxmax()
    for idx in [start_idx, island_idx, end_idx]:
        if idx not in chosen:
            chosen.append(idx)
    return path.loc[chosen].sort_values("generation_min")


def generation_label(row: pd.Series) -> str:
    g0 = int(row["generation_min"])
    g1 = int(row["generation_max"])
    if g1 - g0 >= 40:
        return f"g{g0}-{g1}"
    return f"g{g0}"


def annotate_key_steps(ax, path: pd.DataFrame, train: pd.DataFrame, xcol: str, ycol: str, color: str, *, fontsize: float) -> None:
    labels = key_step_rows(path, train, xcol, ycol)
    for _, row in labels.iterrows():
        x = float(row[xcol])
        y = float(row[ycol])
        ax.scatter([x], [y], s=78, facecolors="white", edgecolors=color, linewidths=1.15, zorder=7)
        ax.annotate(
            generation_label(row),
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=fontsize,
            color=TOKENS["dark"],
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "lw": 0.55, "alpha": 0.82},
            zorder=8,
        )

def plot_wide(df: pd.DataFrame, coord: str, *, local_zoom: bool = True) -> tuple[Path, Path]:
    setup_style()
    xcol, ycol = f"{coord}_x", f"{coord}_y"
    train = df[df["stage"] == "training_background"]

    # Wide two-column group panel: one horizontal strip, compact height.
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 4.5), sharex=not local_zoom, sharey=not local_zoom, facecolor="white")

    for ax, spec in zip(axes, METHODS):
        color = spec["color"]
        sub = df[df["method_key"] == spec["key"]]
        path = sub[sub["stage"] == "encoded_evolution_path"].copy().sort_values("generation_min")
        final = sub[sub["stage"] == "final"].copy()

        ax.set_facecolor(TOKENS["panel"])
        ax.scatter(
            train[xcol], train[ycol],
            s=4.5, c=TOKENS["train"], alpha=0.13, linewidths=0,
            rasterized=True, zorder=1,
        )

        if len(path):
            gen_span = path["generation_max"].fillna(path["generation_min"]).astype(float)
            denom = max(float(gen_span.max() - gen_span.min()), 1.0)
            t = ((gen_span - float(gen_span.min())) / denom).clip(0, 1)
            sizes = 24 + 58 * np.power(t, 0.75)
            colors = [(*lighten(color, 0.68 * (1 - float(v)) + 0.08 * float(v)), 0.48 + 0.38 * float(v)) for v in t]
            ax.scatter(
                path[xcol], path[ycol],
                s=sizes, c=colors, edgecolors=color, linewidths=0.65,
                rasterized=True, zorder=4,
            )
            annotate_key_steps(ax, path, train, xcol, ycol, color, fontsize=8.6)
        if len(final):
            ax.scatter(
                final[xcol], final[ycol],
                s=17 if spec["key"] != "brics" else 30,
                c=color, alpha=0.55, linewidths=0.35, edgecolors="white",
                rasterized=True, zorder=5,
            )

        ax.set_title(spec["short"], color=color, weight="bold", pad=5, fontsize=12)
        ax.grid(True, color=TOKENS["grid"], linewidth=0.55, alpha=0.75)
        ax.tick_params(width=0.65, length=2.5, color=TOKENS["axis"], labelcolor=TOKENS["muted"])
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
        ax.set_xlabel(f"{coord.upper()}-1", labelpad=2, color=TOKENS["ink"])
        if ax is axes[0]:
            ax.set_ylabel(f"{coord.upper()}-2", labelpad=2, color=TOKENS["ink"])
        else:
            ax.set_ylabel("")

        if local_zoom:
            focus = pd.concat([path[[xcol, ycol]], final[[xcol, ycol]]], ignore_index=True)
            set_limits(ax, focus[xcol], focus[ycol], pad=0.16)

    handles = [
        Line2D([], [], marker="o", color=TOKENS["train"], linestyle="None", markersize=5, alpha=0.45, label="QM9 background"),
        Line2D([], [], marker="o", color=TOKENS["dark"], linestyle="None", markersize=5, lw=0, alpha=0.65, label="Evolution states"),
        Line2D([], [], marker="o", color=TOKENS["dark"], linestyle="None", markersize=5, alpha=0.70, label="Final population"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.02), ncol=3, frameon=False, handlelength=1.7, columnspacing=1.7)
    fig.suptitle(f"{coord.upper()} chemical-space evolution across methods", x=0.02, y=1.02, ha="left", fontsize=15, weight="bold", color=TOKENS["ink"])
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.16, top=0.82, wspace=0.12)

    suffix = "wide_local" if local_zoom else "wide_global"
    png = OUT / f"fig2_{coord}_space_distribution_seed42_{suffix}.png"
    pdf = OUT / f"fig2_{coord}_space_distribution_seed42_{suffix}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf



def plot_wide_grid(df: pd.DataFrame, coord: str, *, local_zoom: bool = True) -> tuple[Path, Path]:
    setup_style()
    xcol, ycol = f"{coord}_x", f"{coord}_y"
    train = df[df["stage"] == "training_background"]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.3), sharex=not local_zoom, sharey=not local_zoom, facecolor="white")
    axes_flat = list(axes.flat)

    for ax, spec in zip(axes_flat[:5], METHODS):
        color = spec["color"]
        sub = df[df["method_key"] == spec["key"]]
        path = sub[sub["stage"] == "encoded_evolution_path"].copy().sort_values("generation_min")
        final = sub[sub["stage"] == "final"].copy()
        ax.set_facecolor(TOKENS["panel"])
        ax.scatter(train[xcol], train[ycol], s=4.5, c=TOKENS["train"], alpha=0.12, linewidths=0, rasterized=True, zorder=1)
        if len(path):
            gen_span = path["generation_max"].fillna(path["generation_min"]).astype(float)
            denom = max(float(gen_span.max() - gen_span.min()), 1.0)
            t = ((gen_span - float(gen_span.min())) / denom).clip(0, 1)
            sizes = 22 + 54 * np.power(t, 0.75)
            colors = [(*lighten(color, 0.68 * (1 - float(v)) + 0.08 * float(v)), 0.48 + 0.38 * float(v)) for v in t]
            ax.scatter(path[xcol], path[ycol], s=sizes, c=colors, edgecolors=color, linewidths=0.6, rasterized=True, zorder=4)
            annotate_key_steps(ax, path, train, xcol, ycol, color, fontsize=9.0)
        if len(final):
            ax.scatter(final[xcol], final[ycol], s=16 if spec["key"] != "brics" else 28, c=color, alpha=0.55, linewidths=0.35, edgecolors="white", rasterized=True, zorder=5)
        ax.set_title(spec["short"], color=color, weight="bold", pad=4, fontsize=13)
        ax.grid(True, color=TOKENS["grid"], linewidth=0.5, alpha=0.75)
        ax.tick_params(width=0.6, length=2.2, color=TOKENS["axis"], labelcolor=TOKENS["muted"])
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
        ax.set_xlabel(f"{coord.upper()}-1", labelpad=1.5, color=TOKENS["ink"])
        ax.set_ylabel(f"{coord.upper()}-2", labelpad=1.5, color=TOKENS["ink"])
        if local_zoom:
            focus = pd.concat([path[[xcol, ycol]], final[[xcol, ycol]]], ignore_index=True)
            set_limits(ax, focus[xcol], focus[ycol], pad=0.16)

    legend_ax = axes_flat[5]
    legend_ax.axis("off")
    handles = [
        Line2D([], [], marker="o", color=TOKENS["train"], linestyle="None", markersize=5, alpha=0.45, label="QM9 background"),
        Line2D([], [], marker="o", color=TOKENS["dark"], linestyle="None", markersize=5, lw=0, alpha=0.65, label="Evolution states"),
        Line2D([], [], marker="o", color=TOKENS["dark"], linestyle="None", markersize=5, alpha=0.70, label="Final population"),
        Line2D([], [], marker="o", color=TOKENS["dark"], linestyle="None", markersize=4, alpha=0.28, label="Early path state"),
        Line2D([], [], marker="o", color=TOKENS["dark"], linestyle="None", markersize=8, alpha=0.88, label="Late path state"),
    ]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, handlelength=1.8, borderaxespad=0)
    legend_ax.text(0.0, 0.18, "Colors match Fig. 2. Larger/darker points are later states; labels mark key generations.", transform=legend_ax.transAxes, fontsize=10.0, color=TOKENS["muted"], va="top", wrap=True)

    fig.suptitle(f"{coord.upper()} chemical-space evolution across methods", x=0.055, y=0.985, ha="left", fontsize=15, weight="bold", color=TOKENS["ink"])
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.10, top=0.88, wspace=0.16, hspace=0.34)
    suffix = "wide_grid_local" if local_zoom else "wide_grid_global"
    png = OUT / f"fig2_{coord}_space_distribution_seed42_{suffix}.png"
    pdf = OUT / f"fig2_{coord}_space_distribution_seed42_{suffix}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf

def main() -> None:
    df = pd.read_csv(PROJECTION_CSV)
    outputs = []
    for coord in ("pca", "umap"):
        outputs.extend(plot_wide(df, coord, local_zoom=True))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
