#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "64")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
EXP = ROOT / "Main_results_202604_LLM_GA"
OUT = EXP / "figures"
PROJECTION_CSV = OUT / "seed42_space_distribution_projection.csv"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "neutral_light": "#E2E5EA",
    "neutral_mid": "#7A828F",
}

METHODS = [
    ("random", "Random Latent Search", "C0"),
    ("brics", "BRICS-based SMILES GA", "C2"),
    ("latent_ga", "Latent GA", "C1"),
    ("llm_init", "LLM-Initialized Latent GA", "C3"),
    ("ours", "Iterative LLM-Guided Latent GA", "C4"),
]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 16,
        "axes.labelsize": 17,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.dpi": 180,
        "savefig.dpi": 600,
    })


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=86, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=118, break_long_words=False)
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=20, weight="bold", color=TOKENS["ink"])
    fig.text(left, 0.922, subtitle, ha="left", va="top", fontsize=12.5, color=TOKENS["muted"])


def set_limits(ax, x: pd.Series, y: pd.Series, pad: float = 0.075) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xr = xmax - xmin if xmax > xmin else 1.0
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_xlim(xmin - xr * pad, xmax + xr * pad)
    ax.set_ylim(ymin - yr * pad, ymax + yr * pad)


def encode_generation_style(path_df: pd.DataFrame) -> pd.DataFrame:
    out = path_df.copy()
    gen = out["generation_max"].fillna(out["generation_min"]).astype(float)
    gmin = float(gen.min())
    gmax = float(gen.max())
    denom = gmax - gmin if gmax > gmin else 1.0
    t = ((gen - gmin) / denom).clip(0.0, 1.0)

    out["generation_style_t"] = t
    out["point_size"] = 22 + 120 * np.power(t, 0.75)
    out["point_alpha"] = 0.20 + 0.70 * np.power(t, 0.85)
    return out


def plot_projection(df: pd.DataFrame, coord: str) -> tuple[Path, Path]:
    setup_style()
    method_colors = dict((k, c) for k, _, c in METHODS)
    xcol, ycol = f"{coord}_x", f"{coord}_y"

    path = encode_generation_style(df[df["stage"] == "encoded_evolution_path"])
    train = df[df["stage"] == "training_background"]

    fig, ax = plt.subplots(figsize=(11.8, 9.0), facecolor=TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])

    # Keep the QM9 training manifold as context, but quiet enough that evolution remains primary.
    ax.scatter(
        train[xcol],
        train[ycol],
        s=8,
        c=TOKENS["neutral_light"],
        alpha=0.16,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # Draw early generations first and later generations last, so darker/larger points stay visible.
    path = path.sort_values(["generation_style_t", "method_key"])
    for key, _, color in METHODS:
        sub = path[path["method_key"] == key]
        if sub.empty:
            continue
        rgba = np.array([mpl.colors.to_rgba(color, a) for a in sub["point_alpha"]])
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=sub["point_size"],
            c=rgba,
            edgecolors=mpl.colors.to_rgba(color, 0.82),
            linewidths=0.55,
            rasterized=True,
            zorder=3,
        )

    set_limits(ax, path[xcol], path[ycol], pad=0.09)
    ax.grid(True, color=TOKENS["grid"], linewidth=0.8, alpha=0.8)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(color=TOKENS["axis"], labelcolor=TOKENS["muted"])
    ax.set_xlabel(f"{coord.upper()}-1", color=TOKENS["ink"])
    ax.set_ylabel(f"{coord.upper()}-2", color=TOKENS["ink"])

    method_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.82,
            label=name,
        )
        for _, name, color in METHODS
    ]
    gen_handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=5, color=TOKENS["neutral_mid"], alpha=0.28, label="Early generation"),
        Line2D([], [], marker="o", linestyle="None", markersize=9, color=TOKENS["neutral_mid"], alpha=0.58, label="Middle generation"),
        Line2D([], [], marker="o", linestyle="None", markersize=12, color=TOKENS["neutral_mid"], alpha=0.90, label="Late generation"),
    ]

    legend1 = ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.1,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=gen_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    add_chart_header(
        fig,
        ax,
        f"{coord.upper()} evolution of molecular search trajectories",
        "Each method uses a fixed color. Within every method, later generations are drawn with larger, darker, more opaque bubbles; no arrows or connecting lines are used.",
    )

    png = OUT / f"fig2_{coord}_evolution_bubbles_seed42.png"
    pdf = OUT / f"fig2_{coord}_evolution_bubbles_seed42.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)
    return png, pdf



def encode_generation_style_by_method(path_df: pd.DataFrame) -> pd.DataFrame:
    out = path_df.copy()
    styled = []
    for key, sub in out.groupby("method_key", sort=False):
        sub = sub.copy()
        gen = sub["generation_max"].fillna(sub["generation_min"]).astype(float)
        gmin = float(gen.min())
        gmax = float(gen.max())
        denom = gmax - gmin if gmax > gmin else 1.0
        t = ((gen - gmin) / denom).clip(0.0, 1.0)
        sub["generation_style_t"] = t
        sub["point_size"] = 22 + 120 * np.power(t, 0.75)
        sub["point_alpha"] = 0.20 + 0.70 * np.power(t, 0.85)
        styled.append(sub)
    return pd.concat(styled, ignore_index=True)


def plot_projection_faceted(df: pd.DataFrame, coord: str) -> tuple[Path, Path]:
    setup_style()
    xcol, ycol = f"{coord}_x", f"{coord}_y"
    path = encode_generation_style_by_method(df[df["stage"] == "encoded_evolution_path"])
    train = df[df["stage"] == "training_background"]

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.8), facecolor=TOKENS["surface"])
    axes_flat = list(axes.flat)

    for ax, (key, name, color) in zip(axes_flat[:5], METHODS):
        sub = path[path["method_key"] == key].sort_values("generation_style_t")
        ax.set_facecolor(TOKENS["panel"])
        ax.scatter(
            train[xcol],
            train[ycol],
            s=6,
            c=TOKENS["neutral_light"],
            alpha=0.12,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        if not sub.empty:
            rgba = np.array([mpl.colors.to_rgba(color, a) for a in sub["point_alpha"]])
            ax.scatter(
                sub[xcol],
                sub[ycol],
                s=sub["point_size"],
                c=rgba,
                edgecolors=mpl.colors.to_rgba(color, 0.82),
                linewidths=0.55,
                rasterized=True,
                zorder=3,
            )
            set_limits(ax, sub[xcol], sub[ycol], pad=0.15)
        ax.set_title(name, fontsize=12.5, color=mpl.colors.to_rgba(color, 1), weight="bold", pad=8)
        ax.grid(True, color=TOKENS["grid"], linewidth=0.75, alpha=0.75)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])
        ax.tick_params(color=TOKENS["axis"], labelcolor=TOKENS["muted"], labelsize=10)
        ax.set_xlabel(f"{coord.upper()}-1", color=TOKENS["ink"], fontsize=11)
        ax.set_ylabel(f"{coord.upper()}-2", color=TOKENS["ink"], fontsize=11)

    legend_ax = axes_flat[5]
    legend_ax.axis("off")
    handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=5, color=TOKENS["neutral_mid"], alpha=0.28, label="Early"),
        Line2D([], [], marker="o", linestyle="None", markersize=9, color=TOKENS["neutral_mid"], alpha=0.58, label="Middle"),
        Line2D([], [], marker="o", linestyle="None", markersize=12, color=TOKENS["neutral_mid"], alpha=0.90, label="Late"),
    ]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, title="Relative generation\nwithin each method")
    legend_ax.text(
        0.0,
        0.30,
        "Separate panels reduce overlap, so the reader can compare where each method moves in chemical space. Colors match the existing Fig. 2 palette.",
        transform=legend_ax.transAxes,
        fontsize=11.5,
        color=TOKENS["muted"],
        va="top",
        wrap=True,
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.08, wspace=0.18, hspace=0.34)
    fig.text(0.055, 0.965, f"{coord.upper()} evolution trajectories by method", ha="left", va="top", fontsize=20, weight="bold", color=TOKENS["ink"])
    fig.text(0.055, 0.925, "Each panel uses the same projection data and the same generation encoding; later generations are larger and more opaque. No arrows or connecting lines are used.", ha="left", va="top", fontsize=12.5, color=TOKENS["muted"])

    png = OUT / f"fig2_{coord}_evolution_bubbles_faceted_seed42.png"
    pdf = OUT / f"fig2_{coord}_evolution_bubbles_faceted_seed42.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)
    return png, pdf

def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECTION_CSV)
    required = {"stage", "method_key", "generation_min", "generation_max", "pca_x", "pca_y", "umap_x", "umap_y"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{PROJECTION_CSV} missing columns: {missing}")

    outputs = []
    for coord in ("pca", "umap"):
        outputs.extend(plot_projection(df, coord))
        outputs.extend(plot_projection_faceted(df, coord))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
