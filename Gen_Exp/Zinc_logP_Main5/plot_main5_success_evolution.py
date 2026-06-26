#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot success-count evolution curves for the formal ZINC logP five-group experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    {
        "name": "Random Latent Search",
        "pattern": "random_latent_formal_seed{seed}/progress_metrics_random_latent.csv",
        "kind": "random",
        "color": "#6f6f6f",
    },
    {
        "name": "ZINC-Seeded Latent GA",
        "pattern": "train_random_zinc_seeded_formal_seed{seed}/progress_metrics_decode_aware.csv",
        "kind": "ga",
        "color": "#2f6fbb",
    },
    {
        "name": "LLM-Initialized Latent GA",
        "pattern": "llm_llm_initialized_formal_seed{seed}/progress_metrics_decode_aware.csv",
        "kind": "ga",
        "color": "#d8872b",
    },
    {
        "name": "Iterative LLM-Guided Latent GA",
        "pattern": "llm_iterative_llm_guided_formal_seed{seed}/progress_metrics_decode_aware.csv",
        "kind": "ga",
        "color": "#2b9a66",
    },
]


def load_curve(results_dir: Path, pattern: str, seed: int, kind: str) -> pd.DataFrame:
    path = results_dir / pattern.format(seed=seed)
    df = pd.read_csv(path)
    if kind == "ga":
        out = pd.DataFrame({
            "evolution": df["generation"].astype(int) + 1,
            "success_count": pd.to_numeric(df["archive_rdkit_success_unique"], errors="coerce"),
            "seed": seed,
        })
    else:
        out = pd.DataFrame({
            "evolution": (pd.to_numeric(df["evaluations"], errors="coerce") / 100.0).round().astype(int),
            "success_count": pd.to_numeric(df["archive_rdkit_success_unique"], errors="coerce"),
            "seed": seed,
        })
    return out


def mean_std(curves: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(curves, ignore_index=True)
    stat = df.groupby("evolution")["success_count"].agg(["mean", "std"]).reset_index()
    stat["std"] = stat["std"].fillna(0.0)
    return stat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/results"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/figures"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(args.results_dir / "zinc_logp_main5_formal_detail.csv")
    direct = detail[detail["method"] == "LLM-Generated Molecules"]
    direct_mean = float(pd.to_numeric(direct["archive_rdkit_success_unique"], errors="coerce").mean())
    direct_std = float(pd.to_numeric(direct["archive_rdkit_success_unique"], errors="coerce").std(ddof=1))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 160,
    })

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    all_export = []

    for method in METHODS:
        curves = [load_curve(args.results_dir, method["pattern"], seed, method["kind"]) for seed in [42, 43, 44]]
        stat = mean_std(curves)
        stat["method"] = method["name"]
        all_export.append(stat)

        x = stat["evolution"].to_numpy()
        y = stat["mean"].to_numpy()
        sd = stat["std"].to_numpy()
        ax.plot(x, y, lw=2.4, color=method["color"], label=method["name"])
        ax.fill_between(x, np.clip(y - sd, 0, None), y + sd, color=method["color"], alpha=0.13, lw=0)

    x_ref = np.arange(1, 101)
    ax.plot(
        x_ref,
        np.full_like(x_ref, direct_mean, dtype=float),
        lw=2.2,
        ls="--",
        color="#9a4fb0",
        label="LLM-Generated Molecules",
    )
    ax.fill_between(
        x_ref,
        np.full_like(x_ref, max(0.0, direct_mean - direct_std), dtype=float),
        np.full_like(x_ref, direct_mean + direct_std, dtype=float),
        color="#9a4fb0",
        alpha=0.12,
        lw=0,
    )

    ax.set_xlabel("Evolution step")
    ax.set_ylabel("RDKit-success molecule count")
    ax.set_title("Success Count Evolution on ZINC logP Transfer")
    ax.set_xlim(1, 100)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    png = args.out_dir / "zinc_logp_main5_success_evolution.png"
    pdf = args.out_dir / "zinc_logp_main5_success_evolution.pdf"
    csv = args.out_dir / "zinc_logp_main5_success_evolution_data.csv"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    export = pd.concat(all_export, ignore_index=True)
    direct_df = pd.DataFrame({
        "evolution": x_ref,
        "mean": direct_mean,
        "std": direct_std,
        "method": "LLM-Generated Molecules",
    })
    pd.concat([export, direct_df], ignore_index=True).to_csv(csv, index=False)

    print(f"saved {png}")
    print(f"saved {pdf}")
    print(f"saved {csv}")


if __name__ == "__main__":
    main()
