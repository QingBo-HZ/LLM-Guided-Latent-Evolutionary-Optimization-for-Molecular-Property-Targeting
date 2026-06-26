#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Revised C figure: count-based RDKit-success logP distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUCCESS_LOW = 2.5
SUCCESS_HIGH = 3.5
TARGET_LOGP = 3.0

METHOD_COLORS = {
    "Random Latent Search": "#6f6f6f",
    "ZINC-Seeded Latent GA": "#2f6fbb",
    "LLM-Generated Molecules": "#9a4fb0",
    "LLM-Initialized Latent GA": "#d8872b",
    "Iterative LLM-Guided Latent GA": "#2b9a66",
}

GA_DIRS = {
    "ZINC-Seeded Latent GA": "train_random_zinc_seeded_formal_seed{seed}",
    "LLM-Initialized Latent GA": "llm_llm_initialized_formal_seed{seed}",
    "Iterative LLM-Guided Latent GA": "llm_iterative_llm_guided_formal_seed{seed}",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 160,
    })


def load_method_logp(results_dir: Path, llm_smiles_dir: Path, method: str) -> pd.DataFrame:
    frames = []
    if method == "LLM-Generated Molecules":
        for seed in [42, 43, 44]:
            files = sorted(llm_smiles_dir.glob(f"zinc_logp_llm_direct_*_seed{seed}_*_accepted_ranked.csv"))
            df = pd.read_csv(files[-1])
            frames.append(pd.DataFrame({"rdkit_logP": df["rdkit_logP"], "seed": seed, "method": method}))
    elif method == "Random Latent Search":
        for seed in [42, 43, 44]:
            df = pd.read_csv(results_dir / f"random_latent_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv")
            frames.append(pd.DataFrame({"rdkit_logP": df["rdkit_logP"], "seed": seed, "method": method}))
    else:
        for seed in [42, 43, 44]:
            df = pd.read_csv(results_dir / GA_DIRS[method].format(seed=seed) / "decoded_molecule_unique_ranked.csv")
            frames.append(pd.DataFrame({"rdkit_logP": df["rdkit_logP"], "seed": seed, "method": method}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/results"))
    parser.add_argument("--llm_smiles_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM/smiles"))
    parser.add_argument("--out_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    methods = [
        "Random Latent Search",
        "ZINC-Seeded Latent GA",
        "LLM-Generated Molecules",
        "LLM-Initialized Latent GA",
        "Iterative LLM-Guided Latent GA",
    ]

    all_rows = []
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    bins = np.linspace(SUCCESS_LOW, SUCCESS_HIGH, 31)

    for method in methods:
        df = load_method_logp(args.results_dir, args.llm_smiles_dir, method)
        values = pd.to_numeric(df["rdkit_logP"], errors="coerce").dropna()
        success = values[(values >= SUCCESS_LOW) & (values <= SUCCESS_HIGH)]
        ax.hist(
            success,
            bins=bins,
            histtype="step",
            density=False,
            lw=2.8,
            color=METHOD_COLORS[method],
            label=f"{method} (success n={len(success)})",
        )
        tmp = pd.DataFrame({"rdkit_logP": success, "method": method})
        all_rows.append(tmp)

    ax.axvspan(SUCCESS_LOW, SUCCESS_HIGH, color="#9ccf8f", alpha=0.11, label="Success range")
    ax.axvline(TARGET_LOGP, color="#1f1f1f", lw=2.0, ls="--", label="Target logP")
    ax.set_xlim(SUCCESS_LOW, SUCCESS_HIGH)
    ax.set_xlabel("RDKit Crippen MolLogP within success range")
    ax.set_ylabel("RDKit-success molecule count")
    ax.set_title("RDKit-success Molecule Count Distribution")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    png = args.out_dir / "C_rdkit_success_count_distribution.png"
    pdf = args.out_dir / "C_rdkit_success_count_distribution.pdf"
    csv = args.out_dir / "C_rdkit_success_count_distribution_data.csv"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    pd.concat(all_rows, ignore_index=True).to_csv(csv, index=False)

    print(f"saved {png}")
    print(f"saved {pdf}")
    print(f"saved {csv}")


if __name__ == "__main__":
    main()
