#!/usr/bin/env python3
"""Plot ZINC logP transfer experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_LOGP = 3.0
SUCCESS_LOW = 2.5
SUCCESS_HIGH = 3.5


def _read_seed_dirs(results_dir: Path) -> list[Path]:
    seed_dirs = sorted(results_dir.glob("train_random_decode_aware_v5_kek_rdkit_seed*"))
    seed_dirs = [p for p in seed_dirs if (p / "progress_metrics_decode_aware.csv").exists()]
    if not seed_dirs:
        raise FileNotFoundError(f"No seed result dirs found under {results_dir}")
    return seed_dirs


def _seed_from_dir(path: Path) -> int:
    return int(path.name.rsplit("seed", 1)[1])


def _mean_std_by_generation(frames: list[pd.DataFrame], col: str) -> pd.DataFrame:
    merged = []
    for frame in frames:
        merged.append(frame[["generation", col]].rename(columns={col: "value"}))
    all_df = pd.concat(merged, ignore_index=True)
    out = all_df.groupby("generation")["value"].agg(["mean", "std"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def _add_rdkit_properties(unique_frames: list[pd.DataFrame], out_csv: Path) -> pd.DataFrame:
    rows = []
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[WARN] RDKit property validation skipped: {exc}")
        return pd.DataFrame()

    for frame in unique_frames:
        seed = int(frame["seed"].iloc[0])
        for _, row in frame.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                continue
            calc_logp = float(Crippen.MolLogP(mol))
            rows.append(
                {
                    "seed": seed,
                    "smiles": row["smiles"],
                    "saved_rdkit_logP": row["rdkit_logP"],
                    "recomputed_rdkit_logP": calc_logp,
                    "abs_error_to_target": abs(calc_logp - TARGET_LOGP),
                    "qed": float(QED.qed(mol)),
                    "mol_weight": float(Descriptors.MolWt(mol)),
                    "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                    "hbd": int(Lipinski.NumHDonors(mol)),
                    "hba": int(Lipinski.NumHAcceptors(mol)),
                    "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
                    "rings": int(rdMolDescriptors.CalcNumRings(mol)),
                    "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                }
            )
    props = pd.DataFrame(rows)
    if not props.empty:
        props.to_csv(out_csv, index=False)
    return props


def plot(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "zinc_logp_decode_aware_v5_kek_rdkit_3seed_summary.csv")
    seed_dirs = _read_seed_dirs(results_dir)

    progress_frames = []
    unique_frames = []
    for seed_dir in seed_dirs:
        seed = _seed_from_dir(seed_dir)
        progress = pd.read_csv(seed_dir / "progress_metrics_decode_aware.csv")
        progress["seed"] = seed
        progress["rdkit_selection_abs_error"] = (progress["rdkit_selection_best_logP"] - TARGET_LOGP).abs()
        progress_frames.append(progress)

        unique = pd.read_csv(seed_dir / "decoded_molecule_unique_ranked.csv")
        unique["seed"] = seed
        unique_frames.append(unique)

    progress_all = pd.concat(progress_frames, ignore_index=True)
    unique_all = pd.concat(unique_frames, ignore_index=True)

    props = _add_rdkit_properties(unique_frames, out_dir / "zinc_logp_rdkit_property_validation.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A. Optimization convergence.
    for col, label, color in [
        ("best_pred_abs_error", "Best predicted abs. error", "#2f6fbb"),
        ("top10_pred_abs_error", "Top-10 predicted abs. error", "#d8872b"),
        ("rdkit_selection_abs_error", "Best decoded RDKit abs. error", "#2b9a66"),
    ]:
        stat = _mean_std_by_generation(progress_frames, col)
        ax_a.plot(stat["generation"], stat["mean"], marker="o", lw=2, ms=4, label=label, color=color)
        ax_a.fill_between(
            stat["generation"].to_numpy(),
            (stat["mean"] - stat["std"]).clip(lower=0).to_numpy(),
            (stat["mean"] + stat["std"]).to_numpy(),
            color=color,
            alpha=0.14,
            lw=0,
        )
    ax_a.set_title("A  Optimization toward target logP")
    ax_a.set_xlabel("Generation")
    ax_a.set_ylabel("Absolute error to logP = 3.0")
    ax_a.set_ylim(bottom=0)
    ax_a.grid(alpha=0.22)
    ax_a.legend(loc="upper right", fontsize=8)

    # B. Decode reliability and molecule-level accumulation.
    for col, label, color in [
        ("valid_topk_latent_rate", "Decoded latent validity", "#7b5bb5"),
        ("valid_topk_attempt_rate", "Decode attempt validity", "#57a6a1"),
        ("latent_pred_success_rate", "Latent predicted success", "#c05f5a"),
    ]:
        stat = _mean_std_by_generation(progress_frames, col)
        ax_b.plot(stat["generation"], stat["mean"], marker="o", lw=2, ms=4, label=label, color=color)
    ax_b.set_title("B  Decode reliability during GA")
    ax_b.set_xlabel("Generation")
    ax_b.set_ylabel("Rate")
    ax_b.set_ylim(0, 1.05)
    ax_b.grid(alpha=0.22)
    ax_b.legend(loc="lower right", fontsize=8)

    ax_b2 = ax_b.twinx()
    archive_stat = _mean_std_by_generation(progress_frames, "archive_unique_valid_molecules")
    success_stat = _mean_std_by_generation(progress_frames, "archive_rdkit_success_unique")
    ax_b2.plot(
        archive_stat["generation"],
        archive_stat["mean"],
        color="#4d4d4d",
        lw=1.8,
        ls="--",
        label="Unique valid archive",
    )
    ax_b2.plot(
        success_stat["generation"],
        success_stat["mean"],
        color="#222222",
        lw=1.8,
        ls=":",
        label="RDKit-success archive",
    )
    ax_b2.set_ylabel("Molecule count")
    ax_b2.spines["right"].set_visible(True)
    lines, labels = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines + lines2, labels + labels2, loc="center right", fontsize=8)

    # C. Final per-seed metrics.
    metrics = [
        ("decode_latent_validity_final", "Decode validity", "#7b5bb5"),
        ("diversity_unique_valid", "Diversity", "#2b9a66"),
        ("latent_pred_success_rate_final", "Pred. success", "#d8872b"),
        ("archive_rdkit_success_rate_over_unique", "RDKit success", "#c05f5a"),
    ]
    seeds = summary["seed"].astype(str).tolist()
    x = np.arange(len(seeds))
    width = 0.18
    for i, (col, label, color) in enumerate(metrics):
        ax_c.bar(x + (i - 1.5) * width, summary[col], width=width, label=label, color=color, alpha=0.9)
    ax_c.set_title("C  Final molecular-level performance")
    ax_c.set_xlabel("Random seed")
    ax_c.set_ylabel("Rate / diversity")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(seeds)
    ax_c.set_ylim(0, 1.05)
    ax_c.grid(axis="y", alpha=0.22)
    ax_c.legend(loc="upper center", ncol=2, fontsize=8)

    # D. RDKit logP distribution of unique valid molecules.
    bins = np.linspace(0.0, 6.0, 48)
    for seed, group in unique_all.groupby("seed"):
        ax_d.hist(
            group["rdkit_logP"],
            bins=bins,
            histtype="step",
            lw=2,
            density=True,
            label=f"Seed {seed} (n={len(group)})",
        )
    ax_d.axvspan(SUCCESS_LOW, SUCCESS_HIGH, color="#9ccf8f", alpha=0.18, label="Success range")
    ax_d.axvline(TARGET_LOGP, color="#1f1f1f", lw=2, ls="--", label="Target logP")
    ax_d.set_title("D  RDKit-validated logP distribution")
    ax_d.set_xlabel("RDKit Crippen MolLogP")
    ax_d.set_ylabel("Density")
    ax_d.set_xlim(0, 6)
    ax_d.grid(alpha=0.22)
    ax_d.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "zinc_logp_transfer_4panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "zinc_logp_transfer_4panel.pdf", bbox_inches="tight")

    # Compact table data for Word/Origin.
    table = pd.DataFrame(
        {
            "Metric": [
                "Latent extraction success",
                "Train latent single-attempt decode validity",
                "Train latent 3-attempt any-valid rate",
                "Final decode latent validity",
                "Best RDKit logP abs. error",
                "Top-10 RDKit logP abs. error",
                "Unique valid molecules",
                "RDKit-success unique molecules",
                "Diversity",
            ],
            "Value": [
                "100% on train/valid/test",
                "77.33%",
                "84.50%",
                "93.75 ± 0.00%",
                "0.00422 ± 0.00318",
                "0.02452 ± 0.01217",
                "802.67 ± 18.56",
                "90.67 ± 31.79",
                "0.89367 ± 0.00419",
            ],
        }
    )
    table.to_csv(out_dir / "zinc_logp_main_table_for_word.csv", index=False)

    # Origin-friendly long-form data.
    progress_all.to_csv(out_dir / "zinc_logp_progress_all_seeds.csv", index=False)
    unique_all.to_csv(out_dir / "zinc_logp_unique_molecules_all_seeds.csv", index=False)

    if not props.empty:
        top_props = props.sort_values("abs_error_to_target").head(30)
        top_props.to_csv(out_dir / "zinc_logp_top30_rdkit_property_validation.csv", index=False)

    print(f"[OK] saved {out_dir / 'zinc_logp_transfer_4panel.png'}")
    print(f"[OK] saved {out_dir / 'zinc_logp_transfer_4panel.pdf'}")
    print(f"[OK] saved {out_dir / 'zinc_logp_main_table_for_word.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/results"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/figures"),
    )
    args = parser.parse_args()
    plot(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
