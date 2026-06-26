#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path("/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA")
RESULT_ROOT = ROOT / "sweet_ga_results_0622_v8_hard_metrics"
DATA_DIR = RESULT_ROOT / "nature_style_panels" / "data"
AUDIT_DIR = RESULT_ROOT / "docking_audit"
OUT_DIR = RESULT_ROOT / "nature_style_panels" / "main_4panels"
MOL_DIR = OUT_DIR / "top5_molecule_svgs"

METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "Random-Seeded Latent GA",
    "B": "SweetDB-Seeded Latent GA",
    "C": "LLM-Initialized Latent GA",
    "D": "Iterative LLM-Guided Latent GA",
}
SHORT_LABELS = {
    "A": "Random",
    "B": "SweetDB seed",
    "C": "LLM init",
    "D": "LLM iterative",
}
COLORS = {
    "A": "#6B7280",
    "B": "#3B73B9",
    "C": "#E9783A",
    "D": "#2C9B63",
    "purple": "#9B59B6",
    "orange": "#E9783A",
    "ink": "#1E2430",
    "muted": "#667085",
    "grid": "#E7EAF1",
    "axis": "#D3D8E5",
}


def setup_style():
    if sns is not None:
        sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "Segoe UI", "sans-serif"],
            "axes.unicode_minus": True,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 21,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 14.5,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        }
    )


def polish(ax):
    ax.grid(axis="y", color=COLORS["grid"], lw=1.0, alpha=0.92)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.75, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])


def plot_llm_feedback_mechanism():
    summary = pd.read_csv(DATA_DIR / "v8_d_group_injection_summary.csv")
    gain = pd.read_csv(DATA_DIR / "v8_d_vs_c_top5_logsw_gain_summary.csv")
    summary["generation"] = pd.to_numeric(summary["generation"], errors="coerce")
    summary["generated_basic_gate_count"] = pd.to_numeric(summary["generated_basic_gate_count"], errors="coerce")
    summary["strict_bpe_injected_count"] = pd.to_numeric(summary["strict_bpe_injected_count"], errors="coerce")
    gain["generation"] = pd.to_numeric(gain["generation"], errors="coerce")
    gain["mean_gain"] = pd.to_numeric(gain["mean_gain"], errors="coerce")
    gain["sd_gain"] = pd.to_numeric(gain["sd_gain"], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    width = 0.28
    x = summary["generation"]
    ax.bar(
        x - width / 2,
        summary["generated_basic_gate_count"],
        width=width,
        color=COLORS["purple"],
        alpha=0.28,
        edgecolor=COLORS["purple"],
        linewidth=1.5,
        label="LLM candidates",
    )
    ax.bar(
        x + width / 2,
        summary["strict_bpe_injected_count"],
        width=width,
        color=COLORS["purple"],
        alpha=0.82,
        edgecolor=COLORS["purple"],
        linewidth=1.5,
        label="Injected seeds",
    )
    ax.set_xlim(1, 12.25)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Generation")
    ax.set_ylabel("LLM feedback molecules")
    ax.set_ylim(0, max(16, float(summary["generated_basic_gate_count"].max()) + 3))
    polish(ax)

    ax2 = ax.twinx()
    gx = gain["generation"]
    gy = gain["mean_gain"]
    gsd = gain["sd_gain"]
    ax2.axhline(0, color=COLORS["ink"], lw=1.2, ls="--", alpha=0.45)
    ax2.plot(gx, gy, color=COLORS["D"], lw=3.0, marker="D", markersize=7.0, markerfacecolor="white", markeredgewidth=1.8, label="D-C logSw gain")
    ax2.fill_between(gx, gy - gsd, gy + gsd, color=COLORS["D"], alpha=0.12, linewidth=0)
    ax2.set_ylabel("D minus C top-5 logSw")
    ax2.tick_params(axis="y", colors=COLORS["muted"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["axis"])

    for gen in [3, 6, 9]:
        ax.axvline(gen, color=COLORS["purple"], lw=1.45, ls=(0, (4, 4)), alpha=0.55, zorder=0)

    ax.set_title("LLM feedback intervention", loc="left", fontweight="bold", color=COLORS["ink"])
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "main_v8_llm_feedback_intervention_effect.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def select_top5():
    df = pd.read_csv(AUDIT_DIR / "final_docking_candidate_pool.csv")
    numeric_cols = ["final_score", "pred_logsw_reencoded", "p_sweet_reencoded", "d_ood_reencoded", "rank"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pre_docking_goldlike"] = df["pre_docking_goldlike"].astype(bool)
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].copy()
        sub = sub.sort_values(
            ["pre_docking_goldlike", "final_score", "pred_logsw_reencoded", "p_sweet_reencoded"],
            ascending=[False, False, False, False],
        )
        sub = sub.drop_duplicates("smiles").head(5).copy()
        sub["method_short_label"] = SHORT_LABELS[method]
        sub["molecule_panel_rank"] = range(1, len(sub) + 1)
        rows.append(sub)
    top = pd.concat(rows, ignore_index=True)
    MOL_DIR.mkdir(parents=True, exist_ok=True)
    top.to_csv(MOL_DIR / "v8_abcd_top5_molecules_for_svg.csv", index=False)
    return top


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.rdDepictor.Compute2DCoords(mol)
    return mol


def write_group_svgs(top: pd.DataFrame):
    for method in METHOD_ORDER:
        sub = top[top["method"] == method].sort_values("molecule_panel_rank")
        mols = [mol_from_smiles(s) for s in sub["smiles"]]
        legends = [
            f"{method}{int(row.molecule_panel_rank)} | logSw {float(row.pred_logsw_reencoded):.2f}\nP {float(row.p_sweet_reencoded):.2f}"
            for row in sub.itertuples()
        ]
        svg = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(250, 220),
            legends=legends,
            useSVG=True,
        )
        (MOL_DIR / f"v8_{method}_{SHORT_LABELS[method].replace(' ', '_').lower()}_top5_molecules.svg").write_text(svg, encoding="utf-8")
        png = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(250, 220),
            legends=legends,
            useSVG=False,
        )
        png.save(MOL_DIR / f"v8_{method}_{SHORT_LABELS[method].replace(' ', '_').lower()}_top5_molecules_preview.png")

    # Optional combined SVG for quick browsing.
    mols = []
    legends = []
    for row in top.sort_values(["method", "molecule_panel_rank"]).itertuples():
        mols.append(mol_from_smiles(row.smiles))
        legends.append(f"{row.method}{int(row.molecule_panel_rank)} | logSw {float(row.pred_logsw_reencoded):.2f}")
    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(230, 210),
        legends=legends,
        useSVG=True,
    )
    (MOL_DIR / "v8_ABCD_top5_molecules_combined.svg").write_text(svg, encoding="utf-8")
    png = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(230, 210),
        legends=legends,
        useSVG=False,
    )
    png.save(MOL_DIR / "v8_ABCD_top5_molecules_combined_preview.png")


def write_readme():
    (OUT_DIR / "main_4panels_readme.md").write_text(
        """# v8 main four-panel figure set

Recommended four chart panels:

1. `main_v8_cumulative_hard_pass_evolution_llm_marked.png`
2. `main_v8_top5_logsw_evolution_llm_marked.png` as supplementary/mechanistic potency curve, not the sole success metric
3. `main_v8_final_logsw_vs_predicted_vina_surrogate.png` until real Vina scores are backfilled
4. `main_v8_llm_feedback_intervention_effect.png`

Top-5 molecule SVGs are in `top5_molecule_svgs/`.
Molecules are selected from the v8 final external pool by hard-pass first, then final_score, then predicted logSw, with duplicate SMILES removed within each method.
""",
        encoding="utf-8",
    )


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_llm_feedback_mechanism()
    top = select_top5()
    write_group_svgs(top)
    write_readme()
    print(f"Wrote fourth panel and molecule SVGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
