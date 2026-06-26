#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export one SVG per method for Figure 4E.

Each SVG contains the seed-42/43/44 molecule with the minimum RDKit
|logP - 3.0| for that method. The drawing code intentionally follows
Main_results_202604_LLM_GA/draw_smiles_grid_svg.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
MAIN_DIR = ROOT / "Gen_Exp" / "Zinc_logP_Main5"
RESULTS_DIR = MAIN_DIR / "results"
LLM_SMILES_DIR = ROOT / "Gen_Exp" / "Zinc_logP_LLM" / "smiles"
OUT_DIR = MAIN_DIR / "figures" / "E_group_seed_min_svgs"

TARGET_LOGP = 3.0
SEEDS = [42, 43, 44]
SUB_IMG_SIZE = 760


METHODS = [
    (
        "Random Latent Search",
        "random_latent_search",
        lambda seed: RESULTS_DIR / f"random_latent_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
    (
        "ZINC-Seeded Latent GA",
        "zinc_seeded_latent_ga",
        lambda seed: RESULTS_DIR / f"train_random_zinc_seeded_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
    (
        "LLM-Generated Molecules",
        "llm_generated_molecules",
        lambda seed: next(LLM_SMILES_DIR.glob(f"zinc_logp_llm_direct_gpt-5.4-mini_seed{seed}_*_accepted_ranked.csv")),
    ),
    (
        "LLM-Initialized Latent GA",
        "llm_initialized_latent_ga",
        lambda seed: RESULTS_DIR / f"llm_llm_initialized_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
    (
        "Iterative LLM-Guided Latent GA",
        "iterative_llm_guided_latent_ga",
        lambda seed: RESULTS_DIR / f"llm_iterative_llm_guided_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
]


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def prepare_mol(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    return mol


def load_seed_best(method: str, slug: str, seed: int, path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "smiles" not in df.columns or "rdkit_logP" not in df.columns:
        raise ValueError(f"{path} must contain smiles and rdkit_logP columns")
    if "rdkit_abs_error" not in df.columns:
        df["rdkit_abs_error"] = (df["rdkit_logP"] - TARGET_LOGP).abs()
    if "rdkit_success" in df.columns:
        df = df[df["rdkit_success"].astype(bool)].copy()
    else:
        df = df[df["rdkit_abs_error"] <= 0.5].copy()

    df = df[df["smiles"].notna()].copy()
    df["canonical_smiles"] = df["smiles"].map(canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    if df.empty:
        raise RuntimeError(f"No valid success molecules for {method}, seed={seed}")

    df = df.sort_values(["rdkit_abs_error", "rdkit_logP"], ascending=[True, True])
    row = df.iloc[0].copy()
    row["method"] = method
    row["method_slug"] = slug
    row["seed"] = seed
    row["source_file"] = str(path)
    return row


def draw_molecule_grid_svg(smiles_list, legends, out_svg, mols_per_row=3, sub_img_size=(SUB_IMG_SIZE, SUB_IMG_SIZE)):
    mols = []
    valid_legends = []

    for smi, legend in zip(smiles_list, legends):
        mol = prepare_mol(smi)
        if mol is not None:
            mols.append(mol)
            valid_legends.append(legend)
        else:
            print(f"[WARN] invalid SMILES skipped: {smi}")

    if len(mols) == 0:
        raise ValueError("No valid molecules to draw.")

    n_mols = len(mols)
    rows = (n_mols + mols_per_row - 1) // mols_per_row
    width = mols_per_row * sub_img_size[0]
    height = rows * sub_img_size[1]

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height, sub_img_size[0], sub_img_size[1])

    opts = drawer.drawOptions()
    opts.legendFontSize = 30
    opts.bondLineWidth = 2
    opts.maxFontSize = 34
    opts.minFontSize = 18
    opts.fixedBondLength = 34
    opts.padding = 0.12
    opts.centreMoleculesBeforeDrawing = True
    opts.clearBackground = False
    opts.additionalAtomLabelPadding = 0.15

    drawer.DrawMolecules(mols, legends=valid_legends)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText().replace("svg:", "")
    Path(out_svg).write_text(svg, encoding="utf-8")
    print(f"[OK] SVG saved to: {out_svg}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for method, slug, path_fn in METHODS:
        rows = []
        for seed in SEEDS:
            row = load_seed_best(method, slug, seed, path_fn(seed))
            rows.append(row)
            all_rows.append(row)

        selected = pd.DataFrame(rows)
        selected_csv = OUT_DIR / f"E_{slug}_seed_min_selected.csv"
        selected.to_csv(selected_csv, index=False)

        smiles_list = selected["smiles"].tolist()
        legends = [
            f"Seed {int(row.seed)}\nlogP = {float(row.rdkit_logP):.3f}\nerr = {float(row.rdkit_abs_error):.4f}"
            for row in selected.itertuples(index=False)
        ]
        out_svg = OUT_DIR / f"E_{slug}_seed_min.svg"
        draw_molecule_grid_svg(smiles_list, legends, out_svg)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(OUT_DIR / "E_all_methods_seed_min_selected.csv", index=False)
    print(f"[OK] All selected rows saved to: {OUT_DIR / 'E_all_methods_seed_min_selected.csv'}")


if __name__ == "__main__":
    main()
