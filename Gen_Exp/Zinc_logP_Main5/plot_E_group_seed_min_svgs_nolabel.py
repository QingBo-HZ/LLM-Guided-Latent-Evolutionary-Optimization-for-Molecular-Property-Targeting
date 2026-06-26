#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export unlabeled per-method SVGs for Figure 4E.

Each SVG contains the seed-42/43/44 molecule with the minimum RDKit
|logP - 3.0| for that method. No external legends, method names, seed labels,
or metric text are drawn.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from plot_E_group_seed_min_svgs import METHODS, SEEDS, load_seed_best


OUT_DIR = Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/figures/E_group_seed_min_svgs_nolabel")
SUB_IMG_SIZE = 760


def prepare_mol(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    return mol


def draw_molecule_grid_svg(smiles_list, out_svg, mols_per_row=3, sub_img_size=(SUB_IMG_SIZE, SUB_IMG_SIZE)):
    mols = []
    for smi in smiles_list:
        mol = prepare_mol(smi)
        if mol is not None:
            mols.append(mol)
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
    opts.bondLineWidth = 2
    opts.maxFontSize = 34
    opts.minFontSize = 18
    opts.fixedBondLength = 34
    opts.padding = 0.12
    opts.centreMoleculesBeforeDrawing = True
    opts.clearBackground = False
    opts.additionalAtomLabelPadding = 0.15

    drawer.DrawMolecules(mols)
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

        out_svg = OUT_DIR / f"E_{slug}_seed_min_nolabel.svg"
        draw_molecule_grid_svg(selected["smiles"].tolist(), out_svg)

    all_df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "E_all_methods_seed_min_selected.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"[OK] All selected rows saved to: {out_csv}")


if __name__ == "__main__":
    main()
