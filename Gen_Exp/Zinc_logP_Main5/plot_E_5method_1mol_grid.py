#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw one representative molecule per method for ZINC logP transfer Figure 4E.

The SVG uses the same RDKit MolDraw2DSVG grid style as the QM9 main-result
script. A high-resolution PNG is also generated directly with PIL because this
environment does not provide RDKit Cairo or an external SVG-to-PNG converter.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from plot_E_molecule_path_png import draw_atom_label, draw_bond, font, transform_points


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
MAIN_DIR = ROOT / "Gen_Exp" / "Zinc_logP_Main5"
RESULTS_DIR = MAIN_DIR / "results"
LLM_SMILES_DIR = ROOT / "Gen_Exp" / "Zinc_logP_LLM" / "smiles"
OUT_DIR = MAIN_DIR / "figures"

TARGET_LOGP = 3.0
SUB_IMG_SIZE = 430

METHOD_SOURCES = [
    (
        "Random Latent\nSearch",
        "Random Latent Search",
        list(RESULTS_DIR.glob("random_latent_formal_seed*/decoded_molecule_unique_ranked.csv")),
    ),
    (
        "ZINC-Seeded\nLatent GA",
        "ZINC-Seeded Latent GA",
        list(RESULTS_DIR.glob("train_random_zinc_seeded_formal_seed*/decoded_molecule_unique_ranked.csv")),
    ),
    (
        "LLM-Generated\nMolecules",
        "LLM-Generated Molecules",
        list(LLM_SMILES_DIR.glob("zinc_logp_llm_direct_gpt-5.4-mini_seed*_accepted_ranked.csv")),
    ),
    (
        "LLM-Initialized\nLatent GA",
        "LLM-Initialized Latent GA",
        list(RESULTS_DIR.glob("llm_llm_initialized_formal_seed*/decoded_molecule_unique_ranked.csv")),
    ),
    (
        "Iterative LLM-\nGuided Latent GA",
        "Iterative LLM-Guided Latent GA",
        list(RESULTS_DIR.glob("llm_iterative_llm_guided_formal_seed*/decoded_molecule_unique_ranked.csv")),
    ),
]


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


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


def load_best_row(display_method: str, table_method: str, paths: list[Path]) -> pd.Series:
    frames = []
    for path in sorted(paths):
        df = pd.read_csv(path)
        if "smiles" not in df.columns or "rdkit_logP" not in df.columns:
            continue
        if "rdkit_abs_error" not in df.columns:
            df["rdkit_abs_error"] = (df["rdkit_logP"] - TARGET_LOGP).abs()
        if "rdkit_success" in df.columns:
            df = df[df["rdkit_success"].astype(bool)].copy()
        else:
            df = df[df["rdkit_abs_error"] <= 0.5].copy()
        if df.empty:
            continue
        df["display_method"] = display_method
        df["method"] = table_method
        df["source_file"] = str(path)
        df["seed"] = seed_from_path(path)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No valid molecules found for {table_method}")

    out = pd.concat(frames, ignore_index=True)
    out["canonical_smiles"] = out["smiles"].map(canonical_smiles)
    out = out.dropna(subset=["canonical_smiles"])
    out = out.sort_values(["rdkit_abs_error", "rdkit_logP"], ascending=[True, True])
    out = out.drop_duplicates("canonical_smiles", keep="first")
    return out.iloc[0]


def draw_svg(selected: pd.DataFrame, out_svg: Path) -> None:
    mols = []
    legends = []
    for _, row in selected.iterrows():
        mol = prepare_mol(row["smiles"])
        if mol is None:
            continue
        mols.append(mol)
        method_line = str(row["display_method"])
        legends.append(f"{method_line}\nlogP={row['rdkit_logP']:.3f}, err={row['rdkit_abs_error']:.4f}")

    width = SUB_IMG_SIZE * len(mols)
    height = SUB_IMG_SIZE
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height, SUB_IMG_SIZE, SUB_IMG_SIZE)
    opts = drawer.drawOptions()
    opts.legendFontSize = 18
    opts.bondLineWidth = 2
    opts.maxFontSize = 28
    opts.minFontSize = 16
    opts.fixedBondLength = 30
    opts.padding = 0.08
    opts.centreMoleculesBeforeDrawing = True
    opts.clearBackground = False
    opts.additionalAtomLabelPadding = 0.15
    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()
    out_svg.write_text(drawer.GetDrawingText().replace("svg:", ""), encoding="utf-8")


def draw_plain_molecule(smiles: str, box_size: int, legend_lines: list[str]) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol)

    legend_h = 94
    img = Image.new("RGB", (box_size, box_size), "white")
    draw = ImageDraw.Draw(img)
    pts = transform_points(mol, box_size, box_size - legend_h, pad=42)
    shifted = [(x, y + 10) for x, y in pts]

    for bond in mol.GetBonds():
        draw_bond(draw, shifted[bond.GetBeginAtomIdx()], shifted[bond.GetEndAtomIdx()], bond)
    for atom in mol.GetAtoms():
        draw_atom_label(draw, shifted[atom.GetIdx()], "")
        draw_atom_label(draw, shifted[atom.GetIdx()], atom.GetSymbol() if atom.GetSymbol() != "C" else "")

    font_method = font(21, bold=True)
    font_metric = font(18, bold=False)
    y = box_size - legend_h + 10
    for i, line in enumerate(legend_lines):
        f = font_method if i < len(legend_lines) - 1 else font_metric
        bbox = draw.textbbox((0, 0), line, font=f)
        draw.text(((box_size - (bbox[2] - bbox[0])) / 2, y), line, font=f, fill=(30, 30, 30))
        y += (bbox[3] - bbox[1]) + 5
    return img


def draw_png(selected: pd.DataFrame, out_png: Path) -> None:
    cells = []
    for _, row in selected.iterrows():
        method_lines = str(row["display_method"]).split("\n")
        metric_line = f"logP={row['rdkit_logP']:.3f}, err={row['rdkit_abs_error']:.4f}"
        cells.append(draw_plain_molecule(row["smiles"], SUB_IMG_SIZE, method_lines + [metric_line]))

    canvas = Image.new("RGB", (SUB_IMG_SIZE * len(cells), SUB_IMG_SIZE), "white")
    for i, cell in enumerate(cells):
        canvas.paste(cell, (i * SUB_IMG_SIZE, 0))
    canvas.save(out_png, dpi=(300, 300))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    for display_method, table_method, paths in METHOD_SOURCES:
        row = load_best_row(display_method, table_method, paths).copy()
        row["display_method"] = display_method
        row["method"] = table_method
        selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    out_csv = OUT_DIR / "E_5method_1mol_grid_data.csv"
    selected[
        [
            "method",
            "seed",
            "smiles",
            "canonical_smiles",
            "rdkit_logP",
            "rdkit_abs_error",
            "source_file",
        ]
    ].to_csv(out_csv, index=False)

    out_svg = OUT_DIR / "E_5method_1mol_grid.svg"
    out_png = OUT_DIR / "E_5method_1mol_grid.png"
    draw_svg(selected, out_svg)
    draw_png(selected, out_png)

    print(f"saved {out_svg}")
    print(f"saved {out_png}")
    print(f"saved {out_csv}")
    print(f"png_size_px={(SUB_IMG_SIZE * len(selected), SUB_IMG_SIZE)}")


if __name__ == "__main__":
    main()
