#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw Figure 4E as a 5 methods x 3 seeds representative molecule grid."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from plot_E_molecule_path_png import atom_label, draw_atom_label, draw_bond, font, transform_points


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
MAIN_DIR = ROOT / "Gen_Exp" / "Zinc_logP_Main5"
RESULTS_DIR = MAIN_DIR / "results"
LLM_SMILES_DIR = ROOT / "Gen_Exp" / "Zinc_logP_LLM" / "smiles"
OUT_DIR = MAIN_DIR / "figures"

TARGET_LOGP = 3.0
SEEDS = [42, 43, 44]
CELL_W = 900
CELL_H = 660

try:
    from rdkit.Chem import rdCoordGen
except Exception:  # pragma: no cover - depends on RDKit build
    rdCoordGen = None


METHODS = [
    (
        "Random Latent\nSearch",
        "Random Latent Search",
        lambda seed: RESULTS_DIR / f"random_latent_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
    (
        "ZINC-Seeded\nLatent GA",
        "ZINC-Seeded Latent GA",
        lambda seed: RESULTS_DIR / f"train_random_zinc_seeded_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
    (
        "LLM-Generated\nMolecules",
        "LLM-Generated Molecules",
        lambda seed: next(LLM_SMILES_DIR.glob(f"zinc_logp_llm_direct_gpt-5.4-mini_seed{seed}_*_accepted_ranked.csv")),
    ),
    (
        "LLM-Initialized\nLatent GA",
        "LLM-Initialized Latent GA",
        lambda seed: RESULTS_DIR / f"llm_llm_initialized_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
    (
        "Iterative LLM-\nGuided Latent GA",
        "Iterative LLM-Guided Latent GA",
        lambda seed: RESULTS_DIR / f"llm_iterative_llm_guided_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv",
    ),
]


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def add_2d_coords(mol: Chem.Mol) -> Chem.Mol:
    if rdCoordGen is not None:
        rdCoordGen.AddCoords(mol)
    else:
        rdDepictor.Compute2DCoords(mol)
    return mol


def prepare_mol(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return add_2d_coords(mol)


def load_seed_best(display_method: str, method: str, seed: int, path: Path) -> pd.Series:
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
    row["display_method"] = display_method
    row["method"] = method
    row["seed"] = seed
    row["source_file"] = str(path)
    return row


def select_grid_rows() -> pd.DataFrame:
    rows = []
    for display_method, method, path_fn in METHODS:
        for seed in SEEDS:
            rows.append(load_seed_best(display_method, method, seed, path_fn(seed)))
    return pd.DataFrame(rows)


def draw_svg(selected: pd.DataFrame, out_svg: Path) -> None:
    mols = []
    legends = []
    for _, row in selected.iterrows():
        mol = prepare_mol(row["smiles"])
        if mol is None:
            raise ValueError(f"Invalid SMILES: {row['smiles']}")
        mols.append(mol)
        method = str(row["display_method"])
        legends.append(f"{method}\nSeed {int(row['seed'])}: logP={row['rdkit_logP']:.3f}, err={row['rdkit_abs_error']:.4f}")

    width = 3 * CELL_W
    height = 5 * CELL_H
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height, CELL_W, CELL_H)
    opts = drawer.drawOptions()
    opts.legendFontSize = 24
    opts.bondLineWidth = 2
    opts.maxFontSize = 34
    opts.minFontSize = 18
    opts.fixedBondLength = 34
    opts.padding = 0.14
    opts.centreMoleculesBeforeDrawing = True
    opts.clearBackground = False
    opts.additionalAtomLabelPadding = 0.18
    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()
    out_svg.write_text(drawer.GetDrawingText().replace("svg:", ""), encoding="utf-8")


def draw_plain_cell(row: pd.Series) -> Image.Image:
    mol = Chem.MolFromSmiles(str(row["smiles"]))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {row['smiles']}")
    mol = Chem.Mol(mol)
    add_2d_coords(mol)

    legend_h = 132
    img = Image.new("RGB", (CELL_W, CELL_H), "white")
    draw = ImageDraw.Draw(img)

    pts = transform_points(mol, CELL_W, CELL_H - legend_h, pad=78)
    shifted = [(x, y + 10) for x, y in pts]

    for bond in mol.GetBonds():
        draw_bond(draw, shifted[bond.GetBeginAtomIdx()], shifted[bond.GetEndAtomIdx()], bond)
    for atom in mol.GetAtoms():
        draw_atom_label(draw, shifted[atom.GetIdx()], atom_label(atom))

    method_lines = str(row["display_method"]).split("\n")
    metric_line = f"Seed {int(row['seed'])}: logP={row['rdkit_logP']:.3f}, err={row['rdkit_abs_error']:.4f}"
    legend_lines = method_lines + [metric_line]

    font_method = font(30, bold=True)
    font_metric = font(25, bold=False)
    y = CELL_H - legend_h + 18
    for i, line in enumerate(legend_lines):
        f = font_method if i < len(legend_lines) - 1 else font_metric
        bbox = draw.textbbox((0, 0), line, font=f)
        draw.text(((CELL_W - (bbox[2] - bbox[0])) / 2, y), line, font=f, fill=(30, 30, 30))
        y += (bbox[3] - bbox[1]) + 8

    return img


def draw_png(selected: pd.DataFrame, out_png: Path) -> None:
    cells = [draw_plain_cell(row) for _, row in selected.iterrows()]
    canvas = Image.new("RGB", (3 * CELL_W, 5 * CELL_H), "white")
    for i, cell in enumerate(cells):
        x = (i % 3) * CELL_W
        y = (i // 3) * CELL_H
        canvas.paste(cell, (x, y))

    draw = ImageDraw.Draw(canvas)
    for col in range(1, 3):
        draw.line((col * CELL_W, 0, col * CELL_W, canvas.height), fill=(230, 232, 236), width=2)
    for row in range(1, 5):
        draw.line((0, row * CELL_H, canvas.width, row * CELL_H), fill=(230, 232, 236), width=2)

    canvas.save(out_png, dpi=(300, 300))


def convert_svg_to_png(out_svg: Path, out_png: Path) -> bool:
    try:
        import cairosvg
    except Exception:
        return False

    cairosvg.svg2png(
        url=str(out_svg),
        write_to=str(out_png),
        output_width=2 * 3 * CELL_W,
        output_height=2 * 5 * CELL_H,
    )
    return True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_grid_rows()

    keep_cols = [
        "method",
        "seed",
        "smiles",
        "canonical_smiles",
        "rdkit_logP",
        "rdkit_abs_error",
        "source_file",
    ]
    out_csv = OUT_DIR / "E_5method_3seed_min_grid_data.csv"
    selected[keep_cols].to_csv(out_csv, index=False)

    out_svg = OUT_DIR / "E_5method_3seed_min_grid.svg"
    out_png = OUT_DIR / "E_5method_3seed_min_grid.png"
    draw_svg(selected, out_svg)
    if not convert_svg_to_png(out_svg, out_png):
        draw_png(selected, out_png)

    print(f"saved {out_svg}")
    print(f"saved {out_png}")
    print(f"saved {out_csv}")
    print(f"png_size_px={(3 * CELL_W, 5 * CELL_H)}")


if __name__ == "__main__":
    main()
