#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw representative 2D molecules for all five ZINC logP transfer methods."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem

from plot_E_molecule_path_png import draw_molecule, font


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
MAIN_DIR = ROOT / "Gen_Exp" / "Zinc_logP_Main5"
RESULTS_DIR = MAIN_DIR / "results"
LLM_SMILES_DIR = ROOT / "Gen_Exp" / "Zinc_logP_LLM" / "smiles"
OUT_DIR = MAIN_DIR / "figures"

TARGET_LOGP = 3.0
N_PER_METHOD = 3

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


def load_method_rows(display_method: str, table_method: str, paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in sorted(paths):
        df = pd.read_csv(path)
        if "smiles" not in df.columns:
            continue
        if "rdkit_logP" not in df.columns:
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
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["canonical_smiles"] = out["smiles"].map(canonical_smiles)
    out = out.dropna(subset=["canonical_smiles"])
    out = out.sort_values(["rdkit_abs_error", "rdkit_logP"], ascending=[True, True])
    out = out.drop_duplicates("canonical_smiles", keep="first")
    return out.head(N_PER_METHOD).copy()


def make_cell(smiles: str, logp: float, err: float, seed: int | None) -> Image.Image:
    seed_text = f"seed {seed}" if seed is not None else "LLM"
    legend = f"{seed_text} | logP={logp:.3f} | err={err:.4f}"
    return draw_molecule(smiles, legend, box_w=700, box_h=390)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_frames = []
    for display_method, table_method, paths in METHOD_SOURCES:
        selected = load_method_rows(display_method, table_method, paths)
        if len(selected) < N_PER_METHOD:
            raise RuntimeError(f"Not enough molecules for {table_method}: {len(selected)}")
        selected_frames.append(selected)

    selected_df = pd.concat(selected_frames, ignore_index=True)
    data_out = OUT_DIR / "E_representative_molecules_by_method_data.csv"
    keep_cols = [
        "method",
        "seed",
        "smiles",
        "canonical_smiles",
        "rdkit_logP",
        "rdkit_abs_error",
        "source_file",
    ]
    selected_df[keep_cols].to_csv(data_out, index=False)

    cell_w, cell_h = 700, 390
    label_w = 430
    title_h = 108
    rows = len(METHOD_SOURCES)
    cols = N_PER_METHOD
    canvas = Image.new("RGB", (label_w + cols * cell_w, title_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = font(48, bold=True)
    font_label = font(34, bold=True)
    font_small = font(24, bold=False)
    title = "Representative Top-Ranked Molecules from Five Search Strategies"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, 22), title, font=font_title, fill=(18, 18, 18))
    subtitle = "Target RDKit logP = 3.0; each cell reports RDKit logP and absolute error"
    sb = draw.textbbox((0, 0), subtitle, font=font_small)
    draw.text(((canvas.width - (sb[2] - sb[0])) / 2, 76), subtitle, font=font_small, fill=(88, 88, 88))

    for row_idx, (display_method, table_method, _paths) in enumerate(METHOD_SOURCES):
        y0 = title_h + row_idx * cell_h
        bg = (248, 250, 252) if row_idx % 2 == 0 else (255, 255, 255)
        draw.rectangle((0, y0, canvas.width, y0 + cell_h), fill=bg)
        draw.line((0, y0, canvas.width, y0), fill=(220, 224, 230), width=2)

        lines = display_method.split("\n")
        total_h = sum(draw.textbbox((0, 0), line, font=font_label)[3] for line in lines) + (len(lines) - 1) * 8
        ty = y0 + (cell_h - total_h) / 2
        for line in lines:
            lb = draw.textbbox((0, 0), line, font=font_label)
            draw.text(((label_w - (lb[2] - lb[0])) / 2, ty), line, font=font_label, fill=(25, 25, 25))
            ty += (lb[3] - lb[1]) + 8

        method_df = selected_df[selected_df["method"] == table_method].reset_index(drop=True)
        for col_idx, mol_row in method_df.iterrows():
            cell = make_cell(
                mol_row["smiles"],
                float(mol_row["rdkit_logP"]),
                float(mol_row["rdkit_abs_error"]),
                None if pd.isna(mol_row["seed"]) else int(mol_row["seed"]),
            )
            x = label_w + col_idx * cell_w
            canvas.paste(cell, (x, y0))

    draw.line((0, title_h + rows * cell_h - 1, canvas.width, title_h + rows * cell_h - 1), fill=(220, 224, 230), width=2)

    out = OUT_DIR / "E_representative_molecules_by_method.png"
    canvas.save(out, dpi=(300, 300))
    print(f"saved {out}")
    print(f"saved {data_out}")
    print(f"size_px={canvas.size}")


if __name__ == "__main__":
    main()
