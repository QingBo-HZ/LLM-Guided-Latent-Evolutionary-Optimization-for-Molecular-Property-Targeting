#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def prepare_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    return mol


def pick_evenly_spaced_indices(n_rows: int, n_pick: int = 5):
    if n_rows <= n_pick:
        return list(range(n_rows))
    idx = np.linspace(0, n_rows - 1, n_pick, dtype=int)
    return list(dict.fromkeys(idx.tolist()))


def load_representative_rows(csv_path: str, mode: str = "best_path", n_pick: int = 5):
    df = pd.read_csv(csv_path)

    if mode == "best_path":
        required = ["smiles", "gap"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"{csv_path} 缺少列: {col}")

        # 去掉无效 smiles
        df = df[df["smiles"].notna()].copy()
        df = df[df["smiles"].astype(str).str.len() > 0].copy()

        # 如果有 evaluations 就按 evaluations 排，没有就按 step/generation 排
        if "evaluations" in df.columns:
            df = df.sort_values("evaluations").reset_index(drop=True)
        elif "step" in df.columns:
            df = df.sort_values("step").reset_index(drop=True)
        elif "generation" in df.columns:
            df = df.sort_values("generation").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        picked_idx = pick_evenly_spaced_indices(len(df), n_pick=n_pick)
        picked = df.iloc[picked_idx].copy()
        return picked

    elif mode == "topk":
        required = ["smiles", "gap"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"{csv_path} 缺少列: {col}")

        df = df[df["smiles"].notna()].copy()
        df = df[df["smiles"].astype(str).str.len() > 0].copy()

        # 每个 step/generation 取 rank=1 或最小 gap
        if "rank" in df.columns:
            df = df.sort_values(["rank", "gap"]).copy()
            if "step" in df.columns:
                df = df.groupby("step", as_index=False).first()
                df = df.sort_values("step").reset_index(drop=True)
            elif "generation" in df.columns:
                df = df.groupby("generation", as_index=False).first()
                df = df.sort_values("generation").reset_index(drop=True)
        else:
            if "step" in df.columns:
                df = df.sort_values(["step", "gap"]).groupby("step", as_index=False).first()
                df = df.sort_values("step").reset_index(drop=True)
            elif "generation" in df.columns:
                df = df.sort_values(["generation", "gap"]).groupby("generation", as_index=False).first()
                df = df.sort_values("generation").reset_index(drop=True)

        picked_idx = pick_evenly_spaced_indices(len(df), n_pick=n_pick)
        picked = df.iloc[picked_idx].copy()
        return picked

    else:
        raise ValueError("mode 只能是 best_path 或 topk")


def draw_molecule_grid_svg(smiles_list, legends, out_svg, mols_per_row=5, sub_img_size=(320, 320)):
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
        raise ValueError("没有可绘制的有效分子。")

    n_mols = len(mols)
    rows = (n_mols + mols_per_row - 1) // mols_per_row
    width = mols_per_row * sub_img_size[0]
    height = rows * sub_img_size[1]

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height, sub_img_size[0], sub_img_size[1])

    opts = drawer.drawOptions()
    opts.legendFontSize = 20
    opts.bondLineWidth = 2
    opts.maxFontSize = 28
    opts.minFontSize = 16
    opts.fixedBondLength = 30
    opts.padding = 0.08
    opts.centreMoleculesBeforeDrawing = True
    opts.clearBackground = False
    opts.additionalAtomLabelPadding = 0.15

    drawer.DrawMolecules(mols, legends=valid_legends)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText().replace("svg:", "")

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"[OK] SVG saved to: {out_svg}")


def main():
    parser = argparse.ArgumentParser(description="从 evolution csv 自动挑 5 个代表性分子并画 SVG")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="输入 csv，如 evolution_path_full.csv 或 topk_evolution_paths.csv")
    parser.add_argument("--mode", type=str, default="best_path", choices=["best_path", "topk"],
                        help="best_path: 用 evolution_path_full.csv；topk: 用 topk_evolution_paths.csv")
    parser.add_argument("--n_pick", type=int, default=5, help="挑几个代表性分子")
    parser.add_argument("--out_svg", type=str, required=True, help="输出 SVG 路径")
    parser.add_argument("--out_csv", type=str, default=None, help="输出挑选后的 csv 路径")
    parser.add_argument("--mols_per_row", type=int, default=5)
    parser.add_argument("--sub_img_size", type=int, default=320)

    args = parser.parse_args()

    picked = load_representative_rows(args.csv_path, mode=args.mode, n_pick=args.n_pick)

    smiles_list = picked["smiles"].tolist()

    legends = []
    for _, row in picked.iterrows():
        if "step" in picked.columns:
            tag = f"Step {int(row['step'])}"
        elif "generation" in picked.columns:
            tag = f"Gen {int(row['generation'])}"
        elif "evaluations" in picked.columns:
            tag = f"Eval {int(row['evaluations'])}"
        else:
            tag = "Selected"

        legends.append(f"{tag}\ngap = {float(row['gap']):.4f}")

    draw_molecule_grid_svg(
        smiles_list=smiles_list,
        legends=legends,
        out_svg=args.out_svg,
        mols_per_row=args.mols_per_row,
        sub_img_size=(args.sub_img_size, args.sub_img_size),
    )

    if args.out_csv is None:
        root, _ = os.path.splitext(args.out_svg)
        out_csv = root + "_selected.csv"
    else:
        out_csv = args.out_csv

    picked.to_csv(out_csv, index=False)
    print(f"[OK] Selected rows saved to: {out_csv}")


if __name__ == "__main__":
    main()