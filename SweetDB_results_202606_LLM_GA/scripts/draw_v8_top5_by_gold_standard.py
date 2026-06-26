#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "Random-Seeded Latent GA",
    "B": "SweetDB-Seeded Latent GA",
    "C": "LLM-Initialized Latent GA",
    "D": "Iterative LLM-Guided Latent GA",
}
SHORT_LABELS = {
    "A": "random",
    "B": "sweetdb_seed",
    "C": "llm_init",
    "D": "llm_iterative",
}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is not None:
        Chem.rdDepictor.Compute2DCoords(mol)
    return mol


def select_top5(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"method", "smiles", "gold_real_vina", "pre_docking_goldlike", "vina_kcal_mol"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    for col in ["vina_kcal_mol", "pred_logsw_reencoded", "p_sweet_reencoded", "d_ood_reencoded", "final_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["gold_real_vina", "pre_docking_goldlike", "real_vina_supported", "sweet_gate_pass", "pred_logsw_pass", "ood_pass"]:
        if col in df.columns:
            df[col] = as_bool(df[col])

    df = df.dropna(subset=["smiles"]).copy()
    df["canonical_smiles"] = df["smiles"].map(canonicalize)
    df = df.dropna(subset=["canonical_smiles"])

    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].copy()
        sub = sub.sort_values(
            [
                "gold_real_vina",
                "pre_docking_goldlike",
                "final_score",
                "pred_logsw_reencoded",
                "p_sweet_reencoded",
                "vina_kcal_mol",
            ],
            ascending=[False, False, False, False, False, True],
        )
        sub = sub.drop_duplicates("canonical_smiles").head(5).copy()
        sub["method_label"] = METHOD_LABELS[method]
        sub["molecule_panel_rank"] = range(1, len(sub) + 1)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def write_group_svgs(top: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in METHOD_ORDER:
        sub = top[top["method"] == method].sort_values("molecule_panel_rank")
        mols = [mol_from_smiles(s) for s in sub["smiles"]]
        legends = []
        for row in sub.itertuples():
            legends.append(
                f"{method}{int(row.molecule_panel_rank)} | logSw {float(row.pred_logsw_reencoded):.2f}\n"
                f"P {float(row.p_sweet_reencoded):.2f} | Vina {float(row.vina_kcal_mol):.2f}"
            )
        stem = f"v8_{method}_{SHORT_LABELS[method]}_top5_gold_standard"
        svg = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(270, 235),
            legends=legends,
            useSVG=True,
        )
        (out_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")
        png = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(270, 235),
            legends=legends,
            useSVG=False,
        )
        png.save(out_dir / f"{stem}_preview.png")

    all_mols = []
    all_legends = []
    for row in top.sort_values(["method", "molecule_panel_rank"]).itertuples():
        all_mols.append(mol_from_smiles(row.smiles))
        all_legends.append(
            f"{row.method}{int(row.molecule_panel_rank)} | logSw {float(row.pred_logsw_reencoded):.2f}\n"
            f"Vina {float(row.vina_kcal_mol):.2f}"
        )
    svg = Draw.MolsToGridImage(
        all_mols,
        molsPerRow=5,
        subImgSize=(245, 220),
        legends=all_legends,
        useSVG=True,
    )
    (out_dir / "v8_ABCD_top5_gold_standard_combined.svg").write_text(svg, encoding="utf-8")
    png = Draw.MolsToGridImage(
        all_mols,
        molsPerRow=5,
        subImgSize=(245, 220),
        legends=all_legends,
        useSVG=False,
    )
    png.save(out_dir / "v8_ABCD_top5_gold_standard_combined_preview.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    top = select_top5(args.input_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out_dir / "v8_abcd_top5_gold_standard_for_svg.csv", index=False)
    write_group_svgs(top, args.out_dir)
    print(top[["method", "molecule_panel_rank", "ID", "gold_real_vina", "pred_logsw_reencoded", "p_sweet_reencoded", "vina_kcal_mol", "smiles"]].to_string(index=False))
    print(f"Wrote gold-standard molecule SVGs to {args.out_dir}")


if __name__ == "__main__":
    main()
