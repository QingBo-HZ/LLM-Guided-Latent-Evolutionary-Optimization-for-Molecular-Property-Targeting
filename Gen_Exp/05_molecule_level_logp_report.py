#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smi):
    try:
        if smi is None or pd.isna(smi):
            return None
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def calc_logp(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return float(Descriptors.MolLogP(mol))


def compute_diversity(smiles_list):
    mols = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            mols.append(mol)
    if len(mols) < 2:
        return 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - float(np.mean(sims)) if sims else 0.0


def load_candidates(result_dir):
    result_dir = Path(result_dir)
    frames = []

    for name in ["final_population", "topk_evolution_paths", "evolution_path", "best_candidates_over_time"]:
        paths = sorted(result_dir.glob(f"{name}*.csv"))
        for path in paths:
            df = pd.read_csv(path)
            if "smiles" not in df.columns:
                if "best_smiles" in df.columns:
                    df = df.rename(columns={"best_smiles": "smiles"})
                else:
                    continue
            df["source_file"] = path.name
            df["source_type"] = name
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No candidate csv with smiles found in {result_dir}")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    return raw


def main():
    parser = argparse.ArgumentParser("Build molecule-level RDKit logP report from ZINC logP GA outputs")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--target_logp", type=float, default=3.0)
    parser.add_argument("--success_low", type=float, default=2.5)
    parser.add_argument("--success_high", type=float, default=3.5)
    parser.add_argument("--out_prefix", default="molecule_level")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    raw = load_candidates(result_dir)

    rows = []
    for i, row in raw.iterrows():
        smi = canonicalize_smiles(row.get("smiles"))
        if smi is None:
            continue
        rdkit_logp = calc_logp(smi)
        if rdkit_logp is None:
            continue
        pred_logp = row.get("pred_logP", np.nan)
        try:
            pred_logp = float(pred_logp)
        except Exception:
            pred_logp = np.nan
        rows.append({
            "canonical_smiles": smi,
            "rdkit_logP": rdkit_logp,
            "rdkit_abs_error": abs(rdkit_logp - args.target_logp),
            "rdkit_success": bool(args.success_low <= rdkit_logp <= args.success_high),
            "pred_logP": pred_logp,
            "pred_abs_error": abs(pred_logp - args.target_logp) if np.isfinite(pred_logp) else np.nan,
            "source_file": row.get("source_file"),
            "source_type": row.get("source_type"),
            "generation": row.get("generation", np.nan),
            "rank": row.get("rank", row.get("rank_by_score", np.nan)),
        })

    valid_df = pd.DataFrame(rows)
    if len(valid_df) == 0:
        summary = {
            "result_dir": str(result_dir),
            "decoded_valid_candidates": 0,
            "unique_valid_molecules": 0,
            "rdkit_success_count_unique": 0,
            "rdkit_success_rate_unique": 0.0,
        }
        (result_dir / f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    valid_df = valid_df.sort_values(["rdkit_abs_error", "canonical_smiles"]).reset_index(drop=True)
    valid_df.to_csv(result_dir / f"{args.out_prefix}_all_decoded_candidates.csv", index=False)

    unique_df = valid_df.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    unique_df = unique_df.sort_values(["rdkit_abs_error", "canonical_smiles"]).reset_index(drop=True)
    unique_df["rank_by_rdkit_error"] = np.arange(1, len(unique_df) + 1)
    unique_df.to_csv(result_dir / f"{args.out_prefix}_unique_ranked.csv", index=False)

    top10 = unique_df.head(min(10, len(unique_df)))
    success_unique = unique_df[unique_df["rdkit_success"]]

    summary = {
        "result_dir": str(result_dir),
        "target_logP": args.target_logp,
        "success_low": args.success_low,
        "success_high": args.success_high,
        "decoded_valid_candidate_records": int(len(valid_df)),
        "unique_valid_molecules": int(len(unique_df)),
        "rdkit_success_count_unique": int(len(success_unique)),
        "rdkit_success_rate_unique_over_unique": float(len(success_unique) / len(unique_df)) if len(unique_df) else 0.0,
        "best_smiles_rdkit": unique_df.iloc[0]["canonical_smiles"],
        "best_rdkit_logP": float(unique_df.iloc[0]["rdkit_logP"]),
        "best_rdkit_abs_error": float(unique_df.iloc[0]["rdkit_abs_error"]),
        "top10_rdkit_abs_error_mean": float(top10["rdkit_abs_error"].mean()) if len(top10) else None,
        "top10_rdkit_logP_mean": float(top10["rdkit_logP"].mean()) if len(top10) else None,
        "diversity_unique_valid": compute_diversity(unique_df["canonical_smiles"].tolist()),
        "top10_smiles": top10["canonical_smiles"].tolist(),
    }

    (result_dir / f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
