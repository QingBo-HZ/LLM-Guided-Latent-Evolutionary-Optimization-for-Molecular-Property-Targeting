#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    "direct_regressor": "Sweet-only",
    "docking_surrogate": "Docking-only",
    "gated": "Gate+Sweet",
    "gated_docking": "Gate+Docking",
}


def truthy(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def build_docking_input(result_root, out_dir, top_k, p_threshold, logsw_threshold):
    result_root = Path(result_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    detail = []

    for summary_path in sorted(result_root.glob("*/summary.json")):
        run_dir = summary_path.parent
        with summary_path.open() as handle:
            summary = json.load(handle)
        mode = summary["fitness_mode"]
        seed = int(summary["seed"])
        label = LABELS.get(mode, mode)
        final_path = run_dir / "final_population.csv"
        if not final_path.exists():
            continue

        df = pd.read_csv(final_path)
        for col in ["valid", "reencode_ok"]:
            if col not in df:
                df[col] = False
        for col in ["canonical_smiles", "smiles"]:
            if col not in df:
                df[col] = np.nan

        df = df[truthy(df["valid"]) & truthy(df["reencode_ok"])].copy()
        df["canonical_smiles"] = df["canonical_smiles"].fillna(df["smiles"])
        df = df.dropna(subset=["canonical_smiles"])
        df = df.drop_duplicates("canonical_smiles", keep="first")

        df["p_sweet_reencoded"] = pd.to_numeric(df["p_sweet_reencoded"], errors="coerce")
        df["pred_logsw_reencoded"] = pd.to_numeric(df["pred_logsw_reencoded"], errors="coerce")
        df["d_ood_reencoded"] = pd.to_numeric(df.get("d_ood_reencoded"), errors="coerce")
        df = df.sort_values(
            ["pred_logsw_reencoded", "p_sweet_reencoded"],
            ascending=[False, False],
        ).head(top_k)

        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            mol_id = f"{label.replace('+', '_').replace('-', '_')}_seed{seed}_top{rank}"
            record = {
                "ID": mol_id,
                "smiles": row["canonical_smiles"],
                "method": label,
                "fitness_mode": mode,
                "seed": seed,
                "rank": rank,
                "p_sweet_reencoded": row["p_sweet_reencoded"],
                "pred_logsw_reencoded": row["pred_logsw_reencoded"],
                "d_ood_reencoded": row["d_ood_reencoded"],
                "high_sweet": bool(
                    (row["p_sweet_reencoded"] >= p_threshold)
                    and (row["pred_logsw_reencoded"] >= logsw_threshold)
                ),
            }
            rows.append({"ID": mol_id, "smiles": row["canonical_smiles"]})
            detail.append(record)

    docking_input = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail)
    docking_input.to_csv(out_dir / "docking_input.csv", index=False)
    detail_df.to_csv(out_dir / "gold_standard_candidate_pool.csv", index=False)
    return docking_input, detail_df


def summarize(docking_dir, vina_csv, p_threshold, logsw_threshold, vina_threshold):
    docking_dir = Path(docking_dir)
    detail = pd.read_csv(docking_dir / "gold_standard_candidate_pool.csv")
    vina = pd.read_csv(vina_csv)
    id_col = "mol_id" if "mol_id" in vina.columns else "ID"
    vina = vina.rename(columns={id_col: "ID"})
    merged = detail.merge(vina, on="ID", how="left")
    merged["vina_kcal_mol"] = pd.to_numeric(merged["vina_kcal_mol"], errors="coerce")
    merged["real_vina_supported"] = merged["vina_kcal_mol"] <= vina_threshold
    merged["gold_success"] = merged["high_sweet"] & merged["real_vina_supported"]
    merged.to_csv(docking_dir / "gold_standard_per_molecule.csv", index=False)

    rows = []
    for (method, seed), sub in merged.groupby(["method", "seed"]):
        rows.append({
            "method": method,
            "seed": int(seed),
            "n_evaluated": int(sub["vina_kcal_mol"].notna().sum()),
            "primary_high_sweet": int(sub["high_sweet"].sum()),
            "secondary_real_vina_le_threshold": int(sub["real_vina_supported"].sum()),
            "gold_success_both": int(sub["gold_success"].sum()),
            "mean_real_vina": float(sub["vina_kcal_mol"].mean()),
            "best_real_vina": float(sub["vina_kcal_mol"].min()),
        })
    detail_summary = pd.DataFrame(rows)
    detail_summary.to_csv(docking_dir / "gold_standard_by_seed.csv", index=False)

    grouped = detail_summary.groupby("method").agg(
        n_seeds=("seed", "count"),
        primary_mean=("primary_high_sweet", "mean"),
        primary_std=("primary_high_sweet", "std"),
        secondary_mean=("secondary_real_vina_le_threshold", "mean"),
        secondary_std=("secondary_real_vina_le_threshold", "std"),
        gold_mean=("gold_success_both", "mean"),
        gold_std=("gold_success_both", "std"),
        mean_real_vina=("mean_real_vina", "mean"),
        best_real_vina=("best_real_vina", "min"),
    ).reset_index()
    grouped = grouped.sort_values(
        ["gold_mean", "secondary_mean", "primary_mean", "mean_real_vina"],
        ascending=[False, False, False, True],
    )
    grouped.to_csv(docking_dir / "gold_standard_summary.csv", index=False)
    return merged, detail_summary, grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--docking_dir", required=True)
    parser.add_argument("--vina_csv", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--p_threshold", type=float, default=0.90)
    parser.add_argument("--logsw_threshold", type=float, default=2.80)
    parser.add_argument("--vina_threshold", type=float, default=-7.0)
    args = parser.parse_args()

    if args.vina_csv:
        _, _, summary = summarize(
            args.docking_dir,
            args.vina_csv,
            args.p_threshold,
            args.logsw_threshold,
            args.vina_threshold,
        )
        print(summary.to_string(index=False))
    else:
        docking_input, detail = build_docking_input(
            args.result_root,
            args.docking_dir,
            args.top_k,
            args.p_threshold,
            args.logsw_threshold,
        )
        print(f"wrote {len(docking_input)} docking molecules")
        print(detail.groupby("method")["high_sweet"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
