#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "group_a_random": "Random latent GA",
    "group_b_dataset": "SweetDB-seeded GA",
    "group_c_llm": "LLM-initialized GA",
    "group_d_llm_iterative": "Reflection-guided LLM GA",
}
METHOD_SHORT = {
    "group_a_random": "A",
    "group_b_dataset": "B",
    "group_c_llm": "C",
    "group_d_llm_iterative": "D",
}


def truthy(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def read_summary(run_dir):
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    mode = str(summary.get("init_mode", run_dir.name.split("_ABCD")[0]))
    seed = int(summary.get("seed", re.search(r"seed(\d+)", run_dir.name).group(1)))
    return summary, mode, seed


def build_generation_audit(result_root, out_dir, top_k, p_threshold, logsw_threshold, ood_threshold):
    rows = []
    detail = []
    for summary_path in sorted(Path(result_root).glob("*/summary.json")):
        run_dir = summary_path.parent
        _, mode, seed = read_summary(run_dir)
        method_short = METHOD_SHORT.get(mode, mode)
        method_label = METHOD_LABELS.get(mode, mode)
        topk_path = run_dir / "topk_archive.csv"
        if not topk_path.exists():
            continue
        topk = pd.read_csv(topk_path)
        if topk.empty:
            continue
        topk["valid"] = truthy(topk.get("valid", pd.Series(False, index=topk.index)))
        topk["smiles"] = topk.get("smiles", pd.Series(np.nan, index=topk.index))
        topk = topk[topk["valid"] & topk["smiles"].notna()].copy()
        if topk.empty:
            continue
        for col in ["score_ga", "p_sweet", "pred_logsw", "d_ood"]:
            topk[col] = pd.to_numeric(topk[col], errors="coerce")
        for generation, gen_df in topk.groupby("generation"):
            gen_df = gen_df.drop_duplicates("smiles", keep="first")
            gen_df = gen_df.sort_values(
                ["pred_logsw", "p_sweet", "score_ga"],
                ascending=[False, False, False],
            ).head(top_k)
            for audit_rank, (_, row) in enumerate(gen_df.iterrows(), start=1):
                mol_id = f"{method_short}_seed{seed}_gen{int(generation):03d}_top{audit_rank}"
                rec = {
                    "ID": mol_id,
                    "smiles": row["smiles"],
                    "method": method_short,
                    "method_label": method_label,
                    "init_mode": mode,
                    "seed": seed,
                    "generation": int(generation),
                    "rank": int(audit_rank),
                    "source_rank": int(row.get("rank", audit_rank)),
                    "score_ga": safe_float(row.get("score_ga")),
                    "p_sweet": safe_float(row.get("p_sweet")),
                    "pred_logsw": safe_float(row.get("pred_logsw")),
                    "d_ood": safe_float(row.get("d_ood")),
                }
                rec["sweet_gate_pass"] = bool(rec["p_sweet"] >= p_threshold)
                rec["pred_logsw_pass"] = bool(rec["pred_logsw"] >= logsw_threshold)
                rec["ood_pass"] = bool(rec["d_ood"] <= ood_threshold)
                rec["pre_docking_goldlike"] = bool(
                    rec["sweet_gate_pass"] and rec["pred_logsw_pass"] and rec["ood_pass"]
                )
                rows.append({"ID": mol_id, "smiles": row["smiles"]})
                detail.append(rec)
    docking_input = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail)
    docking_input.to_csv(out_dir / "generation_docking_input.csv", index=False)
    detail_df.to_csv(out_dir / "generation_docking_candidate_pool.csv", index=False)
    return docking_input, detail_df


def build_final_audit(result_root, out_dir, top_k, p_threshold, logsw_threshold, ood_threshold):
    rows = []
    detail = []
    for summary_path in sorted(Path(result_root).glob("*/summary.json")):
        run_dir = summary_path.parent
        _, mode, seed = read_summary(run_dir)
        method_short = METHOD_SHORT.get(mode, mode)
        method_label = METHOD_LABELS.get(mode, mode)
        final_path = run_dir / "final_population.csv"
        if not final_path.exists():
            continue
        df = pd.read_csv(final_path)
        for col in ["valid", "reencode_ok"]:
            if col not in df:
                df[col] = False
        df = df[truthy(df["valid"]) & truthy(df["reencode_ok"])].copy()
        if df.empty:
            continue
        if "canonical_smiles" not in df:
            df["canonical_smiles"] = df.get("smiles", np.nan)
        df["canonical_smiles"] = df["canonical_smiles"].fillna(df.get("smiles", np.nan))
        df = df.dropna(subset=["canonical_smiles"])
        df = df.drop_duplicates("canonical_smiles", keep="first")
        for col in ["p_sweet_reencoded", "pred_logsw_reencoded", "d_ood_reencoded", "final_score"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df = df.sort_values(
            ["pred_logsw_reencoded", "p_sweet_reencoded", "final_score"],
            ascending=[False, False, False],
        ).head(top_k)
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            mol_id = f"{method_short}_seed{seed}_final_top{rank}"
            rec = {
                "ID": mol_id,
                "smiles": row["canonical_smiles"],
                "method": method_short,
                "method_label": method_label,
                "init_mode": mode,
                "seed": seed,
                "rank": int(rank),
                "p_sweet_reencoded": safe_float(row.get("p_sweet_reencoded")),
                "pred_logsw_reencoded": safe_float(row.get("pred_logsw_reencoded")),
                "d_ood_reencoded": safe_float(row.get("d_ood_reencoded")),
                "final_score": safe_float(row.get("final_score")),
            }
            rec["sweet_gate_pass"] = bool(rec["p_sweet_reencoded"] >= p_threshold)
            rec["pred_logsw_pass"] = bool(rec["pred_logsw_reencoded"] >= logsw_threshold)
            rec["ood_pass"] = bool(rec["d_ood_reencoded"] <= ood_threshold)
            rec["pre_docking_goldlike"] = bool(
                rec["sweet_gate_pass"] and rec["pred_logsw_pass"] and rec["ood_pass"]
            )
            rows.append({"ID": mol_id, "smiles": row["canonical_smiles"]})
            detail.append(rec)
    docking_input = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail)
    docking_input.to_csv(out_dir / "final_docking_input.csv", index=False)
    detail_df.to_csv(out_dir / "final_docking_candidate_pool.csv", index=False)
    return docking_input, detail_df


def merge_vina(out_dir, vina_csv, vina_threshold):
    vina = pd.read_csv(vina_csv)
    id_col = "mol_id" if "mol_id" in vina.columns else "ID"
    vina = vina.rename(columns={id_col: "ID"})
    if "vina_kcal_mol" not in vina.columns and "vina_affinity_kcal_mol" in vina.columns:
        vina = vina.rename(columns={"vina_affinity_kcal_mol": "vina_kcal_mol"})
    vina["vina_kcal_mol"] = pd.to_numeric(vina["vina_kcal_mol"], errors="coerce")

    outputs = {}
    for stem in ["generation", "final"]:
        pool_path = out_dir / f"{stem}_docking_candidate_pool.csv"
        if not pool_path.exists():
            continue
        detail = pd.read_csv(pool_path)
        merged = detail.merge(vina[["ID", "vina_kcal_mol"]], on="ID", how="left")
        merged["real_vina_supported"] = merged["vina_kcal_mol"] <= vina_threshold
        merged["gold_success"] = merged["pre_docking_goldlike"] & merged["real_vina_supported"]
        scored_path = out_dir / f"{stem}_docking_scored.csv"
        merged.to_csv(scored_path, index=False)
        outputs[stem] = merged

    if "final" in outputs:
        final = outputs["final"]
        by_seed = final.groupby(["method", "seed"]).agg(
            n_evaluated=("vina_kcal_mol", lambda x: int(x.notna().sum())),
            pre_docking_goldlike=("pre_docking_goldlike", "sum"),
            vina_supported=("real_vina_supported", "sum"),
            gold_success=("gold_success", "sum"),
            mean_real_vina=("vina_kcal_mol", "mean"),
            best_real_vina=("vina_kcal_mol", "min"),
        ).reset_index()
        by_seed.to_csv(out_dir / "final_docking_by_seed.csv", index=False)
        summary = by_seed.groupby("method").agg(
            n_seeds=("seed", "count"),
            gold_success_mean=("gold_success", "mean"),
            gold_success_sd=("gold_success", "std"),
            vina_supported_mean=("vina_supported", "mean"),
            vina_supported_sd=("vina_supported", "std"),
            mean_real_vina=("mean_real_vina", "mean"),
            best_real_vina=("best_real_vina", "min"),
        ).reset_index()
        summary.to_csv(out_dir / "final_docking_summary.csv", index=False)

    if "generation" in outputs:
        gen = outputs["generation"]
        by_gen = gen.groupby(["method", "seed", "generation"]).agg(
            n_evaluated=("vina_kcal_mol", lambda x: int(x.notna().sum())),
            pre_docking_goldlike=("pre_docking_goldlike", "sum"),
            vina_supported=("real_vina_supported", "sum"),
            gold_success=("gold_success", "sum"),
            mean_real_vina=("vina_kcal_mol", "mean"),
            best_real_vina=("vina_kcal_mol", "min"),
        ).reset_index()
        by_gen.to_csv(out_dir / "generation_docking_by_generation_seed.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k_generation", type=int, default=5)
    parser.add_argument("--top_k_final", type=int, default=10)
    parser.add_argument("--p_threshold", type=float, default=0.80)
    parser.add_argument("--logsw_threshold", type=float, default=2.60)
    parser.add_argument("--ood_threshold", type=float, default=7.225)
    parser.add_argument("--vina_threshold", type=float, default=-6.8)
    parser.add_argument("--vina_csv", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_input, gen_detail = build_generation_audit(
        args.result_root,
        out_dir,
        args.top_k_generation,
        args.p_threshold,
        args.logsw_threshold,
        args.ood_threshold,
    )
    final_input, final_detail = build_final_audit(
        args.result_root,
        out_dir,
        args.top_k_final,
        args.p_threshold,
        args.logsw_threshold,
        args.ood_threshold,
    )
    pd.concat(
        [
            gen_input.assign(audit_stage="generation"),
            final_input.assign(audit_stage="final"),
        ],
        ignore_index=True,
    ).drop_duplicates("ID").to_csv(out_dir / "all_docking_input.csv", index=False)

    manifest = {
        "result_root": str(args.result_root),
        "top_k_generation": args.top_k_generation,
        "top_k_final": args.top_k_final,
        "p_threshold": args.p_threshold,
        "logsw_threshold": args.logsw_threshold,
        "ood_threshold": args.ood_threshold,
        "vina_threshold": args.vina_threshold,
        "generation_molecules": int(len(gen_input)),
        "final_molecules": int(len(final_input)),
        "all_molecules": int(len(gen_input) + len(final_input)),
    }
    with (out_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    if args.vina_csv:
        merge_vina(out_dir, args.vina_csv, args.vina_threshold)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not gen_detail.empty:
        print("\nGeneration pre-docking pass:")
        print(gen_detail.groupby("method")["pre_docking_goldlike"].agg(["count", "sum"]).to_string())
    if not final_detail.empty:
        print("\nFinal pre-docking pass:")
        print(final_detail.groupby("method")["pre_docking_goldlike"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
