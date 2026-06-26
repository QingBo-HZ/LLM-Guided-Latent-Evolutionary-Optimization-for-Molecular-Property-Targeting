#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize QM9 main experiment Table 1 over seeds 42/43/44.

The original paper table reports Eval@Gap<0.15 and generations in units of
100 evaluations because all methods use an effective population/batch size of
100 in the selected Table 1 runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA")
SEEDS = [42, 43, 44]
THRESHOLD = 0.15
EVAL_UNIT = 100.0


METHODS = [
    {
        "method": "Random Latent Search",
        "dirs": {
            42: ROOT / "1_random_search/random_search_random_search_V2",
            43: ROOT / "1_random_search/random_search_random_search_V2_seed43",
            44: ROOT / "1_random_search/random_search_random_search_V2_seed44",
        },
        "progress_best_col": "best_gap_so_far",
    },
    {
        "method": "BRICS-based SMILES GA",
        "dirs": {
            42: ROOT / "2_smiles_GA/fragment_ga_smiles_childselect_v1",
            43: ROOT / "2_smiles_GA/fragment_ga_smiles_childselect_v1_seed43",
            44: ROOT / "2_smiles_GA/fragment_ga_smiles_childselect_v1_seed44",
        },
        "progress_best_col": "best_gap_so_far",
    },
    {
        "method": "Latent GA",
        "dirs": {
            42: ROOT / "3_latent_GA_noLLM/psvae_train_random_V2",
            43: ROOT / "3_latent_GA_noLLM/psvae_train_random_V2_seed43",
            44: ROOT / "3_latent_GA_noLLM/psvae_train_random_V2_seed44",
        },
        "progress_best_col": "best_gap_so_far",
    },
    {
        "method": "LLM-Initialized Latent GA",
        "dirs": {
            42: ROOT / "4_latent_GA_LLM/llm_llm_V2",
            43: ROOT / "4_latent_GA_LLM/llm_llm_V2_seed43",
            44: ROOT / "4_latent_GA_LLM/llm_llm_V2_seed44",
        },
        "progress_best_col": "best_gap_so_far",
    },
    {
        "method": "Iterative LLM-Guided Latent GA",
        "dirs": {
            42: ROOT / "5_Ours/llm_ours_V2",
            43: ROOT / "5_Ours/llm_ours_V2_seed43",
            44: ROOT / "5_Ours/llm_ours_V2_seed44",
        },
        "progress_best_col": "best_gap_so_far",
    },
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_at_threshold(progress_path: Path, best_col: str, max_eval_unit: float) -> tuple[float, bool]:
    df = pd.read_csv(progress_path)
    if "evaluations" not in df.columns or best_col not in df.columns:
        return max_eval_unit, False
    hit = df[pd.to_numeric(df[best_col], errors="coerce") < THRESHOLD].copy()
    if hit.empty:
        return max_eval_unit, False
    return float(pd.to_numeric(hit["evaluations"], errors="coerce").iloc[0] / EVAL_UNIT), True


def one_run(method: str, seed: int, run_dir: Path, best_col: str) -> dict:
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "progress_metrics.csv"
    if not summary_path.exists() or not progress_path.exists():
        return {
            "method": method,
            "seed": seed,
            "run_dir": str(run_dir),
            "status": "missing",
        }

    s = read_json(summary_path)
    n_eval = float(s.get("n_evaluations_total", np.nan))
    max_eval_unit = n_eval / EVAL_UNIT if np.isfinite(n_eval) else np.nan
    eval_at, eval_reached = eval_at_threshold(progress_path, best_col, max_eval_unit)
    return {
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir),
        "status": "ready",
        "best_gap": float(s.get("best_gap_final", np.nan)),
        "top10_mean": float(s.get("top10_mean_gap_final", np.nan)),
        "success_rate_pct": float(s.get("success_rate_final", np.nan)) * 100.0,
        "eval_at_gap_lt_0_15": eval_at,
        "eval_reached_gap_lt_0_15": bool(eval_reached),
        "diversity": float(s.get("diversity", np.nan)),
        "generations_to_convergence": max_eval_unit,
        "time_sec_total": float(s.get("time_sec_total", np.nan)),
    }


def summarize_metric(df: pd.DataFrame, col: str) -> tuple[float, float]:
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return np.nan, np.nan
    if len(vals) == 1:
        return float(vals.mean()), 0.0
    return float(vals.mean()), float(vals.std(ddof=1))


def fmt_mean_std(mean: float, std: float, digits: int = 6, suffix: str = "") -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}{suffix}"


def main() -> None:
    rows = []
    for spec in METHODS:
        for seed, run_dir in spec["dirs"].items():
            rows.append(one_run(spec["method"], seed, run_dir, spec["progress_best_col"]))

    detail = pd.DataFrame(rows)
    out_dir = ROOT / "summary_three_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "table1_three_seed_detail.csv", index=False)

    ready = detail[detail["status"] == "ready"].copy()
    summary_rows = []
    for method, sub in ready.groupby("method", sort=False):
        row = {"method": method, "n_runs": int(len(sub))}
        for col in [
            "best_gap",
            "top10_mean",
            "success_rate_pct",
            "eval_at_gap_lt_0_15",
            "diversity",
            "generations_to_convergence",
            "time_sec_total",
        ]:
            mean, std = summarize_metric(sub, col)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "table1_three_seed_mean_std.csv", index=False)

    word_rows = []
    for _, row in summary.iterrows():
        word_rows.append({
            "Method": row["method"],
            "Best gap ↓": fmt_mean_std(row["best_gap_mean"], row["best_gap_std"], 6),
            "Top-10 mean ↓": fmt_mean_std(row["top10_mean_mean"], row["top10_mean_std"], 6),
            "Success rate ↑": fmt_mean_std(row["success_rate_pct_mean"], row["success_rate_pct_std"], 2, "%"),
            "Eval@Gap<0.15 ↓": fmt_mean_std(row["eval_at_gap_lt_0_15_mean"], row["eval_at_gap_lt_0_15_std"], 1),
            "Diversity ↑": fmt_mean_std(row["diversity_mean"], row["diversity_std"], 4),
            "Generations to convergence ↓": fmt_mean_std(row["generations_to_convergence_mean"], row["generations_to_convergence_std"], 1),
            "n": int(row["n_runs"]),
        })
    pd.DataFrame(word_rows).to_csv(out_dir / "table1_three_seed_word_table.csv", index=False)

    print(f"saved {out_dir / 'table1_three_seed_detail.csv'}")
    print(f"saved {out_dir / 'table1_three_seed_mean_std.csv'}")
    print(f"saved {out_dir / 'table1_three_seed_word_table.csv'}")
    missing = detail[detail["status"] != "ready"]
    if not missing.empty:
        print("\n[MISSING]")
        print(missing[["method", "seed", "run_dir", "status"]].to_string(index=False))
    else:
        print("\nAll 15 runs are ready.")


if __name__ == "__main__":
    main()
