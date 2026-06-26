#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize formal five-group ZINC logP transfer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SEEDS = [42, 43, 44]
CONVERGENCE_ABS_ERROR = 0.05

METHOD_ORDER = [
    "Random Latent Search",
    "ZINC-Seeded Latent GA",
    "LLM-Generated Molecules",
    "LLM-Initialized Latent GA",
    "Iterative LLM-Guided Latent GA",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_generation_summary(smiles_dir: Path, mode: str, seed: int) -> Path | None:
    files = sorted(
        smiles_dir.glob(f"zinc_logp_llm_{mode}_*_seed{seed}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def first_convergence_from_progress(path: Path, pop_size: int | None = None) -> int | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "best_rdkit_abs_error" in df.columns:
        hit = df[pd.to_numeric(df["best_rdkit_abs_error"], errors="coerce") <= CONVERGENCE_ABS_ERROR]
        if len(hit):
            if "evaluations" in hit.columns:
                return int(hit.iloc[0]["evaluations"])
            if "generation" in hit.columns and pop_size:
                return int((int(hit.iloc[0]["generation"]) + 1) * pop_size)
    if "rdkit_selection_best_logP" in df.columns:
        err = (pd.to_numeric(df["rdkit_selection_best_logP"], errors="coerce") - 3.0).abs()
        hit = df[err <= CONVERGENCE_ABS_ERROR]
        if len(hit):
            if "evaluations" in hit.columns:
                return int(hit.iloc[0]["evaluations"])
            if "generation" in hit.columns and pop_size:
                return int((int(hit.iloc[0]["generation"]) + 1) * pop_size)
    return None


def row_from_summary(method: str, seed: int, summary_path: Path) -> dict:
    data = read_json(summary_path)
    out_dir = summary_path.parent
    progress = out_dir / "progress_metrics_decode_aware.csv"
    if not progress.exists():
        progress = out_dir / "progress_metrics_random_latent.csv"

    pop_size = data.get("pop_size")
    evaluations_to_convergence = data.get("evaluations_to_convergence")
    if evaluations_to_convergence is None:
        evaluations_to_convergence = first_convergence_from_progress(progress, pop_size=pop_size)

    evaluations_total = data.get("evaluations_total")
    if evaluations_total is None:
        n_gen_completed = data.get("n_gen_completed")
        if pop_size is not None and n_gen_completed is not None:
            evaluations_total = int(pop_size) * int(n_gen_completed)

    return {
        "method": method,
        "seed": seed,
        "source_file": str(summary_path),
        "best_smiles_rdkit": data.get("best_smiles_rdkit"),
        "best_rdkit_logP": data.get("best_rdkit_logP"),
        "best_rdkit_abs_error": data.get("best_rdkit_abs_error"),
        "top10_rdkit_abs_error_mean": data.get("top10_rdkit_abs_error_mean"),
        "success_rate": data.get("archive_rdkit_success_rate_over_unique"),
        "archive_rdkit_success_unique": data.get("archive_rdkit_success_unique"),
        "archive_unique_valid_molecules": data.get("archive_unique_valid_molecules"),
        "decode_latent_validity_final": data.get("decode_latent_validity_final"),
        "decode_success_final": data.get("decode_success_final"),
        "diversity_unique_valid": data.get("diversity_unique_valid"),
        "evaluations_total": evaluations_total,
        "evaluations_to_convergence": evaluations_to_convergence,
        "time_sec_total": data.get("time_sec_total"),
        "n_gen_completed": data.get("n_gen_completed"),
    }


def direct_llm_rows(llm_base_dir: Path) -> list[dict]:
    rows = []
    smiles_dir = llm_base_dir / "smiles"
    for seed in SEEDS:
        path = latest_generation_summary(smiles_dir, "direct", seed)
        if path is None:
            continue
        data = read_json(path)
        rows.append({
            "method": "LLM-Generated Molecules",
            "seed": seed,
            "source_file": str(path),
            "best_smiles_rdkit": data.get("best_smiles"),
            "best_rdkit_logP": data.get("best_rdkit_logP"),
            "best_rdkit_abs_error": data.get("best_rdkit_abs_error"),
            "top10_rdkit_abs_error_mean": data.get("top10_rdkit_abs_error_mean"),
            "success_rate": data.get("rdkit_success_rate"),
            "archive_rdkit_success_unique": data.get("rdkit_success_count"),
            "archive_unique_valid_molecules": data.get("accepted_total"),
            "decode_latent_validity_final": None,
            "decode_success_final": None,
            "diversity_unique_valid": None,
            "evaluations_total": data.get("accepted_total"),
            "evaluations_to_convergence": None,
            "time_sec_total": None,
            "n_gen_completed": None,
        })
    return rows


def collect_rows(main5_dir: Path, llm_base_dir: Path) -> pd.DataFrame:
    results_dir = main5_dir / "results"
    rows = []
    for seed in SEEDS:
        candidates = {
            "Random Latent Search": results_dir / f"random_latent_formal_seed{seed}" / "summary_decode_aware.json",
            "ZINC-Seeded Latent GA": results_dir / f"train_random_zinc_seeded_formal_seed{seed}" / "summary_decode_aware.json",
            "LLM-Initialized Latent GA": results_dir / f"llm_llm_initialized_formal_seed{seed}" / "summary_decode_aware.json",
            "Iterative LLM-Guided Latent GA": results_dir / f"llm_iterative_llm_guided_formal_seed{seed}" / "summary_decode_aware.json",
        }
        for method, path in candidates.items():
            if path.exists():
                rows.append(row_from_summary(method, seed, path))
    rows.extend(direct_llm_rows(llm_base_dir))

    df = pd.DataFrame(rows)
    if len(df):
        df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
        df = df.sort_values(["method", "seed"]).reset_index(drop=True)
        df["method"] = df["method"].astype(str)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_rdkit_abs_error",
        "top10_rdkit_abs_error_mean",
        "success_rate",
        "archive_rdkit_success_unique",
        "archive_unique_valid_molecules",
        "decode_latent_validity_final",
        "diversity_unique_valid",
        "evaluations_total",
        "evaluations_to_convergence",
        "time_sec_total",
        "n_gen_completed",
    ]
    rows = []
    for method in METHOD_ORDER:
        group = df[df["method"] == method]
        if len(group) == 0:
            continue
        out = {"method": method, "n_runs": int(group["seed"].nunique())}
        for metric in metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            if len(values) == 0:
                continue
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main5_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5"))
    parser.add_argument("--llm_base_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM"))
    args = parser.parse_args()

    out_dir = args.main5_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = collect_rows(args.main5_dir, args.llm_base_dir)
    mean_std = summarize(detail)

    detail_path = out_dir / "zinc_logp_main5_formal_detail.csv"
    mean_std_path = out_dir / "zinc_logp_main5_formal_mean_std.csv"
    word_path = out_dir / "zinc_logp_main5_formal_word_table.csv"

    detail.to_csv(detail_path, index=False)
    mean_std.to_csv(mean_std_path, index=False)

    compact = mean_std.copy()
    rename = {
        "best_rdkit_abs_error_mean": "Best logP error mean",
        "best_rdkit_abs_error_std": "Best logP error std",
        "top10_rdkit_abs_error_mean_mean": "Top-10 logP error mean",
        "top10_rdkit_abs_error_mean_std": "Top-10 logP error std",
        "success_rate_mean": "Success rate mean",
        "success_rate_std": "Success rate std",
        "archive_rdkit_success_unique_mean": "Success count mean",
        "diversity_unique_valid_mean": "Diversity mean",
        "evaluations_to_convergence_mean": "Evaluations to convergence mean",
        "time_sec_total_mean": "Time mean (s)",
    }
    compact = compact.rename(columns=rename)
    compact.to_csv(word_path, index=False)

    print(mean_std.to_string(index=False) if len(mean_std) else "No completed rows found.")
    print(f"saved {detail_path}")
    print(f"saved {mean_std_path}")
    print(f"saved {word_path}")


if __name__ == "__main__":
    main()
