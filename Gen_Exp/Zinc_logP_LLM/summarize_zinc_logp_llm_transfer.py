#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize ZINC logP LLM transfer experiment groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "accepted_total",
    "rdkit_success_count",
    "rdkit_success_rate",
    "decode_latent_validity_final",
    "decode_success_final",
    "archive_unique_valid_molecules",
    "archive_rdkit_success_unique",
    "archive_rdkit_success_rate_over_unique",
    "best_rdkit_logP",
    "best_rdkit_abs_error",
    "top10_rdkit_abs_error_mean",
    "diversity_unique_valid",
    "latent_pred_success_rate_final",
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


def direct_rows(base_dir: Path) -> list[dict]:
    rows = []
    smiles_dir = base_dir / "smiles"
    for seed in [42, 43, 44]:
        path = latest_generation_summary(smiles_dir, "direct", seed)
        if path is None:
            continue
        data = read_json(path)
        rows.append({
            "method": "LLM-Generated Molecules",
            "seed": seed,
            "source_file": str(path),
            "accepted_total": data.get("accepted_total", 0),
            "rdkit_success_count": data.get("rdkit_success_count", 0),
            "rdkit_success_rate": data.get("rdkit_success_rate", 0.0),
            "best_smiles_rdkit": data.get("best_smiles"),
            "best_rdkit_logP": data.get("best_rdkit_logP"),
            "best_rdkit_abs_error": data.get("best_rdkit_abs_error"),
            "top10_rdkit_abs_error_mean": data.get("top10_rdkit_abs_error_mean"),
        })
    return rows


def ga_rows(base_dir: Path, pattern: str, method: str) -> list[dict]:
    rows = []
    results_dir = base_dir / "results"
    for path in sorted(results_dir.glob(pattern)):
        summary_path = path / "summary_decode_aware.json"
        if not summary_path.exists():
            continue
        data = read_json(summary_path)
        rows.append({
            "method": method,
            "seed": data.get("seed"),
            "source_file": str(summary_path),
            "accepted_total": None,
            "rdkit_success_count": None,
            "rdkit_success_rate": None,
            "best_smiles_rdkit": data.get("best_smiles_rdkit"),
            "decode_latent_validity_final": data.get("decode_latent_validity_final"),
            "decode_success_final": data.get("decode_success_final"),
            "archive_unique_valid_molecules": data.get("archive_unique_valid_molecules"),
            "archive_rdkit_success_unique": data.get("archive_rdkit_success_unique"),
            "archive_rdkit_success_rate_over_unique": data.get("archive_rdkit_success_rate_over_unique"),
            "best_rdkit_logP": data.get("best_rdkit_logP"),
            "best_rdkit_abs_error": data.get("best_rdkit_abs_error"),
            "top10_rdkit_abs_error_mean": data.get("top10_rdkit_abs_error_mean"),
            "diversity_unique_valid": data.get("diversity_unique_valid"),
            "latent_pred_success_rate_final": data.get("latent_pred_success_rate_final"),
            "n_gen_completed": data.get("n_gen_completed"),
            "time_sec_total": data.get("time_sec_total"),
        })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        out = {"method": method, "n_runs": int(group["seed"].nunique())}
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) == 0:
                continue
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM"))
    args = parser.parse_args()

    rows = []
    rows.extend(direct_rows(args.base_dir))
    rows.extend(ga_rows(args.base_dir, "llm_llm_initialized_seed*", "LLM-Initialized Latent GA"))
    rows.extend(ga_rows(args.base_dir, "llm_iterative_llm_guided_seed*", "Iterative LLM-Guided Latent GA"))

    out_dir = args.base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.DataFrame(rows)
    detail_path = out_dir / "zinc_logp_llm_transfer_3groups_detail.csv"
    mean_std_path = out_dir / "zinc_logp_llm_transfer_3groups_mean_std.csv"
    detail.to_csv(detail_path, index=False)
    summarize(detail).to_csv(mean_std_path, index=False)

    print(detail.to_string(index=False) if len(detail) else "No completed rows found.")
    print(f"saved {detail_path}")
    print(f"saved {mean_std_path}")


if __name__ == "__main__":
    main()
