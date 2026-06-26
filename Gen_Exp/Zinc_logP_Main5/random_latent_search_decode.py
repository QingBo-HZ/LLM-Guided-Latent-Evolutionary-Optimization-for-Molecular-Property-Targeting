#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Random latent search baseline for ZINC logP transfer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
GA_BASE_PATH = ROOT / "Gen_Exp" / "03_optimize_logp_latent_ga.py"
spec = importlib.util.spec_from_file_location("zinc_logp_ga_base", str(GA_BASE_PATH))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def decode_one(model_psvae, z, device, args):
    failures = 0
    for attempt in range(1, args.decode_attempts_per_latent + 1):
        smi = base.latent_to_smiles(
            model_psvae,
            z,
            device=device,
            max_atom_num=args.max_atom_num,
            add_edge_th=args.add_edge_th,
            temperature=args.temperature,
        )
        if smi is None:
            failures += 1
            continue
        rdkit_logp = base.calc_rdkit_logp(smi)
        if rdkit_logp is None:
            failures += 1
            continue
        return smi, rdkit_logp, attempt, failures
    return None, None, args.decode_attempts_per_latent, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zinc_psvae_ckpt", default=str(ROOT / "QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt"))
    parser.add_argument("--predictor_ckpt", default=str(ROOT / "Gen_Exp/Zinc_logP_kek/logp_predictor/best_logp_predictor.pt"))
    parser.add_argument("--latent_pool", default=str(ROOT / "Gen_Exp/Zinc_logP_kek/train/zinc_logp_latent.npy"))
    parser.add_argument("--output_root", default=str(ROOT / "Gen_Exp/Zinc_logP_Main5/results"))
    parser.add_argument("--version", default="random_latent_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--target_logp", type=float, default=3.0)
    parser.add_argument("--success_low", type=float, default=2.5)
    parser.add_argument("--success_high", type=float, default=3.5)
    parser.add_argument("--score_sigma", type=float, default=0.5)
    parser.add_argument("--convergence_abs_error", type=float, default=0.05)
    parser.add_argument("--max_atom_num", type=int, default=80)
    parser.add_argument("--add_edge_th", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=0.30)
    parser.add_argument("--decode_attempts_per_latent", type=int, default=3)
    args = parser.parse_args()

    base.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.time()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_root) / f"random_latent_{args.version}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_train = base.safe_np_load(args.latent_pool, "latent_pool")
    lb = latent_train.min(axis=0).astype(np.float32)
    ub = latent_train.max(axis=0).astype(np.float32)

    predictor = base.LogPPredictorAPI(args.predictor_ckpt, device=device)
    model_psvae = base.load_psvae(args.zinc_psvae_ckpt, device=device)

    rows = []
    progress = []
    valid_latents = 0
    attempts_total = 0
    best_abs = float("inf")
    convergence_eval = None

    for start_idx in range(0, args.n_samples, args.batch_size):
        end_idx = min(args.n_samples, start_idx + args.batch_size)
        batch = rng.uniform(lb, ub, size=(end_idx - start_idx, latent_train.shape[1])).astype(np.float32)
        pred_logp = predictor.predict(batch)
        pred_abs = np.abs(pred_logp - args.target_logp)
        scores = base.score_logp(pred_logp, target_logp=args.target_logp, score_sigma=args.score_sigma)

        for j, z in enumerate(batch):
            idx = start_idx + j
            smi, rdkit_logp, attempts_used, failures = decode_one(model_psvae, z, device, args)
            attempts_total += attempts_used
            if smi is None:
                continue
            valid_latents += 1
            rdkit_abs = abs(rdkit_logp - args.target_logp)
            if rdkit_abs < best_abs:
                best_abs = rdkit_abs
            if convergence_eval is None and rdkit_abs <= args.convergence_abs_error:
                convergence_eval = idx + 1
            rows.append({
                "idx": idx,
                "smiles": smi,
                "pred_logP": float(pred_logp[j]),
                "pred_abs_error": float(pred_abs[j]),
                "score": float(scores[j]),
                "rdkit_logP": float(rdkit_logp),
                "rdkit_abs_error": float(rdkit_abs),
                "rdkit_success": bool(args.success_low <= rdkit_logp <= args.success_high),
                "decode_attempts_used": int(attempts_used),
                "decode_failures_before_success": int(failures),
            })

        if rows:
            df_tmp = pd.DataFrame(rows).drop_duplicates("smiles")
            top10 = df_tmp.sort_values("rdkit_abs_error").head(min(10, len(df_tmp)))
            progress.append({
                "evaluations": end_idx,
                "valid_latent_count": valid_latents,
                "decode_latent_validity": valid_latents / end_idx,
                "archive_unique_valid_molecules": int(df_tmp["smiles"].nunique()),
                "archive_rdkit_success_unique": int(df_tmp[df_tmp["rdkit_success"]]["smiles"].nunique()),
                "best_rdkit_abs_error": float(df_tmp["rdkit_abs_error"].min()),
                "top10_rdkit_abs_error_mean": float(top10["rdkit_abs_error"].mean()),
            })

        print(
            f"[Random {end_idx:05d}/{args.n_samples}] valid={valid_latents}/{end_idx} "
            f"best_abs={best_abs:.4f}",
            flush=True,
        )

    archive_df = pd.DataFrame(rows)
    if len(archive_df):
        unique_df = archive_df.sort_values(["rdkit_abs_error", "idx", "smiles"]).drop_duplicates("smiles")
        unique_df = unique_df.reset_index(drop=True)
        unique_df["rank_by_rdkit_error"] = np.arange(1, len(unique_df) + 1)
        success_unique = unique_df[unique_df["rdkit_success"]]
        top10 = unique_df.head(min(10, len(unique_df)))
        mol_summary = {
            "archive_unique_valid_molecules": int(len(unique_df)),
            "archive_rdkit_success_unique": int(len(success_unique)),
            "archive_rdkit_success_rate_over_unique": float(len(success_unique) / len(unique_df)) if len(unique_df) else 0.0,
            "best_smiles_rdkit": unique_df.iloc[0]["smiles"],
            "best_rdkit_logP": float(unique_df.iloc[0]["rdkit_logP"]),
            "best_rdkit_abs_error": float(unique_df.iloc[0]["rdkit_abs_error"]),
            "top10_rdkit_abs_error_mean": float(top10["rdkit_abs_error"].mean()) if len(top10) else None,
            "diversity_unique_valid": base.compute_diversity(unique_df["smiles"].tolist()),
        }
    else:
        unique_df = pd.DataFrame()
        mol_summary = {
            "archive_unique_valid_molecules": 0,
            "archive_rdkit_success_unique": 0,
            "archive_rdkit_success_rate_over_unique": 0.0,
            "best_smiles_rdkit": None,
            "best_rdkit_logP": None,
            "best_rdkit_abs_error": None,
            "top10_rdkit_abs_error_mean": None,
            "diversity_unique_valid": 0.0,
        }

    archive_df.to_csv(out_dir / "decoded_molecule_archive.csv", index=False)
    unique_df.to_csv(out_dir / "decoded_molecule_unique_ranked.csv", index=False)
    pd.DataFrame(progress).to_csv(out_dir / "progress_metrics_random_latent.csv", index=False)

    summary = {
        "task": "zinc_logp_random_latent_search",
        "method": "Random Latent Search",
        "seed": args.seed,
        "n_samples": args.n_samples,
        "evaluations_total": args.n_samples,
        "evaluations_to_convergence": convergence_eval,
        "decode_latent_validity_final": float(valid_latents / args.n_samples) if args.n_samples else 0.0,
        "decode_success_final": int(valid_latents),
        "decode_attempts_final": int(attempts_total),
        "target_logP": args.target_logp,
        "success_low": args.success_low,
        "success_high": args.success_high,
        "convergence_abs_error": args.convergence_abs_error,
        "time_sec_total": float(time.time() - start),
        **mol_summary,
    }
    save_json(summary, out_dir / "summary_decode_aware.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
