#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decode-aware ZINC logP latent GA.

This script keeps the old latent-level predictor optimization as a diagnostic
signal, but the paper-level result is built from decoded valid molecules and
RDKit MolLogP. It is intended as the corrected migration experiment entry point.
"""

import os
import json
import time
import argparse
import importlib.util

import numpy as np
import pandas as pd
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_GA_PATH = os.path.join(BASE_DIR, "03_optimize_logp_latent_ga.py")

spec = importlib.util.spec_from_file_location("zinc_logp_ga_base", OLD_GA_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def decode_latent_multi(model_psvae, z, device, args):
    attempts = max(1, int(args.decode_attempts_per_latent))
    failures = 0
    for attempt in range(1, attempts + 1):
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
    return None, None, attempts, failures


def add_molecule_archive_rows(
    archive,
    population,
    pred_logp,
    scores,
    abs_error,
    indices,
    generation,
    evaluations,
    source,
    model_psvae,
    device,
    args,
):
    attempted = 0
    valid_attempts_used = 0
    valid_latents = 0

    for rank, idx in enumerate(indices, start=1):
        smi, rdkit_logp, attempts_used, failures = decode_latent_multi(
            model_psvae, population[idx], device, args
        )
        attempted += int(attempts_used)

        if smi is None or rdkit_logp is None:
            continue

        valid_latents += 1
        valid_attempts_used += 1
        archive.append({
            "generation": generation,
            "evaluations": evaluations,
            "source": source,
            "rank_in_source": rank,
            "population_idx": int(idx),
            "decode_attempts_used": int(attempts_used),
            "decode_failures_before_success": int(failures),
            "smiles": smi,
            "pred_logP": float(pred_logp[idx]),
            "pred_abs_error": float(abs_error[idx]),
            "score": float(scores[idx]),
            "rdkit_logP": float(rdkit_logp),
            "rdkit_abs_error": float(abs(rdkit_logp - args.target_logp)),
            "rdkit_success": bool(args.success_low <= rdkit_logp <= args.success_high),
        })

    return attempted, valid_attempts_used, valid_latents


def build_anchor_pool(latent_train, anchor_size, seed):
    latent_train = np.asarray(latent_train, dtype=np.float32)
    if anchor_size <= 0 or anchor_size >= len(latent_train):
        return latent_train
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(latent_train), size=anchor_size, replace=False)
    return latent_train[idx].astype(np.float32)


def pull_to_manifold(z, anchors, blend, lb, ub):
    if anchors is None or len(anchors) == 0 or blend <= 0:
        return np.clip(z, lb, ub).astype(np.float32)
    diff = anchors - z[None, :]
    idx = int(np.argmin(np.sum(diff * diff, axis=1)))
    projected = (1.0 - blend) * z + blend * anchors[idx]
    return np.clip(projected, lb, ub).astype(np.float32)


def summarize_molecule_archive(archive_df, target_logp, success_low, success_high):
    if len(archive_df) == 0:
        return pd.DataFrame(), {
            "archive_valid_records": 0,
            "archive_unique_valid_molecules": 0,
            "archive_rdkit_success_unique": 0,
            "archive_rdkit_success_rate_over_unique": 0.0,
            "best_smiles_rdkit": None,
            "best_rdkit_logP": None,
            "best_rdkit_abs_error": None,
            "top10_rdkit_abs_error_mean": None,
            "top10_rdkit_logP_mean": None,
            "diversity_unique_valid": 0.0,
        }

    archive_df = archive_df.sort_values(
        ["rdkit_abs_error", "generation", "rank_in_source", "smiles"]
    ).reset_index(drop=True)
    unique_df = archive_df.drop_duplicates("smiles", keep="first").reset_index(drop=True)
    unique_df = unique_df.sort_values(
        ["rdkit_abs_error", "generation", "rank_in_source", "smiles"]
    ).reset_index(drop=True)
    unique_df["rank_by_rdkit_error"] = np.arange(1, len(unique_df) + 1)

    success_unique = unique_df[
        (unique_df["rdkit_logP"] >= success_low) &
        (unique_df["rdkit_logP"] <= success_high)
    ]
    top10 = unique_df.head(min(10, len(unique_df)))

    summary = {
        "archive_valid_records": int(len(archive_df)),
        "archive_unique_valid_molecules": int(len(unique_df)),
        "archive_rdkit_success_unique": int(len(success_unique)),
        "archive_rdkit_success_rate_over_unique": (
            float(len(success_unique) / len(unique_df)) if len(unique_df) else 0.0
        ),
        "best_smiles_rdkit": unique_df.iloc[0]["smiles"],
        "best_rdkit_logP": float(unique_df.iloc[0]["rdkit_logP"]),
        "best_rdkit_abs_error": float(unique_df.iloc[0]["rdkit_abs_error"]),
        "top10_rdkit_abs_error_mean": (
            float(top10["rdkit_abs_error"].mean()) if len(top10) else None
        ),
        "top10_rdkit_logP_mean": (
            float(top10["rdkit_logP"].mean()) if len(top10) else None
        ),
        "diversity_unique_valid": base.compute_diversity(unique_df["smiles"].tolist()),
    }

    return unique_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Decode-aware paper-grade ZINC logP latent GA"
    )

    parser.add_argument("--zinc_psvae_ckpt", type=str, default=base.DEFAULT_ZINC_PSVAE_CKPT)
    parser.add_argument("--predictor_ckpt", type=str, default=base.DEFAULT_PREDICTOR_CKPT)
    parser.add_argument("--latent_pool", type=str, default=base.DEFAULT_TRAIN_LATENT_POOL)

    parser.add_argument(
        "--init_mode",
        type=str,
        default="train_random",
        choices=["train_random", "llm", "psvae", "hybrid"],
    )
    parser.add_argument("--llm_latent_path", type=str, default=None)
    parser.add_argument("--psvae_latent_path", type=str, default=None)
    parser.add_argument("--hybrid_latent_path", type=str, default=None)
    parser.add_argument("--hybrid_sigma", type=float, default=0.2)
    parser.add_argument("--hybrid_keep_ratio", type=float, default=0.5)
    parser.add_argument("--hybrid_expand_ratio", type=float, default=0.0)

    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_gen", type=int, default=30)
    parser.add_argument("--elite_size", type=int, default=20)
    parser.add_argument("--cross_prob", type=float, default=0.30)
    parser.add_argument("--mut_prob", type=float, default=0.05)
    parser.add_argument("--mut_eta", type=float, default=20.0)
    parser.add_argument("--tourn_size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--immigrant_ratio", type=float, default=0.20)
    parser.add_argument("--manifold_anchor_size", type=int, default=5000)
    parser.add_argument("--manifold_blend", type=float, default=0.35)
    parser.add_argument("--selection_metric", choices=["pred", "rdkit_hybrid"], default="pred")
    parser.add_argument("--rdkit_selection_weight", type=float, default=0.75)

    parser.add_argument("--target_logp", type=float, default=3.0, help="RDKit molecule-level target logP")
    parser.add_argument("--pred_target_logp", type=float, default=None, help="Optional predictor-space target logP for GA scoring")
    parser.add_argument("--score_sigma", type=float, default=0.5)
    parser.add_argument("--success_low", type=float, default=2.5)
    parser.add_argument("--success_high", type=float, default=3.5)

    parser.add_argument("--max_atom_num", type=int, default=80)
    parser.add_argument("--add_edge_th", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=0.30)
    parser.add_argument("--decode_topk_per_gen", type=int, default=20)
    parser.add_argument("--decode_attempts_per_latent", type=int, default=3)

    parser.add_argument("--output_root", type=str, default=base.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--version", type=str, default="zinc_logp_decode_aware_v1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    base.set_seed(args.seed)
    start_wall_time = time.time()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    out_dir = os.path.join(args.output_root, f"{args.init_mode}_{args.version}")
    base.ensure_dir(out_dir)

    if args.pred_target_logp is None:
        args.pred_target_logp = args.target_logp

    print("\n========== DECODE-AWARE CONFIG ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"[INFO] device = {device}")
    print(f"[INFO] output_dir = {out_dir}")

    latent_train = base.safe_np_load(args.latent_pool, "latent_pool")
    latent_dim = latent_train.shape[1]
    lb = latent_train.min(axis=0).astype(np.float32)
    ub = latent_train.max(axis=0).astype(np.float32)
    anchor_pool = build_anchor_pool(latent_train, args.manifold_anchor_size, args.seed)
    print(f"[INFO] manifold anchor pool shape = {anchor_pool.shape}")

    predictor = base.LogPPredictorAPI(args.predictor_ckpt, device=device)
    if predictor.dim_feature != latent_dim:
        raise ValueError(
            f"Predictor latent dim ({predictor.dim_feature}) != latent dim ({latent_dim})"
        )

    model_psvae = base.load_psvae(args.zinc_psvae_ckpt, device=device)

    population = base.initialize_population(
        init_mode=args.init_mode,
        pop_size=args.pop_size,
        latent_train=latent_train,
        llm_latent_path=args.llm_latent_path,
        psvae_latent_path=args.psvae_latent_path,
        hybrid_latent_path=args.hybrid_latent_path,
        hybrid_sigma=args.hybrid_sigma,
        hybrid_keep_ratio=args.hybrid_keep_ratio,
        hybrid_expand_ratio=args.hybrid_expand_ratio,
        lb=lb,
        ub=ub,
        seed=args.seed,
    )
    np.save(os.path.join(out_dir, "initial_population_latent.npy"), population)

    progress_records = []
    molecule_archive = []
    best_score_monitor = -float("inf")
    no_improve_count = 0

    for gen in range(args.n_gen):
        pred_logp = predictor.predict(population)
        scores = base.score_logp(
            pred_logp,
            target_logp=args.pred_target_logp,
            score_sigma=args.score_sigma,
        )
        abs_error = np.abs(pred_logp - args.pred_target_logp)
        fitness = -scores

        best_idx = int(np.argmax(scores))
        topk = min(args.decode_topk_per_gen, len(population))
        top_idx = np.argsort(scores)[::-1][:topk]
        evaluations = int((gen + 1) * len(population))

        rdkit_scores = np.zeros(len(population), dtype=np.float32)
        rdkit_valid_mask = np.zeros(len(population), dtype=bool)
        rdkit_logp_for_selection = np.full(len(population), np.nan, dtype=np.float32)

        if args.selection_metric == "rdkit_hybrid":
            decoded_indices = np.arange(len(population))
            source_name = "generation_population"
        else:
            decoded_indices = top_idx
            source_name = "generation_topk"

        archive_start = len(molecule_archive)
        decoded_attempts_n, valid_attempts_n, valid_latents_n = add_molecule_archive_rows(
            archive=molecule_archive,
            population=population,
            pred_logp=pred_logp,
            scores=scores,
            abs_error=abs_error,
            indices=decoded_indices,
            generation=gen,
            evaluations=evaluations,
            source=source_name,
            model_psvae=model_psvae,
            device=device,
            args=args,
        )

        if args.selection_metric == "rdkit_hybrid":
            for row in molecule_archive[archive_start:]:
                pidx = int(row["population_idx"])
                rdkit_logp = float(row["rdkit_logP"])
                rdkit_logp_for_selection[pidx] = rdkit_logp
                rdkit_valid_mask[pidx] = True
                rdkit_scores[pidx] = float(np.exp(
                    -0.5 * ((rdkit_logp - args.target_logp) / args.score_sigma) ** 2
                ))
            selection_scores = (
                args.rdkit_selection_weight * rdkit_scores +
                (1.0 - args.rdkit_selection_weight) * scores
            ).astype(np.float32)
            fitness = -selection_scores
        else:
            selection_scores = scores

        pred_success = base.success_mask_logp(
            pred_logp,
            low=args.success_low,
            high=args.success_high,
        )
        archive_df_tmp = pd.DataFrame(molecule_archive)
        unique_archive_count = (
            int(archive_df_tmp["smiles"].nunique()) if len(archive_df_tmp) else 0
        )
        archive_success_unique = 0
        if len(archive_df_tmp):
            success_unique_df = archive_df_tmp[archive_df_tmp["rdkit_success"]]
            archive_success_unique = int(success_unique_df["smiles"].nunique())

        avg_score = float(np.mean(scores))
        best_score = float(scores[best_idx])
        best_pred = float(pred_logp[best_idx])
        best_pred_error = float(abs_error[best_idx])
        top10_pred_error = float(np.mean(np.sort(abs_error)[:min(10, len(abs_error))]))

        progress_records.append({
            "generation": gen,
            "evaluations": evaluations,
            "elapsed_time_sec": float(time.time() - start_wall_time),
            "avg_pred_logP": float(np.mean(pred_logp)),
            "avg_score": avg_score,
            "best_pred_logP": best_pred,
            "best_pred_abs_error": best_pred_error,
            "best_score": best_score,
            "top10_pred_abs_error": top10_pred_error,
            "latent_pred_success_count": int(np.sum(pred_success)),
            "latent_pred_success_rate": float(np.mean(pred_success)),
            "decoded_topk_attempts": int(decoded_attempts_n),
            "valid_topk_attempt_successes": int(valid_attempts_n),
            "valid_topk_latent_count": int(valid_latents_n),
            "valid_topk_attempt_rate": float(valid_attempts_n / decoded_attempts_n) if decoded_attempts_n else 0.0,
            "valid_topk_latent_rate": float(valid_latents_n / len(decoded_indices)) if len(decoded_indices) else 0.0,
            "rdkit_selection_valid_count": int(np.sum(rdkit_valid_mask)),
            "rdkit_selection_best_score": float(np.max(rdkit_scores)) if len(rdkit_scores) else 0.0,
            "rdkit_selection_best_logP": float(rdkit_logp_for_selection[np.nanargmax(rdkit_scores)]) if np.any(rdkit_valid_mask) else np.nan,
            "archive_unique_valid_molecules": unique_archive_count,
            "archive_rdkit_success_unique": archive_success_unique,
        })

        print(
            f"[Gen {gen:03d}] "
            f"pred_best={best_pred:.4f}, "
            f"pred_err={best_pred_error:.4f}, "
            f"pred_success={int(np.sum(pred_success))}/{len(pred_success)}, "
            f"decoded_valid_latent={valid_latents_n}/{len(decoded_indices)}, "
            f"archive_unique={unique_archive_count}, "
            f"archive_success={archive_success_unique}",
            flush=True,
        )

        if avg_score > best_score_monitor:
            best_score_monitor = avg_score
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= args.patience:
            print(f"[Early Stop] no avg-score improvement for {args.patience} generations.")
            break

        sorted_idx = np.argsort(fitness)
        elites = population[sorted_idx[:args.elite_size]].copy()
        new_population = list(elites)

        n_immigrants = int(round(args.pop_size * args.immigrant_ratio))
        n_immigrants = max(0, min(n_immigrants, args.pop_size - len(new_population)))
        if n_immigrants > 0:
            immigrants = base.sample_population_from_pool(latent_train, n_immigrants)
            new_population.extend([z.copy() for z in immigrants])

        while len(new_population) < args.pop_size:
            p1 = base.tournament_selection(population, fitness, tourn_size=args.tourn_size)
            p2 = base.tournament_selection(population, fitness, tourn_size=args.tourn_size)
            c1, c2 = base.arithmetic_crossover(p1, p2, args.cross_prob, lb, ub)
            c1 = base.polynomial_mutation(c1, args.mut_prob, args.mut_eta, lb, ub)
            c2 = base.polynomial_mutation(c2, args.mut_prob, args.mut_eta, lb, ub)
            c1 = pull_to_manifold(c1, anchor_pool, args.manifold_blend, lb, ub)
            c2 = pull_to_manifold(c2, anchor_pool, args.manifold_blend, lb, ub)
            new_population.append(c1)
            if len(new_population) < args.pop_size:
                new_population.append(c2)

        population = np.asarray(new_population, dtype=np.float32)

    final_pred_logp = predictor.predict(population)
    final_scores = base.score_logp(
        final_pred_logp,
        target_logp=args.pred_target_logp,
        score_sigma=args.score_sigma,
    )
    final_abs_error = np.abs(final_pred_logp - args.pred_target_logp)
    final_pred_success = base.success_mask_logp(
        final_pred_logp,
        low=args.success_low,
        high=args.success_high,
    )

    final_indices = np.arange(len(population))
    final_decoded_attempts_n, final_valid_attempts_n, final_valid_latents_n = add_molecule_archive_rows(
        archive=molecule_archive,
        population=population,
        pred_logp=final_pred_logp,
        scores=final_scores,
        abs_error=final_abs_error,
        indices=final_indices,
        generation=-1,
        evaluations=int(len(progress_records) * args.pop_size),
        source="final_population",
        model_psvae=model_psvae,
        device=device,
        args=args,
    )

    final_rows = []
    for i in range(len(population)):
        final_rows.append({
            "idx": i,
            "pred_logP": float(final_pred_logp[i]),
            "pred_abs_error": float(final_abs_error[i]),
            "score": float(final_scores[i]),
            "pred_success": bool(final_pred_success[i]),
        })
    final_df = pd.DataFrame(final_rows).sort_values(
        ["score", "pred_abs_error"], ascending=[False, True]
    ).reset_index(drop=True)
    final_df["rank_by_pred_score"] = np.arange(1, len(final_df) + 1)

    archive_df = pd.DataFrame(molecule_archive)
    if len(archive_df):
        archive_df = archive_df.sort_values(
            ["rdkit_abs_error", "generation", "rank_in_source", "smiles"]
        ).reset_index(drop=True)

    unique_df, mol_summary = summarize_molecule_archive(
        archive_df,
        target_logp=args.target_logp,
        success_low=args.success_low,
        success_high=args.success_high,
    )

    progress_df = pd.DataFrame(progress_records)
    progress_df.to_csv(os.path.join(out_dir, "progress_metrics_decode_aware.csv"), index=False)
    final_df.to_csv(os.path.join(out_dir, "final_population_pred_diagnostic.csv"), index=False)
    archive_df.to_csv(os.path.join(out_dir, "decoded_molecule_archive.csv"), index=False)
    unique_df.to_csv(os.path.join(out_dir, "decoded_molecule_unique_ranked.csv"), index=False)

    np.save(os.path.join(out_dir, "final_population_latent.npy"), population)
    np.save(os.path.join(out_dir, "final_population_pred_logp.npy"), final_pred_logp)
    np.save(os.path.join(out_dir, "final_population_score.npy"), final_scores)

    summary = {
        "task": "zinc_logp_target_optimization_decode_aware",
        "paper_grade_note": (
            "Main molecular metrics are computed from decoded valid SMILES and RDKit MolLogP. "
            "Predicted logP metrics are diagnostic only."
        ),
        "init_mode": args.init_mode,
        "version": args.version,
        "seed": args.seed,
        "zinc_psvae_ckpt": args.zinc_psvae_ckpt,
        "predictor_ckpt": args.predictor_ckpt,
        "latent_pool": args.latent_pool,
        "pop_size": args.pop_size,
        "n_gen_requested": args.n_gen,
        "n_gen_completed": int(len(progress_records)),
        "elite_size": args.elite_size,
        "cross_prob": args.cross_prob,
        "mut_prob": args.mut_prob,
        "mut_eta": args.mut_eta,
        "tourn_size": args.tourn_size,
        "patience": args.patience,
        "immigrant_ratio": args.immigrant_ratio,
        "selection_metric": args.selection_metric,
        "rdkit_selection_weight": args.rdkit_selection_weight,
        "manifold_anchor_size": args.manifold_anchor_size,
        "manifold_blend": args.manifold_blend,
        "target_logP": float(args.target_logp),
        "pred_target_logP": float(args.pred_target_logp),
        "score_sigma": float(args.score_sigma),
        "success_low": float(args.success_low),
        "success_high": float(args.success_high),
        "decode_topk_per_gen": args.decode_topk_per_gen,
        "decode_attempts_per_latent": args.decode_attempts_per_latent,
        "max_atom_num": args.max_atom_num,
        "add_edge_th": args.add_edge_th,
        "temperature": args.temperature,
        "latent_pred_success_count_final": int(np.sum(final_pred_success)),
        "latent_pred_success_rate_final": float(np.mean(final_pred_success)),
        "latent_best_pred_logP_final": float(final_pred_logp[np.argmax(final_scores)]),
        "latent_best_pred_abs_error_final": float(np.min(final_abs_error)),
        "latent_top10_pred_abs_error_final": float(
            np.mean(np.sort(final_abs_error)[:min(10, len(final_abs_error))])
        ),
        "decode_attempt_validity_raw_final": (
            float(final_valid_attempts_n / final_decoded_attempts_n) if final_decoded_attempts_n else 0.0
        ),
        "decode_latent_validity_final": (
            float(final_valid_latents_n / len(population)) if len(population) else 0.0
        ),
        "decode_success_final": int(final_valid_latents_n),
        "decode_attempt_successes_final": int(final_valid_attempts_n),
        "decode_attempts_final": int(final_decoded_attempts_n),
        "time_sec_total": float(time.time() - start_wall_time),
        **mol_summary,
    }
    summary["archive_rdkit_success_rate_over_pop_size"] = (
        float(summary["archive_rdkit_success_unique"] / args.pop_size)
        if args.pop_size else 0.0
    )

    save_json(summary, os.path.join(out_dir, "summary_decode_aware.json"))

    print("\n========== DECODE-AWARE DONE ==========")
    print(f"Output dir: {out_dir}")
    print(f"Latent predicted success final: {summary['latent_pred_success_count_final']}/{args.pop_size}")
    print(f"Final decode latent validity: {summary['decode_success_final']}/{args.pop_size}")
    print(f"Final decode attempt validity: {summary['decode_attempt_successes_final']}/{summary['decode_attempts_final']}")
    print(f"Unique valid molecules in archive: {summary['archive_unique_valid_molecules']}")
    print(f"RDKit success unique in archive: {summary['archive_rdkit_success_unique']}")
    print(f"Best RDKit SMILES: {summary['best_smiles_rdkit']}")
    print(f"Best RDKit logP: {summary['best_rdkit_logP']}")
    print(f"Best RDKit abs error: {summary['best_rdkit_abs_error']}")


if __name__ == "__main__":
    main()
