#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import pandas as pd


def calc_success_rate(final_csv_path: str, threshold: float, gap_col: str):
    df = pd.read_csv(final_csv_path)

    if gap_col not in df.columns:
        raise ValueError(
            f"[final csv] 列 '{gap_col}' 不存在，可用列: {list(df.columns)}"
        )

    gap = pd.to_numeric(df[gap_col], errors="coerce")
    gap = gap.dropna()

    total = len(gap)
    success_mask = gap < threshold
    success_count = int(success_mask.sum())
    success_rate = float(success_count / total) if total > 0 else math.nan

    return {
        "total_samples": total,
        "success_count": success_count,
        "success_rate": success_rate,
    }


def calc_eval_at_threshold(progress_csv_path: str, threshold: float,
                           eval_col: str, best_gap_col: str):
    df = pd.read_csv(progress_csv_path)

    if eval_col not in df.columns:
        raise ValueError(
            f"[progress csv] 列 '{eval_col}' 不存在，可用列: {list(df.columns)}"
        )
    if best_gap_col not in df.columns:
        raise ValueError(
            f"[progress csv] 列 '{best_gap_col}' 不存在，可用列: {list(df.columns)}"
        )

    evals = pd.to_numeric(df[eval_col], errors="coerce")
    best_gap = pd.to_numeric(df[best_gap_col], errors="coerce")

    valid = (~evals.isna()) & (~best_gap.isna())
    df2 = pd.DataFrame({
        "evaluations": evals[valid].astype(int),
        "best_gap_so_far": best_gap[valid].astype(float),
    }).sort_values("evaluations").reset_index(drop=True)

    hit = df2[df2["best_gap_so_far"] < threshold]

    if len(hit) == 0:
        return {
            "eval_at_threshold": None,
            "reached": False,
        }

    first_row = hit.iloc[0]
    return {
        "eval_at_threshold": int(first_row["evaluations"]),
        "reached": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="同时计算 SR@threshold 和 Eval@threshold"
    )
    parser.add_argument(
        "--final_csv", type=str, required=True,
        help="最终种群文件，如 final_population_random.csv"
    )
    parser.add_argument(
        "--progress_csv", type=str, required=True,
        help="过程文件，如 progress_metrics.csv"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="阈值，例如 0.15 / 0.03 / 0.02 / 0.015"
    )
    parser.add_argument(
        "--final_gap_col", type=str, default="pred_gap",
        help="final csv 里的 gap 列名，默认 pred_gap"
    )
    parser.add_argument(
        "--progress_eval_col", type=str, default="evaluations",
        help="progress csv 里的 evaluations 列名，默认 evaluations"
    )
    parser.add_argument(
        "--progress_best_gap_col", type=str, default="best_gap_so_far",
        help="progress csv 里的 best gap 列名，默认 best_gap_so_far"
    )

    args = parser.parse_args()

    sr_result = calc_success_rate(
        final_csv_path=args.final_csv,
        threshold=args.threshold,
        gap_col=args.final_gap_col
    )

    eval_result = calc_eval_at_threshold(
        progress_csv_path=args.progress_csv,
        threshold=args.threshold,
        eval_col=args.progress_eval_col,
        best_gap_col=args.progress_best_gap_col
    )

    print("========== RESULT ==========")
    print(f"Threshold: {args.threshold}")
    print("")
    print("[SR]")
    print(f"Total samples: {sr_result['total_samples']}")
    print(f"Success count: {sr_result['success_count']}")
    print(f"Success rate: {sr_result['success_rate']:.6f} ({sr_result['success_rate']*100:.4f}%)")
    print("")
    print("[Eval]")
    if eval_result["reached"]:
        print(f"Eval@{args.threshold}: {eval_result['eval_at_threshold']}")
    else:
        print(f"Eval@{args.threshold}: NOT REACHED")


if __name__ == "__main__":
    main()