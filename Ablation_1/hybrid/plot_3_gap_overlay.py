#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_avg_gap_history(summary_path):
    print(f"[INFO] 读取: {summary_path}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"文件不存在: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "avg_gap_history" not in data:
        raise KeyError(f"{summary_path} 中不存在字段 'avg_gap_history'")

    hist = data["avg_gap_history"]
    if not isinstance(hist, list):
        raise TypeError(f"{summary_path} 中 'avg_gap_history' 不是 list")

    hist = np.array([float(x) for x in hist], dtype=np.float32)
    print(f"[INFO] 长度: {len(hist)}")
    return hist


def moving_average(x, window=1):
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < window:
        return x.copy()

    kernel = np.ones(window, dtype=np.float32) / window
    y = np.convolve(x, kernel, mode="valid")

    # 为了保持和原始长度一致，前面补齐
    pad_len = len(x) - len(y)
    if pad_len > 0:
        pad = np.full(pad_len, y[0], dtype=np.float32)
        y = np.concatenate([pad, y], axis=0)
    return y


def truncate_to_min_length(arr_list):
    min_len = min(len(x) for x in arr_list)
    return [x[:min_len] for x in arr_list], min_len


def summarize_group(paths, smooth_window=1):
    curves = [load_avg_gap_history(p) for p in paths]
    if smooth_window > 1:
        curves = [moving_average(c, smooth_window) for c in curves]

    # 不再截断
    return curves

color_psvae = "#4c72b0"   # 黑
color_llm = "#dd8452"     # 橙
color_hybrid = "#55a868"  # 绿

def plot_overlay_with_std(
    psvae_paths,
    llm_paths,
    hybrid_paths,
    out_path,
    title,
    smooth_window=1,
):
    print("[INFO] 开始汇总三组方法...")

    psvae_curves = summarize_group(psvae_paths, smooth_window)
    llm_curves = summarize_group(llm_paths, smooth_window)
    hybrid_curves = summarize_group(hybrid_paths, smooth_window)

    print(f"[INFO] 开始绘图，PSVAE长度={len(psvae_curves)}, LLM长度={len(llm_curves)}, Hybrid长度={len(hybrid_curves)}")


    plt.figure(figsize=(8, 5))
    
    # ===== QM9 =====
    for i, curve in enumerate(psvae_curves):
        x = np.arange(len(curve))
        plt.plot(
            x,
            curve,
            color=color_psvae,
            linewidth=1.5,
            alpha=0.9 if i == 0 else 0.5,
            label="QM9-Seeded" if i == 0 else None,
        )

    # QM9 阴影（需要对齐长度 → 只能取 min length）
    min_len = min(len(c) for c in psvae_curves)
    psvae_stack = np.stack([c[:min_len] for c in psvae_curves])
    plt.fill_between(
        np.arange(min_len),
        psvae_stack.min(axis=0),
        psvae_stack.max(axis=0),
        color=color_psvae,
        alpha=0.15,
    )


    # ===== LLM =====
    for i, curve in enumerate(llm_curves):
        x = np.arange(len(curve))
        plt.plot(
            x,
            curve,
            color=color_llm,
            linewidth=1.5,
            alpha=0.9 if i == 0 else 0.5,
            label="LLM-Seeded" if i == 0 else None,
        )

    min_len = min(len(c) for c in llm_curves)
    llm_stack = np.stack([c[:min_len] for c in llm_curves])
    plt.fill_between(
        np.arange(min_len),
        llm_stack.min(axis=0),
        llm_stack.max(axis=0),
        color=color_llm,
        alpha=0.15,
    )


    # ===== Hybrid =====
    for i, curve in enumerate(hybrid_curves):
        x = np.arange(len(curve))
        plt.plot(
            x,
            curve,
            color=color_hybrid,
            linewidth=1.5,
            alpha=0.9 if i == 0 else 0.5,
            label="Hybrid-Seeded" if i == 0 else None,
        )

    min_len = min(len(c) for c in hybrid_curves)
    hybrid_stack = np.stack([c[:min_len] for c in hybrid_curves])
    plt.fill_between(
        np.arange(min_len),
        hybrid_stack.min(axis=0),
        hybrid_stack.max(axis=0),
        color=color_hybrid,
        alpha=0.15,
    )   



    plt.xlabel("Generation")
    plt.ylabel("Average Predicted Gap")
    plt.title(title)
    plt.xlim(0, 999)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] 图片已保存到: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot paper-style overlay of avg_gap_history with optional smoothing and std shading."
    )

    parser.add_argument(
        "--psvae_summaries",
        type=str,
        nargs="+",
        required=True,
        help="PS-VAE 的一个或多个 summary.json 路径",
    )
    parser.add_argument(
        "--llm_summaries",
        type=str,
        nargs="+",
        required=True,
        help="LLM 的一个或多个 summary.json 路径",
    )
    parser.add_argument(
        "--hybrid_summaries",
        type=str,
        nargs="+",
        required=True,
        help="Hybrid 的一个或多个 summary.json 路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出图片路径",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Average Predicted Gap Curves Across Initialization Strategies",
        help="图片标题",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="滑动平均窗口大小，1 表示不平滑，推荐 3 或 5",
    )

    args = parser.parse_args()

    print("[INFO] 参数解析完成")
    plot_overlay_with_std(
        psvae_paths=args.psvae_summaries,
        llm_paths=args.llm_summaries,
        hybrid_paths=args.hybrid_summaries,
        out_path=args.out,
        title=args.title,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()