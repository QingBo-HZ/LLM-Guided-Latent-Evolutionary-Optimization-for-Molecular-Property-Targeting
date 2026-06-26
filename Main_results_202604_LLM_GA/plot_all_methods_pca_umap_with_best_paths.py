#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ============================================================
# Environment
# ============================================================

os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"


# ============================================================
# Paths
# ============================================================

TRAIN_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy"

METHODS = [
    {
        "key": "random",
        "name": "Random Latent Search",
        "path": "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/1_random_search/random_search_random_search_V2",
    },
    {
        "key": "brics",
        "name": "BRICS-based SMILES GA",
        "path": "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA/fragment_ga_smiles_childselect_v1",
    },
    {
        "key": "latent_ga",
        "name": "Latent GA",
        "path": "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/3_latent_GA_noLLM/psvae_train_random_V2",
    },
    {
        "key": "llm_init",
        "name": "LLM-Initialized Latent GA",
        "path": "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/4_latent_GA_LLM/llm_llm_V2",
    },
    {
        "key": "ours",
        "name": "Iterative LLM-Guided Latent GA",
        "path": "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/5_Ours/llm_ours_V2",
    },
]

DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/fig2_space_all_methods"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_existing_file(method_dir, candidates):
    for name in candidates:
        path = os.path.join(method_dir, name)
        if os.path.exists(path):
            return path
    return None


def find_final_latent_file(method_dir):
    """
    优先读取最终种群 latent 文件。
    不通过 SMILES 重新编码。
    """
    candidates = [
        "final_population_latent.npy",
        "final_population_latents.npy",
        "final_latents.npy",
        "generated_latents.npy",
        "population_latents.npy",
        "final_population.npy",
    ]

    path = find_existing_file(method_dir, candidates)
    if path is not None:
        return path

    npy_files = [
        os.path.join(method_dir, f)
        for f in os.listdir(method_dir)
        if f.endswith(".npy")
    ]

    priority_keywords = [
        ("final", "population", "latent"),
        ("final", "latent"),
        ("population", "latent"),
    ]

    for keys in priority_keywords:
        for p in npy_files:
            base = os.path.basename(p).lower()
            if all(k in base for k in keys):
                return p

    return None


def load_latents_from_npy(npy_path, max_n=None):
    """
    读取 latent，并在数量过多时随机抽样。
    返回：
    - arr: 抽样后的 latent
    - n_raw: 原始有效 latent 数量
    """
    arr = np.load(npy_path).astype(np.float32)

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim != 2:
        raise ValueError(
            f"Latent array must be 2D, got shape={arr.shape} from {npy_path}"
        )

    finite_mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite_mask]

    n_raw = len(arr)

    if n_raw == 0:
        raise ValueError(f"No finite latent vectors found in {npy_path}")

    if max_n is not None and n_raw > max_n:
        idx = np.random.choice(n_raw, max_n, replace=False)
        arr = arr[idx]

    return arr.astype(np.float32), n_raw


def get_method_style():
    return {
        "random": {
            "color": "C0",
            "marker": "o",
            "name": "Random Latent Search",
        },
        "brics": {
            "color": "C2",
            "marker": "^",
            "name": "BRICS-based SMILES GA",
        },
        "latent_ga": {
            "color": "C1",
            "marker": "D",
            "name": "Latent GA",
        },
        "llm_init": {
            "color": "C3",
            "marker": "v",
            "name": "LLM-Initialized Latent GA",
        },
        "ours": {
            "color": "C4",
            "marker": "*",
            "name": "Iterative LLM-Guided Latent GA",
        },
    }


def set_axis_margin(ax, x, y, margin=0.05):
    """
    根据当前所有可视化点自动压缩坐标轴边缘空白。
    不改变数据，只减少外侧留白。
    """
    x = np.asarray(x)
    y = np.asarray(y)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) == 0 or len(y) == 0:
        return

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    x_range = xmax - xmin
    y_range = ymax - ymin

    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    xpad = x_range * margin
    ypad = y_range * margin

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        "Combined PCA/UMAP for five molecular optimization methods using final latent populations"
    )

    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_latent_path", type=str, default=TRAIN_LATENT_PATH)

    parser.add_argument("--seed", type=int, default=42)

    # 为了可视化公平性，训练集背景和每组方法都抽样
    parser.add_argument("--n_train_vis", type=int, default=2000)
    parser.add_argument("--max_final_per_method", type=int, default=100)

    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.15)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    print("\n========== CONFIG ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"[INFO] Output dir = {args.output_dir}")

    # ========================================================
    # Load training latent background
    # ========================================================

    print("\n[INFO] Loading training latent background...")
    latent_train = np.load(args.train_latent_path).astype(np.float32)

    if latent_train.ndim != 2:
        raise ValueError(f"Training latent must be 2D, got shape={latent_train.shape}")

    print(f"[INFO] latent_train shape = {latent_train.shape}")

    n_train_vis = min(args.n_train_vis, len(latent_train))
    train_idx = np.random.choice(len(latent_train), n_train_vis, replace=False)
    train_vis = latent_train[train_idx].astype(np.float32)

    latent_dim = train_vis.shape[1]

    print(f"[INFO] latent_dim = {latent_dim}")
    print(f"[INFO] train_vis shape = {train_vis.shape}")

    # ========================================================
    # Load final latent populations
    # ========================================================

    all_final_latents = []
    all_final_method = []
    all_final_method_key = []
    all_latent_index = []

    summary_rows = []

    for m in METHODS:
        method_key = m["key"]
        method_name = m["name"]
        method_dir = m["path"]

        print("\n" + "=" * 80)
        print(f"[INFO] Method: {method_name}")
        print(f"[INFO] Directory: {method_dir}")

        if not os.path.exists(method_dir):
            print(f"[WARN] Directory not found: {method_dir}")
            summary_rows.append({
                "method": method_name,
                "method_key": method_key,
                "status": "missing_dir",
                "final_latent_file": "",
                "n_raw_latent": 0,
                "n_final_latent": 0,
                "latent_dim": "",
            })
            continue

        final_latent_file = find_final_latent_file(method_dir)

        if final_latent_file is None:
            print(f"[WARN] Final latent file not found for {method_name}")
            summary_rows.append({
                "method": method_name,
                "method_key": method_key,
                "status": "missing_final_latent",
                "final_latent_file": "",
                "n_raw_latent": 0,
                "n_final_latent": 0,
                "latent_dim": "",
            })
            continue

        print(f"[INFO] Final latent file: {final_latent_file}")

        latents, n_raw_latent = load_latents_from_npy(
            final_latent_file,
            max_n=args.max_final_per_method,
        )

        if latents.shape[1] != latent_dim:
            raise ValueError(
                f"Latent dimension mismatch for {method_name}: "
                f"{latents.shape[1]} vs training latent_dim {latent_dim}. "
                f"File: {final_latent_file}"
            )

        print(f"[INFO] Raw final latents: {n_raw_latent}")
        print(f"[INFO] Used final latents: {latents.shape}")

        all_final_latents.append(latents)
        all_final_method.extend([method_name] * len(latents))
        all_final_method_key.extend([method_key] * len(latents))
        all_latent_index.extend(list(range(len(latents))))

        # 单独保存每组抽样后的 latent，便于后续检查
        np.save(
            os.path.join(args.output_dir, f"{method_key}_final_latents.npy"),
            latents,
        )

        summary_rows.append({
            "method": method_name,
            "method_key": method_key,
            "status": "ok",
            "final_latent_file": final_latent_file,
            "n_raw_latent": int(n_raw_latent),
            "n_final_latent": int(len(latents)),
            "latent_dim": int(latents.shape[1]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "encoding_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n========== LATENT LOADING SUMMARY ==========")
    print(summary_df)
    print(f"[INFO] Saved summary: {summary_path}")

    if len(all_final_latents) == 0:
        raise RuntimeError("No final latent populations were loaded. Check input directories.")

    final_latents = np.vstack(all_final_latents).astype(np.float32)

    print(f"[INFO] final_latents shape = {final_latents.shape}")

    method_n = {
        row["method_key"]: int(row["n_final_latent"])
        for _, row in summary_df.iterrows()
        if row["status"] == "ok"
    }

    method_raw_n = {
        row["method_key"]: int(row["n_raw_latent"])
        for _, row in summary_df.iterrows()
        if row["status"] == "ok"
    }

    # ========================================================
    # Combined matrix
    # ========================================================

    X_all = np.vstack([train_vis, final_latents]).astype(np.float32)

    n_train = len(train_vis)
    n_final = len(final_latents)

    labels = (
        ["Training set"] * n_train
        + ["Final population"] * n_final
    )

    methods = (
        ["QM9 training set"] * n_train
        + all_final_method
    )

    method_keys = (
        ["train"] * n_train
        + all_final_method_key
    )

    latent_indices = (
        [-1] * n_train
        + all_latent_index
    )

    # ========================================================
    # PCA
    # ========================================================

    print("\n[INFO] Running PCA...")
    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = pca.fit_transform(X_all)

    pca_df = pd.DataFrame({
        "x": pca_coords[:, 0],
        "y": pca_coords[:, 1],
        "label": labels,
        "method": methods,
        "method_key": method_keys,
        "latent_index": latent_indices,
    })

    pca_csv = os.path.join(args.output_dir, "all_methods_chemical_space_pca.csv")
    pca_df.to_csv(pca_csv, index=False)

    print(f"[INFO] Saved PCA CSV: {pca_csv}")
    print(f"[INFO] PCA explained variance ratio = {pca.explained_variance_ratio_}")

    # ========================================================
    # UMAP
    # ========================================================

    if not HAS_UMAP:
        print("[WARN] umap-learn is not installed. UMAP skipped.")
        print("Install with: pip install umap-learn")
        return

    print("\n[INFO] Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
    )

    umap_coords = reducer.fit_transform(X_all)

    umap_df = pd.DataFrame({
        "x": umap_coords[:, 0],
        "y": umap_coords[:, 1],
        "label": labels,
        "method": methods,
        "method_key": method_keys,
        "latent_index": latent_indices,
    })

    umap_csv = os.path.join(args.output_dir, "all_methods_chemical_space_umap.csv")
    umap_df.to_csv(umap_csv, index=False)

    print(f"[INFO] Saved UMAP CSV: {umap_csv}")

    # ========================================================
    # Plot settings
    # ========================================================

    style_map = get_method_style()

    # ========================================================
    # Plot PCA: final latent populations
    # ========================================================

    print("\n[INFO] Plotting PCA...")

    plt.figure(figsize=(7.6, 5.8))

    train_mask = pca_df["method_key"].values == "train"
    plt.scatter(
        pca_df.loc[train_mask, "x"],
        pca_df.loc[train_mask, "y"],
        s=10,
        alpha=0.22,
        c="0.70",
        label="QM9 training set",
        edgecolors="none",
        zorder=1,
    )

    for m in METHODS:
        mk = m["key"]
        style = style_map[mk]

        mask = pca_df["method_key"].values == mk
        if np.sum(mask) == 0:
            continue

        if mk == "random":
            point_alpha = 0.35
            point_size = 22
        elif mk == "brics":
            point_alpha = 0.95
            point_size = 90
        elif mk == "ours":
            point_alpha = 0.95
            point_size = 85
        else:
            point_alpha = 0.80
            point_size = 50

        plt.scatter(
            pca_df.loc[mask, "x"],
            pca_df.loc[mask, "y"],
            s=point_size,
            alpha=point_alpha,
            c=style["color"],
            marker=style["marker"],
            label=style["name"],
            edgecolors="white",
            linewidths=0.4,
            zorder=5 if mk == "ours" else 3,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection of final latent populations")
    plt.legend(frameon=False, fontsize=8.5, loc="upper right")
    plt.grid(True, alpha=0.22)

    ax = plt.gca()
    set_axis_margin(ax, pca_df["x"].values, pca_df["y"].values, margin=0.05)

    plt.tight_layout()

    # 文件名不变
    pca_png = os.path.join(args.output_dir, "fig2c_all_methods_pca.png")
    plt.savefig(pca_png, dpi=600)
    plt.close()

    print(f"[INFO] Saved PCA figure: {pca_png}")

    # ========================================================
    # Plot UMAP: final latent populations only
    # ========================================================

    print("\n[INFO] Plotting UMAP final latent populations...")

    plt.figure(figsize=(7.6, 5.8))

    train_mask = umap_df["method_key"].values == "train"
    plt.scatter(
        umap_df.loc[train_mask, "x"],
        umap_df.loc[train_mask, "y"],
        s=10,
        alpha=0.22,
        c="0.70",
        label="QM9 training set",
        edgecolors="none",
        zorder=1,
    )

    for m in METHODS:
        mk = m["key"]
        style = style_map[mk]

        mask = umap_df["method_key"].values == mk
        if np.sum(mask) == 0:
            continue

        if mk == "random":
            point_alpha = 0.45
            point_size = 26
        elif mk == "brics":
            point_alpha = 0.95
            point_size = 90
        elif mk == "ours":
            point_alpha = 0.95
            point_size = 95
        else:
            point_alpha = 0.75
            point_size = 50

        plt.scatter(
            umap_df.loc[mask, "x"],
            umap_df.loc[mask, "y"],
            s=point_size,
            alpha=point_alpha,
            c=style["color"],
            marker=style["marker"],
            label=style["name"],
            edgecolors="white",
            linewidths=0.45,
            zorder=5 if mk == "ours" else 3,
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP projection of final latent populations")
    plt.legend(frameon=False, fontsize=8.2, loc="upper right")
    plt.grid(True, alpha=0.22)

    ax = plt.gca()
    set_axis_margin(ax, umap_df["x"].values, umap_df["y"].values, margin=0.05)

    plt.tight_layout()

    # 文件名不变
    umap_png = os.path.join(args.output_dir, "fig2d_all_methods_umap_with_best_paths.png")
    plt.savefig(umap_png, dpi=600)
    plt.close()

    print(f"[INFO] Saved UMAP figure: {umap_png}")

    # ========================================================
    # Save final summary
    # ========================================================

    final_summary = {
        "output_dir": args.output_dir,
        "n_train_background": int(n_train),
        "n_final_points": int(n_final),
        "pca_csv": pca_csv,
        "umap_csv": umap_csv,
        "pca_figure": pca_png,
        "umap_figure": umap_png,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "methods": METHODS,
        "method_n_used_for_plot": method_n,
        "method_n_raw_latent": method_raw_n,
        "data_source": (
            "Directly loaded final latent population npy files. "
            "For visualization clarity, 2000 QM9 training latent vectors and "
            "at most 100 final latent vectors per method were randomly sampled."
        ),
    }

    summary_json = os.path.join(args.output_dir, "space_projection_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print("\n========== DONE ==========")
    print(f"PCA CSV:  {pca_csv}")
    print(f"UMAP CSV: {umap_csv}")
    print(f"PCA Fig:  {pca_png}")
    print(f"UMAP Fig: {umap_png}")
    print(f"Summary:  {summary_json}")

    print("\n========== FINAL VISUALIZATION COUNTS ==========")
    print(f"QM9 training background used: {n_train}")
    for m in METHODS:
        mk = m["key"]
        print(
            f"{m['name']}: used n={method_n.get(mk, 0)}, "
            f"raw n={method_raw_n.get(mk, 0)}"
        )


if __name__ == "__main__":
    main()