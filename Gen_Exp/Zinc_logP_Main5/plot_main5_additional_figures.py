#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create separate B-F figures for the formal ZINC logP five-group experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


SEEDS = [42, 43, 44]
TARGET_LOGP = 3.0
SUCCESS_LOW = 2.5
SUCCESS_HIGH = 3.5

METHOD_COLORS = {
    "Random Latent Search": "#6f6f6f",
    "ZINC-Seeded Latent GA": "#2f6fbb",
    "LLM-Generated Molecules": "#9a4fb0",
    "LLM-Initialized Latent GA": "#d8872b",
    "Iterative LLM-Guided Latent GA": "#2b9a66",
}

GA_DIRS = {
    "ZINC-Seeded Latent GA": "train_random_zinc_seeded_formal_seed{seed}",
    "LLM-Initialized Latent GA": "llm_llm_initialized_formal_seed{seed}",
    "Iterative LLM-Guided Latent GA": "llm_iterative_llm_guided_formal_seed{seed}",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 160,
    })


def save_fig(fig, out_dir: Path, stem: str) -> None:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {png}")
    print(f"saved {pdf}")


def cumulative_top10_from_archive(path: Path, max_gen: int = 100) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for gen in range(max_gen):
        sub = df[(df["generation"] >= 0) & (df["generation"] <= gen)].copy()
        if len(sub) == 0:
            rows.append({"evolution": gen + 1, "top10_error": np.nan})
            continue
        sub = sub.sort_values(["rdkit_abs_error", "generation", "smiles"]).drop_duplicates("smiles")
        top10 = sub.head(min(10, len(sub)))
        rows.append({"evolution": gen + 1, "top10_error": float(top10["rdkit_abs_error"].mean())})
    return pd.DataFrame(rows)


def plot_b_top10_error(results_dir: Path, out_dir: Path) -> None:
    all_stats = []
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    random_curves = []
    for seed in SEEDS:
        p = results_dir / f"random_latent_formal_seed{seed}" / "progress_metrics_random_latent.csv"
        df = pd.read_csv(p)
        random_curves.append(pd.DataFrame({
            "evolution": (df["evaluations"] / 100.0).round().astype(int),
            "top10_error": df["top10_rdkit_abs_error_mean"],
            "seed": seed,
        }))
    for method, dir_pattern in GA_DIRS.items():
        curves = []
        for seed in SEEDS:
            p = results_dir / dir_pattern.format(seed=seed) / "decoded_molecule_archive.csv"
            c = cumulative_top10_from_archive(p)
            c["seed"] = seed
            curves.append(c)
        df = pd.concat(curves, ignore_index=True)
        stat = df.groupby("evolution")["top10_error"].agg(["mean", "std"]).reset_index().fillna(0)
        stat["method"] = method
        all_stats.append(stat)
        x = stat["evolution"].to_numpy()
        y = np.clip(stat["mean"].to_numpy(), 1e-5, None)
        sd = stat["std"].to_numpy()
        color = METHOD_COLORS[method]
        ax.plot(x, y, lw=2.4, color=color, label=method)
        ax.fill_between(x, np.clip(y - sd, 1e-5, None), y + sd, color=color, alpha=0.13, lw=0)

    df = pd.concat(random_curves, ignore_index=True)
    stat = df.groupby("evolution")["top10_error"].agg(["mean", "std"]).reset_index().fillna(0)
    stat["method"] = "Random Latent Search"
    all_stats.append(stat)
    x = stat["evolution"].to_numpy()
    y = np.clip(stat["mean"].to_numpy(), 1e-5, None)
    sd = stat["std"].to_numpy()
    color = METHOD_COLORS["Random Latent Search"]
    ax.plot(x, y, lw=2.4, color=color, label="Random Latent Search")
    ax.fill_between(x, np.clip(y - sd, 1e-5, None), y + sd, color=color, alpha=0.13, lw=0)

    detail = pd.read_csv(results_dir / "zinc_logp_main5_formal_detail.csv")
    direct = detail[detail["method"] == "LLM-Generated Molecules"]
    direct_mean = float(direct["top10_rdkit_abs_error_mean"].mean())
    direct_std = float(direct["top10_rdkit_abs_error_mean"].std(ddof=1))
    x_ref = np.arange(1, 101)
    ax.plot(x_ref, np.full_like(x_ref, direct_mean, dtype=float), lw=2.2, ls="--",
            color=METHOD_COLORS["LLM-Generated Molecules"], label="LLM-Generated Molecules")
    ax.fill_between(x_ref, max(1e-5, direct_mean - direct_std), direct_mean + direct_std,
                    color=METHOD_COLORS["LLM-Generated Molecules"], alpha=0.12, lw=0)
    all_stats.append(pd.DataFrame({
        "evolution": x_ref,
        "mean": direct_mean,
        "std": direct_std,
        "method": "LLM-Generated Molecules",
    }))

    ax.set_yscale("log")
    ax.set_xlabel("Evolution step")
    ax.set_ylabel("Top-10 RDKit abs. error to logP=3.0")
    ax.set_title("Top-10 Error Evolution on ZINC logP Transfer")
    ax.set_xlim(1, 100)
    ax.grid(alpha=0.22, which="both")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "B_top10_rdkit_error_evolution")
    pd.concat(all_stats, ignore_index=True).to_csv(out_dir / "B_top10_rdkit_error_evolution_data.csv", index=False)


def load_method_logp(results_dir: Path, llm_dir: Path, method: str) -> pd.DataFrame:
    frames = []
    if method == "LLM-Generated Molecules":
        for seed in SEEDS:
            files = sorted(llm_dir.glob(f"zinc_logp_llm_direct_*_seed{seed}_*_accepted_ranked.csv"))
            df = pd.read_csv(files[-1])
            frames.append(pd.DataFrame({"rdkit_logP": df["rdkit_logP"], "seed": seed, "method": method}))
    elif method == "Random Latent Search":
        for seed in SEEDS:
            df = pd.read_csv(results_dir / f"random_latent_formal_seed{seed}" / "decoded_molecule_unique_ranked.csv")
            frames.append(pd.DataFrame({"rdkit_logP": df["rdkit_logP"], "seed": seed, "method": method}))
    else:
        for seed in SEEDS:
            df = pd.read_csv(results_dir / GA_DIRS[method].format(seed=seed) / "decoded_molecule_unique_ranked.csv")
            frames.append(pd.DataFrame({"rdkit_logP": df["rdkit_logP"], "seed": seed, "method": method}))
    return pd.concat(frames, ignore_index=True)


def plot_c_logp_distribution(results_dir: Path, llm_dir: Path, out_dir: Path) -> None:
    methods = [
        "Random Latent Search",
        "ZINC-Seeded Latent GA",
        "LLM-Generated Molecules",
        "LLM-Initialized Latent GA",
        "Iterative LLM-Guided Latent GA",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    all_df = []
    bins = np.linspace(0, 6, 60)
    for method in methods:
        df = load_method_logp(results_dir, llm_dir, method)
        all_df.append(df)
        values = pd.to_numeric(df["rdkit_logP"], errors="coerce").dropna()
        ax.hist(values, bins=bins, histtype="step", density=True, lw=2.3,
                color=METHOD_COLORS[method], label=f"{method} (n={len(values)})")
    ax.axvspan(SUCCESS_LOW, SUCCESS_HIGH, color="#9ccf8f", alpha=0.16, label="Success range")
    ax.axvline(TARGET_LOGP, color="#1f1f1f", lw=2, ls="--", label="Target logP")
    ax.set_xlim(0, 6)
    ax.set_xlabel("RDKit Crippen MolLogP")
    ax.set_ylabel("Density")
    ax.set_title("RDKit logP Distribution of Generated Molecules")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "C_rdkit_logp_distribution")
    pd.concat(all_df, ignore_index=True).to_csv(out_dir / "C_rdkit_logp_distribution_data.csv", index=False)


def sample_rows(arr: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(arr) <= n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]


def plot_d_latent_umap(results_dir: Path, llm_latent_dir: Path, out_dir: Path) -> None:
    from umap import UMAP

    rng = np.random.default_rng(7)
    train_latent = np.load("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_kek/train/zinc_logp_latent.npy")
    bg = sample_rows(train_latent.astype(np.float32), 3000, 7)
    labels = ["ZINC train background"] * len(bg)
    arrays = [bg]

    lb = train_latent.min(axis=0)
    ub = train_latent.max(axis=0)
    random_latent = rng.uniform(lb, ub, size=(800, train_latent.shape[1])).astype(np.float32)
    arrays.append(random_latent)
    labels += ["Random Latent Search"] * len(random_latent)

    direct = []
    for seed in SEEDS:
        direct.append(np.load(llm_latent_dir / f"direct_seed{seed}" / "llm_init_latent.npy"))
    direct = sample_rows(np.concatenate(direct, axis=0), 1000, 11)
    arrays.append(direct)
    labels += ["LLM-Generated Molecules"] * len(direct)

    for method, pattern in GA_DIRS.items():
        vals = []
        for seed in SEEDS:
            vals.append(np.load(results_dir / pattern.format(seed=seed) / "final_population_latent.npy"))
        vals = np.concatenate(vals, axis=0)
        arrays.append(vals)
        labels += [method] * len(vals)

    X = np.concatenate(arrays, axis=0).astype(np.float32)
    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.15, metric="euclidean", random_state=42)
    emb = reducer.fit_transform(X)
    df = pd.DataFrame({"umap1": emb[:, 0], "umap2": emb[:, 1], "method": labels})
    df.to_csv(out_dir / "D_latent_umap_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    bg_df = df[df["method"] == "ZINC train background"]
    ax.scatter(bg_df["umap1"], bg_df["umap2"], s=4, c="#d3d3d3", alpha=0.28, label="ZINC train background", linewidths=0)
    for method in [
        "Random Latent Search",
        "LLM-Generated Molecules",
        "ZINC-Seeded Latent GA",
        "LLM-Initialized Latent GA",
        "Iterative LLM-Guided Latent GA",
    ]:
        sub = df[df["method"] == method]
        ax.scatter(sub["umap1"], sub["umap2"], s=16, alpha=0.72,
                   c=METHOD_COLORS[method], label=method, linewidths=0)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Latent Space UMAP of Search Regions")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=8, markerscale=1.2)
    fig.tight_layout()
    save_fig(fig, out_dir, "D_latent_space_umap")


def _mol_to_svg_inner(mol, legend: str, width: int = 360, height: int = 260) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.legendFontSize = 18
    opts.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    first = svg.find(">")
    last = svg.rfind("</svg>")
    return svg[first + 1:last]


def plot_e_molecule_path(results_dir: Path, out_dir: Path) -> None:
    archive = pd.read_csv(results_dir / "llm_iterative_llm_guided_formal_seed43" / "decoded_molecule_archive.csv")
    windows = [(0, 9), (10, 29), (30, 49), (50, 69), (70, 89), (90, 99)]
    selected = []
    used = set()
    for lo, hi in windows:
        sub = archive[(archive["generation"] >= lo) & (archive["generation"] <= hi)].copy()
        sub = sub.sort_values(["rdkit_abs_error", "generation", "smiles"])
        row = None
        for _, cand in sub.iterrows():
            if cand["smiles"] not in used:
                row = cand
                used.add(cand["smiles"])
                break
        if row is not None:
            selected.append(row)

    cell_w, cell_h = 360, 260
    n_cols = 3
    n_rows = int(np.ceil(len(selected) / n_cols))
    width = cell_w * n_cols
    height = cell_h * n_rows + 54
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="18" y="32" font-family="DejaVu Sans, Arial" font-size="24" font-weight="600">Representative Molecular Optimization Path</text>',
    ]
    for i, row in enumerate(selected):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        col = i % n_cols
        r = i // n_cols
        x = col * cell_w
        y = 54 + r * cell_h
        legend = f"Gen {int(row['generation'])} | logP={row['rdkit_logP']:.3f} | err={row['rdkit_abs_error']:.4f}"
        inner = _mol_to_svg_inner(mol, legend, cell_w, cell_h)
        parts.append(f'<g transform="translate({x},{y})">{inner}</g>')
    parts.append('</svg>')

    svg_path = out_dir / "E_representative_molecule_path.svg"
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    pd.DataFrame(selected).to_csv(out_dir / "E_representative_molecule_path_data.csv", index=False)
    print(f"saved {svg_path}")

def plot_f_final_summary(results_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_dir / "zinc_logp_main5_formal_mean_std.csv")
    methods = [
        "Random Latent Search",
        "ZINC-Seeded Latent GA",
        "LLM-Generated Molecules",
        "LLM-Initialized Latent GA",
        "Iterative LLM-Guided Latent GA",
    ]
    df = df.set_index("method").loc[methods].reset_index()
    y = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]

    fig, (ax_success, ax_error) = plt.subplots(
        1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    success = df["archive_rdkit_success_unique_mean"].to_numpy()
    success_std = df["archive_rdkit_success_unique_std"].fillna(0).to_numpy()
    ax_success.barh(y, success, xerr=success_std, color=colors, alpha=0.9, capsize=3, height=0.58)
    ax_success.set_yticks(y)
    ax_success.set_yticklabels(methods)
    ax_success.invert_yaxis()
    ax_success.set_xlabel("RDKit-success molecule count")
    ax_success.set_title("Success enrichment")
    ax_success.grid(axis="x", alpha=0.18)
    ax_success.spines["left"].set_visible(False)
    for yi, val in zip(y, success):
        ax_success.text(val + max(success) * 0.025, yi, f"{val:.0f}", va="center", ha="left", fontsize=9)

    top10 = df["top10_rdkit_abs_error_mean_mean"].to_numpy() * 1000.0
    top10_std = df["top10_rdkit_abs_error_mean_std"].fillna(0).to_numpy() * 1000.0
    ax_error.barh(y, top10, xerr=top10_std, color=colors, alpha=0.9, capsize=3, height=0.58)
    ax_error.set_yticks(y)
    ax_error.set_yticklabels([])
    ax_error.invert_yaxis()
    ax_error.set_xlabel(r"Top-10 RDKit error ($\times 10^{-3}$)")
    ax_error.set_title("Optimization accuracy")
    ax_error.grid(axis="x", alpha=0.18)
    ax_error.spines["left"].set_visible(False)
    for yi, val in zip(y, top10):
        ax_error.text(val + max(top10) * 0.035, yi, f"{val:.2f}", va="center", ha="left", fontsize=9)

    ax_success.set_xlim(0, max(success + success_std) * 1.23)
    ax_error.set_xlim(0, max(top10 + top10_std) * 1.25)
    fig.suptitle("Final Performance Summary", y=0.99, fontsize=15)
    fig.tight_layout()
    save_fig(fig, out_dir, "F_final_performance_summary")
    df.to_csv(out_dir / "F_final_performance_summary_data.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/results"))
    parser.add_argument("--llm_smiles_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM/smiles"))
    parser.add_argument("--llm_latent_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_LLM/latent"))
    parser.add_argument("--out_dir", type=Path, default=Path("/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP_Main5/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_b_top10_error(args.results_dir, args.out_dir)
    plot_c_logp_distribution(args.results_dir, args.llm_smiles_dir, args.out_dir)
    plot_d_latent_umap(args.results_dir, args.llm_latent_dir, args.out_dir)
    plot_e_molecule_path(args.results_dir, args.out_dir)
    plot_f_final_summary(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
