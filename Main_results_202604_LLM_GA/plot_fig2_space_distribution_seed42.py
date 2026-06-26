#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from sklearn.decomposition import PCA
from umap import UMAP


ROOT = Path("/root/autodl-tmp/sweeteners_evolve")
EXP = ROOT / "Main_results_202604_LLM_GA"
OUT = EXP / "figures"
TRAIN_LATENT_PATH = ROOT / "QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy"
CKPT_PSVAE = ROOT / "QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt"
PSVAE_ROOT = ROOT / "QM9_test/PS-VAE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(str(PSVAE_ROOT / "src"))

from data.bpe_dataset import BPEMolDataset
from data.mol_bpe import Tokenizer
from pl_models import PSVAEModel
from rdkit.Chem.rdchem import BondType
from utils.chem_utils import GeneralVocab, smiles2molecule
import torch.serialization


RDLogger.DisableLog("rdApp.*")
SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


METHODS = [
    {
        "key": "random",
        "name": "Random Latent Search",
        "short": "Random Latent",
        "color": "C0",
        "result_dir": EXP / "1_random_search/random_search_random_search_V2",
        "path_csv": EXP / "1_random_search/random_search_random_search_V2/evolution_path_full.csv",
        "generation_col": "step",
    },
    {
        "key": "brics",
        "name": "BRICS-based SMILES GA",
        "short": "BRICS SMILES GA",
        "color": "C2",
        "result_dir": EXP / "2_smiles_GA/fragment_ga_smiles_childselect_v1",
        "path_csv": EXP / "2_smiles_GA/fragment_ga_smiles_childselect_v1/evolution_path_full.csv",
        "generation_col": "generation",
    },
    {
        "key": "latent_ga",
        "name": "Latent GA",
        "short": "Latent GA",
        "color": "C1",
        "result_dir": EXP / "3_latent_GA_noLLM/psvae_train_random_V2",
        "path_csv": EXP / "3_latent_GA_noLLM/psvae_train_random_V2/evolution_path.csv",
        "generation_col": "generation",
    },
    {
        "key": "llm_init",
        "name": "LLM-Initialized Latent GA",
        "short": "LLM-Initialized",
        "color": "C3",
        "result_dir": EXP / "4_latent_GA_LLM/llm_llm_V2",
        "path_csv": EXP / "4_latent_GA_LLM/llm_llm_V2/evolution_path.csv",
        "generation_col": "generation",
    },
    {
        "key": "ours",
        "name": "Iterative LLM-Guided Latent GA",
        "short": "Iterative LLM-Guided",
        "color": "C4",
        "result_dir": EXP / "5_Ours/llm_ours_V2",
        "path_csv": EXP / "5_Ours/llm_ours_V2/evolution_path.csv",
        "generation_col": "generation",
    },
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smi: str | float | None) -> str | None:
    try:
        if pd.isna(smi):
            return None
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def find_final_latent_file(result_dir: Path) -> Path:
    candidates = [
        "final_population_latent.npy",
        "final_population_latents.npy",
        "final_latents.npy",
        "generated_latents.npy",
        "population_latents.npy",
        "final_population.npy",
    ]
    for name in candidates:
        path = result_dir / name
        if path.exists():
            return path
    for path in sorted(result_dir.glob("*.npy")):
        base = path.name.lower()
        if "final" in base and "latent" in base:
            return path
    raise FileNotFoundError(f"No final latent file found under {result_dir}")


def load_final_latents(path: Path, max_n: int = 100) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) > max_n:
        idx = np.random.choice(len(arr), max_n, replace=False)
        arr = arr[idx]
    return arr.astype(np.float32)


def get_z_mean_from_mol(model: PSVAEModel, mol: Chem.Mol) -> torch.Tensor:
    step1_res = BPEMolDataset.process_step1(mol, model.tokenizer)
    step2_res = BPEMolDataset.process_step2(step1_res, model.tokenizer)
    batch = BPEMolDataset.process_step3([step2_res], model.tokenizer, device=model.device)
    x, edge_index, edge_attr = batch["x"], batch["edge_index"], batch["edge_attr"]
    x_pieces, x_pos = batch["x_pieces"], batch["x_pos"]
    x = model.decoder.embed_atom(x, x_pieces, x_pos)
    batch_size, node_num, node_dim = x.shape
    graph_ids = torch.repeat_interleave(torch.arange(0, batch_size, device=x.device), node_num)
    _, all_x = model.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
    graph_embedding = model.encoder.embed_graph(all_x, graph_ids, batch["atom_mask"].flatten())
    return model.decoder.W_mean(graph_embedding).squeeze(0)


def smiles_to_latent(model: PSVAEModel, smi: str) -> np.ndarray | None:
    mol = smiles2molecule(smi, kekulize=True)
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        with torch.no_grad():
            z = get_z_mean_from_mol(model, mol)
        return z.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def load_path_frames() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    all_smiles = []
    for spec in METHODS:
        df = pd.read_csv(spec["path_csv"])
        gen_col = spec["generation_col"]
        if gen_col not in df.columns:
            raise ValueError(f"{spec['path_csv']} missing {gen_col}")
        if "smiles" not in df.columns:
            raise ValueError(f"{spec['path_csv']} missing smiles")
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row["smiles"])
            if smi is None:
                continue
            gen = int(row[gen_col])
            rows.append({
                "method_key": spec["key"],
                "method": spec["name"],
                "generation": gen,
                "smiles": smi,
                "gap": float(row["gap"]) if "gap" in df.columns and pd.notna(row["gap"]) else np.nan,
            })
            all_smiles.append(smi)
    path_df = pd.DataFrame(rows)
    unique_smiles = list(dict.fromkeys(all_smiles))
    return path_df, unique_smiles


def encode_or_load_path_latents(model: PSVAEModel, smiles: list[str], out_dir: Path) -> pd.DataFrame:
    cache_csv = out_dir / "seed42_full_path_latent_index.csv"
    cache_npy = out_dir / "seed42_full_path_latents.npy"
    if cache_csv.exists() and cache_npy.exists():
        index_df = pd.read_csv(cache_csv)
        arr = np.load(cache_npy)
        if len(index_df) == len(arr):
            return index_df

    kept, failed, latents = [], [], []
    for i, smi in enumerate(smiles):
        z = smiles_to_latent(model, smi)
        if z is None or not np.all(np.isfinite(z)):
            failed.append(smi)
            continue
        kept.append(smi)
        latents.append(z)
        if (i + 1) % 50 == 0:
            print(f"[INFO] encoded path smiles {i + 1}/{len(smiles)}")

    if not latents:
        raise RuntimeError("No path SMILES were encoded.")

    arr = np.vstack(latents).astype(np.float32)
    index_df = pd.DataFrame({"smiles": kept, "latent_row": np.arange(len(kept))})
    index_df.to_csv(cache_csv, index=False)
    np.save(cache_npy, arr)
    if failed:
        pd.DataFrame({"failed_smiles": failed}).to_csv(out_dir / "seed42_full_path_failed_smiles.csv", index=False)
    print(f"[INFO] encoded path smiles: {len(kept)}/{len(smiles)}, failed={len(failed)}")
    return index_df


def build_projection_dataframe(seed: int = 42, n_train: int = 2000, max_final: int = 100) -> tuple[pd.DataFrame, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    train_latents = np.load(TRAIN_LATENT_PATH).astype(np.float32)
    train_idx = np.random.choice(len(train_latents), min(n_train, len(train_latents)), replace=False)
    train_vis = train_latents[train_idx].astype(np.float32)

    final_arrays, final_rows = [], []
    for spec in METHODS:
        final_file = find_final_latent_file(spec["result_dir"])
        arr = load_final_latents(final_file, max_n=max_final)
        start = sum(len(x) for x in final_arrays)
        final_arrays.append(arr)
        for i in range(len(arr)):
            final_rows.append({
                "method_key": spec["key"],
                "method": spec["name"],
                "stage": "final",
                "smiles": "",
                "generation": np.nan,
                "gap": np.nan,
                "latent_source": "saved_final_population",
                "local_idx": i,
                "global_final_idx": start + i,
            })

    path_df, unique_path_smiles = load_path_frames()

    print("[INFO] loading PS-VAE for path SMILES encoding...")
    model = PSVAEModel.load_from_checkpoint(str(CKPT_PSVAE), map_location=DEVICE)
    model.eval()
    model.to(DEVICE)
    path_index = encode_or_load_path_latents(model, unique_path_smiles, OUT)
    path_latents = np.load(OUT / "seed42_full_path_latents.npy").astype(np.float32)

    path_df = path_df.merge(path_index, on="smiles", how="inner")
    path_counts = (
        path_df.groupby(["method_key", "smiles", "latent_row"], as_index=False)
        .agg(
            method=("method", "first"),
            generation_min=("generation", "min"),
            generation_max=("generation", "max"),
            n_generations=("generation", "size"),
            gap_min=("gap", "min"),
        )
    )

    final_latents = np.vstack(final_arrays).astype(np.float32)
    X = np.vstack([train_vis, final_latents, path_latents]).astype(np.float32)

    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(X)

    reducer = UMAP(
        n_components=2,
        n_neighbors=35,
        min_dist=0.12,
        metric="euclidean",
        random_state=seed,
    )
    umap_coords = reducer.fit_transform(X)

    rows = []
    for i in range(len(train_vis)):
        rows.append({
            "method_key": "train",
            "method": "QM9 training set",
            "stage": "training_background",
            "smiles": "",
            "generation_min": np.nan,
            "generation_max": np.nan,
            "n_generations": np.nan,
            "gap_min": np.nan,
            "pca_x": pca_coords[i, 0],
            "pca_y": pca_coords[i, 1],
            "umap_x": umap_coords[i, 0],
            "umap_y": umap_coords[i, 1],
        })

    offset_final = len(train_vis)
    for i, row in enumerate(final_rows):
        j = offset_final + i
        out = dict(row)
        out.update({
            "generation_min": np.nan,
            "generation_max": np.nan,
            "n_generations": np.nan,
            "gap_min": np.nan,
            "pca_x": pca_coords[j, 0],
            "pca_y": pca_coords[j, 1],
            "umap_x": umap_coords[j, 0],
            "umap_y": umap_coords[j, 1],
        })
        rows.append(out)

    offset_path = len(train_vis) + len(final_latents)
    for _, row in path_counts.iterrows():
        j = offset_path + int(row["latent_row"])
        rows.append({
            "method_key": row["method_key"],
            "method": row["method"],
            "stage": "encoded_evolution_path",
            "smiles": row["smiles"],
            "generation_min": int(row["generation_min"]),
            "generation_max": int(row["generation_max"]),
            "n_generations": int(row["n_generations"]),
            "gap_min": float(row["gap_min"]),
            "pca_x": pca_coords[j, 0],
            "pca_y": pca_coords[j, 1],
            "umap_x": umap_coords[j, 0],
            "umap_y": umap_coords[j, 1],
        })

    proj_df = pd.DataFrame(rows)
    proj_df.to_csv(OUT / "seed42_space_distribution_projection.csv", index=False)

    summary = {
        "seed": seed,
        "n_train_background": int(len(train_vis)),
        "n_final_points": int(len(final_latents)),
        "n_unique_path_smiles_raw": int(len(unique_path_smiles)),
        "n_unique_path_smiles_encoded": int(len(path_index)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "method_path_counts": path_counts.groupby("method_key")["smiles"].count().to_dict(),
        "output_projection_csv": str(OUT / "seed42_space_distribution_projection.csv"),
    }
    with open(OUT / "seed42_space_distribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return proj_df, summary


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.titlesize": 23,
        "axes.labelsize": 23,
        "axes.linewidth": 1.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 18,
        "figure.dpi": 180,
        "savefig.dpi": 600,
    })


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb * (1 - amount) + np.ones(3) * amount)


def set_local_limits(ax, xs, ys, pad: float = 0.10) -> None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) == 0:
        return
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xr = xmax - xmin if xmax > xmin else 1.0
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_xlim(xmin - xr * pad, xmax + xr * pad)
    ax.set_ylim(ymin - yr * pad, ymax + yr * pad)


def plot_faceted(df: pd.DataFrame, coord: str, out_png: Path, out_pdf: Path, *, local_zoom: bool = False) -> None:
    setup_style()
    xcol, ycol = f"{coord}_x", f"{coord}_y"
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(21.5, 13.2),
        sharex=not local_zoom,
        sharey=not local_zoom,
    )
    axes_flat = axes.flat
    train = df[df["stage"] == "training_background"]

    for ax, spec in zip(axes_flat[:5], METHODS):
        color = spec["color"]
        sub = df[df["method_key"] == spec["key"]]
        path = sub[sub["stage"] == "encoded_evolution_path"].copy()
        final = sub[sub["stage"] == "final"].copy()
        path = path.sort_values("generation_min")

        ax.scatter(
            train[xcol],
            train[ycol],
            s=8,
            c="#bfc4cc",
            alpha=0.18,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        if len(path) > 1:
            ax.plot(
                path[xcol],
                path[ycol],
                color=lighten(color, 0.15),
                lw=2.1,
                alpha=0.55,
                zorder=3,
            )
        if len(path):
            sizes = 42 + 16 * np.sqrt(path["n_generations"].clip(lower=1))
            ax.scatter(
                path[xcol],
                path[ycol],
                s=sizes,
                facecolors=lighten(color, 0.58),
                edgecolors=color,
                linewidths=1.05,
                alpha=0.78,
                rasterized=True,
                label="Evolution path states",
                zorder=4,
            )
        if len(final):
            ax.scatter(
                final[xcol],
                final[ycol],
                s=62 if spec["key"] != "brics" else 118,
                c=color,
                alpha=0.70,
                linewidths=0.55,
                edgecolors="white",
                rasterized=True,
                label="Final population",
                zorder=5,
            )

        ax.set_title(spec["short"], color=color, pad=11, weight="bold")
        ax.grid(alpha=0.22, linewidth=1.0)
        ax.set_xlabel(coord.upper() + "-1")
        ax.set_ylabel(coord.upper() + "-2")
        ax.tick_params(width=1.5, length=5)
        if local_zoom:
            plot_points = pd.concat([path[[xcol, ycol]], final[[xcol, ycol]]], ignore_index=True)
            set_local_limits(ax, plot_points[xcol], plot_points[ycol], pad=0.16)

    ax_legend = axes_flat[5]
    ax_legend.axis("off")
    bg_handle = mpl.lines.Line2D([], [], color="#bfc4cc", marker="o", linestyle="None", markersize=9, alpha=0.5)
    path_handle = mpl.lines.Line2D([], [], color="#7A828F", marker="o", linestyle="-", markersize=10, lw=2.2)
    final_handle = mpl.lines.Line2D([], [], color="#464C55", marker="o", linestyle="None", markersize=11)
    ax_legend.legend(
        [bg_handle, path_handle, final_handle],
        ["QM9 training background", "Encoded evolution path states", "Final population"],
        loc="center left",
        frameon=False,
        handlelength=2.4,
        borderaxespad=0,
    )
    ax_legend.text(
        0.0,
        0.24,
        "Marker size for path states reflects how many generations the same best molecule persists.",
        transform=ax_legend.transAxes,
        fontsize=18,
        color="#464C55",
        va="top",
        wrap=True,
    )

    fig.suptitle(
        f"{coord.upper()} distribution of seed-42 search trajectories and final populations"
        + (" (local zoom)" if local_zoom else ""),
        fontsize=30,
        y=0.985,
        weight="bold",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.07, wspace=0.12, hspace=0.25)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] saved {out_png}")
    print(f"[INFO] saved {out_pdf}")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    set_seed(42)
    df, summary = build_projection_dataframe(seed=42)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    plot_faceted(
        df,
        "umap",
        OUT / "fig2d_all_methods_umap_distribution_seed42_highres.png",
        OUT / "fig2d_all_methods_umap_distribution_seed42_highres.pdf",
    )
    plot_faceted(
        df,
        "umap",
        OUT / "fig2d_all_methods_umap_distribution_seed42_zoom_highres.png",
        OUT / "fig2d_all_methods_umap_distribution_seed42_zoom_highres.pdf",
        local_zoom=True,
    )
    plot_faceted(
        df,
        "pca",
        OUT / "fig2c_all_methods_pca_distribution_seed42_highres.png",
        OUT / "fig2c_all_methods_pca_distribution_seed42_highres.pdf",
    )
    plot_faceted(
        df,
        "pca",
        OUT / "fig2c_all_methods_pca_distribution_seed42_zoom_highres.png",
        OUT / "fig2c_all_methods_pca_distribution_seed42_zoom_highres.pdf",
        local_zoom=True,
    )


if __name__ == "__main__":
    main()
