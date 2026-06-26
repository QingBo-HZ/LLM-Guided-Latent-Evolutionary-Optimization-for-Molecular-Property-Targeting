#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from rdkit import Chem
from rdkit import RDLogger
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
RDLogger.DisableLog("rdApp.*")


# =========================
# Default paths
# =========================

PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

CKPT_PSVAE = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt"
TRAIN_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy"

DEFAULT_OUT_DIR = "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA/fragment_ga_smiles_childselect_v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Load PS-VAE related classes
# =========================

from pl_models import PSVAEModel
from utils.chem_utils import smiles2molecule, GeneralVocab
from data.mol_bpe import Tokenizer
from rdkit.Chem.rdchem import BondType
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


# =========================
# Utilities
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def canonicalize_smiles(smi):
    try:
        if pd.isna(smi):
            return None
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def find_existing_file(out_dir, candidates):
    for name in candidates:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            return path
    return None


def detect_smiles_column(df):
    candidates = [
        "smiles",
        "canonical_smiles",
        "best_smiles",
        "best_smiles_final",
        "SMILES",
        "Canonicalized SMILES",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        if "smiles" in c.lower():
            return c

    raise ValueError(f"No SMILES column found. Available columns: {list(df.columns)}")


def load_smiles_from_csv(path, max_n=None, label_name="generated"):
    df = pd.read_csv(path)
    smiles_col = detect_smiles_column(df)

    smiles = []
    for s in df[smiles_col].tolist():
        cs = canonicalize_smiles(s)
        if cs is not None:
            smiles.append(cs)

    smiles = list(dict.fromkeys(smiles))

    if max_n is not None and len(smiles) > max_n:
        smiles = smiles[:max_n]

    print(f"[INFO] Loaded {len(smiles)} valid unique SMILES from {label_name}: {path}")
    print(f"[INFO] SMILES column = {smiles_col}")
    return smiles


def smiles_to_latent(model_psvae, smi, device):
    mol = smiles2molecule(smi, kekulize=True)
    if mol is None:
        return None

    try:
        mol = Chem.RemoveHs(mol)
        with torch.no_grad():
            z = model_psvae.get_z_from_mol(mol)
            if z.dim() > 1:
                z = z.squeeze(0)
            return z.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def encode_smiles_list(model_psvae, smiles_list, device, label_name="generated"):
    latents = []
    kept_smiles = []
    failed = []

    for i, smi in enumerate(smiles_list):
        z = smiles_to_latent(model_psvae, smi, device=device)
        if z is None or not np.all(np.isfinite(z)):
            failed.append(smi)
            continue

        latents.append(z)
        kept_smiles.append(smi)

        if (i + 1) % 100 == 0:
            print(f"[INFO] Encoding {label_name}: {i + 1}/{len(smiles_list)}")

    if len(latents) == 0:
        raise RuntimeError(f"No latent vectors were successfully encoded for {label_name}.")

    latents = np.vstack(latents).astype(np.float32)

    print(f"[INFO] Encoded {len(kept_smiles)}/{len(smiles_list)} SMILES for {label_name}")
    print(f"[INFO] Failed {len(failed)} SMILES for {label_name}")

    return kept_smiles, latents, failed


def save_failed_smiles(failed, path):
    if len(failed) == 0:
        return
    pd.DataFrame({"failed_smiles": failed}).to_csv(path, index=False)


def plot_pca(
    train_latents,
    generated_latents,
    generated_smiles,
    path_csv,
    path_png,
    seed=42,
    title="Chemical Space Visualization by PCA (BRICS-based SMILES GA)",
):
    X_all = np.vstack([train_latents, generated_latents])
    labels = (["train"] * len(train_latents)) + (["generated"] * len(generated_latents))

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_all)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
        "smiles": [""] * len(train_latents) + generated_smiles,
    })
    df.to_csv(path_csv, index=False)

    train_mask = np.array(labels) == "train"
    gen_mask = np.array(labels) == "generated"

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[train_mask, 0], coords[train_mask, 1], s=8, alpha=0.35, label="QM9 training set")
    plt.scatter(coords[gen_mask, 0], coords[gen_mask, 1], s=22, alpha=0.85, label="BRICS-based SMILES GA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_png, dpi=600)
    plt.close()

    print(f"[INFO] Saved PCA CSV: {path_csv}")
    print(f"[INFO] Saved PCA figure: {path_png}")
    print(f"[INFO] PCA explained variance ratio: {pca.explained_variance_ratio_}")


def plot_umap(
    train_latents,
    generated_latents,
    generated_smiles,
    path_csv,
    path_png,
    seed=42,
    title="Chemical Space Visualization by UMAP (BRICS-based SMILES GA)",
):
    if not HAS_UMAP:
        print("[WARN] umap-learn is not installed. Skip UMAP.")
        return

    X_all = np.vstack([train_latents, generated_latents])
    labels = (["train"] * len(train_latents)) + (["generated"] * len(generated_latents))

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=30,
        min_dist=0.15,
        metric="euclidean",
    )
    coords = reducer.fit_transform(X_all)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
        "smiles": [""] * len(train_latents) + generated_smiles,
    })
    df.to_csv(path_csv, index=False)

    train_mask = np.array(labels) == "train"
    gen_mask = np.array(labels) == "generated"

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[train_mask, 0], coords[train_mask, 1], s=8, alpha=0.35, label="QM9 training set")
    plt.scatter(coords[gen_mask, 0], coords[gen_mask, 1], s=22, alpha=0.85, label="BRICS-based SMILES GA")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_png, dpi=600)
    plt.close()

    print(f"[INFO] Saved UMAP CSV: {path_csv}")
    print(f"[INFO] Saved UMAP figure: {path_png}")


def plot_pca_with_evolution_path(
    train_latents,
    generated_latents,
    generated_smiles,
    path_latents,
    path_smiles,
    path_csv,
    path_png,
    seed=42,
):
    X_all = np.vstack([train_latents, generated_latents, path_latents])
    labels = (
        ["train"] * len(train_latents)
        + ["generated"] * len(generated_latents)
        + ["evolution_path"] * len(path_latents)
    )

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_all)

    train_n = len(train_latents)
    gen_n = len(generated_latents)

    train_coords = coords[:train_n]
    gen_coords = coords[train_n:train_n + gen_n]
    path_coords = coords[train_n + gen_n:]

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
        "smiles": (
            [""] * len(train_latents)
            + generated_smiles
            + path_smiles
        ),
    })
    df.to_csv(path_csv, index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(train_coords[:, 0], train_coords[:, 1], s=8, alpha=0.30, label="QM9 training set")
    plt.scatter(gen_coords[:, 0], gen_coords[:, 1], s=22, alpha=0.75, label="Final population")
    plt.plot(path_coords[:, 0], path_coords[:, 1], "-o", linewidth=1.8, markersize=4, label="Best-molecule path")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Chemical Space with Evolution Path (BRICS-based SMILES GA)")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_png, dpi=600)
    plt.close()

    print(f"[INFO] Saved PCA path CSV: {path_csv}")
    print(f"[INFO] Saved PCA path figure: {path_png}")


def main():
    parser = argparse.ArgumentParser("Post-hoc PCA/UMAP for BRICS-based SMILES GA")

    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--psvae_ckpt", type=str, default=CKPT_PSVAE)
    parser.add_argument("--train_latent_path", type=str, default=TRAIN_LATENT_PATH)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train_vis", type=int, default=2000)
    parser.add_argument("--max_generated", type=int, default=None)

    parser.add_argument("--final_population_file", type=str, default="")
    parser.add_argument("--evolution_path_file", type=str, default="")

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir
    ensure_dir(out_dir)

    print("\n========== CONFIG ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"[INFO] device = {DEVICE}")

    # =========================
    # Locate files
    # =========================

    if args.final_population_file:
        final_pop_path = args.final_population_file
    else:
        final_pop_path = find_existing_file(
            out_dir,
            [
                "final_population_fragment_ga.csv",
                "final_population.csv",
                "final_population_smiles_ga.csv",
            ],
        )

    if final_pop_path is None or not os.path.exists(final_pop_path):
        raise FileNotFoundError(
            f"Cannot find final population CSV in {out_dir}. "
            f"Expected final_population_fragment_ga.csv or similar."
        )

    if args.evolution_path_file:
        evo_path = args.evolution_path_file
    else:
        evo_path = find_existing_file(
            out_dir,
            [
                "evolution_path_full.csv",
                "evolution_path.csv",
                "best_path.csv",
            ],
        )

    print(f"[INFO] final population file = {final_pop_path}")
    print(f"[INFO] evolution path file = {evo_path}")

    # =========================
    # Load PS-VAE
    # =========================

    print("[INFO] loading PS-VAE...")
    model_psvae = PSVAEModel.load_from_checkpoint(args.psvae_ckpt, map_location=DEVICE)
    model_psvae.eval()
    model_psvae.to(DEVICE)

    # =========================
    # Load train latent
    # =========================

    print("[INFO] loading train latent...")
    latent_train = np.load(args.train_latent_path).astype(np.float32)
    print(f"[INFO] latent_train shape = {latent_train.shape}")

    n_train_vis = min(args.n_train_vis, len(latent_train))
    train_vis_idx = np.random.choice(len(latent_train), n_train_vis, replace=False)
    train_vis = latent_train[train_vis_idx]

    # =========================
    # Load and encode final population
    # =========================

    final_smiles = load_smiles_from_csv(
        final_pop_path,
        max_n=args.max_generated,
        label_name="final population",
    )

    generated_smiles, generated_latents, failed_gen = encode_smiles_list(
        model_psvae=model_psvae,
        smiles_list=final_smiles,
        device=DEVICE,
        label_name="final population",
    )

    np.save(os.path.join(out_dir, "final_population_latents.npy"), generated_latents)

    pd.DataFrame({
        "smiles": generated_smiles,
    }).to_csv(os.path.join(out_dir, "final_population_encoded_smiles.csv"), index=False)

    save_failed_smiles(
        failed_gen,
        os.path.join(out_dir, "failed_final_population_smiles.csv"),
    )

    # =========================
    # PCA and UMAP for final population
    # =========================

    plot_pca(
        train_latents=train_vis,
        generated_latents=generated_latents,
        generated_smiles=generated_smiles,
        path_csv=os.path.join(out_dir, "chemical_space_pca.csv"),
        path_png=os.path.join(out_dir, "fig5_pca_chemical_space_brics_smiles_ga.png"),
        seed=args.seed,
    )

    plot_umap(
        train_latents=train_vis,
        generated_latents=generated_latents,
        generated_smiles=generated_smiles,
        path_csv=os.path.join(out_dir, "chemical_space_umap.csv"),
        path_png=os.path.join(out_dir, "fig5_umap_chemical_space_brics_smiles_ga.png"),
        seed=args.seed,
    )

    # =========================
    # Optional evolution path projection
    # =========================

    if evo_path is not None and os.path.exists(evo_path):
        evo_smiles = load_smiles_from_csv(
            evo_path,
            max_n=None,
            label_name="evolution path",
        )

        # 去重会破坏路径顺序，这里重新按原 CSV 顺序读取
        evo_df = pd.read_csv(evo_path)
        evo_col = detect_smiles_column(evo_df)
        evo_smiles_ordered = []
        seen = set()
        for s in evo_df[evo_col].tolist():
            cs = canonicalize_smiles(s)
            if cs is None:
                continue
            # 连续重复可以去掉，否则路径会很密
            if len(evo_smiles_ordered) > 0 and cs == evo_smiles_ordered[-1]:
                continue
            # 不强制全局去重，保留可能回访轨迹
            evo_smiles_ordered.append(cs)

        print(f"[INFO] Ordered evolution path molecules: {len(evo_smiles_ordered)}")

        path_smiles, path_latents, failed_path = encode_smiles_list(
            model_psvae=model_psvae,
            smiles_list=evo_smiles_ordered,
            device=DEVICE,
            label_name="evolution path",
        )

        np.save(os.path.join(out_dir, "evolution_path_latents.npy"), path_latents)

        path_df = pd.DataFrame({
            "path_index": np.arange(len(path_smiles)),
            "smiles": path_smiles,
        })
        path_df.to_csv(os.path.join(out_dir, "evolution_path_encoded_smiles.csv"), index=False)

        save_failed_smiles(
            failed_path,
            os.path.join(out_dir, "failed_evolution_path_smiles.csv"),
        )

        plot_pca_with_evolution_path(
            train_latents=train_vis,
            generated_latents=generated_latents,
            generated_smiles=generated_smiles,
            path_latents=path_latents,
            path_smiles=path_smiles,
            path_csv=os.path.join(out_dir, "chemical_space_pca_with_evolution_path.csv"),
            path_png=os.path.join(out_dir, "fig5_pca_chemical_space_with_evolution_path_brics_smiles_ga.png"),
            seed=args.seed,
        )
    else:
        print("[WARN] evolution path file not found. Skip path projection.")

    print("\n========== DONE ==========")
    print(f"Output dir: {out_dir}")
    print(f"Encoded final population: {len(generated_smiles)}")
    print(f"Saved: chemical_space_pca.csv")
    if HAS_UMAP:
        print(f"Saved: chemical_space_umap.csv")
    else:
        print("UMAP skipped because umap-learn is unavailable.")


if __name__ == "__main__":
    main()