#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZINC logP latent-space GA with unified outputs for paper figures/tables.

This script follows the QM9 paper-style latent GA logic:
- initialization modes: train_random / llm / psvae / hybrid
- elite selection
- tournament selection
- arithmetic crossover
- polynomial mutation
- top-k archive per generation
- evolution path
- final population decoding
- summary.json
- progress_metrics.csv
- evolution_path.csv
- topk_evolution_paths.csv
- best_candidates_over_time.csv
- PCA/UMAP chemical-space visualization

Task objective:
- optimize predicted logP toward a target interval/range
- default target_logp = 3.0
- default success range = [2.5, 3.5]
- score = exp(-0.5 * ((pred_logP - target_logP) / score_sigma)^2)
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.serialization

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.rdchem import BondType
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

RDLogger.DisableLog("rdApp.*")


# ======================
# PS-VAE root
# ======================

PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

from pl_models import PSVAEModel
from utils.chem_utils import molecule2smiles, GeneralVocab
from data.mol_bpe import Tokenizer

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


# ======================
# Default paths
# ======================

DEFAULT_ZINC_PSVAE_CKPT = (
    "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/"
    "ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt"
)

DEFAULT_PREDICTOR_CKPT = (
    "/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/"
    "logp_predictor/best_logp_predictor.pt"
)

DEFAULT_TRAIN_LATENT_POOL = (
    "/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/"
    "train/zinc_logp_latent.npy"
)

DEFAULT_OUTPUT_ROOT = (
    "/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/results"
)


# ======================
# General utilities
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def canonicalize_smiles(smi):
    try:
        if smi is None:
            return None
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def calc_rdkit_logp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return None


def compute_diversity(smiles_list):
    mols = []
    for s in smiles_list:
        if s is None:
            continue
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            mols.append(m)

    if len(mols) < 2:
        return 0.0

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        for m in mols
    ]

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    if len(sims) == 0:
        return 0.0

    return 1.0 - float(np.mean(sims))


def safe_np_load(path, name="array"):
    if path is None:
        raise ValueError(f"{name} path is None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}: {path}")
    return arr


# ======================
# Predictor
# Must match 02_train_logp_predictor_fixed_split.py high-R2 version
# ======================

class LogPPredictor(nn.Module):
    def __init__(self, dim_feature, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_feature, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class LogPPredictorAPI:
    def __init__(self, predictor_ckpt, device):
        self.device = torch.device(device)

        ckpt = torch.load(
            predictor_ckpt,
            map_location=self.device,
            weights_only=False
        )

        self.dim_feature = int(ckpt["dim_feature"])
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.dropout = float(ckpt.get("dropout", 0.0))
        self.target_name = ckpt.get("target_name", "logP")

        self.model = LogPPredictor(
            dim_feature=self.dim_feature,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.X_mean = np.asarray(ckpt["X_mean"], dtype=np.float32)
        self.X_std = np.asarray(ckpt["X_std"], dtype=np.float32)
        self.y_mean = float(ckpt["y_mean"])
        self.y_std = float(ckpt["y_std"])

        if self.X_mean.ndim == 1:
            self.X_mean = self.X_mean[None, :]
        if self.X_std.ndim == 1:
            self.X_std = self.X_std[None, :]

        print("[INFO] LogP predictor loaded.")
        print(f"[INFO] dim_feature = {self.dim_feature}")
        print(f"[INFO] hidden_dim  = {self.hidden_dim}")
        print(f"[INFO] dropout     = {self.dropout}")
        print(f"[INFO] target_name = {self.target_name}")

    def predict(self, z):
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]

        if z.shape[1] != self.dim_feature:
            raise ValueError(
                f"Latent dim mismatch: z dim={z.shape[1]}, "
                f"predictor dim={self.dim_feature}"
            )

        zn = (z - self.X_mean) / self.X_std
        x = torch.tensor(zn, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_norm = self.model(x).detach().cpu().numpy().reshape(-1)

        pred = pred_norm * self.y_std + self.y_mean
        return pred.astype(np.float32)


# ======================
# Model loading
# ======================

def load_psvae(ckpt_path, device):
    print(f"[INFO] Loading ZINC PS-VAE checkpoint: {ckpt_path}", flush=True)

    old_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    try:
        model = PSVAEModel.load_from_checkpoint(
            ckpt_path,
            map_location=device
        )
    finally:
        torch.load = old_torch_load

    model.eval()
    model.to(device)

    print("[INFO] ZINC PS-VAE loaded.", flush=True)
    return model


# ======================
# Decode
# ======================

def latent_to_smiles(
    model_psvae,
    z,
    device,
    max_atom_num=80,
    add_edge_th=0.55,
    temperature=0.5
):
    try:
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32, device=device)

            graph = model_psvae.inference_single_z(
                z_t,
                max_atom_num=max_atom_num,
                add_edge_th=add_edge_th,
                temperature=temperature
            )

            mol = model_psvae.return_data_to_mol(graph)
            smi = molecule2smiles(mol)

        return canonicalize_smiles(smi)
    except Exception:
        return None


# ======================
# logP scoring
# ======================

def score_logp(pred_logp, target_logp=3.0, score_sigma=0.5):
    pred_logp = np.asarray(pred_logp, dtype=np.float32)
    score = np.exp(-0.5 * ((pred_logp - target_logp) / score_sigma) ** 2)
    return score.astype(np.float32)


def success_mask_logp(pred_logp, low=2.5, high=3.5):
    pred_logp = np.asarray(pred_logp, dtype=np.float32)
    return (pred_logp >= low) & (pred_logp <= high)


# ======================
# GA operations: QM9 logic style
# ======================

def tournament_selection(pop, fitness, tourn_size=2):
    indices = np.random.choice(len(pop), tourn_size, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx].copy()


def arithmetic_crossover(p1, p2, cross_prob, lb, ub):
    if np.random.random() > cross_prob:
        return p1.copy(), p2.copy()

    alpha = np.random.random()
    c1 = alpha * p1 + (1.0 - alpha) * p2
    c2 = (1.0 - alpha) * p1 + alpha * p2

    c1 = np.clip(c1, lb, ub).astype(np.float32)
    c2 = np.clip(c2, lb, ub).astype(np.float32)

    return c1, c2


def polynomial_mutation(individual, prob, eta, low, up):
    mutated = individual.copy().astype(np.float32)

    for i in range(len(mutated)):
        if np.random.random() < prob:
            r = np.random.random()
            if r < 0.5:
                delta = (2.0 * r) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))

            mutated[i] += delta * (up[i] - low[i])
            mutated[i] = np.clip(mutated[i], low[i], up[i])

    return mutated.astype(np.float32)


# ======================
# Initialization
# ======================

def sample_population_from_pool(pool, pop_size):
    pool = np.asarray(pool, dtype=np.float32)
    n = len(pool)

    if n == 0:
        raise ValueError("latent pool is empty")

    if n >= pop_size:
        indices = np.random.choice(n, pop_size, replace=False)
        return pool[indices].copy()

    extra_indices = np.random.choice(n, pop_size - n, replace=True)
    extra = pool[extra_indices].copy()
    return np.concatenate([pool.copy(), extra], axis=0).astype(np.float32)


def farthest_point_sample(pool, n_select, seed=42):
    pool = np.asarray(pool, dtype=np.float32)
    n = len(pool)

    if n <= n_select:
        return pool.copy()

    rng = np.random.default_rng(seed)
    selected_idx = [int(rng.integers(0, n))]

    dist = np.linalg.norm(pool - pool[selected_idx[0]], axis=1)

    for _ in range(1, n_select):
        idx = int(np.argmax(dist))
        selected_idx.append(idx)

        new_dist = np.linalg.norm(pool - pool[idx], axis=1)
        dist = np.minimum(dist, new_dist)

    return pool[selected_idx].copy()


def select_diverse_fill(pool, existing, n_select):
    pool = np.asarray(pool, dtype=np.float32)
    existing = np.asarray(existing, dtype=np.float32)

    if len(pool) <= n_select:
        return pool.copy()

    if len(existing) == 0:
        return farthest_point_sample(pool, n_select)

    dist_matrix = np.linalg.norm(
        pool[:, None, :] - existing[None, :, :],
        axis=2
    )
    min_dist = dist_matrix.min(axis=1)

    idx = np.argsort(min_dist)[::-1][:n_select]
    return pool[idx].copy()


def build_hybrid_population(
    llm_latent,
    psvae_latent,
    latent_train,
    lb,
    ub,
    pop_size,
    sigma_scale=0.15,
    llm_keep_ratio=0.5,
    llm_expand_ratio=0.0,
    local_k=8,
    seed=42,
):
    llm_latent = np.asarray(llm_latent, dtype=np.float32)
    psvae_latent = np.asarray(psvae_latent, dtype=np.float32)
    latent_train = np.asarray(latent_train, dtype=np.float32)

    n_keep = max(1, int(pop_size * llm_keep_ratio))
    n_expand = max(0, int(pop_size * llm_expand_ratio))
    n_fill = pop_size - n_keep - n_expand

    if n_fill < 0:
        n_fill = 0

    keep_part = farthest_point_sample(llm_latent, n_keep, seed=seed)

    expand_list = []
    parent_pool = keep_part if len(keep_part) > 0 else llm_latent

    for i in range(n_expand):
        parent = parent_pool[i % len(parent_pool)].copy()

        dist2 = ((latent_train - parent[None, :]) ** 2).sum(axis=1)
        nn_idx = np.argsort(dist2)[:max(local_k, 2)]
        local_neighbors = latent_train[nn_idx]

        local_std = local_neighbors.std(axis=0).astype(np.float32)
        local_std = np.maximum(local_std, 1e-6)

        noise = np.random.normal(
            loc=0.0,
            scale=local_std * sigma_scale,
            size=parent.shape
        ).astype(np.float32)

        child = parent + noise
        child = np.clip(child, lb, ub)
        expand_list.append(child.astype(np.float32))

    if len(expand_list) > 0:
        expand_part = np.vstack(expand_list).astype(np.float32)
    else:
        expand_part = np.zeros((0, latent_train.shape[1]), dtype=np.float32)

    existing_part = np.concatenate([keep_part, expand_part], axis=0)

    if n_fill > 0:
        fill_part = select_diverse_fill(psvae_latent, existing_part, n_fill)
        population = np.concatenate([keep_part, expand_part, fill_part], axis=0)
    else:
        population = existing_part

    if len(population) > pop_size:
        population = population[:pop_size]
    elif len(population) < pop_size:
        extra = sample_population_from_pool(psvae_latent, pop_size - len(population))
        population = np.concatenate([population, extra], axis=0)

    return population.astype(np.float32)


def initialize_population(
    init_mode,
    pop_size,
    latent_train,
    llm_latent_path,
    psvae_latent_path,
    hybrid_latent_path,
    hybrid_sigma,
    hybrid_keep_ratio,
    hybrid_expand_ratio,
    lb,
    ub,
    seed,
):
    init_mode = init_mode.lower()

    if init_mode == "train_random":
        print("[INFO] Initialization: train_random")
        return sample_population_from_pool(latent_train, pop_size)

    if init_mode == "llm":
        if not os.path.exists(llm_latent_path):
            raise FileNotFoundError(f"LLM latent file not found: {llm_latent_path}")
        llm_latent = safe_np_load(llm_latent_path, "llm_latent")
        print(f"[INFO] Initialization: llm, shape={llm_latent.shape}")
        return sample_population_from_pool(llm_latent, pop_size)

    if init_mode == "psvae":
        if psvae_latent_path is None:
            print("[WARN] psvae_latent_path is None. Fallback to train_random.")
            return sample_population_from_pool(latent_train, pop_size)
        if not os.path.exists(psvae_latent_path):
            print(f"[WARN] psvae latent not found: {psvae_latent_path}. Fallback to train_random.")
            return sample_population_from_pool(latent_train, pop_size)
        psvae_latent = safe_np_load(psvae_latent_path, "psvae_latent")
        print(f"[INFO] Initialization: psvae, shape={psvae_latent.shape}")
        return sample_population_from_pool(psvae_latent, pop_size)

    if init_mode == "hybrid":
        if hybrid_latent_path is not None and os.path.exists(hybrid_latent_path):
            hybrid_latent = safe_np_load(hybrid_latent_path, "hybrid_latent")
            print(f"[INFO] Initialization: existing hybrid, shape={hybrid_latent.shape}")
            return sample_population_from_pool(hybrid_latent, pop_size)

        if not os.path.exists(llm_latent_path):
            raise FileNotFoundError(
                f"Hybrid mode requires llm latent, but not found: {llm_latent_path}"
            )

        llm_latent = safe_np_load(llm_latent_path, "llm_latent")

        if psvae_latent_path is not None and os.path.exists(psvae_latent_path):
            psvae_latent = safe_np_load(psvae_latent_path, "psvae_latent")
        else:
            print("[WARN] psvae latent not found for hybrid. Using latent_train as fill pool.")
            psvae_latent = latent_train.copy()

        return build_hybrid_population(
            llm_latent=llm_latent,
            psvae_latent=psvae_latent,
            latent_train=latent_train,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            sigma_scale=hybrid_sigma,
            llm_keep_ratio=hybrid_keep_ratio,
            llm_expand_ratio=hybrid_expand_ratio,
            local_k=8,
            seed=seed,
        )

    raise ValueError(f"Unknown init_mode: {init_mode}")


# ======================
# Main
# ======================

def main():
    parser = argparse.ArgumentParser(
        description="Paper-style ZINC logP latent-space GA"
    )

    # Core inputs
    parser.add_argument("--zinc_psvae_ckpt", type=str, default=DEFAULT_ZINC_PSVAE_CKPT)
    parser.add_argument("--predictor_ckpt", type=str, default=DEFAULT_PREDICTOR_CKPT)
    parser.add_argument("--latent_pool", type=str, default=DEFAULT_TRAIN_LATENT_POOL)

    # Initialization
    parser.add_argument(
        "--init_mode",
        type=str,
        default="train_random",
        choices=["train_random", "llm", "psvae", "hybrid"]
    )
    parser.add_argument("--llm_latent_path", type=str, default=None)
    parser.add_argument("--psvae_latent_path", type=str, default=None)
    parser.add_argument("--hybrid_latent_path", type=str, default=None)
    parser.add_argument("--hybrid_sigma", type=float, default=0.2)
    parser.add_argument("--hybrid_keep_ratio", type=float, default=0.5)
    parser.add_argument("--hybrid_expand_ratio", type=float, default=0.0)

    # GA parameters
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--n_gen", type=int, default=30)
    parser.add_argument("--elite_size", type=int, default=20)
    parser.add_argument("--cross_prob", type=float, default=0.30)
    parser.add_argument("--mut_prob", type=float, default=0.05)
    parser.add_argument("--mut_eta", type=float, default=20.0)
    parser.add_argument("--tourn_size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)

    # logP objective
    parser.add_argument("--target_logp", type=float, default=3.0)
    parser.add_argument("--score_sigma", type=float, default=0.5)
    parser.add_argument("--success_low", type=float, default=2.5)
    parser.add_argument("--success_high", type=float, default=3.5)

    # Decode settings
    parser.add_argument("--max_atom_num", type=int, default=80)
    parser.add_argument("--add_edge_th", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=0.5)

    # Outputs
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--version", type=str, default="zinc_logp_paper_v1")
    parser.add_argument("--topk_archive", type=int, default=10)

    # Misc
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train_vis", type=int, default=1000)

    args = parser.parse_args()

    set_seed(args.seed)
    start_wall_time = time.time()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    out_dir = os.path.join(args.output_root, f"{args.init_mode}_{args.version}")
    ensure_dir(out_dir)

    print("\n========== CONFIG ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"[INFO] device = {device}")
    print(f"[INFO] output_dir = {out_dir}")

    # ======================
    # Load data / models
    # ======================

    print("\n[INFO] Loading latent pool...")
    latent_train = safe_np_load(args.latent_pool, "latent_pool")
    latent_dim = latent_train.shape[1]

    print(f"[INFO] latent_train shape = {latent_train.shape}")
    print(f"[INFO] latent_dim = {latent_dim}")

    lb = latent_train.min(axis=0).astype(np.float32)
    ub = latent_train.max(axis=0).astype(np.float32)

    predictor = LogPPredictorAPI(
        predictor_ckpt=args.predictor_ckpt,
        device=device
    )

    if predictor.dim_feature != latent_dim:
        raise ValueError(
            f"Predictor latent dim ({predictor.dim_feature}) != "
            f"latent pool dim ({latent_dim})"
        )

    model_psvae = load_psvae(
        ckpt_path=args.zinc_psvae_ckpt,
        device=device
    )

    # ======================
    # Initialization
    # ======================

    population = initialize_population(
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

    if population.shape[1] != latent_dim:
        raise ValueError(
            f"Initial population dim mismatch: {population.shape}, "
            f"expected (*, {latent_dim})"
        )

    print(f"[INFO] Initial population shape = {population.shape}")

    np.save(os.path.join(out_dir, "initial_population_latent.npy"), population)

    # ======================
    # Histories / archives
    # ======================

    progress_records = []
    evolution_path = []
    topk_archive = []
    best_candidates_over_time = []

    avg_score_history = []
    avg_logp_history = []
    best_score_so_far_history = []
    best_logp_so_far_history = []
    best_abs_error_so_far_history = []
    top10_mean_score_history = []
    top10_mean_abs_error_history = []
    success_count_history = []
    success_rate_history = []
    eval_count_history = []
    elapsed_time_history = []

    best_score_monitor = -float("inf")
    no_improve_count = 0

    best_score_so_far = -float("inf")
    best_logp_so_far = None
    best_abs_error_so_far = float("inf")
    best_z_so_far = None
    best_generation_so_far = -1

    best_latents_history = []

    # ======================
    # Evolution loop
    # ======================

    for gen in range(args.n_gen):
        pred_logp = predictor.predict(population)

        scores = score_logp(
            pred_logp,
            target_logp=args.target_logp,
            score_sigma=args.score_sigma
        )

        abs_error = np.abs(pred_logp - args.target_logp)
        fitness = -scores

        avg_score = float(np.mean(scores))
        avg_logp = float(np.mean(pred_logp))

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_logp = float(pred_logp[best_idx])
        best_abs_error = float(abs_error[best_idx])
        best_z = population[best_idx].copy()

        if best_score > best_score_so_far:
            best_score_so_far = best_score
            best_logp_so_far = best_logp
            best_abs_error_so_far = best_abs_error
            best_z_so_far = best_z.copy()
            best_generation_so_far = gen

        topk = min(10, len(scores))
        top10_mean_score = float(np.mean(np.sort(scores)[::-1][:topk]))
        top10_mean_abs_error = float(np.mean(np.sort(abs_error)[:topk]))

        success_mask = success_mask_logp(
            pred_logp,
            low=args.success_low,
            high=args.success_high
        )
        success_count = int(np.sum(success_mask))
        success_rate = float(success_count / len(pred_logp))

        evaluations = int((gen + 1) * len(population))
        elapsed = float(time.time() - start_wall_time)

        # Decode only the best and top-k candidates per generation for archives
        best_smi = latent_to_smiles(
            model_psvae,
            best_z,
            device=device,
            max_atom_num=args.max_atom_num,
            add_edge_th=args.add_edge_th,
            temperature=args.temperature
        )

        evolution_path.append({
            "generation": gen,
            "evaluations": evaluations,
            "smiles": best_smi,
            "pred_logP": best_logp,
            "target_abs_error": best_abs_error,
            "score": best_score,
            "success": bool(args.success_low <= best_logp <= args.success_high),
        })

        best_candidates_over_time.append({
            "generation": gen,
            "evaluations": evaluations,
            "best_smiles": best_smi,
            "best_pred_logP": best_logp,
            "best_abs_error_to_target": best_abs_error,
            "best_score": best_score,
            "best_score_so_far": float(best_score_so_far),
            "best_logP_so_far": float(best_logp_so_far),
            "best_abs_error_so_far": float(best_abs_error_so_far),
            "avg_logP": avg_logp,
            "avg_score": avg_score,
            "top10_mean_score": top10_mean_score,
            "top10_mean_abs_error_to_target": top10_mean_abs_error,
            "success_count": success_count,
            "success_rate": success_rate,
        })

        top_idx = np.argsort(scores)[::-1][:args.topk_archive]
        for rank, idx in enumerate(top_idx):
            smi = latent_to_smiles(
                model_psvae,
                population[idx],
                device=device,
                max_atom_num=args.max_atom_num,
                add_edge_th=args.add_edge_th,
                temperature=args.temperature
            )

            topk_archive.append({
                "generation": gen,
                "evaluations": evaluations,
                "rank": rank + 1,
                "smiles": smi,
                "pred_logP": float(pred_logp[idx]),
                "target_abs_error": float(abs_error[idx]),
                "score": float(scores[idx]),
                "success": bool(args.success_low <= pred_logp[idx] <= args.success_high),
            })

        progress_records.append({
            "generation": gen,
            "evaluations": evaluations,
            "elapsed_time_sec": elapsed,
            "avg_logP": avg_logp,
            "avg_score": avg_score,
            "best_logP": best_logp,
            "best_abs_error_to_target": best_abs_error,
            "best_score": best_score,
            "best_score_so_far": float(best_score_so_far),
            "best_logP_so_far": float(best_logp_so_far),
            "best_abs_error_so_far": float(best_abs_error_so_far),
            "top10_mean_score": top10_mean_score,
            "top10_mean_abs_error_to_target": top10_mean_abs_error,
            "success_count": success_count,
            "success_rate": success_rate,
        })

        avg_score_history.append(avg_score)
        avg_logp_history.append(avg_logp)
        best_score_so_far_history.append(float(best_score_so_far))
        best_logp_so_far_history.append(float(best_logp_so_far))
        best_abs_error_so_far_history.append(float(best_abs_error_so_far))
        top10_mean_score_history.append(top10_mean_score)
        top10_mean_abs_error_history.append(top10_mean_abs_error)
        success_count_history.append(success_count)
        success_rate_history.append(success_rate)
        eval_count_history.append(evaluations)
        elapsed_time_history.append(elapsed)

        best_latents_history.append(best_z.copy())

        print(
            f"[Gen {gen:03d}] "
            f"avg_score={avg_score:.6f}, "
            f"best_score={best_score:.6f}, "
            f"best_logP={best_logp:.4f}, "
            f"best_abs_error={best_abs_error:.4f}, "
            f"success={success_count}/{len(pred_logp)}, "
            f"no_improve={no_improve_count}",
            flush=True
        )

        # Early stopping monitors average score, same style as the QM9 script
        if avg_score > best_score_monitor:
            best_score_monitor = avg_score
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= args.patience:
            print(
                f"[Early Stop] no average-score improvement for {args.patience} generations.",
                flush=True
            )
            break

        # GA update
        sorted_idx = np.argsort(fitness)
        elites = population[sorted_idx[:args.elite_size]].copy()

        new_population = list(elites)

        while len(new_population) < args.pop_size:
            p1 = tournament_selection(
                population,
                fitness,
                tourn_size=args.tourn_size
            )
            p2 = tournament_selection(
                population,
                fitness,
                tourn_size=args.tourn_size
            )

            c1, c2 = arithmetic_crossover(
                p1,
                p2,
                args.cross_prob,
                lb,
                ub
            )

            c1 = polynomial_mutation(
                c1,
                args.mut_prob,
                args.mut_eta,
                lb,
                ub
            )

            c2 = polynomial_mutation(
                c2,
                args.mut_prob,
                args.mut_eta,
                lb,
                ub
            )

            new_population.append(c1)

            if len(new_population) < args.pop_size:
                new_population.append(c2)

        population = np.asarray(new_population, dtype=np.float32)

    # ======================
    # Save evolution outputs
    # ======================

    progress_df = pd.DataFrame(progress_records)
    progress_df.to_csv(os.path.join(out_dir, "progress_metrics.csv"), index=False)

    evo_df = pd.DataFrame(evolution_path)
    evo_df.to_csv(os.path.join(out_dir, "evolution_path.csv"), index=False)

    topk_df = pd.DataFrame(topk_archive)
    topk_df.to_csv(os.path.join(out_dir, "topk_evolution_paths.csv"), index=False)

    best_candidates_df = pd.DataFrame(best_candidates_over_time)
    best_candidates_df.to_csv(
        os.path.join(out_dir, "best_candidates_over_time.csv"),
        index=False
    )

    if len(best_latents_history) > 0:
        np.save(
            os.path.join(out_dir, "best_latents_history.npy"),
            np.vstack(best_latents_history).astype(np.float32)
        )

    if best_z_so_far is not None:
        np.save(
            os.path.join(out_dir, "best_latent_so_far.npy"),
            best_z_so_far.astype(np.float32)
        )

    progress_df.sort_values("best_score_so_far", ascending=False).to_csv(
        os.path.join(out_dir, "progress_sorted_by_best_score.csv"),
        index=False
    )

    if len(topk_df) > 0:
        topk_df.sort_values(
            ["score", "generation", "rank"],
            ascending=[False, True, True]
        ).to_csv(
            os.path.join(out_dir, "topk_sorted_by_score.csv"),
            index=False
        )

    # ======================
    # Final evaluation
    # ======================

    print("\n[INFO] Final population prediction...", flush=True)

    final_pred_logp = predictor.predict(population)
    final_scores = score_logp(
        final_pred_logp,
        target_logp=args.target_logp,
        score_sigma=args.score_sigma
    )
    final_abs_error = np.abs(final_pred_logp - args.target_logp)
    final_success = success_mask_logp(
        final_pred_logp,
        low=args.success_low,
        high=args.success_high
    )

    print("[INFO] Decoding final population...", flush=True)

    decoded_smiles = []
    decode_success = 0

    for i in range(len(population)):
        smi = latent_to_smiles(
            model_psvae,
            population[i],
            device=device,
            max_atom_num=args.max_atom_num,
            add_edge_th=args.add_edge_th,
            temperature=args.temperature
        )

        decoded_smiles.append(smi)

        if smi is not None:
            decode_success += 1

    valid_smiles = [s for s in decoded_smiles if s is not None]
    unique_valid = len(set(valid_smiles)) if len(valid_smiles) > 0 else 0
    validity = float(decode_success / len(decoded_smiles)) if len(decoded_smiles) > 0 else 0.0
    uniqueness = float(unique_valid / len(valid_smiles)) if len(valid_smiles) > 0 else 0.0
    diversity = compute_diversity(valid_smiles) if len(valid_smiles) > 1 else 0.0

    final_rows = []
    for i in range(len(population)):
        rdkit_logp = calc_rdkit_logp(decoded_smiles[i]) if decoded_smiles[i] is not None else None

        final_rows.append({
            "idx": i,
            "smiles": decoded_smiles[i],
            "pred_logP": float(final_pred_logp[i]),
            "target_logP": float(args.target_logp),
            "target_abs_error": float(final_abs_error[i]),
            "score": float(final_scores[i]),
            "success": bool(final_success[i]),
            "rdkit_logP_decoded": rdkit_logp,
        })

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(
        ["score", "target_abs_error"],
        ascending=[False, True]
    ).reset_index(drop=True)
    final_df["rank_by_score"] = np.arange(1, len(final_df) + 1)

    final_csv_path = os.path.join(out_dir, f"final_population_{args.version}.csv")
    final_df.to_csv(final_csv_path, index=False)

    np.save(os.path.join(out_dir, "final_population_latent.npy"), population)
    np.save(os.path.join(out_dir, "final_population_pred_logp.npy"), final_pred_logp)
    np.save(os.path.join(out_dir, "final_population_score.npy"), final_scores)
    np.save(os.path.join(out_dir, "final_population_abs_error.npy"), final_abs_error)

    # ======================
    # Summary
    # ======================

    final_success_count = int(np.sum(final_success))
    final_success_rate = float(final_success_count / len(final_success))

    best_final_row = final_df.iloc[0]
    topk_final = min(10, len(final_df))

    top10_mean_score_final = float(final_df["score"].head(topk_final).mean())
    top10_mean_abs_error_final = float(final_df["target_abs_error"].head(topk_final).mean())

    total_time_sec = float(time.time() - start_wall_time)
    total_evaluations = int(len(progress_df) * args.pop_size)

    best_ever_success = int(
        args.success_low <= float(best_logp_so_far) <= args.success_high
    ) if best_logp_so_far is not None else 0

    summary = {
        "task": "zinc_logp_target_optimization",
        "init_mode": args.init_mode,
        "version": args.version,
        "seed": args.seed,

        "zinc_psvae_ckpt": args.zinc_psvae_ckpt,
        "predictor_ckpt": args.predictor_ckpt,
        "latent_pool": args.latent_pool,

        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "elite_size": args.elite_size,
        "cross_prob": args.cross_prob,
        "mut_prob": args.mut_prob,
        "mut_eta": args.mut_eta,
        "tourn_size": args.tourn_size,
        "patience": args.patience,

        "target_logP": float(args.target_logp),
        "score_sigma": float(args.score_sigma),
        "success_low": float(args.success_low),
        "success_high": float(args.success_high),

        "success_count_final": final_success_count,
        "success_rate_final": final_success_rate,
        "best_ever_success": best_ever_success,

        "best_score_final": float(best_final_row["score"]),
        "best_logP_final": float(best_final_row["pred_logP"]),
        "best_abs_error_final": float(best_final_row["target_abs_error"]),

        "best_score_so_far": float(best_score_so_far),
        "best_logP_so_far": float(best_logp_so_far),
        "best_abs_error_so_far": float(best_abs_error_so_far),
        "best_generation_so_far": int(best_generation_so_far),

        "avg_logP_final": float(np.mean(final_pred_logp)),
        "median_logP_final": float(np.median(final_pred_logp)),
        "avg_score_final": float(np.mean(final_scores)),
        "median_score_final": float(np.median(final_scores)),
        "top10_mean_score_final": top10_mean_score_final,
        "top10_mean_abs_error_final": top10_mean_abs_error_final,

        "validity": validity,
        "decode_success": int(decode_success),
        "uniqueness": uniqueness,
        "diversity": diversity,

        "time_sec_total": total_time_sec,
        "n_evaluations_total": total_evaluations,

        "best_smiles_final": best_final_row["smiles"],
        "best_pred_logP_final": float(best_final_row["pred_logP"]),
        "best_rdkit_logP_decoded": (
            None if pd.isna(best_final_row["rdkit_logP_decoded"])
            else float(best_final_row["rdkit_logP_decoded"])
        ),

        "avg_score_history": [float(x) for x in avg_score_history],
        "avg_logp_history": [float(x) for x in avg_logp_history],
        "best_score_so_far_history": [float(x) for x in best_score_so_far_history],
        "best_logp_so_far_history": [float(x) for x in best_logp_so_far_history],
        "best_abs_error_so_far_history": [float(x) for x in best_abs_error_so_far_history],
        "top10_mean_score_history": [float(x) for x in top10_mean_score_history],
        "top10_mean_abs_error_history": [float(x) for x in top10_mean_abs_error_history],
        "success_count_history": [int(x) for x in success_count_history],
        "success_rate_history": [float(x) for x in success_rate_history],
        "eval_count_history": [int(x) for x in eval_count_history],
        "elapsed_time_history": [float(x) for x in elapsed_time_history],
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))

    # ======================
    # Figures
    # ======================

    # Fig 2: convergence by evaluations
    plt.figure(figsize=(8, 5))
    plt.plot(
        eval_count_history,
        best_score_so_far_history,
        marker="o",
        label="Best-so-far score"
    )
    plt.plot(
        eval_count_history,
        top10_mean_score_history,
        marker="s",
        label="Top-10 mean score"
    )
    plt.xlabel("Molecular evaluations")
    plt.ylabel("logP target score")
    plt.title(f"Convergence curve ({args.init_mode})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_convergence_curve.png"), dpi=300)
    plt.close()

    # Additional absolute-error curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        eval_count_history,
        best_abs_error_so_far_history,
        marker="o",
        label="Best-so-far |logP - target|"
    )
    plt.plot(
        eval_count_history,
        top10_mean_abs_error_history,
        marker="s",
        label="Top-10 mean |logP - target|"
    )
    plt.xlabel("Molecular evaluations")
    plt.ylabel("|predicted logP - target logP|")
    plt.title(f"Target-error curve ({args.init_mode})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2b_target_error_curve.png"), dpi=300)
    plt.close()

    # Fig 3A: score vs evaluations
    plt.figure(figsize=(8, 5))
    plt.plot(eval_count_history, avg_score_history, marker="o")
    plt.xlabel("Molecular evaluations")
    plt.ylabel("Average logP target score")
    plt.title(f"Efficiency curve: score vs evaluations ({args.init_mode})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3a_score_vs_evaluations.png"), dpi=300)
    plt.close()

    # Fig 3B: success count vs evaluations
    plt.figure(figsize=(8, 5))
    plt.step(eval_count_history, success_count_history, where="post")
    plt.xlabel("Molecular evaluations")
    plt.ylabel("Success count")
    plt.title(f"Success count vs evaluations ({args.init_mode})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3b_success_vs_evaluations.png"), dpi=300)
    plt.close()

    # Fig 3C: success count vs time
    plt.figure(figsize=(8, 5))
    plt.step(elapsed_time_history, success_count_history, where="post")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Success count")
    plt.title(f"Success count vs time ({args.init_mode})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3c_success_vs_time.png"), dpi=300)
    plt.close()

    # Fig 5: PCA / UMAP
    n_train_vis = min(args.n_train_vis, len(latent_train))
    train_vis_idx = np.random.choice(len(latent_train), n_train_vis, replace=False)
    train_vis = latent_train[train_vis_idx]
    gen_vis = population.copy()

    X_all = np.vstack([train_vis, gen_vis])
    labels = (["train"] * len(train_vis)) + (["generated"] * len(gen_vis))

    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(X_all)

    space_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels
    })
    space_df.to_csv(os.path.join(out_dir, "chemical_space_pca.csv"), index=False)

    plt.figure(figsize=(8, 6))
    train_mask = np.array(labels) == "train"
    gen_mask = np.array(labels) == "generated"

    plt.scatter(
        coords[train_mask, 0],
        coords[train_mask, 1],
        s=8,
        alpha=0.25,
        c="0.70",
        label="ZINC training latent"
    )
    plt.scatter(
        coords[gen_mask, 0],
        coords[gen_mask, 1],
        s=18,
        alpha=0.85,
        label="Final population"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA projection of final latent population ({args.init_mode})")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_pca_chemical_space.png"), dpi=300)
    plt.close()

    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            random_state=args.seed,
            n_neighbors=30,
            min_dist=0.15,
            metric="euclidean"
        )
        umap_coords = reducer.fit_transform(X_all)

        umap_df = pd.DataFrame({
            "x": umap_coords[:, 0],
            "y": umap_coords[:, 1],
            "label": labels
        })
        umap_df.to_csv(os.path.join(out_dir, "chemical_space_umap.csv"), index=False)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            umap_coords[train_mask, 0],
            umap_coords[train_mask, 1],
            s=8,
            alpha=0.25,
            c="0.70",
            label="ZINC training latent"
        )
        plt.scatter(
            umap_coords[gen_mask, 0],
            umap_coords[gen_mask, 1],
            s=18,
            alpha=0.85,
            label="Final population"
        )
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title(f"UMAP projection of final latent population ({args.init_mode})")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig5_umap_chemical_space.png"), dpi=300)
        plt.close()

    print("\n========== DONE ==========")
    print(f"Mode: {args.init_mode}")
    print(f"Output dir: {out_dir}")
    print(f"Best final score: {summary['best_score_final']:.6f}")
    print(f"Best final predicted logP: {summary['best_logP_final']:.6f}")
    print(f"Best final abs error: {summary['best_abs_error_final']:.6f}")
    print(f"Best-so-far score: {summary['best_score_so_far']:.6f}")
    print(f"Best-so-far predicted logP: {summary['best_logP_so_far']:.6f}")
    print(f"Final success count: {final_success_count}")
    print(f"Final success rate: {final_success_rate:.4f}")
    print(f"Validity: {validity:.4f}")
    print(f"Uniqueness: {uniqueness:.4f}")
    print(f"Diversity: {diversity:.4f}")
    print(f"Total time: {total_time_sec:.2f} s")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Best final SMILES: {summary['best_smiles_final']}")


if __name__ == "__main__":
    main()