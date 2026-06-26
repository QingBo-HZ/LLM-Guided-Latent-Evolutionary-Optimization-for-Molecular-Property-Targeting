#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
RDLogger.DisableLog("rdApp.*")

PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

from pl_models import PSVAEModel
from data.bpe_dataset import BPEMolDataset
from utils.chem_utils import smiles2molecule, GeneralVocab
from data.mol_bpe import Tokenizer
from rdkit.Chem.rdchem import BondType
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


CKPT_PSVAE = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt"
PREDICTOR_CKPT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/best_predictor.pt"
MEAN_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_mean.npy"
STD_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_std.npy"
TRAIN_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy"

DEFAULT_TRAIN_SMILES_CSV = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/train_labeled.csv"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def canonicalize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def gap_to_score(gap, threshold=0.15, scale=0.03):
    return float(1.0 / (1.0 + np.exp((gap - threshold) / scale)))


def compute_diversity(smiles_list):
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s)) if s is not None else None
        if m is not None:
            mols.append(m)

    if len(mols) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    return 1.0 - float(np.mean(sims))


class Predictor(nn.Module):
    def __init__(self, dim_feature, dim_hidden, num_property, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_feature, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(dim_hidden, num_property)

    def forward(self, x):
        return self.output(self.mlp(x))


class QM9PredictorAPI:
    def __init__(self, predictor_ckpt, mean_path, std_path, device="cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(predictor_ckpt, map_location=self.device)

        self.model = Predictor(
            dim_feature=ckpt["dim_feature"],
            dim_hidden=ckpt["hidden_dim"],
            num_property=ckpt["num_property"],
            dropout=ckpt.get("dropout", 0.0),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.property_names = ckpt["property_names"]
        self.gap_idx = self.property_names.index("gap")
        self.y_mean = np.load(mean_path)
        self.y_std = np.load(std_path)

    def enforce_physical_constraints(self, pred, margin=1e-6):
        pred = pred.copy()
        if pred.shape[1] >= 3:
            homo = pred[:, 0]
            lumo = pred[:, 1]
            gap = pred[:, 2]

            gap = np.maximum(gap, margin)
            bad = lumo <= homo + margin
            lumo[bad] = homo[bad] + gap[bad]
            gap = lumo - homo

            pred[:, 1] = lumo
            pred[:, 2] = gap

        return pred

    def predict_array(self, z):
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]

        x = torch.tensor(z, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy()

        pred = pred_norm * self.y_std + self.y_mean
        return self.enforce_physical_constraints(pred)


print(f"[INFO] device = {DEVICE}")

print("[INFO] loading PS-VAE...")
model_psvae = PSVAEModel.load_from_checkpoint(CKPT_PSVAE, map_location=DEVICE)
model_psvae.eval()
model_psvae.to(DEVICE)

print("[INFO] loading predictor...")
predictor = QM9PredictorAPI(
    predictor_ckpt=PREDICTOR_CKPT,
    mean_path=MEAN_PATH,
    std_path=STD_PATH,
    device=DEVICE,
)
print(f"[INFO] predictor properties = {predictor.property_names}")

print("[INFO] loading train latent...")
latent_train = np.load(TRAIN_LATENT_PATH).astype(np.float32)
print(f"[INFO] latent_train shape = {latent_train.shape}")

LATENT_CACHE = {}
EVAL_CACHE = {}


def get_z_mean_from_mol(mol):
    step1_res = BPEMolDataset.process_step1(mol, model_psvae.tokenizer)
    step2_res = BPEMolDataset.process_step2(step1_res, model_psvae.tokenizer)
    batch = BPEMolDataset.process_step3(
        [step2_res],
        model_psvae.tokenizer,
        device=model_psvae.device,
    )

    x, edge_index, edge_attr = batch["x"], batch["edge_index"], batch["edge_attr"]
    x_pieces, x_pos = batch["x_pieces"], batch["x_pos"]
    x = model_psvae.decoder.embed_atom(x, x_pieces, x_pos)
    batch_size, node_num, node_dim = x.shape
    graph_ids = torch.repeat_interleave(
        torch.arange(0, batch_size, device=x.device),
        node_num,
    )
    _, all_x = model_psvae.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
    graph_embedding = model_psvae.encoder.embed_graph(
        all_x,
        graph_ids,
        batch["atom_mask"].flatten(),
    )
    return model_psvae.decoder.W_mean(graph_embedding).squeeze(0)


def smiles_to_latent(smi):
    can_smi = canonicalize_smiles(smi)
    if can_smi is None:
        return None
    if can_smi in LATENT_CACHE:
        return LATENT_CACHE[can_smi].copy()

    mol = smiles2molecule(can_smi, kekulize=True)
    if mol is None:
        return None

    try:
        mol = Chem.RemoveHs(mol)
        with torch.no_grad():
            z = get_z_mean_from_mol(mol)
            if z.dim() > 1:
                z = z.squeeze(0)
            z_arr = z.detach().cpu().numpy().astype(np.float32)
            LATENT_CACHE[can_smi] = z_arr
            return z_arr.copy()
    except Exception:
        return None


def load_train_smiles(train_smiles_csv, smiles_col):
    if train_smiles_csv.endswith(".csv"):
        df = pd.read_csv(train_smiles_csv)
    elif train_smiles_csv.endswith(".tsv") or train_smiles_csv.endswith(".txt"):
        df = pd.read_csv(train_smiles_csv, sep="\t")
    else:
        raise ValueError("Only csv/tsv/txt are supported.")

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}; columns={list(df.columns)}")

    smiles = [canonicalize_smiles(s) for s in df[smiles_col].astype(str).tolist()]
    smiles = [s for s in smiles if s is not None]
    smiles = list(dict.fromkeys(smiles))
    return smiles


def inspect_labeled_gap(train_smiles_csv, smiles_col, success_threshold):
    df = pd.read_csv(train_smiles_csv)

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}; columns={list(df.columns)}")
    if "gap" not in df.columns:
        print("[WARN] no gap column found in labeled file.")
        return None

    df = df[[smiles_col, "gap"]].dropna().copy()
    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.drop_duplicates("canonical_smiles")

    print(f"[INFO] labeled molecules: {len(df)}")
    print(f"[INFO] labeled gap min: {df['gap'].min():.6f}")
    print(f"[INFO] labeled gap mean: {df['gap'].mean():.6f}")
    print(f"[INFO] labeled gap median: {df['gap'].median():.6f}")
    print(f"[INFO] labeled gap < {success_threshold}: {int((df['gap'] < success_threshold).sum())}")
    return df


def smiles_to_brics_fragments(smi, min_frags=1, max_frags=8):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        frags = list(BRICSDecompose(mol, returnMols=False))
        frags = [canonicalize_smiles(f) for f in frags]
        frags = [f for f in frags if f is not None]
        frags = list(dict.fromkeys(frags))

        if len(frags) < min_frags or len(frags) > max_frags:
            return None

        return frags
    except Exception:
        return None


def build_fragment_library(train_smiles, max_mols_for_library=5000):
    lib = []
    subset = train_smiles[:max_mols_for_library]

    for smi in subset:
        frags = smiles_to_brics_fragments(smi)
        if frags is not None:
            lib.extend(frags)

    lib = [canonicalize_smiles(f) for f in lib]
    lib = [f for f in lib if f is not None]
    lib = list(dict.fromkeys(lib))
    return lib


def safe_brics_build(fragment_smiles_list, max_products=20):
    try:
        frag_mols = []
        for fs in fragment_smiles_list:
            m = Chem.MolFromSmiles(fs)
            if m is not None:
                frag_mols.append(m)

        if len(frag_mols) == 0:
            return None

        builder = BRICSBuild(
            frag_mols,
            maxDepth=3,
            scrambleReagents=True,
            uniquify=True,
        )

        products = []
        for prod in builder:
            try:
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(prod, canonical=True)
                if smi is not None:
                    products.append(smi)
            except Exception:
                pass

            if len(products) >= max_products:
                break

        if len(products) == 0:
            return None

        return random.choice(products)
    except Exception:
        return None


def fragment_mutation(parent_smi, fragment_library):
    parent_frags = smiles_to_brics_fragments(parent_smi)

    if parent_frags is None or len(parent_frags) == 0:
        return canonicalize_smiles(parent_smi)

    child_frags = parent_frags.copy()
    replace_idx = random.randrange(len(child_frags))
    child_frags[replace_idx] = random.choice(fragment_library)

    child_smi = safe_brics_build(child_frags, max_products=20)
    if child_smi is None:
        return canonicalize_smiles(parent_smi)

    return canonicalize_smiles(child_smi)


def fragment_crossover(parent1_smi, parent2_smi):
    frags1 = smiles_to_brics_fragments(parent1_smi)
    frags2 = smiles_to_brics_fragments(parent2_smi)

    if frags1 is None or len(frags1) == 0:
        return canonicalize_smiles(parent1_smi)
    if frags2 is None or len(frags2) == 0:
        return canonicalize_smiles(parent2_smi)

    take1 = max(1, len(frags1) // 2)
    take2 = max(1, len(frags2) // 2)

    sel1 = random.sample(frags1, min(take1, len(frags1)))
    sel2 = random.sample(frags2, min(take2, len(frags2)))

    mixed = list(dict.fromkeys(sel1 + sel2))
    child_smi = safe_brics_build(mixed, max_products=20)

    if child_smi is None:
        return canonicalize_smiles(random.choice([parent1_smi, parent2_smi]))

    return canonicalize_smiles(child_smi)


def init_population_smiles(train_smiles, pop_size):
    idx = np.random.choice(
        len(train_smiles),
        pop_size,
        replace=len(train_smiles) < pop_size,
    )
    return [train_smiles[i] for i in idx]


def init_population_from_label_gap_no_leakage(
    train_smiles_csv,
    smiles_col,
    pop_size,
    warm_frac=0.8,
    success_threshold=0.15,
    gap_upper=0.25,
):
    df = pd.read_csv(train_smiles_csv)

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}; columns={list(df.columns)}")
    if "gap" not in df.columns:
        raise ValueError(f"gap column not found; columns={list(df.columns)}")

    df = df[[smiles_col, "gap"]].dropna().copy()
    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.drop_duplicates("canonical_smiles")

    near_df = df[(df["gap"] >= success_threshold) & (df["gap"] <= gap_upper)].copy()

    if len(near_df) < pop_size:
        print(f"[WARN] near-threshold pool too small: {len(near_df)}; fallback to gap >= threshold.")
        near_df = df[df["gap"] >= success_threshold].copy()

    if len(near_df) == 0:
        raise RuntimeError("No warm-start molecules found.")

    near_df = near_df.sort_values("gap", ascending=True).reset_index(drop=True)

    warm_n = int(pop_size * warm_frac)
    warm_n = max(1, min(warm_n, len(near_df)))

    warm_smiles = near_df["canonical_smiles"].head(warm_n).tolist()

    rest_n = pop_size - len(warm_smiles)
    if rest_n > 0:
        rest_pool = near_df["canonical_smiles"].iloc[warm_n:].tolist()
        if len(rest_pool) == 0:
            rest_pool = near_df["canonical_smiles"].tolist()

        idx = np.random.choice(len(rest_pool), rest_n, replace=len(rest_pool) < rest_n)
        rest_smiles = [rest_pool[i] for i in idx]
    else:
        rest_smiles = []

    population = warm_smiles + rest_smiles
    random.shuffle(population)

    print("[INFO] warm-start from label gap WITHOUT label leakage")
    print(f"[INFO] success_threshold = {success_threshold}")
    print(f"[INFO] gap_upper = {gap_upper}")
    print(f"[INFO] candidate pool size = {len(near_df)}")
    print(f"[INFO] best seed label-gap = {near_df['gap'].min():.6f}")
    print(f"[INFO] top10 seed label-gap mean = {near_df['gap'].head(10).mean():.6f}")
    print(f"[INFO] warm-start size = {len(warm_smiles)}")

    return population[:pop_size]


def evaluate_population(pop_smiles, success_threshold=0.15):
    rows = []

    for smi in pop_smiles:
        can_smi = canonicalize_smiles(smi)

        if can_smi is None:
            rows.append({
                "smiles": smi,
                "canonical_smiles": None,
                "source": "invalid",
                "latent_ok": 0,
                "gap": np.inf,
                "score": 0.0,
            })
            continue

        if can_smi in EVAL_CACHE:
            rows.append(EVAL_CACHE[can_smi].copy())
            continue

        z = smiles_to_latent(can_smi)
        if z is None:
            row = {
                "smiles": can_smi,
                "canonical_smiles": can_smi,
                "source": "predictor_failed",
                "latent_ok": 0,
                "gap": np.inf,
                "score": 0.0,
            }
            EVAL_CACHE[can_smi] = row.copy()
            rows.append(row)
            continue

        pred = predictor.predict_array(z[None, :])[0]
        gap = float(pred[predictor.gap_idx])
        score = gap_to_score(gap, threshold=success_threshold, scale=0.03)

        row = {
            "smiles": can_smi,
            "canonical_smiles": can_smi,
            "source": "predictor",
            "latent_ok": 1,
            "gap": gap,
            "score": score,
        }

        for j, p in enumerate(predictor.property_names):
            row[p] = float(pred[j])

        EVAL_CACHE[can_smi] = row.copy()
        rows.append(row)

    return rows


def tournament_selection(pop_smiles, fitness, tourn_size=5):
    idx = np.random.choice(len(pop_smiles), tourn_size, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pop_smiles[best]


def propose_child(p1, p2, fragment_library, train_smiles, cross_prob, mut_prob):
    if random.random() < cross_prob:
        child = fragment_crossover(p1, p2)
    else:
        child = p1

    if random.random() < mut_prob:
        mutated = fragment_mutation(child, fragment_library)
        if mutated is not None:
            child = mutated

    if child is None:
        child = random.choice(train_smiles)

    return canonicalize_smiles(child)


def propose_best_child(
    population,
    fitness_full,
    fragment_library,
    train_smiles,
    cross_prob,
    mut_prob,
    success_threshold,
    child_trials=10,
    tourn_size=5,
):
    candidates = []

    for _ in range(child_trials):
        p1 = tournament_selection(population, fitness_full, tourn_size=tourn_size)
        p2 = tournament_selection(population, fitness_full, tourn_size=tourn_size)

        child = propose_child(
            p1=p1,
            p2=p2,
            fragment_library=fragment_library,
            train_smiles=train_smiles,
            cross_prob=cross_prob,
            mut_prob=mut_prob,
        )

        if child is not None:
            candidates.append(child)

    if len(candidates) == 0:
        return random.choice(train_smiles), np.inf, 0.0

    eval_rows = evaluate_population(candidates, success_threshold=success_threshold)
    df = pd.DataFrame(eval_rows)
    df = df[df["latent_ok"] == 1].copy()

    if len(df) == 0:
        return random.choice(train_smiles), np.inf, 0.0

    best_row = df.sort_values("gap", ascending=True).iloc[0]
    return best_row["smiles"], float(best_row["gap"]), float(best_row["score"])


def main():
    parser = argparse.ArgumentParser("QM9 SMILES/Fragment-GA with child selection")

    parser.add_argument("--train_smiles_csv", type=str, default=DEFAULT_TRAIN_SMILES_CSV)
    parser.add_argument("--smiles_col", type=str, default="smiles")

    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_gen", type=int, default=1000)
    parser.add_argument("--elite_size", type=int, default=30)

    parser.add_argument("--mut_prob", type=float, default=0.20)
    parser.add_argument("--cross_prob", type=float, default=0.20)

    parser.add_argument("--child_trials", type=int, default=10)
    parser.add_argument("--tourn_size", type=int, default=5)
    parser.add_argument("--random_immigrant_frac", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="smiles_childselect_v1")
    parser.add_argument("--success_threshold", type=float, default=0.15)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--fragment_lib_max_mols", type=int, default=50000)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--warm_start_frac", type=float, default=0.8)
    parser.add_argument("--warm_start_gap_upper", type=float, default=0.25)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = os.path.join(args.output_root, f"fragment_ga_{args.version}")
    ensure_dir(out_dir)

    print("\n========== CONFIG ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    inspect_labeled_gap(args.train_smiles_csv, args.smiles_col, args.success_threshold)

    train_smiles = load_train_smiles(args.train_smiles_csv, args.smiles_col)
    print(f"[INFO] usable train SMILES: {len(train_smiles)}")

    fragment_library = build_fragment_library(
        train_smiles,
        max_mols_for_library=args.fragment_lib_max_mols,
    )
    print(f"[INFO] BRICS fragment library size: {len(fragment_library)}")

    if len(fragment_library) == 0:
        raise RuntimeError("fragment_library is empty.")

    if args.warm_start:
        population = init_population_from_label_gap_no_leakage(
            train_smiles_csv=args.train_smiles_csv,
            smiles_col=args.smiles_col,
            pop_size=args.pop_size,
            warm_frac=args.warm_start_frac,
            success_threshold=args.success_threshold,
            gap_upper=args.warm_start_gap_upper,
        )
    else:
        population = init_population_smiles(train_smiles, args.pop_size)

    init_df = pd.DataFrame(evaluate_population(population, success_threshold=args.success_threshold))
    init_valid = init_df[np.isfinite(init_df["gap"])].copy()
    print("\n[DEBUG] initial predictor-gap statistics:")
    print(init_valid["gap"].describe())
    print("\n[DEBUG] initial top10:")
    print(init_valid.sort_values("gap").head(10)[["smiles", "gap", "score", "source"]])

    start_wall = time.time()

    avg_score_history = []
    avg_gap_history = []
    best_gap_history = []
    best_score_history = []
    best_gap_so_far_history = []
    top10_mean_gap_history = []
    success_count_history = []
    success_rate_history = []
    eval_count_history = []
    elapsed_time_history = []
    best_smiles_history = []

    best_gap_so_far = float("inf")
    total_evaluations = 0

    for gen in range(args.n_gen):
        eval_rows = evaluate_population(population, success_threshold=args.success_threshold)
        total_evaluations += len(population)

        df_eval = pd.DataFrame(eval_rows)
        valid_df = df_eval[df_eval["latent_ok"] == 1].copy()

        if len(valid_df) == 0:
            print(f"[Gen {gen:03d}] no valid molecules. stop.")
            break

        gaps = valid_df["gap"].values.astype(np.float32)
        scores = valid_df["score"].values.astype(np.float32)

        avg_gap = float(np.mean(gaps))
        avg_score = float(np.mean(scores))
        best_gap = float(np.min(gaps))
        best_score = float(np.max(scores))
        topk = min(10, len(gaps))
        top10_mean_gap = float(np.mean(np.sort(gaps)[:topk]))

        success_count = int(np.sum(gaps < args.success_threshold))
        success_rate = float(success_count / len(gaps))

        if best_gap < best_gap_so_far:
            best_gap_so_far = best_gap

        best_idx = int(np.argmin(gaps))
        best_smiles = valid_df.iloc[best_idx]["smiles"]

        avg_gap_history.append(avg_gap)
        avg_score_history.append(avg_score)
        best_gap_history.append(best_gap)
        best_score_history.append(best_score)
        best_gap_so_far_history.append(best_gap_so_far)
        top10_mean_gap_history.append(top10_mean_gap)
        success_count_history.append(success_count)
        success_rate_history.append(success_rate)
        eval_count_history.append(total_evaluations)
        elapsed_time_history.append(float(time.time() - start_wall))
        best_smiles_history.append(best_smiles)

        source_counts = valid_df["source"].value_counts().to_dict()

        print(
            f"[Gen {gen:03d}] "
            f"avg_gap={avg_gap:.6f}, "
            f"best_gap={best_gap:.6f}, "
            f"best_so_far={best_gap_so_far:.6f}, "
            f"top10={top10_mean_gap:.6f}, "
            f"success={success_count}/{len(gaps)}, "
            f"sources={source_counts}"
        )

        fitness_full = np.full(len(population), 1e6, dtype=np.float32)
        valid_idx = valid_df.index.to_numpy()
        fitness_full[valid_idx] = -valid_df["score"].values.astype(np.float32)

        sorted_idx = np.argsort(fitness_full)
        elites = [population[i] for i in sorted_idx[:args.elite_size]]

        new_population = list(elites)

        immigrant_n = int(args.pop_size * args.random_immigrant_frac)
        immigrant_n = max(0, min(immigrant_n, args.pop_size - len(new_population)))

        while len(new_population) < args.pop_size - immigrant_n:
            child, _, _ = propose_best_child(
                population=population,
                fitness_full=fitness_full,
                fragment_library=fragment_library,
                train_smiles=train_smiles,
                cross_prob=args.cross_prob,
                mut_prob=args.mut_prob,
                success_threshold=args.success_threshold,
                child_trials=args.child_trials,
                tourn_size=args.tourn_size,
            )
            new_population.append(child)

        while len(new_population) < args.pop_size:
            new_population.append(random.choice(train_smiles))

        population = new_population[:args.pop_size]

    final_eval = evaluate_population(population, success_threshold=args.success_threshold)
    final_df = pd.DataFrame(final_eval)
    final_valid_df = final_df[final_df["latent_ok"] == 1].copy()

    final_df.to_csv(os.path.join(out_dir, "final_population_fragment_ga.csv"), index=False)

    diversity = compute_diversity(final_df["smiles"].dropna().tolist())
    validity = float(np.mean(final_df["latent_ok"].values))

    if len(final_valid_df) == 0:
        best_gap_final = np.inf
        avg_gap_final = np.inf
        median_gap_final = np.inf
        top10_mean_gap_final = np.inf
        best_score_final = 0.0
        avg_score_final = 0.0
        top10_mean_gap_score_final = 0.0
        best_smiles_final = None
        best_properties_final = {}
        final_success_count = 0
        final_success_rate = 0.0
    else:
        final_gap = final_valid_df["gap"].values.astype(np.float32)
        final_score = final_valid_df["score"].values.astype(np.float32)

        best_gap_final = float(np.min(final_gap))
        avg_gap_final = float(np.mean(final_gap))
        median_gap_final = float(np.median(final_gap))
        topk = min(10, len(final_gap))
        top10_mean_gap_final = float(np.mean(np.sort(final_gap)[:topk]))

        best_score_final = float(np.max(final_score))
        avg_score_final = float(np.mean(final_score))
        top10_mean_gap_score_final = float(np.mean(np.sort(final_score)[::-1][:topk]))

        best_idx = int(np.argmin(final_gap))
        best_smiles_final = final_valid_df.iloc[best_idx]["smiles"]

        z_best = smiles_to_latent(best_smiles_final)
        if z_best is not None:
            pred_best = predictor.predict_array(z_best[None, :])[0]
            best_properties_final = {
                p: float(pred_best[j]) for j, p in enumerate(predictor.property_names)
            }
        else:
            best_properties_final = {}

        final_success_count = int(np.sum(final_gap < args.success_threshold))
        final_success_rate = float(final_success_count / len(final_gap))

    total_time = float(time.time() - start_wall)

    progress_df = pd.DataFrame({
        "generation": np.arange(len(avg_gap_history)),
        "evaluations": eval_count_history,
        "elapsed_time_sec": elapsed_time_history,
        "avg_gap": avg_gap_history,
        "avg_score": avg_score_history,
        "best_gap": best_gap_history,
        "best_score": best_score_history,
        "best_gap_so_far": best_gap_so_far_history,
        "top10_mean_gap": top10_mean_gap_history,
        "success_count": success_count_history,
        "success_rate": success_rate_history,
    })
    progress_df.to_csv(os.path.join(out_dir, "progress_metrics.csv"), index=False)

    evo_rows = []
    for gid, smi in enumerate(best_smiles_history):
        can = canonicalize_smiles(smi)
        z = smiles_to_latent(can)
        if z is not None:
            pred = predictor.predict_array(z[None, :])[0]
            gap_val = float(pred[predictor.gap_idx])
        else:
            gap_val = np.inf
        evo_rows.append({
            "generation": int(gid),
            "smiles": can,
            "gap": gap_val,
        })
    pd.DataFrame(evo_rows).to_csv(os.path.join(out_dir, "evolution_path_full.csv"), index=False)

    summary = {
        "method": "SMILES-GA / Fragment-GA child-selection",
        "task_definition": "BRICS mutation/crossover with multiple child trials, deterministic PS-VAE mean encoding, and cached predictor evaluation",
        "version": args.version,
        "seed": args.seed,
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "elite_size": args.elite_size,
        "mut_prob": args.mut_prob,
        "cross_prob": args.cross_prob,
        "child_trials": args.child_trials,
        "tourn_size": args.tourn_size,
        "random_immigrant_frac": args.random_immigrant_frac,
        "success_threshold": float(args.success_threshold),
        "deterministic_encoder": True,
        "eval_cache_by_canonical_smiles": True,
        "best_gap_final": best_gap_final,
        "avg_gap_final": avg_gap_final,
        "median_gap_final": median_gap_final,
        "top10_mean_gap_final": top10_mean_gap_final,
        "best_score_final": best_score_final,
        "avg_score_final": avg_score_final,
        "top10_mean_gap_score_final": top10_mean_gap_score_final,
        "success_count_final": final_success_count,
        "success_rate_final": final_success_rate,
        "diversity": diversity,
        "validity": validity,
        "time_sec_total": total_time,
        "n_evaluations_total": total_evaluations,
        "best_smiles_final": best_smiles_final,
        "best_properties_final": best_properties_final,
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))

    plt.figure(figsize=(8, 5))
    plt.plot(progress_df["generation"], progress_df["best_gap_so_far"], label="Best-so-far gap")
    plt.plot(progress_df["generation"], progress_df["top10_mean_gap"], label="Top-10 mean gap")
    plt.xlabel("Generation")
    plt.ylabel("Gap")
    plt.title("Convergence Curve (SMILES/Fragment-GA child-selection)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_convergence_curve.png"), dpi=300)
    plt.close()

    print("\n========== DONE ==========")
    print(f"Output dir: {out_dir}")
    print(f"Final best gap: {best_gap_final:.6f}")
    print(f"Final avg gap: {avg_gap_final:.6f}")
    print(f"Final top10 gap: {top10_mean_gap_final:.6f}")
    print(f"Final success rate: {final_success_rate:.4f}")
    print(f"Validity: {validity:.4f}")
    print(f"Diversity: {diversity:.4f}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Best SMILES: {best_smiles_final}")


if __name__ == "__main__":
    main()