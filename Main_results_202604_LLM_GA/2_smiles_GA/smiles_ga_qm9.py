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


# ====================== 环境设置 ======================
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
RDLogger.DisableLog("rdApp.*")


# ====================== 添加 PS-VAE 模块路径 ======================
PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))
print("[DEBUG] sys.path appended", flush=True)


# ====================== 导入 PS-VAE 模型 ======================
print("[DEBUG] importing PSVAEModel...", flush=True)
from pl_models import PSVAEModel


# ====================== 导入解码与 checkpoint safe globals ======================
from utils.chem_utils import smiles2molecule, GeneralVocab
from data.mol_bpe import Tokenizer
from rdkit.Chem.rdchem import BondType
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)

print(f"[DEBUG] registered safe globals: {[str(x) for x in SAFE_GLOBALS]}", flush=True)


# ====================== 默认路径 ======================
CKPT_PSVAE = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt"

PREDICTOR_CKPT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/best_predictor.pt"

# 注意：如果 predictor_ckpt_V2 目录里有 y_mean.npy / y_std.npy，建议你改成 V2 目录
MEAN_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_mean.npy"
STD_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_std.npy"

TRAIN_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy"

DEFAULT_TRAIN_SMILES_CSV = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/train_labeled.csv"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ====================== 工具函数 ======================
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


def canonicalize_smiles(smi: str):
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
        if s is None:
            continue
        m = Chem.MolFromSmiles(s)
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


# ====================== Predictor ======================
class Predictor(nn.Module):
    def __init__(self, dim_feature, dim_hidden, num_property, dropout=0.2):
        super(Predictor, self).__init__()
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
        hidden = self.mlp(x)
        return self.output(hidden)


class QM9PredictorAPI:
    def __init__(self, predictor_ckpt, mean_path, std_path, device="cpu"):
        self.device = torch.device(device)

        ckpt = torch.load(predictor_ckpt, map_location=self.device)

        self.model = Predictor(
            dim_feature=ckpt["dim_feature"],
            dim_hidden=ckpt["hidden_dim"],
            num_property=ckpt["num_property"],
            dropout=ckpt.get("dropout", 0.0)
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
            bad_mask = lumo <= homo + margin
            lumo[bad_mask] = homo[bad_mask] + gap[bad_mask]
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
        pred = self.enforce_physical_constraints(pred)

        return pred


# ====================== 加载模型 ======================
print(f"使用设备: {DEVICE}")

print("加载 PSVAEModel...")
model_psvae = PSVAEModel.load_from_checkpoint(CKPT_PSVAE, map_location=DEVICE)
model_psvae.eval()
model_psvae.to(DEVICE)
print("PSVAEModel 加载完成")

print("加载 QM9PredictorAPI...")
predictor = QM9PredictorAPI(
    predictor_ckpt=PREDICTOR_CKPT,
    mean_path=MEAN_PATH,
    std_path=STD_PATH,
    device=DEVICE
)
print("QM9PredictorAPI 加载完成")
print(f"[INFO] predictor properties: {predictor.property_names}")

print("加载训练集 latent...")
latent_train = np.load(TRAIN_LATENT_PATH).astype(np.float32)
latent_dim = latent_train.shape[1]
print(f"[INFO] latent_train shape: {latent_train.shape}")


# ====================== 编码：SMILES -> latent ======================
def smiles_to_latent(smi):
    mol = smiles2molecule(smi, kekulize=True)
    if mol is None:
        return None

    try:
        mol = Chem.RemoveHs(mol)
        with torch.no_grad():
            z = model_psvae.get_z_from_mol(mol)
            if z.dim() > 1:
                z = z.squeeze(0)
            z = z.detach().cpu().numpy().astype(np.float32)
        return z
    except Exception:
        return None


# ====================== 训练集读取 ======================
def load_train_smiles(train_smiles_csv, smiles_col):
    if train_smiles_csv.endswith(".csv"):
        df = pd.read_csv(train_smiles_csv)
    elif train_smiles_csv.endswith(".tsv") or train_smiles_csv.endswith(".txt"):
        df = pd.read_csv(train_smiles_csv, sep="\t")
    else:
        raise ValueError("只支持 csv / tsv / txt")

    if smiles_col not in df.columns:
        raise ValueError(f"未找到 smiles 列 {smiles_col}，现有列: {list(df.columns)}")

    smiles = [canonicalize_smiles(s) for s in df[smiles_col].astype(str).tolist()]
    smiles = [s for s in smiles if s is not None]
    smiles = list(dict.fromkeys(smiles))

    return smiles


def inspect_labeled_gap(train_smiles_csv, smiles_col, success_threshold):
    df = pd.read_csv(train_smiles_csv)

    if smiles_col not in df.columns:
        raise ValueError(f"未找到 smiles 列 {smiles_col}，当前列: {list(df.columns)}")

    if "gap" not in df.columns:
        raise ValueError(f"未找到 gap 列，当前列: {list(df.columns)}")

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


# ====================== BRICS 片段工具 ======================
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


def safe_brics_build(fragment_smiles_list, max_products=10):
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
            uniquify=True
        )

        products = []
        for _, prod in enumerate(builder):
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


# ====================== Fragment mutation / crossover ======================
def fragment_mutation(parent_smi, fragment_library):
    parent_frags = smiles_to_brics_fragments(parent_smi)

    if parent_frags is None or len(parent_frags) == 0:
        return canonicalize_smiles(parent_smi)

    child_frags = parent_frags.copy()
    replace_idx = random.randrange(len(child_frags))
    child_frags[replace_idx] = random.choice(fragment_library)

    child_smi = safe_brics_build(child_frags, max_products=10)

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

    mixed = sel1 + sel2
    mixed = list(dict.fromkeys(mixed))

    child_smi = safe_brics_build(mixed, max_products=10)

    if child_smi is None:
        return canonicalize_smiles(random.choice([parent1_smi, parent2_smi]))

    return canonicalize_smiles(child_smi)


# ====================== 初始化、评估、选择 ======================
def init_population_smiles(train_smiles, pop_size):
    if len(train_smiles) >= pop_size:
        idx = np.random.choice(len(train_smiles), pop_size, replace=False)
        return [train_smiles[i] for i in idx]

    idx = np.random.choice(len(train_smiles), pop_size, replace=True)
    return [train_smiles[i] for i in idx]


def init_population_from_gap_labels_no_leakage(
    train_smiles_csv,
    smiles_col,
    pop_size,
    warm_frac=0.8,
    success_threshold=0.15,
    gap_upper=0.20,
):
    """
    用 train_labeled.csv 的 gap 做 warm-start 排序，
    但排除已经达标的分子，避免一开始 success=100%。

    标签只用于初始化，不用于后续 fitness 评价。
    """
    df = pd.read_csv(train_smiles_csv)

    if smiles_col not in df.columns:
        raise ValueError(f"未找到 smiles 列 {smiles_col}，当前列: {list(df.columns)}")

    if "gap" not in df.columns:
        raise ValueError(f"未找到 gap 列，当前列: {list(df.columns)}")

    df = df[[smiles_col, "gap"]].dropna().copy()
    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.drop_duplicates("canonical_smiles")

    near_df = df[
        (df["gap"] >= success_threshold) &
        (df["gap"] <= gap_upper)
    ].copy()

    if len(near_df) < pop_size:
        print(
            f"[WARN] near-threshold pool too small: {len(near_df)}. "
            f"Fallback to all molecules with gap >= {success_threshold}."
        )
        near_df = df[df["gap"] >= success_threshold].copy()

    if len(near_df) == 0:
        raise RuntimeError(
            f"没有 gap >= {success_threshold} 的 warm-start 分子，"
            f"请检查 train_labeled.csv 或调高 success_threshold。"
        )

    near_df = near_df.sort_values("gap", ascending=True).reset_index(drop=True)

    warm_n = int(pop_size * warm_frac)
    warm_n = max(1, min(warm_n, len(near_df)))

    warm_smiles = near_df["canonical_smiles"].head(warm_n).tolist()

    rest_n = pop_size - len(warm_smiles)
    if rest_n > 0:
        rest_pool = near_df["canonical_smiles"].iloc[warm_n:].tolist()
        if len(rest_pool) == 0:
            rest_pool = near_df["canonical_smiles"].tolist()

        idx = np.random.choice(
            len(rest_pool),
            rest_n,
            replace=len(rest_pool) < rest_n
        )
        rest_smiles = [rest_pool[i] for i in idx]
    else:
        rest_smiles = []

    population = warm_smiles + rest_smiles
    random.shuffle(population)

    print("[INFO] warm-start from near-threshold gap labels WITHOUT label leakage")
    print(f"[INFO] success_threshold: {success_threshold}")
    print(f"[INFO] gap_upper: {gap_upper}")
    print(f"[INFO] candidate pool size: {len(near_df)}")
    print(f"[INFO] warm-start best seed label-gap: {near_df['gap'].min():.6f}")
    print(f"[INFO] warm-start top10 seed label-gap mean: {near_df['gap'].head(10).mean():.6f}")
    print(f"[INFO] warm-start size: {len(warm_smiles)}")

    return population[:pop_size]


def evaluate_population(pop_smiles, success_threshold=0.15):
    """
    无标签泄漏评价：
    所有分子统一使用：
    SMILES -> PS-VAE encoder -> latent z -> predictor -> gap

    训练集标签不参与 fitness 评价。
    """
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
                "score": 0.0
            })
            continue

        z = smiles_to_latent(can_smi)

        if z is None:
            rows.append({
                "smiles": can_smi,
                "canonical_smiles": can_smi,
                "source": "predictor_failed",
                "latent_ok": 0,
                "gap": np.inf,
                "score": 0.0
            })
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
            "score": score
        }

        for j, p in enumerate(predictor.property_names):
            row[p] = float(pred[j])

        rows.append(row)

    return rows


def tournament_selection(pop_smiles, fitness, tourn_size=2):
    idx = np.random.choice(len(pop_smiles), tourn_size, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pop_smiles[best]


# ====================== 主函数 ======================
def main():
    parser = argparse.ArgumentParser(
        description="QM9 SMILES/Fragment-GA baseline without label leakage"
    )

    parser.add_argument("--train_smiles_csv", type=str, default=DEFAULT_TRAIN_SMILES_CSV)
    parser.add_argument("--smiles_col", type=str, default="smiles")

    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_gen", type=int, default=1000)
    parser.add_argument("--elite_size", type=int, default=30)

    parser.add_argument("--mut_prob", type=float, default=0.15)
    parser.add_argument("--cross_prob", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="smiles_noleak_v1")
    parser.add_argument("--success_threshold", type=float, default=0.15)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--fragment_lib_max_mols", type=int, default=20000)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--warm_start_frac", type=float, default=0.8)
    parser.add_argument("--warm_start_gap_upper", type=float, default=0.30)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = os.path.join(args.output_root, f"fragment_ga_{args.version}")
    ensure_dir(out_dir)

    print("\n========== 配置 ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    inspect_labeled_gap(
        args.train_smiles_csv,
        args.smiles_col,
        args.success_threshold
    )

    train_smiles = load_train_smiles(args.train_smiles_csv, args.smiles_col)
    print(f"[INFO] 训练集可用 SMILES 数: {len(train_smiles)}")

    fragment_library = build_fragment_library(
        train_smiles,
        max_mols_for_library=args.fragment_lib_max_mols
    )
    print(f"[INFO] BRICS fragment library size: {len(fragment_library)}")

    if len(fragment_library) == 0:
        raise RuntimeError("fragment_library 为空，无法运行 Fragment-GA")

    if args.warm_start:
        population = init_population_from_gap_labels_no_leakage(
            args.train_smiles_csv,
            args.smiles_col,
            pop_size=args.pop_size,
            warm_frac=args.warm_start_frac,
            success_threshold=args.success_threshold,
            gap_upper=args.warm_start_gap_upper
        )
    else:
        population = init_population_smiles(train_smiles, args.pop_size)

    init_eval = evaluate_population(
        population,
        success_threshold=args.success_threshold
    )
    init_df = pd.DataFrame(init_eval)
    init_valid = init_df[np.isfinite(init_df["gap"])].copy()

    print("\n[DEBUG] initial population predictor-gap statistics:")
    print(init_valid["gap"].describe())

    print("\n[DEBUG] initial population top 10 by predictor-gap:")
    print(init_valid.sort_values("gap").head(10)[["smiles", "gap", "score", "source"]])

    start_wall_time = time.time()

    avg_score_history = []
    avg_gap_history = []
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
        eval_rows = evaluate_population(
            population,
            success_threshold=args.success_threshold
        )
        total_evaluations += len(population)

        df_eval = pd.DataFrame(eval_rows)
        valid_df = df_eval[df_eval["latent_ok"] == 1].copy()

        if len(valid_df) == 0:
            print(f"[Gen {gen:03d}] 无有效分子，停止。")
            break

        gaps = valid_df["gap"].values.astype(np.float32)
        scores = valid_df["score"].values.astype(np.float32)

        avg_gap = float(np.mean(gaps))
        avg_score = float(np.mean(scores))
        best_gap = float(np.min(gaps))

        topk = min(10, len(gaps))
        top10_mean_gap = float(np.mean(np.sort(gaps)[:topk]))

        success_count = int(np.sum(gaps < args.success_threshold))
        success_rate = float(success_count / len(gaps))

        if best_gap < best_gap_so_far:
            best_gap_so_far = best_gap

        best_idx = int(np.argmin(gaps))
        best_smiles = valid_df.iloc[best_idx]["smiles"]
        best_smiles_history.append(best_smiles)

        avg_score_history.append(avg_score)
        avg_gap_history.append(avg_gap)
        best_gap_so_far_history.append(best_gap_so_far)
        top10_mean_gap_history.append(top10_mean_gap)
        success_count_history.append(success_count)
        success_rate_history.append(success_rate)
        eval_count_history.append(total_evaluations)
        elapsed_time_history.append(float(time.time() - start_wall_time))

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

        # ========= 选择 =========
        fitness_full = np.full(len(population), 1e6, dtype=np.float32)
        valid_idx = valid_df.index.to_numpy()
        fitness_full[valid_idx] = -valid_df["score"].values.astype(np.float32)

        sorted_idx = np.argsort(fitness_full)
        elites = [population[i] for i in sorted_idx[:args.elite_size]]

        new_population = list(elites)

        # ========= BRICS mutation + crossover =========
        while len(new_population) < args.pop_size:
            p1 = tournament_selection(population, fitness_full, tourn_size=2)
            p2 = tournament_selection(population, fitness_full, tourn_size=2)

            if random.random() < args.cross_prob:
                child = fragment_crossover(p1, p2)
            else:
                child = p1

            if random.random() < args.mut_prob:
                mutated = fragment_mutation(child, fragment_library)
                if mutated is not None:
                    child = mutated

            if child is None:
                child = random.choice(train_smiles)

            new_population.append(child)

        population = new_population[:args.pop_size]

    # ========= 最终评估 =========
    final_eval = evaluate_population(
        population,
        success_threshold=args.success_threshold
    )
    final_df = pd.DataFrame(final_eval)
    final_valid_df = final_df[final_df["latent_ok"] == 1].copy()

    final_csv_path = os.path.join(out_dir, "final_population_fragment_ga.csv")
    final_df.to_csv(final_csv_path, index=False)

    final_smiles = final_df["smiles"].tolist()
    valid_smiles = [s for s in final_smiles if s is not None]
    diversity = compute_diversity(valid_smiles) if len(valid_smiles) > 1 else 0.0
    validity = float(np.mean(final_df["latent_ok"].values))

    if len(final_valid_df) == 0:
        best_gap = np.inf
        avg_gap = np.inf
        median_gap = np.inf
        top10_mean_gap = np.inf
        best_score = 0.0
        avg_score = 0.0
        top10_mean_gap_score = 0.0
        best_smiles_final = None
        best_properties_final = {}
        final_success_count = 0
        final_success_rate = 0.0
    else:
        final_gap = final_valid_df["gap"].values.astype(np.float32)
        final_score = final_valid_df["score"].values.astype(np.float32)

        best_gap = float(np.min(final_gap))
        avg_gap = float(np.mean(final_gap))
        median_gap = float(np.median(final_gap))

        topk = min(10, len(final_gap))
        top10_mean_gap = float(np.mean(np.sort(final_gap)[:topk]))

        best_score = float(np.max(final_score))
        avg_score = float(np.mean(final_score))
        top10_mean_gap_score = float(np.mean(np.sort(final_score)[::-1][:topk]))

        best_idx = int(np.argmax(final_score))
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

    total_time_sec = float(time.time() - start_wall_time)

    progress_df = pd.DataFrame({
        "generation": np.arange(len(avg_gap_history)),
        "evaluations": eval_count_history,
        "elapsed_time_sec": elapsed_time_history,
        "avg_gap": avg_gap_history,
        "avg_score": avg_score_history,
        "best_gap_so_far": best_gap_so_far_history,
        "top10_mean_gap": top10_mean_gap_history,
        "success_count": success_count_history,
        "success_rate": success_rate_history,
    })

    progress_csv_path = os.path.join(out_dir, "progress_metrics.csv")
    progress_df.to_csv(progress_csv_path, index=False)

    summary = {
        "method": "SMILES-GA / Fragment-GA",
        "task_definition": "BRICS fragment-based mutation/crossover; warm-start uses labels only for initialization; all fitness uses latent predictor",
        "version": args.version,
        "seed": args.seed,

        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "elite_size": args.elite_size,
        "mut_prob": args.mut_prob,
        "cross_prob": args.cross_prob,
        "success_threshold": float(args.success_threshold),
        "warm_start_gap_upper": float(args.warm_start_gap_upper),

        "best_gap_final": best_gap,
        "avg_gap_final": avg_gap,
        "median_gap_final": median_gap,
        "top10_mean_gap_final": top10_mean_gap,

        "best_score_final": best_score,
        "avg_score_final": avg_score,
        "top10_mean_gap_score_final": top10_mean_gap_score,

        "success_count_final": final_success_count,
        "success_rate_final": final_success_rate,

        "diversity": diversity,
        "validity": validity,

        "time_sec_total": total_time_sec,
        "n_evaluations_total": total_evaluations,

        "best_smiles_final": best_smiles_final,
        "best_properties_final": best_properties_final,

        "avg_score_history": [float(x) for x in avg_score_history],
        "avg_gap_history": [float(x) for x in avg_gap_history],
        "best_gap_so_far_history": [float(x) for x in best_gap_so_far_history],
        "top10_mean_gap_history": [float(x) for x in top10_mean_gap_history],
        "success_count_history": [int(x) for x in success_count_history],
        "success_rate_history": [float(x) for x in success_rate_history],
        "eval_count_history": [int(x) for x in eval_count_history],
        "elapsed_time_history": [float(x) for x in elapsed_time_history],
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))

    # 图2：主收敛曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(best_gap_so_far_history)), best_gap_so_far_history, marker="o", label="Best-so-far gap")
    plt.plot(range(len(top10_mean_gap_history)), top10_mean_gap_history, marker="s", label="Top-10 mean gap")
    plt.xlabel("Generation")
    plt.ylabel("Gap")
    plt.title("Convergence Curve (SMILES/Fragment-GA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_convergence_curve.png"), dpi=300)
    plt.close()

    # 图3A
    plt.figure(figsize=(8, 5))
    plt.plot(eval_count_history, avg_score_history, marker="o")
    plt.xlabel("Evaluations")
    plt.ylabel("Average Gap Score")
    plt.title("Efficiency Curve: Score vs Evaluations (SMILES/Fragment-GA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3a_score_vs_evaluations.png"), dpi=300)
    plt.close()

    # 图3B
    plt.figure(figsize=(8, 5))
    plt.plot(eval_count_history, success_count_history, marker="o")
    plt.xlabel("Evaluations")
    plt.ylabel("Success Count")
    plt.title("Efficiency Curve: Success Count vs Evaluations (SMILES/Fragment-GA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3b_success_vs_evaluations.png"), dpi=300)
    plt.close()

    # 图3C
    plt.figure(figsize=(8, 5))
    plt.plot(elapsed_time_history, success_count_history, marker="o")
    plt.xlabel("Elapsed Time (sec)")
    plt.ylabel("Success Count")
    plt.title("Efficiency Curve: Success Count vs Time (SMILES/Fragment-GA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3c_success_vs_time.png"), dpi=300)
    plt.close()

    # ====================== 导出 evolution path ======================
    evo_rows = []
    for gid, smi in enumerate(best_smiles_history):
        can_smi = canonicalize_smiles(smi)
        z = smiles_to_latent(can_smi)

        if z is not None:
            pred = predictor.predict_array(z[None, :])[0]
            gap_val = float(pred[predictor.gap_idx])
            source = "predictor"
        else:
            gap_val = np.inf
            source = "failed"

        evo_rows.append({
            "generation": int(gid),
            "smiles": can_smi,
            "gap": gap_val,
            "source": source
        })

    evo_csv_path = os.path.join(out_dir, "evolution_path_full.csv")
    pd.DataFrame(evo_rows).to_csv(evo_csv_path, index=False)
    print(f"完整 evolution path 已保存到: {evo_csv_path}")

    # 图5：PCA / UMAP（基于 latent）
    final_valid_smiles = [s for s in final_smiles if s is not None]
    gen_latents = []

    for s in final_valid_smiles:
        z = smiles_to_latent(s)
        if z is not None:
            gen_latents.append(z)

    if len(gen_latents) > 0:
        gen_latents = np.vstack(gen_latents)

        n_train_vis = min(2000, len(latent_train))
        train_vis_idx = np.random.choice(len(latent_train), n_train_vis, replace=False)
        train_vis = latent_train[train_vis_idx]

        X_all = np.vstack([train_vis, gen_latents])
        labels = (["train"] * len(train_vis)) + (["generated"] * len(gen_latents))

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

        plt.scatter(coords[train_mask, 0], coords[train_mask, 1], s=8, alpha=0.4, label="Train")
        plt.scatter(coords[gen_mask, 0], coords[gen_mask, 1], s=16, alpha=0.8, label="Generated")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Chemical Space Visualization by PCA (SMILES/Fragment-GA)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig5_pca_chemical_space.png"), dpi=300)
        plt.close()

        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=args.seed)
            umap_coords = reducer.fit_transform(X_all)

            umap_df = pd.DataFrame({
                "x": umap_coords[:, 0],
                "y": umap_coords[:, 1],
                "label": labels
            })
            umap_df.to_csv(os.path.join(out_dir, "chemical_space_umap.csv"), index=False)

            plt.figure(figsize=(8, 6))
            plt.scatter(umap_coords[train_mask, 0], umap_coords[train_mask, 1], s=8, alpha=0.4, label="Train")
            plt.scatter(umap_coords[gen_mask, 0], umap_coords[gen_mask, 1], s=16, alpha=0.8, label="Generated")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.title("Chemical Space Visualization by UMAP (SMILES/Fragment-GA)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig5_umap_chemical_space.png"), dpi=300)
            plt.close()

    print("\n========== 运行完成 ==========")
    print(f"输出目录: {out_dir}")
    print(f"最终最优 gap: {best_gap:.6f}")
    print(f"最终平均 gap: {avg_gap:.6f}")
    print(f"最终 top10 gap: {top10_mean_gap:.6f}")
    print(f"最终 success rate: {final_success_rate:.4f}")
    print(f"最终 validity: {validity:.4f}")
    print(f"最终 diversity: {diversity:.4f}")
    print(f"总时间(秒): {total_time_sec:.2f}")
    print(f"总评估次数: {total_evaluations}")
    print(f"最终最优 smiles: {best_smiles_final}")


if __name__ == "__main__":
    main()