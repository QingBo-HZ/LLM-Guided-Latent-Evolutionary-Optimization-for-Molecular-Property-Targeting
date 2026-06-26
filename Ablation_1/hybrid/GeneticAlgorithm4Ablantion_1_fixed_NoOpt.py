#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三模式统一版遗传算法优化 QM9 分子 gap（修正版 + Eval@0.15 记录版）

支持初始化模式：
1. llm          : LLM 生成的 smiles -> latent
2. psvae        : PS-VAE 采样得到的 latent
3. hybrid       : LLM latent + PS-VAE latent + 基于 PS-VAE 分布尺度的局部扩增
4. train_random : 训练集随机初始化（保留作为对照）

说明：
- 当前优化目标默认是最小化 gap
- predictor / PS-VAE / 解码逻辑按你当前工程版本修正
- Hybrid 现在真正使用了 psvae_init_latent.npy
- 新增 progress_metrics.csv，用于计算 Table 2 中的 Eval@0.15
"""

from html import parser
import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ====================== 环境设置 ======================
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# ====================== 添加 PS-VAE 模块路径 ======================
PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))
print("[DEBUG] sys.path appended", flush=True)

# ====================== 导入 PS-VAE 模型 ======================
print("[DEBUG] importing PSVAEModel...", flush=True)
from pl_models import PSVAEModel

# ====================== 导入解码与 checkpoint safe globals ======================
from utils.chem_utils import molecule2smiles, GeneralVocab
from data.mol_bpe import Tokenizer
from rdkit.Chem.rdchem import BondType
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
torch.serialization.add_safe_globals(SAFE_GLOBALS)
print(f"[DEBUG] registered safe globals: {[str(x) for x in SAFE_GLOBALS]}", flush=True)

# ====================== 固定默认参数（可被命令行覆盖） ======================
DEFAULT_POP_SIZE = 200
DEFAULT_N_GEN = 30
DEFAULT_CROSS_PROB = 0.3
DEFAULT_MUT_PROB = 0.05
DEFAULT_ELITE_SIZE = 20
DEFAULT_MUT_ETA = 20
DEFAULT_PATIENCE = 5
DEFAULT_VERSION = "ablation_v1"
DEFAULT_INIT_MODE = "llm"  # train_random / llm / psvae / hybrid

# ====================== 路径配置 ======================
CKPT_PSVAE = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt"
PREDICTOR_CKPT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/best_predictor.pt"
MEAN_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_mean.npy"
STD_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_std.npy"
SMILES_LATENT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent"

# 三种初始化文件
# 不经过多轮优化的 LLM latent（直接从 llm_init_latent.npy 读）
LLM_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_0/llm_init_latent.npy"

# 经过多轮优化的 LLM latent
# LLM_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent/llm_init_latent.npy"

PSVAE_INIT_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/latent/psvae_init_latent.npy"
HYBRID_INIT_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/latent/hybrid_init_latent.npy"

OUTPUT_ROOT = "/root/autodl-tmp/sweeteners_evolve/Ablation_1/results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


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


def compute_diversity(smiles_list):
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)

    if len(mols) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)

    return 1.0 - float(np.mean(sims))


def calc_eval_at_threshold(progress_df, threshold=0.15):
    """
    标准 Eval@threshold:
    首次 best_gap_so_far < threshold 所需 molecular evaluations.
    """
    if len(progress_df) == 0:
        return None

    hit = progress_df[progress_df["best_gap_so_far"] < threshold]
    if len(hit) == 0:
        return None

    return int(hit.iloc[0]["evaluations"])


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
        """
        z: np.ndarray, shape [D] or [N, D]
        return: np.ndarray, shape [N, 7]
        """
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]

        x = torch.tensor(z, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy()

        pred = pred_norm * self.y_std + self.y_mean
        pred = self.enforce_physical_constraints(pred)
        return pred

    def predict_dict(self, z):
        pred = self.predict_array(z)
        out = []
        for row in pred:
            out.append({k: float(v) for k, v in zip(self.property_names, row)})
        return out if len(out) > 1 else out[0]


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


# ====================== 数据准备 ======================
print("加载训练集 latent...")
train_latent_path = os.path.join(SMILES_LATENT, "x_train.npy")
if not os.path.exists(train_latent_path):
    raise FileNotFoundError(f"训练集 latent 不存在: {train_latent_path}")
latent_train = np.load(train_latent_path).astype(np.float32)

# 直接在数据准备阶段把 PS-VAE latent 也读进来
print("加载 PS-VAE 初始化 latent...")
if not os.path.exists(PSVAE_INIT_LATENT_PATH):
    raise FileNotFoundError(f"PS-VAE 初始化 latent 不存在: {PSVAE_INIT_LATENT_PATH}")
psvae_latent_global = np.load(PSVAE_INIT_LATENT_PATH).astype(np.float32)

latent_dim = latent_train.shape[1]
print(f"Latent 维度: {latent_dim}")
print(f"Train latent shape : {latent_train.shape}")
print(f"PSVAE latent shape : {psvae_latent_global.shape}")

if psvae_latent_global.shape[1] != latent_dim:
    raise ValueError(
        f"PS-VAE latent 维度错误: {psvae_latent_global.shape}, expected (*, {latent_dim})"
    )

LB = latent_train.min(axis=0)
UB = latent_train.max(axis=0)


# ====================== 解码函数（按你已验证可用的版本） ======================
def latent_to_smiles(z, max_atom_num=20, add_edge_th=0.5, temperature=0.6):
    with torch.no_grad():
        z_t = torch.tensor(z, dtype=torch.float32, device=DEVICE)
        try:
            graph = model_psvae.inference_single_z(
                z_t,
                max_atom_num=max_atom_num,
                add_edge_th=add_edge_th,
                temperature=temperature
            )
            mol = model_psvae.return_data_to_mol(graph)
            smi = molecule2smiles(mol)
        except Exception:
            smi = None
    return smi


# ====================== 遗传操作函数 ======================
def tournament_selection(pop, fitness, tourn_size=2):
    indices = np.random.choice(len(pop), tourn_size, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx].copy()


def arithmetic_crossover(
    p1,
    p2,
    cross_prob,
):
    if np.random.random() > cross_prob:
        return p1.copy(), p2.copy()

    alpha = np.random.random()
    c1 = alpha * p1 + (1.0 - alpha) * p2
    c2 = (1.0 - alpha) * p1 + alpha * p2

    c1 = np.clip(c1, LB, UB).astype(np.float32)
    c2 = np.clip(c2, LB, UB).astype(np.float32)

    return c1, c2


def polynomial_mutation(
    individual,
    prob,
    eta,
    low,
    up,
):

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


# ====================== 初始化种群函数 ======================
def sample_population_from_pool(pool, pop_size):
    pool = np.asarray(pool, dtype=np.float32)
    n = len(pool)

    if n == 0:
        raise ValueError("初始化 latent pool 为空")

    if n >= pop_size:
        indices = np.random.choice(n, pop_size, replace=False)
        return pool[indices].copy()
    else:
        extra_indices = np.random.choice(n, pop_size - n, replace=True)
        extra = pool[extra_indices].copy()
        return np.concatenate([pool, extra], axis=0)


def farthest_point_sample(pool, n_select, seed=42):
    pool = np.asarray(pool, dtype=np.float32)
    n = len(pool)
    if n <= n_select:
        return pool.copy()

    rng = np.random.default_rng(seed)
    selected_idx = [rng.integers(0, n)]

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

    dist_matrix = np.linalg.norm(pool[:, None, :] - existing[None, :, :], axis=2)
    min_dist = dist_matrix.min(axis=1)

    idx = np.argsort(min_dist)[::-1][:n_select]
    return pool[idx].copy()


def build_hybrid_population(
    llm_latent,
    psvae_latent,
    pop_size,
    sigma_scale=0.15,
    llm_keep_ratio=0.5,
    llm_expand_ratio=0.0,
    ref_pool=latent_train,
    local_k=8,
):
    """
    改进版 Hybrid：
    1) 保留彼此尽量远的 LLM seed
    2) 对 keep_part 做均匀局部扩增
    3) 用与已有 hybrid 点差异更大的 PS-VAE latent 补齐
    """
    llm_latent = np.asarray(llm_latent, dtype=np.float32)
    psvae_latent = np.asarray(psvae_latent, dtype=np.float32)
    ref_pool = np.asarray(ref_pool, dtype=np.float32)

    if len(llm_latent) == 0:
        raise ValueError("llm_latent 为空，无法构建 hybrid 初始化")
    if len(psvae_latent) == 0:
        raise ValueError("psvae_latent 为空，无法构建 hybrid 初始化")
    if llm_latent.ndim != 2 or psvae_latent.ndim != 2:
        raise ValueError(
            f"llm_latent 和 psvae_latent 都必须是二维数组，"
            f"got llm={llm_latent.shape}, psvae={psvae_latent.shape}"
        )
    if llm_latent.shape[1] != psvae_latent.shape[1]:
        raise ValueError(
            f"latent 维度不一致，llm={llm_latent.shape[1]}, psvae={psvae_latent.shape[1]}"
        )

    n_keep = max(1, int(pop_size * llm_keep_ratio))
    n_expand = max(1, int(pop_size * llm_expand_ratio))
    n_fill = pop_size - n_keep - n_expand
    if n_fill < 0:
        n_fill = 0

    # 1) LLM keep：彼此尽量远
    keep_part = farthest_point_sample(llm_latent, n_keep)

    # 2) 均匀局部扩增
    expand_list = []
    parent_pool = keep_part if len(keep_part) > 0 else llm_latent

    for i in range(n_expand):
        parent = parent_pool[i % len(parent_pool)].copy()

        dist2 = ((ref_pool - parent[None, :]) ** 2).sum(axis=1)
        nn_idx = np.argsort(dist2)[:max(local_k, 2)]
        local_neighbors = ref_pool[nn_idx]

        local_std = local_neighbors.std(axis=0).astype(np.float32)
        local_std = np.maximum(local_std, 1e-6)

        noise = np.random.normal(
            loc=0.0,
            scale=local_std * sigma_scale,
            size=parent.shape
        ).astype(np.float32)

        child = parent + noise
        child = np.clip(child, LB, UB)

        expand_list.append(child.astype(np.float32))

    expand_part = np.array(expand_list, dtype=np.float32)

    # 3) PS-VAE 补齐：尽量选与现有 hybrid 点不同的
    if n_fill > 0:
        existing_part = np.concatenate([keep_part, expand_part], axis=0)
        fill_part = select_diverse_fill(psvae_latent, existing_part, n_fill)
        population = np.concatenate([keep_part, expand_part, fill_part], axis=0)
    else:
        population = np.concatenate([keep_part, expand_part], axis=0)

    # 长度校正
    if len(population) > pop_size:
        population = population[:pop_size]
    elif len(population) < pop_size:
        extra = sample_population_from_pool(psvae_latent, pop_size - len(population))
        population = np.concatenate([population, extra], axis=0)

    return population.astype(np.float32)


def initialize_population(
    init_mode,
    pop_size,
    llm_latent_path,
    psvae_latent_path,
    hybrid_latent_path,
    hybrid_sigma,
    hybrid_keep_ratio,
    hybrid_expand_ratio
):
    """
    初始化种群：
    - train_random: 从训练集 latent 中随机抽样
    - llm        : 从 llm_init_latent.npy 抽样
    - psvae      : 从 psvae_init_latent.npy 抽样
    - hybrid     : 优先读 hybrid_init_latent.npy；没有则真正执行 LLM + PS-VAE 混合
    """
    if init_mode == "train_random":
        print("[INFO] 使用训练集随机初始化")
        return sample_population_from_pool(latent_train, pop_size)

    elif init_mode == "llm":
        if not os.path.exists(llm_latent_path):
            raise FileNotFoundError(f"LLM latent 文件不存在: {llm_latent_path}")
        llm_latent = np.load(llm_latent_path).astype(np.float32)
        print(f"[INFO] 使用 LLM 初始化，shape={llm_latent.shape}")
        if llm_latent.shape[1] != latent_dim:
            raise ValueError(f"LLM latent 维度错误: {llm_latent.shape}, 期望 (*, {latent_dim})")
        return sample_population_from_pool(llm_latent, pop_size)

    elif init_mode == "psvae":
        if not os.path.exists(psvae_latent_path):
            raise FileNotFoundError(f"PS-VAE latent 文件不存在: {psvae_latent_path}")
        psvae_latent = np.load(psvae_latent_path).astype(np.float32)
        print(f"[INFO] 使用 PS-VAE 初始化，shape={psvae_latent.shape}")
        if psvae_latent.shape[1] != latent_dim:
            raise ValueError(f"PS-VAE latent 维度错误: {psvae_latent.shape}, 期望 (*, {latent_dim})")
        return sample_population_from_pool(psvae_latent, pop_size)

    elif init_mode == "hybrid":
        if hybrid_latent_path is not None and os.path.exists(hybrid_latent_path):
            hybrid_latent = np.load(hybrid_latent_path).astype(np.float32)
            print(f"[INFO] 使用现成 Hybrid latent 初始化，shape={hybrid_latent.shape}")
            if hybrid_latent.shape[1] != latent_dim:
                raise ValueError(f"Hybrid latent 维度错误: {hybrid_latent.shape}, 期望 (*, {latent_dim})")
            return sample_population_from_pool(hybrid_latent, pop_size)

        if not os.path.exists(llm_latent_path):
            raise FileNotFoundError(f"Hybrid 模式需要 llm latent，但文件不存在: {llm_latent_path}")
        if not os.path.exists(psvae_latent_path):
            raise FileNotFoundError(f"Hybrid 模式需要 psvae latent，但文件不存在: {psvae_latent_path}")

        llm_latent = np.load(llm_latent_path).astype(np.float32)
        psvae_latent = np.load(psvae_latent_path).astype(np.float32)

        print("[INFO] 使用真正的 Hybrid 初始化：LLM seed + PS-VAE latent 补充")
        print(f"[INFO] llm_latent shape   = {llm_latent.shape}")
        print(f"[INFO] psvae_latent shape = {psvae_latent.shape}")

        if llm_latent.shape[1] != latent_dim:
            raise ValueError(f"LLM latent 维度错误: {llm_latent.shape}, 期望 (*, {latent_dim})")
        if psvae_latent.shape[1] != latent_dim:
            raise ValueError(f"PS-VAE latent 维度错误: {psvae_latent.shape}, 期望 (*, {latent_dim})")

        if hybrid_keep_ratio < 0 or hybrid_expand_ratio < 0:
            raise ValueError("hybrid_keep_ratio 和 hybrid_expand_ratio 必须 >= 0")

        if hybrid_keep_ratio + hybrid_expand_ratio > 1.0:
            raise ValueError("hybrid_keep_ratio + hybrid_expand_ratio 不能超过 1.0")

        return build_hybrid_population(
            llm_latent=llm_latent,
            psvae_latent=psvae_latent,
            pop_size=pop_size,
            sigma_scale=hybrid_sigma,
            llm_keep_ratio=hybrid_keep_ratio,
            llm_expand_ratio=hybrid_expand_ratio,
            ref_pool=latent_train,
            local_k=8,
        )

    else:
        raise ValueError(f"未知 init_mode: {init_mode}")


# ====================== 主函数 ======================
def main():
    parser = argparse.ArgumentParser(description="三模式统一版 QM9 latent-space GA（修正版）")
    parser.add_argument("--init_mode", type=str, default=DEFAULT_INIT_MODE,
                        choices=["train_random", "llm", "psvae", "hybrid"])
    parser.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--n_gen", type=int, default=DEFAULT_N_GEN)
    parser.add_argument("--cross_prob", type=float, default=DEFAULT_CROSS_PROB)
    parser.add_argument("--mut_prob", type=float, default=DEFAULT_MUT_PROB)
    parser.add_argument("--elite_size", type=int, default=DEFAULT_ELITE_SIZE)
    parser.add_argument("--mut_eta", type=float, default=DEFAULT_MUT_ETA)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llm_latent_path", type=str, default=LLM_LATENT_PATH)
    parser.add_argument("--psvae_latent_path", type=str, default=PSVAE_INIT_LATENT_PATH)
    parser.add_argument("--hybrid_latent_path", type=str, default=HYBRID_INIT_LATENT_PATH)
    parser.add_argument("--hybrid_sigma", type=float, default=0.2,
                        help="Hybrid 中 LLM 邻域扩增时，按训练流形局部邻域尺度缩放的系数")
    parser.add_argument("--hybrid_keep_ratio", type=float, default=0.5,
                        help="Hybrid 初始化中 LLM 原始种子所占比例")
    parser.add_argument("--hybrid_expand_ratio", type=float, default=0.0,
                        help="Hybrid 初始化中 LLM 扩增种子所占比例")

    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = os.path.join(args.output_root, f"{args.init_mode}_{args.version}")
    ensure_dir(out_dir)

    print("\n========== 配置 ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    # 初始化种群
    population = initialize_population(
        init_mode=args.init_mode,
        pop_size=args.pop_size,
        llm_latent_path=args.llm_latent_path,
        psvae_latent_path=args.psvae_latent_path,
        hybrid_latent_path=args.hybrid_latent_path,
        hybrid_sigma=args.hybrid_sigma,
        hybrid_keep_ratio=args.hybrid_keep_ratio,
        hybrid_expand_ratio=args.hybrid_expand_ratio,
    )
    print(f"[INFO] 初始种群 shape: {population.shape}")

    # ====================== GA 主循环（sigmoid gap-score） ======================
    avg_score_history = []
    avg_gap_history = []
    attr_means_history = []

    # 新增：用于 Table 2 / Eval@0.15 的标准进度指标
    best_gap_history = []
    best_gap_so_far_history = []
    top10_mean_gap_history = []
    success_count_history = []
    success_rate_history = []
    eval_count_history = []

    best_gap_so_far = float("inf")

    best_score_so_far = float("inf")
    no_improve_count = 0

    for gen in range(args.n_gen):
        pred_all = predictor.predict_array(population)
        attr_means = pred_all.mean(axis=0)
        attr_means_history.append(attr_means)

        pred_gap = pred_all[:, predictor.gap_idx].astype(np.float32)

        # ====================== Sigmoid gap-score ======================
        center = 0.15
        scale = 0.03

        gap_score = 1.0 / (1.0 + np.exp((pred_gap - center) / scale))

        # GA 最小化
        fitness = -gap_score

        avg_fitness = float(np.mean(fitness))
        best_fitness = float(np.min(fitness))
        avg_gap = float(np.mean(pred_gap))
        best_gap = float(np.min(pred_gap))

        avg_score = float(-avg_fitness)
        best_score = float(-best_fitness)

        avg_score_history.append(avg_score)
        avg_gap_history.append(avg_gap)

        # ====================== 新增：每代标准指标 ======================
        total_evaluations = int((gen + 1) * args.pop_size)

        if best_gap < best_gap_so_far:
            best_gap_so_far = best_gap

        sorted_gap = np.sort(pred_gap)
        topk = min(10, len(sorted_gap))
        top10_mean_gap = float(np.mean(sorted_gap[:topk]))

        success_count = int(np.sum(pred_gap < 0.15))
        success_rate = float(success_count / len(pred_gap)) if len(pred_gap) > 0 else 0.0

        best_gap_history.append(best_gap)
        best_gap_so_far_history.append(float(best_gap_so_far))
        top10_mean_gap_history.append(top10_mean_gap)
        success_count_history.append(success_count)
        success_rate_history.append(success_rate)
        eval_count_history.append(total_evaluations)

        print(
            f"[Gen {gen:03d}] "
            f"evals={total_evaluations}, "
            f"avg_score={avg_score:.6f}, best_score={best_score:.6f}, "
            f"avg_gap={avg_gap:.6f}, best_gap={best_gap:.6f}, "
            f"best_gap_so_far={best_gap_so_far:.6f}, "
            f"top10_gap={top10_mean_gap:.6f}, "
            f"success={success_count}/{len(pred_gap)}, "
            f"no_improve={no_improve_count}"
        )

        # ====================== Early Stop ======================
        if avg_fitness < best_score_so_far:
            best_score_so_far = avg_fitness
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= args.patience:
            print(f"[Early Stop] 连续 {args.patience} 代 score 未改善，提前停止。")
            break

        # ====================== 精英 ======================
        sorted_idx = np.argsort(fitness)
        elites = population[sorted_idx[:args.elite_size]].copy()

        new_population = list(elites)

        # ====================== 生成新一代 ======================
        while len(new_population) < args.pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)

            c1, c2 = arithmetic_crossover(
                p1, p2, args.cross_prob,
            )

            c1 = polynomial_mutation(
                c1, args.mut_prob, args.mut_eta, LB, UB,
            )

            c2 = polynomial_mutation(
                c2, args.mut_prob, args.mut_eta, LB, UB,
            )

            new_population.append(c1)
            if len(new_population) < args.pop_size:
                new_population.append(c2)

        population = np.array(new_population, dtype=np.float32)

    # ====================== 保存 progress_metrics.csv ======================
    progress_df = pd.DataFrame({
        "generation": np.arange(len(avg_gap_history), dtype=int),
        "evaluations": eval_count_history,
        "avg_gap": avg_gap_history,
        "avg_score": avg_score_history,
        "best_gap": best_gap_history,
        "best_gap_so_far": best_gap_so_far_history,
        "top10_mean_gap": top10_mean_gap_history,
        "success_count": success_count_history,
        "success_rate": success_rate_history,
    })

    progress_csv_path = os.path.join(out_dir, "progress_metrics.csv")
    progress_df.to_csv(progress_csv_path, index=False)
    print(f"[INFO] progress metrics 已保存到: {progress_csv_path}")

    eval_at_gap_lt_015 = calc_eval_at_threshold(progress_df, threshold=0.15)
    print(f"[INFO] Eval@Gap<0.15 = {eval_at_gap_lt_015}")

    # 最终评估
    print("对最终种群进行预测...")
    final_pred = predictor.predict_array(population)
    final_gap = final_pred[:, predictor.gap_idx]

    # 解码最终种群
    print("解码最终种群为 SMILES（如果某些失败，会记为空）...")
    decoded_smiles = []
    decode_success = 0

    for i in range(len(population)):
        smi = latent_to_smiles(population[i])
        decoded_smiles.append(smi)
        if smi is not None:
            decode_success += 1

    valid_smiles = [s for s in decoded_smiles if s is not None]
    diversity = compute_diversity(valid_smiles) if len(valid_smiles) > 1 else 0.0
    decode_rate = decode_success / len(decoded_smiles) if len(decoded_smiles) > 0 else 0.0

    print(f"[INFO] final decode success: {decode_success}/{len(population)}")

    # 保存最终种群信息
    property_names = predictor.property_names
    rows = []
    for i in range(len(population)):
        row = {
            "idx": i,
            "smiles": decoded_smiles[i],
            "pred_gap": float(final_gap[i]),
        }
        for j, p in enumerate(property_names):
            row[p] = float(final_pred[i, j])
        rows.append(row)

    final_csv_path = os.path.join(out_dir, f"final_population_{args.version}.csv")
    pd.DataFrame(rows).to_csv(final_csv_path, index=False)
    print(f"最终种群已保存到: {final_csv_path}")

    # 保存 latent 与预测结果
    np.save(os.path.join(out_dir, "final_population_latent.npy"), population)
    np.save(os.path.join(out_dir, "final_population_pred.npy"), final_pred)
    np.save(os.path.join(out_dir, "final_population_gap.npy"), final_gap)

    # ====================== 最终统计指标 ======================
    final_gap_sorted = np.sort(final_gap)
    best_gap = float(final_gap_sorted[0])
    avg_gap = float(np.mean(final_gap))
    median_gap = float(np.median(final_gap))

    topk = min(10, len(final_gap_sorted))
    top10_mean_gap = float(np.mean(final_gap_sorted[:topk]))

    # score
    gap_score = 1.0 / (1.0 + np.exp((final_gap - 0.15) / 0.03))
    best_score = float(np.max(gap_score))
    avg_score = float(np.mean(gap_score))

    score_sorted = np.sort(gap_score)[::-1]
    top10_mean_gap_score = float(np.mean(score_sorted[:min(10, len(score_sorted))]))

    best_final_idx = int(np.argmax(gap_score))

    final_success_count = int(np.sum(final_gap < 0.15))
    final_success_rate = float(final_success_count / len(final_gap)) if len(final_gap) > 0 else 0.0

    summary = {
        "init_mode": args.init_mode,
        "version": args.version,
        "seed": args.seed,

        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "elite_size": args.elite_size,

        "best_gap_final": best_gap,
        "avg_gap_final": avg_gap,
        "median_gap_final": median_gap,
        "top10_mean_gap_final": top10_mean_gap,

        "best_score_final": best_score,
        "avg_score_final": avg_score,
        "top10_mean_gap_score_final": top10_mean_gap_score,

        "success_count_final": final_success_count,
        "success_rate_final": final_success_rate,

        "eval_at_gap_lt_0.15": eval_at_gap_lt_015,
        "eval_at_gap_lt_0.15_definition": "First molecular evaluations where best_gap_so_far < 0.15",

        "diversity": diversity,
        "validity": decode_rate,

        "best_smiles_final": decoded_smiles[best_final_idx],
        "best_properties_final": {
            p: float(final_pred[best_final_idx, j]) for j, p in enumerate(property_names)
        },

        "decode_success": int(decode_success),

        "progress_metrics_csv": progress_csv_path,
        "final_population_csv": final_csv_path,

        "avg_score_history": [float(x) for x in avg_score_history],
        "avg_gap_history": [float(x) for x in avg_gap_history],
        "best_gap_history": [float(x) for x in best_gap_history],
        "best_gap_so_far_history": [float(x) for x in best_gap_so_far_history],
        "top10_mean_gap_history": [float(x) for x in top10_mean_gap_history],
        "success_count_history": [int(x) for x in success_count_history],
        "success_rate_history": [float(x) for x in success_rate_history],
        "eval_count_history": [int(x) for x in eval_count_history],
    }

    summary_path = os.path.join(out_dir, "summary.json")
    save_json(summary, summary_path)
    print(f"[INFO] summary 已保存到: {summary_path}")

    # 画 gap 平均值曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(avg_score_history)), avg_score_history, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Average Gap Score")
    plt.title(f"GA Score Optimization Curve ({args.init_mode})")

    plt.grid(True)
    plt.tight_layout()
    curve_path = os.path.join(out_dir, "avg_gap_curve.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"平均 gap 曲线已保存到: {curve_path}")

    # 新增：画 best_gap_so_far 和 top10_mean_gap 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(progress_df["evaluations"], progress_df["best_gap_so_far"], label="Best-so-far gap")
    plt.plot(progress_df["evaluations"], progress_df["top10_mean_gap"], label="Top-10 mean gap")
    plt.axhline(0.15, linestyle="--", linewidth=1.0, label="gap = 0.15")
    plt.xlabel("Molecular Evaluations")
    plt.ylabel("Predicted HOMO-LUMO Gap")
    plt.title(f"Convergence Curve ({args.init_mode})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    conv_curve_path = os.path.join(out_dir, "convergence_best_top10_curve.png")
    plt.savefig(conv_curve_path, dpi=300)
    plt.close()
    print(f"收敛曲线已保存到: {conv_curve_path}")

    # 新增：画 success count 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(progress_df["evaluations"], progress_df["success_count"], marker="o")
    plt.xlabel("Molecular Evaluations")
    plt.ylabel("Success Count")
    plt.title(f"Success Count Curve ({args.init_mode})")
    plt.grid(True)
    plt.tight_layout()
    success_curve_path = os.path.join(out_dir, "success_count_curve.png")
    plt.savefig(success_curve_path, dpi=300)
    plt.close()
    print(f"成功分子数量曲线已保存到: {success_curve_path}")

    # 画所有属性均值曲线
    attr_means_arr = np.array(attr_means_history)   # shape: [n_gen, 7]
    plt.figure(figsize=(10, 6))
    for j, pname in enumerate(property_names):
        plt.plot(attr_means_arr[:, j], label=pname)
    plt.xlabel("Generation")
    plt.ylabel("Average Property Value")
    plt.title(f"Average Predicted Properties per Generation ({args.init_mode})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    attr_curve_path = os.path.join(out_dir, "avg_properties_curve.png")
    plt.savefig(attr_curve_path, dpi=300)
    plt.close()
    print(f"属性均值曲线已保存到: {attr_curve_path}")

    print("\n========== 运行完成 ==========")
    print(f"模式: {args.init_mode}")
    print(f"输出目录: {out_dir}")

    print(f"最终最优 gap: {best_gap:.6f}")
    print(f"最终平均 gap: {avg_gap:.6f}")
    print(f"最终 top10 gap: {top10_mean_gap:.6f}")

    print(f"最终最优 score: {best_score:.6f}")
    print(f"最终平均 score: {avg_score:.6f}")
    print(f"最终 top10 score: {top10_mean_gap_score:.6f}")

    print(f"最终 success count: {final_success_count}")
    print(f"最终 success rate: {final_success_rate:.4f}")
    print(f"Eval@Gap<0.15: {eval_at_gap_lt_015}")

    print(f"最终 decode rate (validity): {decode_rate:.4f}")
    print(f"最终 diversity: {diversity:.4f}")

    print(f"最终最优 smiles: {decoded_smiles[best_final_idx]}")


if __name__ == "__main__":
    main()