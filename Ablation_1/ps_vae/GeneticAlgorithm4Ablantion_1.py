#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三模式统一版遗传算法优化 QM9 分子 gap
支持初始化模式：
1. llm      : LLM 生成的 smiles -> latent
2. psvae    : PS-VAE 生成的 latent
3. hybrid   : LLM latent 作为 seed，在邻域扩增
4. train_random : 训练集随机初始化（保留作为对照）

说明：
- 当前优化目标默认是最小化 gap
- 后续 predictor / PS-VAE / 解码逻辑与现有脚本保持一致
"""

import os
import sys
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ====================== 添加 PS-VAE 模块路径 ======================
PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))
print("[DEBUG] sys.path appended", flush=True)

# ====================== 导入 PS-VAE 模型 ======================
print("[DEBUG] importing PSVAEModel...", flush=True)
from pl_models import PSVAEModel

# ====================== 导入解码所需工具 ======================
from utils.chem_utils import molecule2smiles

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

# 文件路径（按你当前工程路径设置）
CSV_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/qm9_latent_ga.csv"
CKPT_PSVAE = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt"
PREDICTOR_CKPT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/best_predictor.pt"
MEAN_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_mean.npy"
STD_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_std.npy"
SMILES_LATENT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent"

# 三种初始化文件
LLM_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent/llm_init_latent.npy"
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
try:
    from utils.chem_utils import GeneralVocab
    from data.mol_bpe import Tokenizer
    from rdkit.Chem.rdchem import BondType
    import torch.serialization

    SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
    torch.serialization.add_safe_globals(SAFE_GLOBALS)
    print(f"[DEBUG] registered safe globals: {[str(x) for x in SAFE_GLOBALS]}")
except ImportError as e:
    print(f"警告：无法导入安全全局类，若后续加载失败请检查路径。错误：{e}")

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

latent_cache_file = os.path.join(SMILES_LATENT, "x_train.npy")
if os.path.exists(latent_cache_file):
    print("加载缓存的训练数据 latent...")
    latent_train = np.load(latent_cache_file)
else:
    raise FileNotFoundError(f"缓存文件 {latent_cache_file} 不存在，请先运行编码步骤。")

latent_dim = latent_train.shape[1]
print(f"Latent 维度: {latent_dim}")
print(f"Latent shape: {latent_train.shape}")

LB = latent_train.min(axis=0)
UB = latent_train.max(axis=0)


# ====================== 解码函数 ======================
def latent_to_smiles(z):
    """
    尝试把 latent 向量解码为 SMILES。
    由于不同版本 PS-VAE API 可能略有不同，这里做多重兼容。
    """
    z_tensor = torch.tensor(z, dtype=torch.float32, device=DEVICE).view(1, -1)

    decode_methods = [
        "inference_single_z",
        "inference_from_z",
        "decode_from_z",
        "sample_from_z",
    ]

    for method_name in decode_methods:
        if hasattr(model_psvae, method_name):
            try:
                method = getattr(model_psvae, method_name)
                out = method(z_tensor)
                if isinstance(out, str):
                    return out
                if isinstance(out, list) and len(out) > 0:
                    if isinstance(out[0], str):
                        return out[0]
                    try:
                        return molecule2smiles(out[0])
                    except Exception:
                        pass
                try:
                    return molecule2smiles(out)
                except Exception:
                    pass
            except Exception:
                continue

    # 再试 model.model
    if hasattr(model_psvae, "model"):
        inner_model = model_psvae.model
        for method_name in decode_methods:
            if hasattr(inner_model, method_name):
                try:
                    method = getattr(inner_model, method_name)
                    out = method(z_tensor)
                    if isinstance(out, str):
                        return out
                    if isinstance(out, list) and len(out) > 0:
                        if isinstance(out[0], str):
                            return out[0]
                        try:
                            return molecule2smiles(out[0])
                        except Exception:
                            pass
                    try:
                        return molecule2smiles(out)
                    except Exception:
                        pass
                except Exception:
                    continue

    return None


# ====================== 适应度函数 ======================
def fitness_function(z):
    """
    输入单个 latent 向量 z，返回预测的 gap（最小化目标）
    """
    pred_dict = predictor.predict_dict(z)
    return pred_dict["gap"]


# ====================== 遗传操作函数 ======================
def tournament_selection(pop, fitness, tourn_size=2):
    indices = np.random.choice(len(pop), tourn_size, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]
    return pop[best_idx].copy()


def arithmetic_crossover(p1, p2, cross_prob):
    if np.random.random() > cross_prob:
        return p1.copy(), p2.copy()

    alpha = np.random.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    c1 = np.clip(c1, LB, UB)
    c2 = np.clip(c2, LB, UB)
    return c1, c2


def polynomial_mutation(individual, prob, eta, low, up):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < prob:
            r = np.random.random()
            if r < 0.5:
                delta = (2 * r) ** (1.0 / (eta + 1)) - 1.0
            else:
                delta = 1.0 - (2 * (1.0 - r)) ** (1.0 / (eta + 1))
            mutated[i] += delta * (up[i] - low[i])
            mutated[i] = np.clip(mutated[i], low[i], up[i])
    return mutated


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


def build_hybrid_population(llm_latent, pop_size, sigma=0.08, seed_ratio=0.5):
    llm_latent = np.asarray(llm_latent, dtype=np.float32)

    n_seed = max(1, int(pop_size * seed_ratio))
    base = sample_population_from_pool(llm_latent, n_seed)

    n_expand = pop_size - n_seed
    expanded = []
    for _ in range(n_expand):
        parent = base[np.random.randint(0, len(base))].copy()
        noise = np.random.normal(0.0, sigma, size=parent.shape).astype(np.float32)
        child = parent + noise
        child = np.clip(child, LB, UB)
        expanded.append(child)

    if len(expanded) > 0:
        expanded = np.stack(expanded, axis=0)
        population = np.concatenate([base, expanded], axis=0)
    else:
        population = base

    return population.astype(np.float32)


def initialize_population(init_mode, pop_size, llm_latent_path, psvae_latent_path, hybrid_latent_path, hybrid_sigma):
    """
    初始化种群：
    - train_random: 从训练集 latent 中随机抽样
    - llm        : 从 llm_init_latent.npy 抽样
    - psvae      : 从 psvae_init_latent.npy 抽样
    - hybrid     : 优先读 hybrid_init_latent.npy；没有则 llm latent + 邻域扩增
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
            raise FileNotFoundError(
                f"Hybrid 模式没有现成 hybrid_init_latent.npy，"
                f"同时 llm latent 也不存在: {llm_latent_path}"
            )
        llm_latent = np.load(llm_latent_path).astype(np.float32)
        print(f"[INFO] 使用 LLM seed + 邻域扩增构建 Hybrid 初始化，llm shape={llm_latent.shape}")
        if llm_latent.shape[1] != latent_dim:
            raise ValueError(f"LLM latent 维度错误: {llm_latent.shape}, 期望 (*, {latent_dim})")
        return build_hybrid_population(llm_latent, pop_size, sigma=hybrid_sigma, seed_ratio=0.5)

    else:
        raise ValueError(f"未知 init_mode: {init_mode}")


# ====================== 主函数 ======================
def main():
    parser = argparse.ArgumentParser(description="三模式统一版 QM9 latent-space GA")
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
    parser.add_argument("--hybrid_sigma", type=float, default=0.08)

    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT)

    args = parser.parse_args()
    set_seed(args.seed)

    # 每种模式单独输出到不同目录
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
    )
    print(f"[INFO] 初始种群 shape: {population.shape}")

    # 遗传算法主循环
    avg_fitness_history = []       # gap 平均值
    attr_means_history = []        # 所有属性平均值
    best_fitness_so_far = float("inf")
    no_improve_count = 0

    for gen in range(args.n_gen):
        pred_all = predictor.predict_array(population)    # shape (POP_SIZE, 7)
        attr_means = pred_all.mean(axis=0)
        attr_means_history.append(attr_means)

        fitness = pred_all[:, predictor.gap_idx]          # 目标：最小化 gap
        avg_fitness = float(np.mean(fitness))
        best_fitness = float(np.min(fitness))
        best_idx = int(np.argmin(fitness))
        best_individual = population[best_idx].copy()
        best_pred = pred_all[best_idx].copy()

        avg_fitness_history.append(avg_fitness)

        print(
            f"[Gen {gen:03d}] avg_gap={avg_fitness:.6f}, "
            f"best_gap={best_fitness:.6f}, "
            f"no_improve={no_improve_count}"
        )

        # 早停逻辑（看平均 gap 是否改善）
        if avg_fitness < best_fitness_so_far:
            best_fitness_so_far = avg_fitness
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= args.patience:
            print(f"[Early Stop] 连续 {args.patience} 代平均 gap 未改善，提前停止。")
            break

        # 精英保留（gap 越小越好）
        sorted_idx = np.argsort(fitness)
        elites = population[sorted_idx[:args.elite_size]].copy()

        # 构造下一代
        new_population = list(elites)

        while len(new_population) < args.pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)

            c1, c2 = arithmetic_crossover(p1, p2, args.cross_prob)
            c1 = polynomial_mutation(c1, args.mut_prob, args.mut_eta, LB, UB)
            c2 = polynomial_mutation(c2, args.mut_prob, args.mut_eta, LB, UB)

            new_population.append(c1)
            if len(new_population) < args.pop_size:
                new_population.append(c2)

        population = np.array(new_population, dtype=np.float32)

    # 最终评估
    print("对最终种群进行预测...")
    final_pred = predictor.predict_array(population)
    final_gap = final_pred[:, predictor.gap_idx]

    # 解码最终种群
    print("解码最终种群为 SMILES（如果某些失败，会记为空）...")
    decoded_smiles = []
    for i in range(len(population)):
        smi = latent_to_smiles(population[i])
        decoded_smiles.append(smi)

    # 保存最终种群信息
    property_names = predictor.property_names
    rows = []
    for i in range(len(population)):
        row = {
            "idx": i,
            "smiles": decoded_smiles[i],
            "fitness_gap": float(final_gap[i]),
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

    # 保存 summary
    best_final_idx = int(np.argmin(final_gap))
    summary = {
        "init_mode": args.init_mode,
        "version": args.version,
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "elite_size": args.elite_size,
        "best_gap_final": float(final_gap[best_final_idx]),
        "best_smiles_final": decoded_smiles[best_final_idx],
        "best_properties_final": {p: float(final_pred[best_final_idx, j]) for j, p in enumerate(property_names)},
        "avg_gap_history": [float(x) for x in avg_fitness_history],
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))

    # 画 gap 平均值曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(avg_fitness_history)), avg_fitness_history, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Average Predicted Gap")
    plt.title(f"GA Optimization Curve ({args.init_mode})")
    plt.grid(True)
    plt.tight_layout()
    curve_path = os.path.join(out_dir, "avg_gap_curve.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"平均 gap 曲线已保存到: {curve_path}")

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
    print(f"最终最优 gap: {float(final_gap[best_final_idx]):.6f}")
    print(f"最终最优 smiles: {decoded_smiles[best_final_idx]}")


if __name__ == "__main__":
    main()