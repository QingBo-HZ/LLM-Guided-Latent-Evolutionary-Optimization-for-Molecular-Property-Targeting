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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Draw, rdFingerprintGenerator
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
from utils.chem_utils import molecule2smiles, GeneralVocab
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
MEAN_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_mean.npy"
STD_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_std.npy"
TRAIN_LATENT_PATH = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy"

DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/1_random_search"
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


def compute_diversity(smiles_list, max_mols=1000):
    """
    防 OOM 版本：
    1) 最多抽样 max_mols 个分子
    2) 不保存全部 pairwise similarity
    3) 使用 MorganGenerator 替代过时接口
    """
    mols = []
    for s in smiles_list:
        if s is None:
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)

    if len(mols) < 2:
        return 0.0

    if len(mols) > max_mols:
        idx = np.random.choice(len(mols), max_mols, replace=False)
        mols = [mols[i] for i in idx]

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpgen.GetFingerprint(m) for m in mols]

    sim_sum = 0.0
    pair_count = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim_sum += DataStructs.TanimotoSimilarity(fps[i], fps[j])
            pair_count += 1

    if pair_count == 0:
        return 0.0

    mean_sim = sim_sum / pair_count
    return 1.0 - float(mean_sim)


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

print("加载训练集 latent...")
if not os.path.exists(TRAIN_LATENT_PATH):
    raise FileNotFoundError(f"训练集 latent 不存在: {TRAIN_LATENT_PATH}")
latent_train = np.load(TRAIN_LATENT_PATH).astype(np.float32)
latent_dim = latent_train.shape[1]

LB = latent_train.min(axis=0)
UB = latent_train.max(axis=0)


# ====================== 解码函数 ======================
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


# ====================== 主函数 ======================
def main():
    parser = argparse.ArgumentParser(description="QM9 Random Search baseline in latent space (stable final)")
    parser.add_argument("--n_samples", type=int, default=6000,
                        help="总评估次数 / 随机采样次数")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="统计进度时的批大小")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="random_v1")
    parser.add_argument("--success_threshold", type=float, default=0.15,
                        help="gap 小于该阈值视为 success")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--max_atom_num", type=int, default=20)
    parser.add_argument("--add_edge_th", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.6)

    parser.add_argument("--topk_per_step", type=int, default=10,
                        help="每个 step 保存前 k 个候选到 topk_evolution_paths.csv")
    parser.add_argument("--diversity_max_mols", type=int, default=1000,
                        help="计算 diversity 时最多抽样多少个分子，防 OOM")

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = os.path.join(args.output_root, f"random_search_{args.version}")
    ensure_dir(out_dir)

    print("\n========== 配置 ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    start_wall_time = time.time()

    # 这里保留 all_*，但对 6000 样本仍是可承受的
    all_latents = []
    all_preds = []
    all_smiles = []

    decode_success = 0

    best_gap_so_far_history = []
    top10_mean_gap_history = []
    success_count_history = []
    success_rate_history = []
    avg_score_history = []
    avg_gap_history = []
    eval_count_history = []
    elapsed_time_history = []

    best_latent_history = []
    best_gap_history = []

    evolution_full_records = []
    topk_records = []

    gaps_seen = []

    n_batches = int(np.ceil(args.n_samples / args.batch_size))

    for b in range(n_batches):
        cur_bs = min(args.batch_size, args.n_samples - b * args.batch_size)
        if cur_bs <= 0:
            break

        z_batch = np.random.uniform(low=LB, high=UB, size=(cur_bs, latent_dim)).astype(np.float32)

        pred_batch = predictor.predict_array(z_batch)
        gap_batch = pred_batch[:, predictor.gap_idx].astype(np.float32)

        smiles_batch = []
        for i in range(cur_bs):
            smi = latent_to_smiles(
                z_batch[i],
                max_atom_num=args.max_atom_num,
                add_edge_th=args.add_edge_th,
                temperature=args.temperature
            )
            smiles_batch.append(smi)
            if smi is not None:
                decode_success += 1

        all_latents.append(z_batch)
        all_preds.append(pred_batch)
        all_smiles.extend(smiles_batch)

        gaps_seen.extend(gap_batch.tolist())
        gaps_seen_np = np.array(gaps_seen, dtype=np.float32)

        gap_score_seen = 1.0 / (1.0 + np.exp((gaps_seen_np - 0.15) / 0.03))

        best_gap_so_far = float(np.min(gaps_seen_np))
        topk = min(10, len(gaps_seen_np))
        top10_mean_gap = float(np.mean(np.sort(gaps_seen_np)[:topk]))

        success_count = int(np.sum(gaps_seen_np < args.success_threshold))
        success_rate = float(success_count / len(gaps_seen_np))

        avg_gap = float(np.mean(gaps_seen_np))
        avg_score = float(np.mean(gap_score_seen))

        flat_latents = np.vstack(all_latents)
        best_idx_global = int(np.argmin(gaps_seen_np))
        best_z = flat_latents[best_idx_global].copy()
        best_gap_val = float(gaps_seen_np[best_idx_global])
        best_smi = latent_to_smiles(
            best_z,
            max_atom_num=args.max_atom_num,
            add_edge_th=args.add_edge_th,
            temperature=args.temperature
        )

        best_latent_history.append(best_z)
        best_gap_history.append(best_gap_val)

        evolution_full_records.append({
            "step": int(b),
            "evaluations": int(len(gaps_seen_np)),
            "smiles": best_smi,
            "gap": best_gap_val
        })

        cur_topk = min(args.topk_per_step, len(gap_batch))
        top_idx_local = np.argsort(gap_batch)[:cur_topk]

        for rank, local_idx in enumerate(top_idx_local):
            smi_local = smiles_batch[local_idx]
            gap_local = float(gap_batch[local_idx])

            topk_records.append({
                "step": int(b),
                "evaluations": int(len(gaps_seen_np)),
                "rank": int(rank + 1),
                "smiles": smi_local,
                "gap": gap_local
            })

        best_gap_so_far_history.append(best_gap_so_far)
        top10_mean_gap_history.append(top10_mean_gap)
        success_count_history.append(success_count)
        success_rate_history.append(success_rate)
        avg_score_history.append(avg_score)
        avg_gap_history.append(avg_gap)
        eval_count_history.append(int(len(gaps_seen_np)))
        elapsed_time_history.append(float(time.time() - start_wall_time))

        print(
            f"[Batch {b:03d}] "
            f"evals={len(gaps_seen_np)}, "
            f"avg_gap={avg_gap:.6f}, "
            f"best_gap={best_gap_so_far:.6f}, "
            f"success={success_count}/{len(gaps_seen_np)}"
        )

    final_latents = np.vstack(all_latents)
    final_pred = np.vstack(all_preds)
    final_gap = final_pred[:, predictor.gap_idx].astype(np.float32)

    valid_smiles = [s for s in all_smiles if s is not None]
    diversity = compute_diversity(valid_smiles, max_mols=args.diversity_max_mols)
    decode_rate = decode_success / len(all_smiles) if len(all_smiles) > 0 else 0.0

    property_names = predictor.property_names
    rows = []
    for i in range(len(final_latents)):
        row = {
            "idx": i,
            "smiles": all_smiles[i],
            "pred_gap": float(final_gap[i]),
        }
        for j, p in enumerate(property_names):
            row[p] = float(final_pred[i, j])
        rows.append(row)

    final_csv_path = os.path.join(out_dir, "final_population_random.csv")
    pd.DataFrame(rows).to_csv(final_csv_path, index=False)
    print(f"最终样本已保存到: {final_csv_path}")

    np.save(os.path.join(out_dir, "final_population_latent.npy"), final_latents)
    np.save(os.path.join(out_dir, "final_population_pred.npy"), final_pred)
    np.save(os.path.join(out_dir, "final_population_gap.npy"), final_gap)

    valid_final_df = pd.DataFrame(rows)
    valid_final_df = valid_final_df[valid_final_df["smiles"].notna()].copy()
    valid_final_df.sort_values("pred_gap", inplace=True)
    valid_final_path = os.path.join(out_dir, "final_population_valid_sorted.csv")
    valid_final_df.to_csv(valid_final_path, index=False)
    print(f"有效最终分子排序结果已保存到: {valid_final_path}")

    progress_df = pd.DataFrame({
        "step": np.arange(len(avg_gap_history)),
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
    print(f"进化过程指标已保存到: {progress_csv_path}")

    evo_full_df = pd.DataFrame(evolution_full_records)
    evo_full_path = os.path.join(out_dir, "evolution_path_full.csv")
    evo_full_df.to_csv(evo_full_path, index=False)
    print(f"完整 evolution path 已保存到: {evo_full_path}")

    if len(evolution_full_records) >= 5:
        pick_ids = [
            0,
            len(evolution_full_records) // 4,
            len(evolution_full_records) // 2,
            3 * len(evolution_full_records) // 4,
            len(evolution_full_records) - 1
        ]
    else:
        pick_ids = list(range(len(evolution_full_records)))

    evo_rows = [evolution_full_records[i] for i in pick_ids]
    evo_df = pd.DataFrame(evo_rows)
    evo_csv_path = os.path.join(out_dir, "evolution_path.csv")
    evo_df.to_csv(evo_csv_path, index=False)
    print(f"精简 evolution path 已保存到: {evo_csv_path}")

    topk_df = pd.DataFrame(topk_records)
    topk_csv_path = os.path.join(out_dir, "topk_evolution_paths.csv")
    topk_df.to_csv(topk_csv_path, index=False)
    print(f"top-k evolution paths 已保存到: {topk_csv_path}")

    progress_df.sort_values("best_gap_so_far").to_csv(
        os.path.join(out_dir, "progress_sorted_by_best_gap.csv"), index=False
    )
    topk_df.sort_values(["gap", "step", "rank"]).to_csv(
        os.path.join(out_dir, "topk_sorted_by_gap.csv"), index=False
    )

    final_gap_sorted = np.sort(final_gap)
    best_gap = float(final_gap_sorted[0])
    avg_gap = float(np.mean(final_gap))
    median_gap = float(np.median(final_gap))

    topk = min(10, len(final_gap_sorted))
    top10_mean_gap = float(np.mean(final_gap_sorted[:topk]))

    gap_score = 1.0 / (1.0 + np.exp((final_gap - 0.15) / 0.03))
    best_score = float(np.max(gap_score))
    avg_score = float(np.mean(gap_score))
    score_sorted = np.sort(gap_score)[::-1]
    top10_mean_gap_score = float(np.mean(score_sorted[:min(10, len(score_sorted))]))

    best_final_idx = int(np.argmax(gap_score))

    final_success_count = int(np.sum(final_gap < args.success_threshold))
    final_success_rate = float(final_success_count / len(final_gap))
    total_time_sec = float(time.time() - start_wall_time)
    total_evaluations = int(len(final_gap))

    summary = {
        "method": "Random Search",
        "task_definition": "Uniform random sampling in latent bounding box -> predictor -> decode",
        "version": args.version,
        "seed": args.seed,

        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "success_threshold": float(args.success_threshold),
        "diversity_max_mols": int(args.diversity_max_mols),

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
        "validity": decode_rate,

        "time_sec_total": total_time_sec,
        "n_evaluations_total": total_evaluations,

        "best_smiles_final": all_smiles[best_final_idx],
        "best_properties_final": {
            p: float(final_pred[best_final_idx, j]) for j, p in enumerate(property_names)
        },

        "decode_success": int(decode_success),

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
    plt.xlabel("Step")
    plt.ylabel("Gap")
    plt.title("Convergence Curve (Random Search)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_convergence_curve.png"), dpi=300)
    plt.close()

    # 图3A：score vs evaluations
    plt.figure(figsize=(8, 5))
    plt.plot(eval_count_history, avg_score_history, marker="o")
    plt.xlabel("Evaluations")
    plt.ylabel("Average Gap Score")
    plt.title("Efficiency Curve: Score vs Evaluations (Random Search)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3a_score_vs_evaluations.png"), dpi=300)
    plt.close()

    # 图3B：success count vs evaluations
    plt.figure(figsize=(8, 5))
    plt.plot(eval_count_history, success_count_history, marker="o")
    plt.xlabel("Evaluations")
    plt.ylabel("Success Count")
    plt.title("Efficiency Curve: Success Count vs Evaluations (Random Search)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3b_success_vs_evaluations.png"), dpi=300)
    plt.close()

    # 图3C：success count vs time
    plt.figure(figsize=(8, 5))
    plt.plot(elapsed_time_history, success_count_history, marker="o")
    plt.xlabel("Elapsed Time (sec)")
    plt.ylabel("Success Count")
    plt.title("Efficiency Curve: Success Count vs Time (Random Search)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3c_success_vs_time.png"), dpi=300)
    plt.close()

    # 图4：从精简版 evolution path 画图
    evo_mols = []
    evo_legends = []

    for row in evo_rows:
        smi = row["smiles"]
        gap_val = row["gap"]
        step_id = row["step"]

        mol = Chem.MolFromSmiles(smi) if smi is not None else None
        evo_mols.append(mol)
        evo_legends.append(f"Step {step_id}\ngap={gap_val:.4f}")

    valid_mols = [m for m in evo_mols if m is not None]
    valid_legends = [evo_legends[i] for i, m in enumerate(evo_mols) if m is not None]

    if len(valid_mols) > 0:
        try:
            img = Draw.MolsToGridImage(
                valid_mols,
                molsPerRow=len(valid_mols),
                subImgSize=(300, 300),
                legends=valid_legends
            )
            fig4_path = os.path.join(out_dir, "fig4_evolution_path.png")
            img.save(fig4_path)
            print(f"图4分子演化路径图已保存到: {fig4_path}")
        except Exception as e:
            print(f"[WARN] 跳过图4分子绘制: {e}")

    # 图5：PCA 化学空间图
    n_train_vis = min(2000, len(latent_train))
    train_vis_idx = np.random.choice(len(latent_train), n_train_vis, replace=False)
    train_vis = latent_train[train_vis_idx]
    gen_vis = final_latents.copy()

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

    plt.scatter(coords[train_mask, 0], coords[train_mask, 1], s=8, alpha=0.4, label="Train")
    plt.scatter(coords[gen_mask, 0], coords[gen_mask, 1], s=16, alpha=0.8, label="Generated")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Chemical Space Visualization by PCA (Random Search)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_pca_chemical_space.png"), dpi=300)
    plt.close()

    # 图5：UMAP 化学空间图（可选）
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
        plt.title("Chemical Space Visualization by UMAP (Random Search)")
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
    print(f"最终 decode rate (validity): {decode_rate:.4f}")
    print(f"最终 diversity: {diversity:.4f}")
    print(f"总时间(秒): {total_time_sec:.2f}")
    print(f"总评估次数: {total_evaluations}")
    print(f"最终最优 smiles: {all_smiles[best_final_idx]}")


if __name__ == "__main__":
    main()