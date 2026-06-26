#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 x_train.npy 中按 predictor 预测属性进行排序采样，生成 QM9-seeded 初始化 latent。

用途：
    为 GA 的 QM9-seeded / dataset-seeded baseline 提供“公平”的初始化种子池。
    默认按 gap 最小化，从训练集 latent 中挑选最优的若干个。

支持：
    1) 严格取前 n_samples 个最优 latent
    2) 先取前 top_pool_size 个，再从中随机抽 n_samples 个（更有多样性）
    3) 可选加微小噪声
    4) 支持最小化/最大化任意 predictor 属性

示例：
    # 直接取 gap 最小的前 1000 个
    python sample_qm9_seeded_latent_by_property.py \
      --n_samples 1000 \
      --property gap \
      --objective min \
      --select_mode topk \
      --out_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/latent/qm9_gapmin_top1000.npy

    # 先取 gap 最小的前 5000 个，再随机抽 1000 个（更公平一些）
    python sample_qm9_seeded_latent_by_property.py \
      --n_samples 1000 \
      --property gap \
      --objective min \
      --select_mode top_pool_random \
      --top_pool_size 5000 \
      --out_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/latent/qm9_gapmin_pool5000_rand1000.npy
"""

import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn


# ====================== Predictor 定义 ======================
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
            dropout=ckpt.get("dropout", 0.0),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.property_names = ckpt["property_names"]
        self.y_mean = np.load(mean_path)
        self.y_std = np.load(std_path)

    def enforce_physical_constraints(self, pred, margin=1e-6):
        """
        与你主 GA 脚本保持一致：
        默认 property_names 中:
            0 -> homo
            1 -> lumo
            2 -> gap
        """
        pred = pred.copy()

        # 如果属性数不足 3，直接返回
        if pred.shape[1] < 3:
            return pred

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

    def predict_array(self, z, batch_size=4096):
        """
        z: np.ndarray [N, D]
        return: np.ndarray [N, num_property]
        """
        z = np.asarray(z, dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"输入 z 必须是二维数组，当前 shape={z.shape}")

        outputs = []
        n = len(z)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x = torch.tensor(z[start:end], dtype=torch.float32, device=self.device)
                pred_norm = self.model(x).cpu().numpy()
                pred = pred_norm * self.y_std + self.y_mean
                pred = self.enforce_physical_constraints(pred)
                outputs.append(pred)

        return np.concatenate(outputs, axis=0)


# ====================== 工具函数 ======================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ====================== 主逻辑 ======================
def main(args):
    set_seed(args.seed)

    ensure_dir(os.path.dirname(args.out_path))
    if args.save_meta:
        ensure_dir(os.path.dirname(args.meta_path))

    print("========== Sample QM9-Seeded Latent By Property ==========")
    print(f"train_latent   : {args.train_latent}")
    print(f"predictor_ckpt : {args.predictor_ckpt}")
    print(f"mean_path      : {args.mean_path}")
    print(f"std_path       : {args.std_path}")
    print(f"property       : {args.property}")
    print(f"objective      : {args.objective}")
    print(f"select_mode    : {args.select_mode}")
    print(f"top_pool_size  : {args.top_pool_size}")
    print(f"n_samples      : {args.n_samples}")
    print(f"mode           : {args.mode}")
    print(f"noise_std      : {args.noise_std}")
    print(f"seed           : {args.seed}")
    print(f"out_path       : {args.out_path}")
    print("=========================================================")

    device = args.device
    print(f"[INFO] using device: {device}")

    # 1) 读取训练 latent
    z_train = np.load(args.train_latent).astype(np.float32)
    if z_train.ndim != 2:
        raise ValueError(f"x_train.npy 必须是二维数组，当前 shape={z_train.shape}")

    n_train, latent_dim = z_train.shape
    print(f"[INFO] train latent shape = {z_train.shape}")

    lb = z_train.min(axis=0)
    ub = z_train.max(axis=0)

    # 2) 加载 predictor
    predictor = QM9PredictorAPI(
        predictor_ckpt=args.predictor_ckpt,
        mean_path=args.mean_path,
        std_path=args.std_path,
        device=device,
    )
    print(f"[INFO] predictor properties = {predictor.property_names}")

    if args.property not in predictor.property_names:
        raise ValueError(
            f"property='{args.property}' 不在 predictor.property_names 中: {predictor.property_names}"
        )

    prop_idx = predictor.property_names.index(args.property)
    print(f"[INFO] selected property index = {prop_idx}")

    # 3) 对全部训练 latent 预测属性
    print("[INFO] predicting properties for x_train.npy ...")
    pred_all = predictor.predict_array(z_train, batch_size=args.batch_size)
    prop_values = pred_all[:, prop_idx].astype(np.float32)

    # 4) 排序（最小化 / 最大化）
    if args.objective == "min":
        sorted_idx = np.argsort(prop_values)         # 从小到大
    elif args.objective == "max":
        sorted_idx = np.argsort(prop_values)[::-1]   # 从大到小
    else:
        raise ValueError(f"未知 objective: {args.objective}")

    # 5) 选择候选池
    if args.select_mode == "topk":
        k = min(args.n_samples, n_train)
        chosen_idx = sorted_idx[:k]

        # 如果 n_samples > n_train，则重复补齐
        if k < args.n_samples:
            extra = np.random.choice(chosen_idx, args.n_samples - k, replace=True)
            chosen_idx = np.concatenate([chosen_idx, extra], axis=0)

    elif args.select_mode == "top_pool_random":
        top_pool_size = min(args.top_pool_size, n_train)
        if top_pool_size < args.n_samples:
            raise ValueError(
                f"top_pool_size={top_pool_size} 小于 n_samples={args.n_samples}，"
                f"请增大 top_pool_size 或减小 n_samples"
            )

        pool_idx = sorted_idx[:top_pool_size]
        chosen_idx = np.random.choice(pool_idx, args.n_samples, replace=False)

    else:
        raise ValueError(f"未知 select_mode: {args.select_mode}")

    Z = z_train[chosen_idx].copy()

    # 6) 可选：加微小噪声
    if args.mode == "sample_noise":
        noise = np.random.normal(
            loc=0.0,
            scale=args.noise_std,
            size=Z.shape
        ).astype(np.float32)
        Z = Z + noise
        Z = np.clip(Z, lb, ub)

    # 7) 保存 latent
    np.save(args.out_path, Z)
    print(f"[DONE] saved latent -> {args.out_path}")
    print(f"[DONE] saved shape  = {Z.shape}")

    # 8) 保存元信息
    if args.save_meta:
        chosen_prop_values = prop_values[chosen_idx]

        if args.objective == "min":
            best_value = float(np.min(chosen_prop_values))
            avg_value = float(np.mean(chosen_prop_values))
            top10_value = float(np.mean(np.sort(chosen_prop_values)[:min(10, len(chosen_prop_values))]))
        else:
            best_value = float(np.max(chosen_prop_values))
            avg_value = float(np.mean(chosen_prop_values))
            top10_value = float(np.mean(np.sort(chosen_prop_values)[::-1][:min(10, len(chosen_prop_values))]))

        meta = {
            "train_latent": args.train_latent,
            "predictor_ckpt": args.predictor_ckpt,
            "mean_path": args.mean_path,
            "std_path": args.std_path,
            "property": args.property,
            "objective": args.objective,
            "select_mode": args.select_mode,
            "top_pool_size": args.top_pool_size,
            "n_samples": args.n_samples,
            "mode": args.mode,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "latent_shape": list(Z.shape),
            "property_names": predictor.property_names,
            "best_selected_property_value": best_value,
            "avg_selected_property_value": avg_value,
            "top10_selected_property_value": top10_value,
            "out_path": args.out_path,
        }

        save_json(meta, args.meta_path)
        print(f"[DONE] saved meta  -> {args.meta_path}")
        print(json.dumps(meta, ensure_ascii=False, indent=2))

    print("=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按 predictor 属性排序，从 QM9 x_train.npy 中采样初始化 latent"
    )

    parser.add_argument(
        "--train_latent",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy",
        help="训练集 latent 文件 x_train.npy"
    )

    parser.add_argument(
        "--predictor_ckpt",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/best_predictor.pt",
        help="predictor checkpoint 路径"
    )

    parser.add_argument(
        "--mean_path",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_mean.npy",
        help="predictor y_mean.npy 路径"
    )

    parser.add_argument(
        "--std_path",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_std.npy",
        help="predictor y_std.npy 路径"
    )

    parser.add_argument(
        "--property",
        type=str,
        default="gap",
        help="按哪个 predictor 属性排序，例如 gap / homo / lumo"
    )

    parser.add_argument(
        "--objective",
        type=str,
        default="min",
        choices=["min", "max"],
        help="min 表示最小化该属性，max 表示最大化该属性"
    )

    parser.add_argument(
        "--select_mode",
        type=str,
        default="topk",
        choices=["topk", "top_pool_random"],
        help=(
            "topk: 直接取前 n_samples 个最优；"
            "top_pool_random: 先取前 top_pool_size 个，再随机抽 n_samples 个"
        )
    )

    parser.add_argument(
        "--top_pool_size",
        type=int,
        default=5000,
        help="当 select_mode=top_pool_random 时，从前 top_pool_size 个里随机抽样"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="输出多少个初始化 latent"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="sample_only",
        choices=["sample_only", "sample_noise"],
        help="sample_only: 纯排序采样; sample_noise: 采样后加微小噪声"
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.02,
        help="当 mode=sample_noise 时的噪声强度"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="predictor 批量预测时的 batch size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu / cuda"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/latent/qm9_gapmin_top1000.npy",
        help="输出 latent 文件路径"
    )

    parser.add_argument(
        "--save_meta",
        action="store_true",
        help="是否额外保存采样元信息 json"
    )

    parser.add_argument(
        "--meta_path",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/latent/qm9_gapmin_top1000_meta.json",
        help="元信息 json 输出路径"
    )

    args = parser.parse_args()
    main(args)