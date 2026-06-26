#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 x_train.npy 中随机采样，生成 PS-VAE 组初始化 latent。

用途：
    为 GA 的 PS-VAE 初始化组提供初始种群池。

输入：
    x_train.npy   shape = (N, latent_dim)

输出：
    psvae_init_latent.npy   shape = (n_samples, latent_dim)
"""

import os
import argparse
import numpy as np


def main(args):
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    print("========== Sample PS-VAE Init Latent ==========")
    print(f"train_latent: {args.train_latent}")
    print(f"n_samples   : {args.n_samples}")
    print(f"mode        : {args.mode}")
    print(f"noise_std   : {args.noise_std}")
    print(f"out_path    : {args.out_path}")

    z_train = np.load(args.train_latent).astype(np.float32)
    if z_train.ndim != 2:
        raise ValueError(f"x_train.npy 必须是二维数组，当前 shape={z_train.shape}")

    n_train, latent_dim = z_train.shape
    print(f"train shape : {z_train.shape}")

    # train latent 的边界，用于可选裁剪
    lb = z_train.min(axis=0)
    ub = z_train.max(axis=0)

    # 随机抽样
    if n_train >= args.n_samples:
        idx = np.random.choice(n_train, args.n_samples, replace=False)
    else:
        idx = np.random.choice(n_train, args.n_samples, replace=True)

    Z = z_train[idx].copy()

    # 可选：加微小噪声
    if args.mode == "sample_noise":
        noise = np.random.normal(
            loc=0.0,
            scale=args.noise_std,
            size=Z.shape
        ).astype(np.float32)
        Z = Z + noise
        Z = np.clip(Z, lb, ub)

    np.save(args.out_path, Z)

    print(f"[DONE] saved -> {args.out_path}")
    print(f"saved shape = {Z.shape}")
    print("===============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_latent",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy",
        help="训练集 latent 文件 x_train.npy"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="输出多少个初始化 latent"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="sample_only",
        choices=["sample_only", "sample_noise"],
        help="sample_only: 纯随机抽样; sample_noise: 抽样后加微小噪声"
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.02,
        help="当 mode=sample_noise 时的噪声强度"
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/latent/psvae_init_latent.npy",
        help="输出文件路径"
    )

    args = parser.parse_args()
    main(args)