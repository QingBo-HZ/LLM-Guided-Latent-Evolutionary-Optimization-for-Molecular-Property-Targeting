#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 路径设置
# =========================
csv_path = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/train_history.csv"
save_dir = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2"

os.makedirs(save_dir, exist_ok=True)

# =========================
# 读取数据
# =========================
df = pd.read_csv(csv_path)

print("历史记录维度:", df.shape)
print("列名:", df.columns.tolist())

# 检查 epoch 数
num_epochs = len(df)
print(f"共记录 {num_epochs} 轮")

# =========================
# 1. train / valid loss 曲线
# =========================
plt.figure(figsize=(8, 6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
plt.plot(df["epoch"], df["valid_loss"], label="Valid Loss", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_fig_path = os.path.join(save_dir, "loss_curve.png")
plt.savefig(loss_fig_path, dpi=300)
plt.close()

print(f"已保存: {loss_fig_path}")

# =========================
# 2. 各性质 val_mae 曲线
# =========================
mae_cols = [
    "val_mae_homo",
    "val_mae_lumo",
    "val_mae_gap",
    "val_mae_u0",
    "val_mae_u298",
    "val_mae_h298",
    "val_mae_g298",
]

plt.figure(figsize=(10, 7))
for col in mae_cols:
    plt.plot(df["epoch"], df[col], label=col, linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Validation MAE (normalized)")
plt.title("Validation MAE of QM9 Properties")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

mae_fig_path = os.path.join(save_dir, "val_mae_curve.png")
plt.savefig(mae_fig_path, dpi=300)
plt.close()

print(f"已保存: {mae_fig_path}")

# =========================
# 3. 分成子图画每个性质
# =========================
fig, axes = plt.subplots(4, 2, figsize=(12, 12))
axes = axes.flatten()

for i, col in enumerate(mae_cols):
    axes[i].plot(df["epoch"], df[col], linewidth=2)
    axes[i].set_title(col)
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Val MAE")
    axes[i].grid(True, alpha=0.3)

# 最后一个空白子图去掉
if len(mae_cols) < len(axes):
    for j in range(len(mae_cols), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()

sub_fig_path = os.path.join(save_dir, "val_mae_subplots.png")
plt.savefig(sub_fig_path, dpi=300)
plt.close()

print(f"已保存: {sub_fig_path}")

print("可视化完成。")