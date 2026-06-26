#!/bin/bash

#Step 1：补充labeled_split的QM9其他数据
python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/rebuild_labeled_split.py

#Step 2：导出 train/valid/test 的 latent
python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/extract_latent_qm9.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt \
  --csv_path /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/train_labeled.csv \
  --split_name train \
  --out_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent \
  --gpu 0

python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/extract_latent_qm9.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt \
  --csv_path /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/valid_labeled.csv \
  --split_name valid \
  --out_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent \
  --gpu 0

python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/extract_latent_qm9.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt \
  --csv_path /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/test_labeled.csv \
  --split_name test \
  --out_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent \
  --gpu 0

#Step 3：检查 latent 和标签维度
train (107039, 56) (107039, 7)
valid (13380, 56) (13380, 7)
test (13380, 56) (13380, 7)

#Step 4：训练 predictor
python -u /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/train_qm9_predictor.py \
  --x_train /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy \
  --x_valid /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_valid.npy \
  --x_test  /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_test.npy \
  --y_train /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/y_train.npy \
  --y_valid /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/y_valid.npy \
  --y_test  /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/y_test.npy \
  --save_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt \
  --hidden_dim 256 \
  --dropout 0.2 \
  --batch_size 256 \
  --epochs 100 \
  --lr 5e-4 \
  --weight_decay 1e-5 \
  --patience 8 \
  --min_delta 1e-4

# Step 6：在代码里调用 predictor
用法示例
------------------------------------------------------------------------------------------------------------------------
from predictor_api_qm9 import QM9PredictorAPI
import numpy as np

api = QM9PredictorAPI(
    predictor_ckpt="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/best_predictor.pt",
    mean_path="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_mean.npy",
    std_path="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt/y_std.npy",
    device="cuda"
)

z = np.random.randn(56)   # 举例，56只是示意，真实维度以你的latent为准
print(api.predict_dict(z))
--------------------------------------------------------------------------------------------------------------------------