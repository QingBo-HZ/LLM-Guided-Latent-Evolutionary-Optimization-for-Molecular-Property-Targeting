#!/bin/bash

# Step0 查看数据分布；得到含固定标签的QM9数据

python /root/autodl-tmp/sweeteners_evolve/QM9_test/qm9_univariate_dist.py
python /root/autodl-tmp/sweeteners_evolve/QM9_test/prep_qm9_lantent_ga_csv.py

# Step1 得到有QM标签的csv数据
python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/rebuild_labeled_split.py \
  --master_csv /root/autodl-tmp/sweeteners_evolve/QM9_test/qm9_latent_ga.csv \
  --split_file /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_qm9/train.txt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split \
  --out_name train_labeled.csv

python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/rebuild_labeled_split.py \
  --master_csv /root/autodl-tmp/sweeteners_evolve/QM9_test/qm9_latent_ga.csv \
  --split_file /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_qm9/valid.txt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split \
  --out_name valid_labeled.csv

python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/rebuild_labeled_split.py \
  --master_csv /root/autodl-tmp/sweeteners_evolve/QM9_test/qm9_latent_ga.csv \
  --split_file /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_qm9/test.txt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split \
  --out_name test_labeled.csv

得到的数据在 /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/，这是预测器训练需要的初始文件

# Step 2：导出 train/valid/test 的 latent
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

# Step 3：检查 latent 和标签维度
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
  --save_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2 \
  --hidden_dim 256 \
  --dropout 0.2 \
  --batch_size 256 \
  --epochs 50 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --patience 8 \
  --lambda_order 2.0 \
  --lambda_consistency 2.0 \
  --lambda_positive 2.0

#Step 5：调用QM9PredictorAPI
from predictor_api_qm9 import QM9PredictorAPI
import numpy as np

api = QM9PredictorAPI(
    predictor_ckpt="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/best_predictor.pt",
    mean_path="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/y_mean.npy",
    std_path="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/predictor_ckpt_V2/y_std.npy",
    device="cuda"
)

z = np.random.randn(56).astype(np.float32)

pred = api.predict_dict(z)
print(pred)

