#!/usr/bin/env bash
set -e

cd /root/autodl-tmp/sweeteners_evolve/Gen_Exp

TRAIN_LATENT="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train/zinc_logp_latent.npy"
TRAIN_META="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train/zinc_logp_meta.csv"

VALID_LATENT="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/valid/zinc_logp_latent.npy"
VALID_META="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/valid/zinc_logp_meta.csv"

TEST_LATENT="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/test/zinc_logp_latent.npy"
TEST_META="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/test/zinc_logp_meta.csv"

# ============================================================
# Run 1: recommended main setting
# Residual MLP + SmoothL1 + cosine scheduler
# ============================================================

python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/02_train_logp_predictor.py \
  --train_latent "${TRAIN_LATENT}" \
  --train_meta "${TRAIN_META}" \
  --valid_latent "${VALID_LATENT}" \
  --valid_meta "${VALID_META}" \
  --test_latent "${TEST_LATENT}" \
  --test_meta "${TEST_META}" \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/logp_predictor \
  --gpu 0 \
  --epochs 300 \
  --batch_size 256

