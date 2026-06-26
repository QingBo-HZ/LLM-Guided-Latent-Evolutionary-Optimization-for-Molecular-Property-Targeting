#!/bin/bash
##########################################################################
# File Name: train_zinc_deep_v2_1gpu_clean.sh
# Purpose: Clean single-GPU ZINC PS-VAE deep training
##########################################################################

set -e

CODE_DIR=`dirname $0`/../src
DATA_DIR=`dirname $0`/../data
CKPT_DIR=`dirname $0`/../ckpts

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH

# 先用单卡确认训练流程正常
export CUDA_VISIBLE_DEVICES=0

# 清理并重新设置线程变量，避免 nthreads 报错
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset NUMEXPR_NUM_THREADS
unset NUMEXPR_MAX_THREADS

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=64

# 不要直接写到总 ckpts，单独建一个新目录，避免读取旧 checkpoint 或旧状态
SAVE_DIR=${CKPT_DIR}/zinc_deep_v2_clean_1gpu_$(date +%Y%m%d_%H%M%S)
mkdir -p ${SAVE_DIR}

echo "========== ZINC PS-VAE Deep Training V2: 1 GPU Clean =========="
echo "CODE_DIR = ${CODE_DIR}"
echo "DATA_DIR = ${DATA_DIR}"
echo "CKPT_DIR = ${CKPT_DIR}"
echo "SAVE_DIR = ${SAVE_DIR}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS = ${MKL_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS = ${NUMEXPR_NUM_THREADS}"
echo "NUMEXPR_MAX_THREADS = ${NUMEXPR_MAX_THREADS}"
echo "==============================================================="

python ${CODE_DIR}/train.py \
  --train_set ${DATA_DIR}/my_zinc/train/train.txt \
  --valid_set ${DATA_DIR}/my_zinc/valid/valid.txt \
  --test_set ${DATA_DIR}/my_zinc/test/test.txt \
  --vocab ${CKPT_DIR}/vocab/my_zinc_bpe_1000.txt \
  --batch_size 32 \
  --shuffle \
  --alpha 0.3 \
  --beta 0 \
  --max_beta 0.03 \
  --step_beta 0.001 \
  --kl_anneal_iter 5000 \
  --kl_warmup 2000 \
  --lr 5e-4 \
  --save_dir ${SAVE_DIR} \
  --grad_clip 5.0 \
  --epochs 60 \
  --patience 12 \
  --gpus 1 \
  --props logp \
  --latent_dim 128 \
  --node_hidden_dim 384 \
  --graph_embedding_dim 512