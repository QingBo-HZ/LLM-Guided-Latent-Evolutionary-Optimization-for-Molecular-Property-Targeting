#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/sweeteners_evolve
PYTHON=/root/miniconda3/envs/molclr_pyg28/bin/python
GA=${ROOT}/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py
LOG_DIR=${ROOT}/Ablation_1/results/run_epoch_gpu2_logs
mkdir -p "${LOG_DIR}"

run_one() {
  local round="$1"
  local latent_path="$2"
  local version="llm_epoch_${round}_std"
  local log_path="${LOG_DIR}/${version}.log"

  echo "[$(date '+%F %T')] START ${version}" | tee -a "${LOG_DIR}/run_all.log"
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${GA}" \
    --init_mode llm \
    --pop_size 100 \
    --n_gen 1000 \
    --elite_size 20 \
    --cross_prob 0.8 \
    --mut_prob 0.08 \
    --mut_eta 20 \
    --patience 20 \
    --seed 42 \
    --version "${version}" \
    --llm_latent_path "${latent_path}" \
    --random_immigrant_frac 0.05 \
    --archive_topk_per_gen 10 \
    --max_archive_decode 3000 \
    --final_decode_temperature 0.8 \
    > "${log_path}" 2>&1
  echo "[$(date '+%F %T')] DONE ${version}" | tee -a "${LOG_DIR}/run_all.log"
}

run_one 1   "${ROOT}/Ablation_1/llm/latent_0/llm_init_latent.npy"
run_one 10  "${ROOT}/Ablation_1/llm/latent_epoch_10/llm_init_latent.npy"
run_one 20  "${ROOT}/Ablation_1/llm/latent_epoch_20/llm_init_latent.npy"
run_one 40  "${ROOT}/Ablation_1/llm/latent_epoch_40/llm_init_latent.npy"
run_one 80  "${ROOT}/Ablation_1/llm/latent_epoch_80/llm_init_latent.npy"
run_one 100 "${ROOT}/Ablation_1/llm/latent_epoch_100/llm_init_latent.npy"

echo "[$(date '+%F %T')] ALL DONE" | tee -a "${LOG_DIR}/run_all.log"
