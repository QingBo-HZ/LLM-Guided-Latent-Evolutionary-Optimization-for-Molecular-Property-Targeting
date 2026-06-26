#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/sweeteners_evolve"
PYTHON="/root/miniconda3/envs/molclr_pyg28/bin/python"
GA_SCRIPT="$ROOT/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py"
LOGDIR="$ROOT/Ablation_1/results/run_epoch_gpu0_logs"
mkdir -p "$LOGDIR"

COMMON_ARGS=(
  --init_mode llm
  --pop_size 100
  --n_gen 1000
  --elite_size 20
  --cross_prob 0.8
  --mut_prob 0.08
  --mut_eta 20
  --patience 20
  --seed 42
  --random_immigrant_frac 0.05
  --archive_topk_per_gen 10
  --max_archive_decode 3000
  --final_decode_temperature 0.8
)

run_group() {
  local version="$1"
  local latent_path="$2"
  local log_file="$LOGDIR/${version}.log"
  echo "[$(date '+%F %T')] START ${version}" | tee -a "$LOGDIR/run_all.log"
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$GA_SCRIPT"     "${COMMON_ARGS[@]}"     --version "$version"     --llm_latent_path "$latent_path"     > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE ${version}" | tee -a "$LOGDIR/run_all.log"
}

run_group llm_epoch_80_std "$ROOT/Ablation_1/llm/latent_epoch_80/llm_init_latent.npy"
run_group llm_epoch_100_std "$ROOT/Ablation_1/llm/latent_epoch_100/llm_init_latent.npy"
