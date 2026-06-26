#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/sweeteners_evolve"
PY="/root/miniconda3/envs/molclr_pyg28/bin/python"
EXP="${ROOT}/Main_results_202604_LLM_GA"

export OMP_NUM_THREADS=1

for seed in 43 44; do
  echo "[START] Random Latent Search seed=${seed} $(date)"
  CUDA_VISIBLE_DEVICES=2 "${PY}" "${EXP}/1_random_search/random_search.py" \
    --n_samples 100000 \
    --batch_size 100 \
    --success_threshold 0.15 \
    --topk_per_step 10 \
    --seed "${seed}" \
    --version "random_search_V2_seed${seed}"
  echo "[DONE] Random Latent Search seed=${seed} $(date)"

  echo "[START] Latent-GA seed=${seed} $(date)"
  CUDA_VISIBLE_DEVICES=2 "${PY}" "${EXP}/3_latent_GA_noLLM/latent_GA_random.py" \
    --init_mode psvae \
    --pop_size 100 \
    --n_gen 1000 \
    --elite_size 20 \
    --cross_prob 0.8 \
    --mut_prob 0.08 \
    --mut_eta 20 \
    --patience 20 \
    --seed "${seed}" \
    --success_threshold 0.15 \
    --version "train_random_V2_seed${seed}"
  echo "[DONE] Latent-GA seed=${seed} $(date)"

  echo "[START] LLM-Initialized Latent GA seed=${seed} $(date)"
  CUDA_VISIBLE_DEVICES=2 "${PY}" "${EXP}/4_latent_GA_LLM/latent_GA_LLM.py" \
    --init_mode llm \
    --pop_size 100 \
    --n_gen 1000 \
    --elite_size 20 \
    --cross_prob 0.8 \
    --mut_prob 0.08 \
    --mut_eta 20 \
    --patience 20 \
    --seed "${seed}" \
    --success_threshold 0.15 \
    --version "llm_V2_seed${seed}"
  echo "[DONE] LLM-Initialized Latent GA seed=${seed} $(date)"

  echo "[START] Iterative LLM-Guided Latent GA seed=${seed} $(date)"
  CUDA_VISIBLE_DEVICES=2 "${PY}" "${EXP}/5_Ours/latent_ours.py" \
    --init_mode llm \
    --pop_size 100 \
    --n_gen 1000 \
    --elite_size 20 \
    --cross_prob 0.8 \
    --mut_prob 0.08 \
    --mut_eta 20 \
    --patience 20 \
    --seed "${seed}" \
    --success_threshold 0.15 \
    --version "ours_V2_seed${seed}"
  echo "[DONE] Iterative LLM-Guided Latent GA seed=${seed} $(date)"
done
