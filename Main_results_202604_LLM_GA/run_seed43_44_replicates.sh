#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/sweeteners_evolve"
PY="/root/miniconda3/envs/molclr_pyg28/bin/python"
EXP="${ROOT}/Main_results_202604_LLM_GA"
LOG_DIR="${EXP}/logs_seed43_44"
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1

run_smiles() {
  local seed="$1"
  local gpu="$2"
  echo "[START] SMILES-GA seed=${seed} gpu=${gpu} $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EXP}/2_smiles_GA/smiles_ga_qm9_childselect.py" \
    --train_smiles_csv "${ROOT}/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/train_labeled.csv" \
    --smiles_col smiles \
    --pop_size 100 \
    --n_gen 1000 \
    --elite_size 30 \
    --mut_prob 0.20 \
    --cross_prob 0.20 \
    --fragment_lib_max_mols 50000 \
    --success_threshold 0.15 \
    --warm_start \
    --warm_start_frac 0.8 \
    --warm_start_gap_upper 0.25 \
    --child_trials 10 \
    --tourn_size 5 \
    --random_immigrant_frac 0.05 \
    --seed "${seed}" \
    --version "smiles_childselect_v1_seed${seed}"
  echo "[DONE] SMILES-GA seed=${seed} $(date)"
}

run_short_jobs() {
  local gpu="$1"
  for seed in 43 44; do
    echo "[START] Random Latent Search seed=${seed} gpu=${gpu} $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EXP}/1_random_search/random_search.py" \
      --n_samples 100000 \
      --batch_size 100 \
      --success_threshold 0.15 \
      --topk_per_step 10 \
      --seed "${seed}" \
      --version "random_search_V2_seed${seed}"
    echo "[DONE] Random Latent Search seed=${seed} $(date)"

    echo "[START] Latent-GA seed=${seed} gpu=${gpu} $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EXP}/3_latent_GA_noLLM/latent_GA_random.py" \
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

    echo "[START] LLM-Initialized Latent GA seed=${seed} gpu=${gpu} $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EXP}/4_latent_GA_LLM/latent_GA_LLM.py" \
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

    echo "[START] Iterative LLM-Guided Latent GA seed=${seed} gpu=${gpu} $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EXP}/5_Ours/latent_ours.py" \
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
}

run_smiles 43 0 > "${LOG_DIR}/smiles_seed43.log" 2>&1 &
pid_smiles43=$!

run_smiles 44 1 > "${LOG_DIR}/smiles_seed44.log" 2>&1 &
pid_smiles44=$!

run_short_jobs 2 > "${LOG_DIR}/short_jobs_gpu2.log" 2>&1 &
pid_short=$!

echo "${pid_smiles43}" > "${LOG_DIR}/pid_smiles43.txt"
echo "${pid_smiles44}" > "${LOG_DIR}/pid_smiles44.txt"
echo "${pid_short}" > "${LOG_DIR}/pid_short_jobs.txt"

echo "[INFO] Started jobs:"
echo "  SMILES seed43 PID=${pid_smiles43} GPU=0 log=${LOG_DIR}/smiles_seed43.log"
echo "  SMILES seed44 PID=${pid_smiles44} GPU=1 log=${LOG_DIR}/smiles_seed44.log"
echo "  Short jobs    PID=${pid_short} GPU=2 log=${LOG_DIR}/short_jobs_gpu2.log"

wait "${pid_smiles43}" "${pid_smiles44}" "${pid_short}"
echo "[ALL DONE] $(date)" | tee "${LOG_DIR}/ALL_DONE.txt"
