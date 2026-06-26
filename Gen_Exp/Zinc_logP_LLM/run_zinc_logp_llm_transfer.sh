#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/sweeteners_evolve"
PY="${PY:-/root/miniconda3/envs/molclr_pyg28/bin/python}"
GPU="${GPU:-0}"

BASE="${ROOT}/Gen_Exp/Zinc_logP_LLM"
SMILES_DIR="${BASE}/smiles"
LATENT_DIR="${BASE}/latent"
RESULTS_DIR="${BASE}/results"
LOG_DIR="${BASE}/logs"

GEN="${BASE}/generate_zinc_logp_smiles.py"
EXTRACT="${ROOT}/Ablation_1/llm/extract_latent4llm.py"
GA="${ROOT}/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py"
SUMMARIZE="${BASE}/summarize_zinc_logp_llm_transfer.py"

ZINC_CKPT="${ROOT}/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt"
PREDICTOR="${ROOT}/Gen_Exp/Zinc_logP_kek/logp_predictor/best_logp_predictor.pt"
LATENT_POOL="${ROOT}/Gen_Exp/Zinc_logP_kek/train/zinc_logp_latent.npy"

mkdir -p "${SMILES_DIR}" "${LATENT_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

latest_smi() {
  local mode="$1"
  local seed="$2"
  ls -t "${SMILES_DIR}"/zinc_logp_llm_"${mode}"_*_seed"${seed}"_*.smi | head -n 1
}

COMMON_GA_ARGS=(
  --zinc_psvae_ckpt "${ZINC_CKPT}"
  --predictor_ckpt "${PREDICTOR}"
  --latent_pool "${LATENT_POOL}"
  --output_root "${RESULTS_DIR}"
  --pop_size 80
  --n_gen 30
  --elite_size 16
  --cross_prob 0.30
  --mut_prob 0.05
  --mut_eta 20
  --patience 8
  --immigrant_ratio 0.20
  --manifold_anchor_size 5000
  --manifold_blend 0.35
  --selection_metric rdkit_hybrid
  --rdkit_selection_weight 0.75
  --target_logp 3.0
  --success_low 2.5
  --success_high 3.5
  --score_sigma 0.5
  --max_atom_num 80
  --add_edge_th 0.45
  --temperature 0.30
  --decode_topk_per_gen 30
  --decode_attempts_per_latent 3
  --gpu "${GPU}"
)

echo "========== ZINC logP LLM transfer pipeline =========="
echo "ROOT=${ROOT}"
echo "GPU=${GPU}"
echo "PY=${PY}"
echo "SMILES_DIR=${SMILES_DIR}"
echo "LATENT_DIR=${LATENT_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "====================================================="

for seed in 42 43 44; do
  run_step "llm_direct_generate_seed${seed}" \
    "${PY}" "${GEN}" \
      --mode direct \
      --model gpt-5.4-mini \
      --seed "${seed}" \
      --generations 20 \
      --n_candidates_per_call 30 \
      --target_total 500 \
      --temperature 0.7 \
      --max_tokens 2200 \
      --out_dir "${SMILES_DIR}" \
      --prefix zinc_logp_llm

  direct_smi="$(latest_smi direct "${seed}")"

  run_step "llm_direct_extract_latent_seed${seed}" \
    "${PY}" "${EXTRACT}" \
      --ckpt "${ZINC_CKPT}" \
      --input_path "${direct_smi}" \
      --out_dir "${LATENT_DIR}/direct_seed${seed}" \
      --gpu "${GPU}"

  run_step "ga_llm_initialized_seed${seed}" \
    "${PY}" "${GA}" \
      "${COMMON_GA_ARGS[@]}" \
      --init_mode llm \
      --llm_latent_path "${LATENT_DIR}/direct_seed${seed}/llm_init_latent.npy" \
      --version "llm_initialized_seed${seed}" \
      --seed "${seed}"
done

for seed in 42 43 44; do
  run_step "llm_iterative_generate_seed${seed}" \
    "${PY}" "${GEN}" \
      --mode iterative \
      --model gpt-5.4-mini \
      --seed "${seed}" \
      --generations 20 \
      --n_candidates_per_call 30 \
      --target_total 500 \
      --temperature 0.7 \
      --max_tokens 2200 \
      --out_dir "${SMILES_DIR}" \
      --prefix zinc_logp_llm

  iterative_smi="$(latest_smi iterative "${seed}")"

  run_step "llm_iterative_extract_latent_seed${seed}" \
    "${PY}" "${EXTRACT}" \
      --ckpt "${ZINC_CKPT}" \
      --input_path "${iterative_smi}" \
      --out_dir "${LATENT_DIR}/iterative_seed${seed}" \
      --gpu "${GPU}"

  run_step "ga_iterative_llm_guided_seed${seed}" \
    "${PY}" "${GA}" \
      "${COMMON_GA_ARGS[@]}" \
      --init_mode llm \
      --llm_latent_path "${LATENT_DIR}/iterative_seed${seed}/llm_init_latent.npy" \
      --version "iterative_llm_guided_seed${seed}" \
      --seed "${seed}"
done

run_step "summarize_llm_transfer" "${PY}" "${SUMMARIZE}" --base_dir "${BASE}"

echo "DONE all ZINC logP LLM transfer steps"
