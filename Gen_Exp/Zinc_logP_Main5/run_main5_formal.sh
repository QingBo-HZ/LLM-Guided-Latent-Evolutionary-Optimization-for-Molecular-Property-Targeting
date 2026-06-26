#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/sweeteners_evolve"
PY="${PY:-/root/miniconda3/envs/molclr_pyg28/bin/python}"
GPU="${GPU:-1}"

BASE="${ROOT}/Gen_Exp/Zinc_logP_Main5"
RESULTS_DIR="${BASE}/results"
LOG_DIR="${BASE}/logs"

RANDOM_SCRIPT="${BASE}/random_latent_search_decode.py"
GA="${ROOT}/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py"
SUMMARIZE="${BASE}/summarize_main5_formal.py"

ZINC_CKPT="${ROOT}/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt"
PREDICTOR="${ROOT}/Gen_Exp/Zinc_logP_kek/logp_predictor/best_logp_predictor.pt"
LATENT_POOL="${ROOT}/Gen_Exp/Zinc_logP_kek/train/zinc_logp_latent.npy"

LLM_LATENT_BASE="${ROOT}/Gen_Exp/Zinc_logP_LLM/latent"
LLM_BASE="${ROOT}/Gen_Exp/Zinc_logP_LLM"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

COMMON_GA_ARGS=(
  --zinc_psvae_ckpt "${ZINC_CKPT}"
  --predictor_ckpt "${PREDICTOR}"
  --latent_pool "${LATENT_POOL}"
  --output_root "${RESULTS_DIR}"
  --pop_size 100
  --n_gen 100
  --elite_size 20
  --cross_prob 0.35
  --mut_prob 0.06
  --mut_eta 20
  --patience 100
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
  --decode_topk_per_gen 100
  --decode_attempts_per_latent 3
  --gpu "${GPU}"
)

echo "========== Formal ZINC logP main-5 pipeline =========="
echo "GPU=${GPU}"
echo "PY=${PY}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "GA params: pop_size=100, n_gen=100, patience=100"
echo "======================================================"

for seed in 42 43 44; do
  run_step "random_latent_search_seed${seed}" \
    "${PY}" "${RANDOM_SCRIPT}" \
      --zinc_psvae_ckpt "${ZINC_CKPT}" \
      --predictor_ckpt "${PREDICTOR}" \
      --latent_pool "${LATENT_POOL}" \
      --output_root "${RESULTS_DIR}" \
      --version formal \
      --seed "${seed}" \
      --gpu "${GPU}" \
      --n_samples 10000 \
      --batch_size 100 \
      --decode_attempts_per_latent 3

  run_step "zinc_seeded_latent_ga_seed${seed}" \
    "${PY}" "${GA}" \
      "${COMMON_GA_ARGS[@]}" \
      --init_mode train_random \
      --version "zinc_seeded_formal_seed${seed}" \
      --seed "${seed}"

  run_step "llm_initialized_latent_ga_seed${seed}" \
    "${PY}" "${GA}" \
      "${COMMON_GA_ARGS[@]}" \
      --init_mode llm \
      --llm_latent_path "${LLM_LATENT_BASE}/direct_seed${seed}/llm_init_latent.npy" \
      --version "llm_initialized_formal_seed${seed}" \
      --seed "${seed}"

  run_step "iterative_llm_guided_latent_ga_seed${seed}" \
    "${PY}" "${GA}" \
      "${COMMON_GA_ARGS[@]}" \
      --init_mode llm \
      --llm_latent_path "${LLM_LATENT_BASE}/iterative_seed${seed}/llm_init_latent.npy" \
      --version "iterative_llm_guided_formal_seed${seed}" \
      --seed "${seed}"
done

run_step "summarize_main5_formal" \
  "${PY}" "${SUMMARIZE}" \
    --main5_dir "${BASE}" \
    --llm_base_dir "${LLM_BASE}"

echo "DONE formal ZINC logP main-5 pipeline"
