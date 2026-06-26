#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA"
PY="/home/jqb/.conda/envs/brc_vae/bin/python"
GA="${ROOT}/train_1/sweet_gated_latent_ga_4groups.py"
AUDIT="${ROOT}/train_1/build_abcd_docking_audit.py"
LATENT_DIR="${ROOT}/latent_evaluator_data_manifold_v2"
PREDICTOR_DIR="${LATENT_DIR}/gated_predictor_scaffold_ensemble_highsweet_v4"
OOD_DIR="${ROOT}/sweet_like_dataset_out/ood_background_manifold_v2_train_only"

OUT="${OUT:-${ROOT}/sweet_ga_results_0622_v8_hard_metrics}"
AUDIT_OUT="${AUDIT_OUT:-${OUT}/docking_audit}"
LLM_LATENT="${ROOT}/llm_sweet_seed_smiles/gpt55_gemini31_fair_v1_latent/llm_gpt55_gemini31_latent.npy"
REPLAY_DIR="${ROOT}/llm_reflection_replay_0622_v8"
TAG="ABCD_v8_hard_metrics_0622"
SEEDS="${SEEDS:-2026 2027 2028 2029 2030}"
RUN_GROUPS="${RUN_GROUPS:-A B C D}"

mkdir -p "$OUT/logs" "$REPLAY_DIR" "$AUDIT_OUT"

# Reuse generated LLM reflection candidates, but trigger and audit them under the new protocol.
cp -f "${ROOT}/sweet_ga_results_corrected/group_d_llm_iterative_ABCD_mainfig_quality_v3_20260609_D_seed2026/llm_online_injection/gen_003_latent.npy" "$REPLAY_DIR/llm_gen_003.npy"
cp -f "${ROOT}/sweet_ga_results_corrected/group_d_llm_iterative_ABCD_mainfig_quality_v3_20260609_D_seed2026/llm_online_injection/gen_006_latent.npy" "$REPLAY_DIR/llm_gen_006.npy"
cp -f "${ROOT}/sweet_ga_results_corrected/group_d_llm_iterative_ABCD_mainfig_quality_v3_20260609_D_seed2026/llm_online_injection/gen_009_latent.npy" "$REPLAY_DIR/llm_gen_009.npy"

common=(
  --latent_dir "${LATENT_DIR}"
  --predictor_dir "${PREDICTOR_DIR}"
  --ood_dir "${OOD_DIR}"
  --output_root "${OUT}"
  --objective constrained_sweetness
  --p_sweet_threshold 0.80
  --logsw_success_threshold 2.60
  --success_p_sweet 0.80
  --success_logsw 2.60
  --final_min_logsw 2.60
  --lambda_reg_uncertainty 0.25
  --pop_size 30
  --n_gen 12
  --elite_size 3
  --cross_prob 0.38
  --mut_prob 0.14
  --mut_sigma 0.22
  --patience 100
  --seed_augment_sigma 0.0
  --seed_pool_size 30
  --archive_topk 8
  --decode_archive
  --decode_final
  --save_population_history
)

run_group() {
  local seed="$1"
  local label="$2"
  local mode="$3"
  shift 3
  local version="${TAG}_${label}_s${seed}"
  local prefix=""
  case "$mode" in
    group_a_random) prefix="group_a_random" ;;
    group_b_dataset) prefix="group_b_dataset" ;;
    group_c_llm) prefix="group_c_llm" ;;
    group_d_llm_iterative) prefix="group_d_llm_iterative" ;;
  esac
  local outdir="${OUT}/${prefix}_${version}_seed${seed}"
  local log="${OUT}/logs/${label}_seed${seed}.log"
  if [[ -f "${outdir}/summary.json" ]]; then
    echo "[$(date '+%F %T')] skip existing ${label}, seed=${seed}" | tee "$log"
    return
  fi
  echo "[$(date '+%F %T')] start ${label}, seed=${seed}" | tee "$log"
  PYTHONUNBUFFERED=1 "${PY}" "${GA}" \
    --init_mode "${mode}" \
    --version "${version}" \
    --seed "${seed}" \
    "${common[@]}" \
    "$@" \
    2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] finish ${label}, seed=${seed}" | tee -a "$log"
}

for seed in ${SEEDS}; do
  [[ " ${RUN_GROUPS} " == *" A "* ]] && run_group "${seed}" A group_a_random
  [[ " ${RUN_GROUPS} " == *" B "* ]] && run_group "${seed}" B group_b_dataset
  [[ " ${RUN_GROUPS} " == *" C "* ]] && run_group "${seed}" C group_c_llm --llm_latent_path "${LLM_LATENT}"
  [[ " ${RUN_GROUPS} " == *" D "* ]] && run_group "${seed}" D group_d_llm_iterative \
    --llm_latent_path "${LLM_LATENT}" \
    --llm_iterative_latent_dir "${REPLAY_DIR}" \
    --llm_feedback_interval 3 \
    --llm_feedback_topn 12 \
    --llm_inject_size 4 \
    --llm_judge_keep 12 \
    --llm_stagnation_window 3 \
    --llm_stagnation_score_delta 999.0 \
    --llm_stagnation_no_improve 99 \
    --llm_inject_min_score_gain -0.08 \
    --llm_trigger_unique_ratio 1.01 \
    --llm_local_refine_steps 6 \
    --llm_local_refine_samples 96 \
    --llm_local_refine_sigma 0.16 \
    --llm_local_refine_sigma_decay 0.82 \
    --llm_local_refine_min_novelty 0.10 \
    --llm_local_refine_novelty_weight 0.04 \
    --llm_inject_min_novelty 0.10 \
    --llm_inject_min_population_quantile 0.45
done

"${PY}" "${AUDIT}" \
  --result_root "${OUT}" \
  --out_dir "${AUDIT_OUT}" \
  --top_k_generation 5 \
  --top_k_final 10 \
  --p_threshold 0.80 \
  --logsw_threshold 2.60 \
  --ood_threshold 7.225 \
  --vina_threshold -6.8

echo "Docking audit inputs are in: ${AUDIT_OUT}"
