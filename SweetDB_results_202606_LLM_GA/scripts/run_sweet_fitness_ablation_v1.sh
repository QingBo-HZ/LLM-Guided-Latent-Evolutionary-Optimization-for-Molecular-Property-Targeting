#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA
PY=/home/jqb/.conda/envs/brc_vae/bin/python
GA="$ROOT/train_1/sweet_fitness_ablation_ga.py"
OUT=${OUT:-"$ROOT/sweet_fitness_ablation_v1"}
DOCKING=${DOCKING:-"$ROOT/latent_evaluator_data_manifold_v2/docking_surrogate_v1"}
VERSION_PREFIX=${VERSION_PREFIX:-"fitness_ablation_v1"}
INIT_MODE=${INIT_MODE:-"group_a_random"}
LLM_LATENT_PATH=${LLM_LATENT_PATH:-""}
ENFORCE_UNIQUE=${ENFORCE_UNIQUE:-"1"}
SEEDS=${SEEDS:-"2026 2027 2028"}

mkdir -p "$OUT/logs"

MODE_SPECS=${MODE_SPECS:-"direct_regressor:Sweet_Predictor docking_surrogate:Docking_Predictor gated:Gated_Sweet gated_docking:Gated_Docking"}

for seed in $SEEDS; do
  for spec in $MODE_SPECS; do
    mode=${spec%%:*}
    label=${spec##*:}
    version="${VERSION_PREFIX}_${label}"
    log="$OUT/logs/${label}_seed${seed}.log"

    echo "[$(date '+%F %T')] starting ${label}, seed=${seed}" | tee "$log"
    llm_args=()
    if [[ "$INIT_MODE" == "group_c_llm" ]]; then
      if [[ -z "$LLM_LATENT_PATH" ]]; then
        echo "LLM_LATENT_PATH is required for group_c_llm" >&2
        exit 2
      fi
      llm_args+=(--llm_latent_path "$LLM_LATENT_PATH")
    fi
    unique_args=()
    if [[ "$ENFORCE_UNIQUE" == "1" ]]; then
      unique_args+=(
        --enforce_unique_smiles
        --unique_elite_candidates 30
        --unique_refill_attempts 48
        --unique_refill_sigma 0.06
        --unique_refill_from_seed_prob 0.0
        --unique_refill_pool current
        --unique_target_ratio 0.70
        --unique_refill_min_p_sweet 0.0
        --unique_refill_min_logsw -99
        --unique_refill_max_ood_ratio 100
      )
    fi

    "$PY" "$GA" \
      --fitness_mode "$mode" \
      --docking_predictor_dir "$DOCKING" \
      --output_root "$OUT" \
      --init_mode "$INIT_MODE" \
      "${llm_args[@]}" \
      --version "$version" \
      --seed "$seed" \
      --objective constrained_sweetness \
      --p_sweet_threshold 0.55 \
      --logsw_success_threshold 2.0 \
      --success_p_sweet 0.55 \
      --final_min_logsw 2.0 \
      --pop_size 30 \
      --n_gen 12 \
      --elite_size 4 \
      --cross_prob 0.35 \
      --mut_prob 0.10 \
      --mut_sigma 0.20 \
      --patience 100 \
      --tournament_size 3 \
      --seed_pool_size 30 \
      --seed_augment_sigma 0.0 \
      --archive_topk 30 \
      --decode_final \
      --save_population_history \
      "${unique_args[@]}" \
      2>&1 | tee -a "$log"
    echo "[$(date '+%F %T')] finished ${label}, seed=${seed}" | tee -a "$log"
  done
done
