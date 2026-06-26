#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA
PY=/home/jqb/.conda/envs/brc_vae/bin/python
GA="$ROOT/train_1/sweet_fitness_ablation_ga.py"

OUT=${OUT:-/home/jqb/sweet_fitness_5seeds_panels/runs_v6b_gold_no_fusion_fast}
DOCKING=${DOCKING:-$ROOT/latent_evaluator_data_manifold_v2/docking_surrogate_raw_vina_v1}
VERSION_PREFIX=${VERSION_PREFIX:-fitness_gold_v6_no_fusion}
INIT_MODE=${INIT_MODE:-group_c_llm}
LLM_LATENT_PATH=${LLM_LATENT_PATH:-$ROOT/llm_sweet_seed_smiles/gpt55_gemini31_fair_v1_top30_decontaminated/llm_gpt55_gemini31_top30_decontaminated_latent.npy}
SEEDS=${SEEDS:-"2026 2027 2028 2029 2030"}

# No fused/combo modes here. These are the four original fitness choices.
MODE_SPECS=${MODE_SPECS:-"direct_regressor:Sweet_Predictor docking_surrogate:Docking_Predictor gated:Gated_Sweet gated_docking:Gated_Docking"}
ENFORCE_UNIQUE=${ENFORCE_UNIQUE:-0}

mkdir -p "$OUT/logs"

for seed in $SEEDS; do
  for spec in $MODE_SPECS; do
    mode=${spec%%:*}
    label=${spec##*:}
    version="${VERSION_PREFIX}_${label}"
    log="$OUT/logs/${label}_seed${seed}.log"

    if [[ -f "$OUT/group_c_llm_${version}_seed${seed}/summary.json" ]]; then
      echo "[$(date '+%F %T')] skip existing ${label}, seed=${seed}" | tee "$log"
      continue
    fi

    echo "[$(date '+%F %T')] starting ${label}, seed=${seed}, mode=${mode}" | tee "$log"

    unique_args=()
    if [[ "$ENFORCE_UNIQUE" == "1" ]]; then
      unique_args+=(
        --enforce_unique_smiles
        --unique_elite_candidates 30
        --unique_refill_attempts 16
        --unique_refill_sigma 0.08
        --unique_refill_from_seed_prob 0.0
        --unique_refill_pool current
        --unique_target_ratio 0.60
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
      --llm_latent_path "$LLM_LATENT_PATH" \
      --version "$version" \
      --seed "$seed" \
      --objective constrained_sweetness \
      --p_sweet_threshold 0.90 \
      --logsw_success_threshold 2.80 \
      --success_p_sweet 0.90 \
      --success_logsw 2.80 \
      --final_min_logsw 2.80 \
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

"$PY" "$ROOT/train_1/summarize_sweet_fitness_ablation.py" --root "$OUT"
