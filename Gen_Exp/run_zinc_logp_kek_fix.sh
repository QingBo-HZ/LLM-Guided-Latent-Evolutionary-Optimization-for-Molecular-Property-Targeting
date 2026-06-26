#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/sweeteners_evolve
PY=/root/miniconda3/envs/molclr_pyg28/bin/python
CKPT=$ROOT/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt
DATA=$ROOT/QM9_test/PS-VAE/data/my_zinc
OUT=$ROOT/Gen_Exp/Zinc_logP_kek
LOG=$OUT/logs
GPU=${GPU:-0}

mkdir -p "$OUT" "$LOG"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$GPU

echo "========== ZINC logP kekulize repair pipeline =========="
echo "start_time=$(date '+%F %T')"
echo "ROOT=$ROOT"
echo "GPU=$GPU"
echo "CKPT=$CKPT"
echo "OUT=$OUT"
echo "========================================================"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name"
  "$@" 2>&1 | tee "$LOG/${name}.log"
  echo "[$(date '+%F %T')] DONE  $name"
}

run_step encode_train_kek \
  "$PY" "$ROOT/Gen_Exp/01_encode_zinc_logp_latent.py" \
    --ckpt "$CKPT" \
    --input "$DATA/train/train.txt" \
    --out_dir "$OUT/train" \
    --gpu 0 \
    --kekulize

run_step encode_valid_kek \
  "$PY" "$ROOT/Gen_Exp/01_encode_zinc_logp_latent.py" \
    --ckpt "$CKPT" \
    --input "$DATA/valid/valid.txt" \
    --out_dir "$OUT/valid" \
    --gpu 0 \
    --kekulize

run_step encode_test_kek \
  "$PY" "$ROOT/Gen_Exp/01_encode_zinc_logp_latent.py" \
    --ckpt "$CKPT" \
    --input "$DATA/test/test.txt" \
    --out_dir "$OUT/test" \
    --gpu 0 \
    --kekulize

run_step audit_train_kek_latent \
  "$PY" "$ROOT/Gen_Exp/08_decode_latent_pool_audit.py" \
    --latent "$OUT/train/zinc_logp_latent.npy" \
    --out_dir "$OUT/psvae_audit/train_latent_n200_a3" \
    --n_samples 200 \
    --attempts 3 \
    --gpu 0 \
    --seed 42

run_step train_logp_predictor_kek \
  "$PY" "$ROOT/Gen_Exp/02_train_logp_predictor.py" \
    --train_latent "$OUT/train/zinc_logp_latent.npy" \
    --train_meta "$OUT/train/zinc_logp_meta.csv" \
    --valid_latent "$OUT/valid/zinc_logp_latent.npy" \
    --valid_meta "$OUT/valid/zinc_logp_meta.csv" \
    --test_latent "$OUT/test/zinc_logp_latent.npy" \
    --test_meta "$OUT/test/zinc_logp_meta.csv" \
    --out_dir "$OUT/logp_predictor" \
    --gpu 0 \
    --hidden_dim 512 \
    --dropout 0.05 \
    --epochs 400 \
    --batch_size 512 \
    --lr 8e-4 \
    --weight_decay 1e-6 \
    --patience 50

for SEED in 42 43 44; do
  run_step "ga_rdkit_kek_seed${SEED}" \
    "$PY" "$ROOT/Gen_Exp/03b_optimize_logp_latent_ga_decode_aware.py" \
      --zinc_psvae_ckpt "$CKPT" \
      --predictor_ckpt "$OUT/logp_predictor/best_logp_predictor.pt" \
      --latent_pool "$OUT/train/zinc_logp_latent.npy" \
      --init_mode train_random \
      --pop_size 80 \
      --n_gen 12 \
      --elite_size 16 \
      --selection_metric rdkit_hybrid \
      --rdkit_selection_weight 0.85 \
      --decode_topk_per_gen 30 \
      --decode_attempts_per_latent 3 \
      --patience 8 \
      --immigrant_ratio 0.30 \
      --manifold_anchor_size 5000 \
      --manifold_blend 0.35 \
      --target_logp 3.0 \
      --pred_target_logp 3.0 \
      --score_sigma 0.7 \
      --success_low 2.5 \
      --success_high 3.5 \
      --max_atom_num 80 \
      --add_edge_th 0.45 \
      --temperature 0.30 \
      --output_root "$OUT/results" \
      --version "decode_aware_v5_kek_rdkit_seed${SEED}" \
      --gpu 0 \
      --seed "$SEED"
done

run_step summarize_kek_3seed \
  "$PY" "$ROOT/Gen_Exp/summarize_zinc_logp_kek_results.py" \
    --results_root "$OUT/results" \
    --versions train_random_decode_aware_v5_kek_rdkit_seed42 train_random_decode_aware_v5_kek_rdkit_seed43 train_random_decode_aware_v5_kek_rdkit_seed44 \
    --out_prefix zinc_logp_decode_aware_v5_kek_rdkit_3seed

echo "end_time=$(date '+%F %T')"
echo "DONE all steps"
