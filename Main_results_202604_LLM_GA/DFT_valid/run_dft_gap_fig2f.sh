#!/usr/bin/env bash
set -e

INPUT_DIR="/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/DFT_valid/DFT_Gaussian_outputs"
OUT_DIR="/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/DFT_valid/DFT_Fig2f_results"

python dft_gap_pipeline_fig2f.py \
  --input_dir "${INPUT_DIR}" \
  --out_dir "${OUT_DIR}" \
  --unit eV \
  --require_normal_termination

echo "DFT gap parsing and Fig. 2f plotting finished."
echo "Results saved to: ${OUT_DIR}"