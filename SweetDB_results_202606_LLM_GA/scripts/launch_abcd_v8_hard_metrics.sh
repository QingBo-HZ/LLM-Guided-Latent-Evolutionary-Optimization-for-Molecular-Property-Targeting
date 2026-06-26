#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA"
OUT="${ROOT}/sweet_ga_results_0622_v8_hard_metrics"
mkdir -p "${OUT}"
cd "${ROOT}/train_1"

nohup ./run_abcd_v8_hard_metrics.sh \
  > "${OUT}/full_run_driver.log" \
  2>&1 \
  < /dev/null &

pid=$!
echo "${pid}" > "${OUT}/full_run.pid"
echo "${pid}"
