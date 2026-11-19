#!/usr/bin/env bash
set -euo pipefail

# This orchestrator is intentionally resumable.  Run training stages in
# separate nohup jobs if desired; evaluation must start after checkpoints exist.
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${PVT_B2_PATH:?Set PVT_B2_PATH}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/reviewer_results}"

echo "[1/8] data audit"
RESULT_ROOT="${RESULT_ROOT}" bash tools/reviewer/run_data_audit.sh
echo "[2/8] train five DG comparisons"
STAGE=dg5 bash tools/reviewer/run_review_training.sh
echo "[3/8] train three original baselines under the fair protocol"
STAGE=baselines3 bash tools/reviewer/run_review_training.sh
echo "[4/8] train six missing factorial ablations"
STAGE=ablation6 bash tools/reviewer/run_review_training.sh
echo "[5/8] region + boundary evaluation and statistics"
RESULT_ROOT="${RESULT_ROOT}/evaluation" STAGE=all bash tools/reviewer/run_boundary_evaluation.sh
echo "[6/8] parameters, FLOPs, latency, VRAM and checkpoint provenance"
RESULT_ROOT="${RESULT_ROOT}" bash tools/reviewer/run_efficiency_and_manifest.sh
echo "[7/8] manuscript figures"
python tools/reviewer/make_review_figures.py \
  --summary-dir "${RESULT_ROOT}/evaluation/summary" --output-dir "${RESULT_ROOT}/figures"
echo "[8/8] archive results"
RESULT_ROOT="${RESULT_ROOT}" bash tools/reviewer/package_review_results.sh
echo "[DONE] ${RESULT_ROOT}"
