#!/usr/bin/env bash
set -euo pipefail

# One-command paper result pipeline. It does not train or overwrite weights.
# It evaluates fixed iter_14000 checkpoints, measures efficiency, creates
# qualitative figures, produces tables/plots, and archives only paper outputs.

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 3407 2026 42 1234}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/work_dirs_polyp_v2/dg_multiseed}"
: "${DATA_ROOT:?Set DATA_ROOT to the polyp_datasets directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/paper_results}"
: "${PVT_B2_PATH:?Set PVT_B2_PATH to pvt_v2_b2.pth}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_EFFICIENCY="${RUN_EFFICIENCY:-1}"
RUN_QUALITATIVE="${RUN_QUALITATIVE:-1}"
QUAL_SEED="${QUAL_SEED:-0}"
QUAL_SAMPLES="${QUAL_SAMPLES:-5}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/tables" \
  "${OUTPUT_ROOT}/figures" "${OUTPUT_ROOT}/efficiency" \
  "${OUTPUT_ROOT}/qualitative"

checkpoint_count=$(find "${WORK_ROOT}" -type f -name 'iter_14000.pth' | wc -l)
echo "[PAPER] checkpoint count=${checkpoint_count} (expected 35)"
if [[ "${checkpoint_count}" -ne 35 ]]; then
  echo "[PAPER][ERROR] training is incomplete; refusing partial paper analysis" >&2
  exit 2
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  env GPU_ID="${GPU_ID}" SEEDS="${SEEDS}" WORK_ROOT="${WORK_ROOT}" \
    DATA_ROOT="${DATA_ROOT}" RESULT_ROOT="${OUTPUT_ROOT}/evaluation" \
    bash tools/paper_analysis/evaluate_all_resumable.sh \
    2>&1 | tee "${OUTPUT_ROOT}/logs/evaluation.log"
fi

if [[ "${RUN_EFFICIENCY}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PVT_B2_PATH="${PVT_B2_PATH}" \
    python tools/paper_analysis/compute_efficiency.py \
      --project-root "${PROJECT_ROOT}" --work-root "${WORK_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/efficiency" --seed 0 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/efficiency.log"
fi

if [[ "${RUN_QUALITATIVE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PVT_B2_PATH="${PVT_B2_PATH}" \
    python tools/paper_analysis/generate_qualitative.py \
      --project-root "${PROJECT_ROOT}" --work-root "${WORK_ROOT}" \
      --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_ROOT}/qualitative" \
      --seed "${QUAL_SEED}" --samples "${QUAL_SAMPLES}" --device cuda:0 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/qualitative.log"
fi

python tools/paper_analysis/make_tables_and_figures.py \
  --results-root "${OUTPUT_ROOT}/evaluation" \
  --efficiency-csv "${OUTPUT_ROOT}/efficiency/model_efficiency.csv" \
  --output-dir "${OUTPUT_ROOT}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/tables_and_figures.log"

eval_count=$(find "${OUTPUT_ROOT}/evaluation" -type f -name 'eval_*.json' | wc -l)
figure_count=$(find "${OUTPUT_ROOT}/figures" -type f | wc -l)
cat > "${OUTPUT_ROOT}/RESULT_MANIFEST.txt" <<EOF
Paper result bundle
===================
Generated: $(date --iso-8601=seconds)
Checkpoints found: ${checkpoint_count}/35
Evaluation JSON: ${eval_count}/175
Figure files: ${figure_count}
Protocol: fixed iter_14000 checkpoints; five seeds; five-domain equal-weight macro
Qualitative selection: lowest-Dice cases ranked by final fair_unet checkpoint
EOF

archive="${PROJECT_ROOT}/paper_results_bundle.tar.gz"
tar -czf "${archive}" -C "$(dirname "${OUTPUT_ROOT}")" "$(basename "${OUTPUT_ROOT}")"
echo "[PAPER] complete"
echo "[PAPER] result folder: ${OUTPUT_ROOT}"
echo "[PAPER] download archive: ${archive}"

