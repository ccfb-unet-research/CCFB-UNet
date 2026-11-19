#!/usr/bin/env bash
set -euo pipefail
CORE_WORK_ROOT="${CORE_WORK_ROOT:-$PWD/work_dirs_polyp_v2/dg_multiseed}"
REVIEW_WORK_ROOT="${REVIEW_WORK_ROOT:-$PWD/work_dirs_polyp_v2/reviewer_multiseed}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/reviewer_results}"
python tools/reviewer/compute_efficiency.py \
  --project-root "$PWD" --work-root "${CORE_WORK_ROOT}" \
  --config-dir my_config/dg --output-dir "${RESULT_ROOT}/efficiency/core" \
  --models fair_unet fair_ibn fair_mixstyle fair_pvt_control_scratch fair_pvt_control_pretrained fair_ccfb_scratch fair_ccfb_pretrained \
  --seed 0 --warmup 50 --repeats 200
python tools/reviewer/compute_efficiency.py \
  --project-root "$PWD" --work-root "${REVIEW_WORK_ROOT}" \
  --config-dir my_config/reviewer --output-dir "${RESULT_ROOT}/efficiency/reviewer" \
  --seed 0 --warmup 50 --repeats 200
python tools/reviewer/checkpoint_manifest.py \
  --work-root "${CORE_WORK_ROOT}" --work-root "${REVIEW_WORK_ROOT}" \
  --output-dir "${RESULT_ROOT}/provenance"

