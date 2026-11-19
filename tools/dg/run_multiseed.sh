#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Examples:
#   bash tools/dg/run_multiseed.sh fair_unet fair_ibn fair_mixstyle fair_ccfb_pretrained
#   SEEDS="0 3407 2026" GPU_ID=1 bash tools/dg/run_multiseed.sh fair_ibn

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 3407 2026 42 1234}"
WORK_ROOT="${WORK_ROOT:-./work_dirs_polyp_v2/dg_multiseed}"

if [[ "$#" -gt 0 ]]; then
  CONFIGS=("$@")
else
  CONFIGS=(fair_unet fair_ibn fair_mixstyle fair_ccfb_pretrained)
fi

for seed in ${SEEDS}; do
  for name in "${CONFIGS[@]}"; do
    config="./my_config/dg/${name}.py"
    run_dir="${WORK_ROOT}/seed_${seed}/${name}"
    log_dir="${WORK_ROOT}/logs"
    mkdir -p "${run_dir}" "${log_dir}"
    echo "[DG] training config=${name} seed=${seed} gpu=${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      ./tools/dist_train.sh "${config}" 1 \
      --seed "${seed}" --deterministic --work-dir "${run_dir}" \
      2>&1 | tee "${log_dir}/${name}_seed${seed}.log"
  done
done

