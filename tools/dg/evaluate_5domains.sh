#!/usr/bin/env bash
set -euo pipefail

# Evaluates the fixed final checkpoint on the five established test domains.
# This deliberately avoids selecting a checkpoint on the Kvasir target set.
# The training layout produced by run_multiseed.sh is assumed.

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 3407 2026 42 1234}"
WORK_ROOT="${WORK_ROOT:-./work_dirs_polyp_v2/dg_multiseed}"
RESULT_ROOT="${RESULT_ROOT:-./dg_results}"
: "${DATA_ROOT:?Set DATA_ROOT to the polyp_datasets directory}"

if [[ "$#" -gt 0 ]]; then
  CONFIGS=("$@")
else
  CONFIGS=(fair_unet fair_ibn fair_mixstyle fair_ccfb_pretrained)
fi

DATASETS=(Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS-Larib)

for seed in ${SEEDS}; do
  for name in "${CONFIGS[@]}"; do
    checkpoint="${WORK_ROOT}/seed_${seed}/${name}/iter_14000.pth"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "[DG] missing checkpoint, skip: ${checkpoint}" >&2
      continue
    fi
    for dataset in "${DATASETS[@]}"; do
      out_dir="${RESULT_ROOT}/seed_${seed}/${name}/${dataset}"
      mkdir -p "${out_dir}"
      echo "[DG] evaluating model=${name} seed=${seed} dataset=${dataset}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/test.py \
        --work_dirs "${WORK_ROOT}/seed_${seed}" \
        --config "${name}" \
        --checkpoint "iter_14000.pth" \
        --data_root "${DATA_ROOT}/test/${dataset}/images" \
        --work-dir "${out_dir}" \
        --eval mIoU mFscore mDice
    done
  done
done

python tools/dg/summarize_multiseed.py \
  --results-root "${RESULT_ROOT}" \
  --reference fair_ccfb_pretrained \
  --output-dir "${RESULT_ROOT}/summary"

