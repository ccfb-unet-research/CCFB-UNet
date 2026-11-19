#!/usr/bin/env bash
set -euo pipefail

# Resumable evaluation of all seven fair-comparison models on five domains.
# Existing eval_*.json files are kept and skipped, so an interrupted run can
# be restarted safely without repeating completed evaluations.

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 3407 2026 42 1234}"
WORK_ROOT="${WORK_ROOT:-$PWD/work_dirs_polyp_v2/dg_multiseed}"
: "${DATA_ROOT:?Set DATA_ROOT to the polyp_datasets directory}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/paper_results/evaluation}"

MODELS=(
  fair_unet
  fair_ibn
  fair_mixstyle
  fair_pvt_control_scratch
  fair_pvt_control_pretrained
  fair_ccfb_scratch
  fair_ccfb_pretrained
)
DATASETS=(Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS-Larib)

mkdir -p "${RESULT_ROOT}"

for seed in ${SEEDS}; do
  for model in "${MODELS[@]}"; do
    checkpoint="${WORK_ROOT}/seed_${seed}/${model}/iter_14000.pth"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "[PAPER][ERROR] missing checkpoint: ${checkpoint}" >&2
      exit 2
    fi
    for dataset in "${DATASETS[@]}"; do
      out_dir="${RESULT_ROOT}/seed_${seed}/${model}/${dataset}"
      mkdir -p "${out_dir}"
      if compgen -G "${out_dir}/eval_*.json" > /dev/null; then
        echo "[PAPER] skip completed model=${model} seed=${seed} dataset=${dataset}"
        continue
      fi
      echo "[PAPER] evaluate model=${model} seed=${seed} dataset=${dataset}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/test.py \
        --work_dirs "${WORK_ROOT}/seed_${seed}" \
        --config "${model}" \
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

count=$(find "${RESULT_ROOT}" -type f -name 'eval_*.json' | wc -l)
echo "[PAPER] evaluation JSON count=${count} (expected 175)"
if [[ "${count}" -ne 175 ]]; then
  echo "[PAPER][ERROR] evaluation is incomplete" >&2
  exit 3
fi

