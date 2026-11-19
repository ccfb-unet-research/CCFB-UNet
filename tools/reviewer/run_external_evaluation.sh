#!/usr/bin/env bash
set -euo pipefail

# Copy external_datasets.example.tsv to external_datasets.tsv and edit paths.
MANIFEST="${MANIFEST:-tools/reviewer/external_datasets.tsv}"
MODEL="${MODEL:-fair_ccfb_pretrained}"
SEED="${SEED:-0}"
CORE_WORK_ROOT="${CORE_WORK_ROOT:-$PWD/work_dirs_polyp_v2/dg_multiseed}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/reviewer_results/external_validation}"
[[ -f "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}; see external_datasets.example.tsv" >&2; exit 2; }
while IFS=$'\t' read -r dataset image_dir label_dir; do
  [[ "${dataset}" == dataset || -z "${dataset}" ]] && continue
  python tools/reviewer/evaluate_checkpoint.py \
    --config "my_config/dg/${MODEL}.py" \
    --checkpoint "${CORE_WORK_ROOT}/seed_${SEED}/${MODEL}/iter_14000.pth" \
    --image-dir "${image_dir}" --label-dir "${label_dir}" \
    --output-dir "${RESULT_ROOT}/seed_${SEED}/${MODEL}/${dataset}" --device cuda:0 --save-masks
done < "${MANIFEST}"
