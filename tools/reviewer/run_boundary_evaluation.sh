#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 42 1234 2026 3407}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to polyp_datasets}"
CORE_WORK_ROOT="${CORE_WORK_ROOT:-$PWD/work_dirs_polyp_v2/dg_multiseed}"
REVIEW_WORK_ROOT="${REVIEW_WORK_ROOT:-$PWD/work_dirs_polyp_v2/reviewer_multiseed}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/reviewer_results/evaluation}"
STAGE="${STAGE:-all}"
SAVE_MASKS="${SAVE_MASKS:-0}"

CORE=(fair_unet fair_ibn fair_mixstyle fair_pvt_control_scratch fair_pvt_control_pretrained fair_ccfb_scratch fair_ccfb_pretrained)
REVIEW=(fair_robustnet_pvt_adapted fair_sansaw_pvt_adapted fair_blindnet_pvt_adapted fair_rein_pvt_adapted fair_dpmformer_pvt_adapted fair_ddcnet_scratch fair_resunetpp_scratch fair_ultraseg_scratch ablation_fd_pretrained ablation_mgd_pretrained ablation_gls_pretrained ablation_fd_mgd_pretrained ablation_fd_gls_pretrained ablation_mgd_gls_pretrained)
DATASETS=(Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS-Larib)

evaluate_group() {
  local config_dir="$1" work_root="$2"; shift 2
  local models=("$@")
  for seed in ${SEEDS}; do
    for model in "${models[@]}"; do
      ckpt="${work_root}/seed_${seed}/${model}/iter_14000.pth"
      [[ -f "${ckpt}" ]] || { echo "[REVIEW][WARN] missing ${ckpt}"; continue; }
      for dataset in "${DATASETS[@]}"; do
        out="${RESULT_ROOT}/seed_${seed}/${model}/${dataset}"
        [[ -f "${out}/summary.json" ]] && { echo "[REVIEW] skip ${model}/${seed}/${dataset}"; continue; }
        extra_args=()
        if [[ "${SAVE_MASKS}" == 1 && "${seed}" == 0 ]]; then extra_args+=(--save-masks); fi
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/reviewer/evaluate_checkpoint.py \
          --config "${config_dir}/${model}.py" --checkpoint "${ckpt}" \
          --image-dir "${DATA_ROOT}/test/${dataset}/images" \
          --label-dir "${DATA_ROOT}/test/${dataset}/labels" \
          --output-dir "${out}" --device cuda:0 "${extra_args[@]}"
      done
    done
  done
}

mkdir -p "${RESULT_ROOT}"
if [[ "${STAGE}" == core || "${STAGE}" == all ]]; then
  evaluate_group ./my_config/dg "${CORE_WORK_ROOT}" "${CORE[@]}"
fi
if [[ "${STAGE}" == reviewer || "${STAGE}" == all ]]; then
  evaluate_group ./my_config/reviewer "${REVIEW_WORK_ROOT}" "${REVIEW[@]}"
fi
python tools/reviewer/aggregate_review_results.py \
  --results-root "${RESULT_ROOT}" --output-dir "${RESULT_ROOT}/summary" \
  --reference fair_ccfb_pretrained --bootstrap 2000

