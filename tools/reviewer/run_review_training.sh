#!/usr/bin/env bash
set -euo pipefail

# Usage from repository root:
# STAGE=dg5|baselines3|ablation6|all bash tools/reviewer/run_review_training.sh
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 42 1234 2026 3407}"
STAGE="${STAGE:-all}"
WORK_ROOT="${WORK_ROOT:-$PWD/work_dirs_polyp_v2/reviewer_multiseed}"

DG5=(fair_robustnet_pvt_adapted fair_sansaw_pvt_adapted fair_blindnet_pvt_adapted fair_rein_pvt_adapted fair_dpmformer_pvt_adapted)
BASELINES3=(fair_ddcnet_scratch fair_resunetpp_scratch fair_ultraseg_scratch)
ABLATION6=(ablation_fd_pretrained ablation_mgd_pretrained ablation_gls_pretrained ablation_fd_mgd_pretrained ablation_fd_gls_pretrained ablation_mgd_gls_pretrained)
case "${STAGE}" in
  dg5) CONFIGS=("${DG5[@]}") ;;
  baselines3) CONFIGS=("${BASELINES3[@]}") ;;
  ablation6) CONFIGS=("${ABLATION6[@]}") ;;
  all) CONFIGS=("${DG5[@]}" "${BASELINES3[@]}" "${ABLATION6[@]}") ;;
  *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac

mkdir -p "${WORK_ROOT}/logs"
for seed in ${SEEDS}; do
  for name in "${CONFIGS[@]}"; do
    run_dir="${WORK_ROOT}/seed_${seed}/${name}"
    checkpoint="${run_dir}/iter_14000.pth"
    if [[ -f "${checkpoint}" ]]; then
      echo "[REVIEW] skip completed ${name} seed=${seed}"
      continue
    fi
    mkdir -p "${run_dir}"
    echo "[REVIEW] train ${name} seed=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" ./tools/dist_train.sh \
      "./my_config/reviewer/${name}.py" 1 \
      --seed "${seed}" --deterministic --work-dir "${run_dir}" \
      2>&1 | tee "${WORK_ROOT}/logs/${name}_seed${seed}.log"
  done
done

