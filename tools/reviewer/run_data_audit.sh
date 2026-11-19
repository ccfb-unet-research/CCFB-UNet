#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to polyp_datasets}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/reviewer_results}"
python tools/reviewer/audit_dataset.py \
  --data-root "${DATA_ROOT}" --output-dir "${RESULT_ROOT}/data_audit" \
  --phash-threshold 4

