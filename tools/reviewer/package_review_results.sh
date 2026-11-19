#!/usr/bin/env bash
set -euo pipefail
RESULT_ROOT="${RESULT_ROOT:-$PWD/reviewer_results}"
ARCHIVE="${ARCHIVE:-$PWD/CCFB_UNet_reviewer_results_$(date +%Y%m%d_%H%M%S).tar.gz}"
[[ -d "${RESULT_ROOT}" ]] || { echo "Missing ${RESULT_ROOT}" >&2; exit 2; }
tar -czf "${ARCHIVE}" \
  "${RESULT_ROOT#"$PWD"/}" \
  work_dirs_polyp_v2/reviewer_multiseed/logs \
  2>/dev/null || tar -czf "${ARCHIVE}" "${RESULT_ROOT#"$PWD"/}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
echo "[PACKAGE] ${ARCHIVE}"
echo "[PACKAGE] ${ARCHIVE}.sha256"
