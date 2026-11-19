#!/usr/bin/env bash
set -euo pipefail

# Optional independent official-code audit.  It is not pooled with the matched
# PVT adaptation table because Rein/DPMFormer use much larger VFM pretraining.
ROOT="${1:-official_dg_repos}"
mkdir -p "${ROOT}"
clone_one() {
  local name="$1" url="$2"
  if [[ ! -d "${ROOT}/${name}/.git" ]]; then git clone --depth 1 "${url}" "${ROOT}/${name}"; fi
  git -C "${ROOT}/${name}" rev-parse HEAD
}
{
  echo -e "method\turl\tcommit"
  for item in \
    "RobustNet|https://github.com/sunghac/RobustNet.git" \
    "SAN-SAW|https://github.com/leolyj/SAN-SAW.git" \
    "BlindNet|https://github.com/root0yang/BlindNet.git" \
    "Rein|https://github.com/w1oves/Rein.git" \
    "DPMFormer|https://github.com/jone1222/DPMFormer.git"; do
      name="${item%%|*}"; url="${item#*|}"; commit="$(clone_one "${name}" "${url}" | tail -n 1)"
      echo -e "${name}\t${url}\t${commit}"
  done
} | tee "${ROOT}/OFFICIAL_COMMITS.tsv"

