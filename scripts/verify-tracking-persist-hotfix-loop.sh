#!/usr/bin/env bash
# R5 — Vérification post-hotfix à T+0, T+5, T+10, T+15 min.
#
# Usage (sur prod, après hotfix) :
#   bash scripts/verify-tracking-persist-hotfix-loop.sh
#   bash scripts/verify-tracking-persist-hotfix-loop.sh --once   # une seule passe
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TS_FILE="/tmp/atmr-hotfix-persist-ts.txt"
VERIFY="${ROOT}/scripts/verify-tracking-persist-hotfix.sh"
PASSES=(0 300 300)
ONCE=0

if [[ "${1:-}" == "--once" ]]; then
  ONCE=1
fi

if [[ ! -f "${TS_FILE}" ]]; then
  echo "[WARN] ${TS_FILE} absent — utiliser HOTFIX_TS manuellement ou relancer le hotfix"
fi

run_pass() {
  local n="$1"
  echo ""
  echo "========== Passe ${n}/4 ($(date -u '+%Y-%m-%d %H:%M:%S UTC')) =========="
  if [[ -f "${TS_FILE}" ]]; then
    export HOTFIX_TS
    HOTFIX_TS="$(cat "${TS_FILE}")"
  fi
  bash "${VERIFY}"
}

run_pass 1
if [[ "${ONCE}" == "1" ]]; then
  exit 0
fi

for i in "${!PASSES[@]}"; do
  wait_s="${PASSES[$i]}"
  if [[ "${wait_s}" -gt 0 ]]; then
    echo ""
    echo "Attente ${wait_s}s avant passe $((i + 2))/4..."
    sleep "${wait_s}"
  fi
  run_pass "$((i + 2))"
done

echo ""
echo "Boucle de vérification terminée. Critères : fresh_10m>=1, trip_tracking_post_hotfix>0, PERSIST=true dans le conteneur."
