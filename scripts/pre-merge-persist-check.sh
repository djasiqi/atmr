#!/usr/bin/env bash
# R7 / R9 — Vérifications avant merge PR1 (fragment + prod si SSH disponible).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRAGMENT="${ROOT}/scripts/env.production.defaults.fragment"
OK=1

echo "=== Pre-merge persist check (R7/R9) ==="

echo "--- R7 Fragment env ---"
if grep -q '^TRACKING_INGEST_PERSIST_ENABLED=true' "${FRAGMENT}"; then
  echo "[OK] TRACKING_INGEST_PERSIST_ENABLED=true dans le fragment"
else
  echo "[FAIL] TRACKING_INGEST_PERSIST_ENABLED=true absent du fragment"
  OK=0
fi

if grep -q 'TRACKING_INGEST_ALLOW_REPUBLISH_ONLY' "${FRAGMENT}"; then
  echo "[OK] ALLOW_REPUBLISH_ONLY documenté (commenté) dans le fragment"
else
  echo "[WARN] ALLOW_REPUBLISH_ONLY non mentionné dans le fragment"
fi

if [[ -f "${ROOT}/.local.deploy.env" ]]; then
  # shellcheck source=/dev/null
  set -a
  source "${ROOT}/.local.deploy.env"
  set +a
fi

if [[ -n "${SERVER_HOST:-}" ]]; then
  echo "--- R9 Prod .env.production ---"
  REMOTE="${SERVER_USER:-deploy}@${SERVER_HOST}"
  if ssh "${REMOTE}" 'grep -q "^TRACKING_INGEST_PERSIST_ENABLED=true" /srv/atmr/.env.production && grep -q "^TRACKING_INGEST_PERSIST_ENABLED=true" /srv/atmr/.env'; then
    echo "[OK] PERSIST=true dans .env.production ET .env sur prod"
  else
    echo "[FAIL] PERSIST=true manquant sur prod (.env ou .env.production)"
    OK=0
  fi
else
  echo "[SKIP] R9 prod — SERVER_HOST non défini (.local.deploy.env)"
fi

if [[ "${OK}" -eq 1 ]]; then
  echo "--- Architecture contract (N0) ---"
  if python scripts/architecture/check_tracking_contract.py; then
    echo "[OK] Architecture contract"
  else
    echo "[FAIL] Architecture contract"
    OK=0
  fi
fi

if [[ "${OK}" -eq 1 ]]; then
  echo "Pre-merge check : OK"
else
  echo "Pre-merge check : FAIL"
  exit 1
fi
