#!/usr/bin/env bash
# B0 — Capture diff fragment env vs prod AVANT hotfix (preuve cause racine).
#
# Usage local (avec SSH) :
#   set -a; . ./.local.deploy.env; set +a
#   bash scripts/capture-b0-diff-production.sh
#
# Usage sur le serveur (/srv/atmr) :
#   bash scripts/capture-b0-diff-production.sh --local
#
# Sortie : docs/incidents/b0-diff-evidence.txt (à coller dans le post-mortem)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRAGMENT="${ROOT}/scripts/env.production.defaults.fragment"
OUT="${ROOT}/docs/incidents/b0-diff-evidence.txt"
LOCAL_MODE=0
SERVER_PATH="${SERVER_PATH:-/srv/atmr}"
SERVER_USER="${SERVER_USER:-deploy}"
SERVER_HOST="${SERVER_HOST:-}"

if [[ "${1:-}" == "--local" ]]; then
  LOCAL_MODE=1
fi

if [[ ! -f "${FRAGMENT}" ]]; then
  echo "Fragment introuvable : ${FRAGMENT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"
{
  echo "# B0 diff evidence — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo ""
} >"${OUT}"

echo "=== B0 — capture diff fragment vs prod ==="

if [[ "${LOCAL_MODE}" == "1" ]]; then
  PROD_ENV="${SERVER_PATH}/.env.production"
  PROD_LOCAL="${SERVER_PATH}/.env.production.local"
  PROD_DOT="${SERVER_PATH}/.env"
  if [[ ! -f "${PROD_ENV}" ]]; then
    echo "[FAIL] ${PROD_ENV} introuvable" | tee -a "${OUT}"
    exit 1
  fi
  {
    echo "## Fichiers prod (local mode)"
    echo ""
    echo "### .env.production.local"
    if [[ -f "${PROD_LOCAL}" ]]; then
      grep -E '^TRACKING_INGEST_|^KAFKA_ENABLED=' "${PROD_LOCAL}" 2>/dev/null || echo "(aucune ligne TRACKING/KAFKA)"
    else
      echo "(absent)"
    fi
    echo ""
    echo "### TRACKING vars .env / .env.production"
    grep -E '^TRACKING_INGEST_|^KAFKA_ENABLED=' "${PROD_DOT}" "${PROD_ENV}" 2>/dev/null || true
    echo ""
    echo "## Diff clés TRACKING/KAFKA (fragment vs .env.production)"
    echo '```diff'
    diff -u \
      <(grep -E '^TRACKING_|^KAFKA_ENABLED=' "${FRAGMENT}" | sort) \
      <(grep -E '^TRACKING_|^KAFKA_ENABLED=' "${PROD_ENV}" | sort) \
      || true
    echo '```'
  } >>"${OUT}"
else
  if [[ -z "${SERVER_HOST}" ]]; then
    echo "[FAIL] SERVER_HOST requis (voir docs/deployment-ssh.md) ou utiliser --local sur le serveur" | tee -a "${OUT}"
    exit 1
  fi
  REMOTE="${SERVER_USER}@${SERVER_HOST}"
  scp "${FRAGMENT}" "${REMOTE}:/tmp/atmr-env-fragment.txt"
  ssh "${REMOTE}" bash -s <<'REMOTE' >>"${OUT}"
set -euo pipefail
echo "## Override .env.production.local"
if [[ -f /srv/atmr/.env.production.local ]]; then
  ls -la /srv/atmr/.env.production.local
  grep -E '^TRACKING_INGEST_|^KAFKA_ENABLED=' /srv/atmr/.env.production.local 2>/dev/null || echo "(aucune ligne TRACKING/KAFKA)"
else
  echo "(absent)"
fi
echo ""
echo "### TRACKING vars .env / .env.production"
grep -E '^TRACKING_INGEST_|^KAFKA_ENABLED=' /srv/atmr/.env /srv/atmr/.env.production 2>/dev/null || true
echo ""
echo "## Diff clés TRACKING/KAFKA (fragment vs .env.production)"
echo '```diff'
diff -u \
  <(grep -E '^TRACKING_|^KAFKA_ENABLED=' /tmp/atmr-env-fragment.txt | sort) \
  <(grep -E '^TRACKING_|^KAFKA_ENABLED=' /srv/atmr/.env.production | sort) \
  || true
echo '```'
REMOTE
fi

echo "Evidence écrite : ${OUT}"
echo "→ Coller le contenu dans docs/incidents/2026-06-21-tracking-persist-disabled.md (section B0)"
