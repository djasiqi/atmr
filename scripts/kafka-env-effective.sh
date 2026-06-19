#!/usr/bin/env bash
set -euo pipefail

# Produit un fichier .env effectif pour deploy/check Kafka :
# .env.production + surcharges KAFKA_* de .env.production.local (dernière valeur gagne).
#
# Usage :
#   eval "$(./scripts/kafka-env-effective.sh)"
#   # ou
#   export ATMR_ENV_FILE="$(./scripts/kafka-env-effective.sh --path)"

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"
LOCAL="${ROOT}/.env.production.local"
OUT="${ROOT}/.env.production.kafka-effective"
TMP="${OUT}.$$"

if [[ ! -f "${BASE}" ]]; then
  echo "Fichier manquant : ${BASE}" >&2
  exit 1
fi

cp "${BASE}" "${TMP}"
chmod 600 "${TMP}"

if [[ -f "${LOCAL}" ]]; then
  {
    echo ""
    echo "# --- Surcharges Kafka depuis .env.production.local (kafka-env-effective.sh)"
    grep -E '^KAFKA_' "${LOCAL}" 2>/dev/null | grep -v '^[[:space:]]*#' || true
  } >> "${TMP}"
fi

mv -f "${TMP}" "${OUT}"

if [[ "${1:-}" == "--path" ]]; then
  printf '%s\n' "${OUT}"
else
  printf 'export ATMR_ENV_FILE=%q\n' "${OUT}"
fi
