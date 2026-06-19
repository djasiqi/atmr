#!/usr/bin/env bash
set -euo pipefail

# Ajoute idempotentement le bloc topics v2 dans .env.production.local (non versionné).
# Modèle : scripts/env.production.local.kafka-v2.example

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOCAL="${ROOT}/.env.production.local"
EXAMPLE="${ROOT}/scripts/env.production.local.kafka-v2.example"

if [[ ! -f "${EXAMPLE}" ]]; then
  echo "Fichier manquant : ${EXAMPLE}" >&2
  exit 1
fi

if [[ -f "${LOCAL}" ]] && grep -qE '^KAFKA_TOPIC_DRIVER_LOCATION_RAW=' "${LOCAL}" 2>/dev/null; then
  echo "Topics v2 déjà présents dans ${LOCAL} — rien à faire."
  exit 0
fi

touch "${LOCAL}"
chmod 600 "${LOCAL}"
{
  echo ""
  echo "# --- Kafka topics v2 (Phase 1 LIRIE) — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  grep -v '^#' "${EXAMPLE}" | grep -v '^[[:space:]]*$' || true
} >> "${LOCAL}"

echo "Bloc topics v2 ajouté dans ${LOCAL}"
