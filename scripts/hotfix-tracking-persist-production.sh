#!/usr/bin/env bash
# Hotfix prod : activer TRACKING_INGEST_PERSIST_ENABLED sur tracking-kafka-consumer.
# Voir docs/incidents/2026-06-21-tracking-persist-disabled.md et docs/ops/gps-tracking-pipeline.md
#
# Usage (sur le serveur prod, depuis /srv/atmr) :
#   bash scripts/capture-b0-diff-production.sh --local   # R1 AVANT toute modif
#   bash scripts/hotfix-tracking-persist-production.sh
# Rollback :
#   bash scripts/hotfix-tracking-persist-production.sh --rollback
#
# Vérification (R5) :
#   bash scripts/verify-tracking-persist-hotfix-loop.sh
set -euo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_PRODUCTION="${ROOT}/.env.production"
ENV_DOT="${ROOT}/.env"
ENV_LOCAL="${ROOT}/.env.production.local"
COMPOSE=(docker compose -f docker-compose.production.yml --profile kafka)
SERVICE=tracking-kafka-consumer
TS_FILE="/tmp/atmr-hotfix-persist-ts.txt"
ROLLBACK=0
SKIP_B0=0

for arg in "$@"; do
  case "${arg}" in
    --rollback) ROLLBACK=1 ;;
    --skip-b0) SKIP_B0=1 ;;
  esac
done

env_files_to_patch() {
  printf '%s\n' "${ENV_DOT}" "${ENV_PRODUCTION}"
  if [[ -f "${ENV_LOCAL}" ]]; then
    printf '%s\n' "${ENV_LOCAL}"
  fi
}

upsert_env_var() {
  local file="$1"
  local key="$2"
  local value="$3"
  [[ -f "${file}" ]] || touch "${file}"
  if grep -qE "^${key}=" "${file}" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${file}"
  else
    echo "${key}=${value}" >>"${file}"
  fi
}

echo "=== Hotfix persistance GPS (TRACKING_INGEST_PERSIST_ENABLED) ==="
echo "Répertoire : ${ROOT}"

# R3 — Avertir si un deploy semble en cours
if pgrep -f 'deploy-production\.sh' >/dev/null 2>&1; then
  echo "[FAIL] deploy-production.sh semble en cours — attendre la fin avant hotfix (R3)" >&2
  exit 2
fi
if [[ -f "${ENV_PRODUCTION}" ]] && find "${ENV_PRODUCTION}" -mmin -2 2>/dev/null | grep -q .; then
  echo "[WARN] .env.production modifié il y a < 2 min — vérifier qu'aucun deploy CI n'est en cours (R3)"
fi

# R1 — B0 diff avant toute modification
if [[ "${SKIP_B0}" != "1" ]] && [[ "${ROLLBACK}" != "1" ]]; then
  echo "--- 0/7 B0 diff (AVANT modification) ---"
  if [[ -x "${ROOT}/scripts/capture-b0-diff-production.sh" ]]; then
    bash "${ROOT}/scripts/capture-b0-diff-production.sh" --local || {
      echo "[WARN] B0 diff échoué — continuer avec --skip-b0 si déjà capturé"
      exit 1
    }
  else
    echo "[WARN] capture-b0-diff-production.sh absent — exécuter manuellement (R1)"
  fi
fi

echo "--- 1/7 Diagnostic conteneur (env actuel) ---"
cid="$("${COMPOSE[@]}" ps -q "${SERVICE}" 2>/dev/null | head -n1 || true)"
if [[ -n "${cid}" ]]; then
  docker exec "${cid}" env | grep -E '^TRACKING_' || true
else
  echo "[WARN] Aucun conteneur ${SERVICE} en cours"
fi

echo "--- 2/7 Source Compose (substitution environment:) ---"
"${COMPOSE[@]}" config "${SERVICE}" 2>/dev/null | grep -E 'TRACKING_INGEST_(ASYNC|PERSIST)_ENABLED|OSRM_SNAP' || true
echo "Note : la substitution \${...} lit ${ENV_DOT} (cp depuis .env.production par deploy-production.sh)"

echo "--- 3/7 Fichiers env sur disque (R2 override local) ---"
for f in "${ENV_DOT}" "${ENV_PRODUCTION}" "${ENV_LOCAL}"; do
  if [[ -f "${f}" ]]; then
    echo "# ${f}"
    grep -E '^TRACKING_INGEST_(ASYNC|PERSIST)_ENABLED=|^TRACKING_INGEST_ALLOW|^OSRM_SNAP_' "${f}" 2>/dev/null || echo "  (absent)"
  elif [[ "${f}" == "${ENV_LOCAL}" ]]; then
    echo "# ${f} (absent — OK)"
  fi
done

if [[ -f "${ENV_LOCAL}" ]]; then
  if grep -qE '^TRACKING_INGEST_PERSIST_ENABLED=false' "${ENV_LOCAL}" 2>/dev/null; then
    echo "[WARN] R2 : .env.production.local force PERSIST=false — sera corrigé avec les autres fichiers"
  fi
fi

echo "--- 4/7 Correction env (.env + .env.production + .env.production.local si présent) ---"
while IFS= read -r f; do
  [[ -n "${f}" ]] || continue
  if [[ "${ROLLBACK}" == "1" ]]; then
    echo "  patch ${f} (rollback republish-only)"
    upsert_env_var "${f}" "TRACKING_INGEST_PERSIST_ENABLED" "false"
    upsert_env_var "${f}" "TRACKING_INGEST_ALLOW_REPUBLISH_ONLY" "true"
  else
    echo "  patch ${f}"
    upsert_env_var "${f}" "TRACKING_INGEST_PERSIST_ENABLED" "true"
    upsert_env_var "${f}" "OSRM_SNAP_TIMEOUT_S" "1.5"
    upsert_env_var "${f}" "OSRM_SNAP_TIMEOUT_ENABLED" "true"
    # Ne pas laisser un override republish-only actif après hotfix normal
    if grep -qE '^TRACKING_INGEST_ALLOW_REPUBLISH_ONLY=' "${f}" 2>/dev/null; then
      upsert_env_var "${f}" "TRACKING_INGEST_ALLOW_REPUBLISH_ONLY" "false"
    fi
  fi
done < <(env_files_to_patch)

echo "--- 5/7 R6 — Vérification .env ET .env.production ---"
grep -E '^TRACKING_INGEST_PERSIST_ENABLED=' "${ENV_DOT}" "${ENV_PRODUCTION}" 2>/dev/null || true
if [[ "${ROLLBACK}" != "1" ]]; then
  grep -q '^TRACKING_INGEST_PERSIST_ENABLED=true' "${ENV_DOT}" || {
    echo "[FAIL] ${ENV_DOT} n'a pas PERSIST=true" >&2
    exit 1
  }
  grep -q '^TRACKING_INGEST_PERSIST_ENABLED=true' "${ENV_PRODUCTION}" || {
    echo "[FAIL] ${ENV_PRODUCTION} n'a pas PERSIST=true" >&2
    exit 1
  }
fi

echo "--- 6/7 Vérification substitution Compose avant recreate ---"
"${COMPOSE[@]}" config "${SERVICE}" | grep -E 'TRACKING_INGEST_PERSIST_ENABLED' || true

echo "--- 7/7 Recreate consumer (--force-recreate, ~10-30s downtime) ---"
"${COMPOSE[@]}" up -d --no-deps --force-recreate "${SERVICE}"

sleep 3
cid="$("${COMPOSE[@]}" ps -q "${SERVICE}" 2>/dev/null | head -n1 || true)"
if [[ -n "${cid}" ]]; then
  echo "Env conteneur après recreate :"
  docker exec "${cid}" env | grep -E '^TRACKING_INGEST_' || true
fi

# R4 — HOTFIX_TS auto
HOTFIX_TS="$(date -u '+%Y-%m-%d %H:%M:%S+00')"
echo "${HOTFIX_TS}" >"${TS_FILE}"
echo ""
echo "✓ Hotfix terminé à ${HOTFIX_TS}"
echo "  Timestamp : ${TS_FILE}"
echo "  Vérifier  : bash scripts/verify-tracking-persist-hotfix-loop.sh"
echo "  (ou une passe : HOTFIX_TS=\"${HOTFIX_TS}\" bash scripts/verify-tracking-persist-hotfix.sh)"
echo ""
echo "Runbook : ne pas lancer deploy-production.sh avant merge PR2 sans vérifier .env.production"
