#!/usr/bin/env bash
# Prouve que SKIP_DB_UPGRADE=1 n'appelle jamais `flask db upgrade`,
# même si le fichier env contient SKIP_DB_UPGRADE=0 et une valeur « per ».
set -o errexit -o nounset -o pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/load_compose_env.sh"
# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/deploy_db_upgrade.sh"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
LOG="${WORKDIR}/migration_exec.log"
ENV_FILE="${WORKDIR}/compose.env"

cat >"$ENV_FILE" <<'EOF'
SKIP_DB_UPGRADE=0
INTERNAL_TRACKING_INGEST_RATE_LIMIT=100 per 15 minutes
EOF

migration_exec() {
  printf '%s\n' "$*" >>"$LOG"
  return 0
}

assert_no_upgrade() {
  if grep -q 'flask db upgrade' "$LOG"; then
    echo "FAIL: flask db upgrade a été invoqué" >&2
    cat "$LOG" >&2
    exit 1
  fi
}

run_case() {
  : >"$LOG"
  capture_requested_skip_db_upgrade
  load_compose_env_file "$ENV_FILE"
  restore_requested_skip_db_upgrade
  if [ "${INTERNAL_TRACKING_INGEST_RATE_LIMIT}" != "100 per 15 minutes" ]; then
    echo "FAIL: rate limit altérée: '${INTERNAL_TRACKING_INGEST_RATE_LIMIT}'" >&2
    exit 1
  fi
  DEPLOY_DB_RETRY_1=0 DEPLOY_DB_RETRY_2=0 run_prod_db_upgrade_cycle
}

echo "=== SKIP_DB_UPGRADE=1 ==="
SKIP_DB_UPGRADE=1
export SKIP_DB_UPGRADE
run_case
if [ "${SKIP_DB_UPGRADE}" != "1" ]; then
  echo "FAIL: le fichier env a écrasé SKIP_DB_UPGRADE (${SKIP_DB_UPGRADE})" >&2
  exit 1
fi
assert_no_upgrade
grep -q 'flask db current' "$LOG"
grep -q 'python -m scripts.preview_auth_sms_02_promotion' "$LOG"
if ! skip_db_upgrade_enabled; then
  echo "FAIL: la stack complète ne doit pas démarrer quand SKIP=1" >&2
  exit 1
fi
echo "PASS: SKIP=1 → preview, aucun flask db upgrade, Celery non démarré"

echo "=== SKIP_DB_UPGRADE absent ==="
unset SKIP_DB_UPGRADE
run_case
if skip_db_upgrade_enabled; then
  echo "FAIL: SKIP absent doit laisser l'upgrade s'exécuter" >&2
  exit 1
fi
if ! grep -q 'flask db upgrade heads' "$LOG"; then
  echo "FAIL: flask db upgrade heads absent" >&2
  cat "$LOG" >&2
  exit 1
fi
echo "PASS: SKIP absent → flask db upgrade heads"

DEPLOY="${ROOT}/scripts/deploy-production.sh"
# shellcheck disable=SC2016
if grep -q 'source "${ENV_FILE_ARG}"' "$DEPLOY"; then
  echo "FAIL: deploy source encore le fichier env" >&2
  exit 1
fi
if grep -qE 'up -d --remove-orphans|down --remove-orphans' "$DEPLOY"; then
  echo "FAIL: --remove-orphans encore présent dans le rollback/up" >&2
  exit 1
fi
if ! grep -q 'run_prod_db_upgrade_cycle' "$DEPLOY"; then
  echo "FAIL: le deploy n'appelle pas run_prod_db_upgrade_cycle" >&2
  exit 1
fi
echo "PASS: deploy sans source .env et sans --remove-orphans"
