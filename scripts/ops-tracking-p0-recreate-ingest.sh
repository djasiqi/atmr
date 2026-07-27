#!/usr/bin/env bash
# P0 ops — stop fanout legacy + recreate ingest seul (pas de up global).
# Aucun défaut pour COMPOSE_FILES / ENV_FILE / COMPOSE_PROJECT_NAME.
#
# Dry-run (aucun stop / up) :
#   DRY_RUN=1 COMPOSE_PROJECT_NAME=atmr \
#     COMPOSE_FILES="-f docker-compose.production.yml -f docker-compose.kafka.yml -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml" \
#     ENV_FILE="--env-file .env.production" \
#     ./scripts/ops-tracking-p0-recreate-ingest.sh
#
# Exécution réelle :
#   EXECUTE_P0_RECREATE=YES COMPOSE_PROJECT_NAME=… COMPOSE_FILES=… ENV_FILE=… \
#     ./scripts/ops-tracking-p0-recreate-ingest.sh
#
set -euo pipefail

: "${COMPOSE_FILES:?COMPOSE_FILES obligatoire (ex: -f docker-compose.production.yml -f docker-compose.kafka.yml -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml)}"
: "${ENV_FILE:?ENV_FILE obligatoire (ex: --env-file .env.production)}"
: "${COMPOSE_PROJECT_NAME:?COMPOSE_PROJECT_NAME obligatoire}"

PROFILE="${PROFILE:---profile kafka}"
DRY_RUN="${DRY_RUN:-0}"
EXECUTE_P0_RECREATE="${EXECUTE_P0_RECREATE:-}"

compose() {
  # shellcheck disable=SC2086
  docker compose -p "${COMPOSE_PROJECT_NAME}" ${ENV_FILE} ${COMPOSE_FILES} ${PROFILE} "$@"
}

umask 077
CFG_TMP="$(mktemp /tmp/atmr-p0-compose.XXXXXX)"
cleanup_cfg() {
  rm -f "${CFG_TMP}"
}
trap cleanup_cfg EXIT

echo "== Préflight config fusionné (mktemp) =="
compose config > "${CFG_TMP}"

assert_env_flag() {
  local svc="$1"
  local key="$2"
  local expected="$3"
  local got
  got="$(
    awk -v svc="$svc" -v key="$key" '
      $0 ~ "^  "svc":" {in_svc=1; next}
      in_svc && /^  [a-zA-Z]/ {in_svc=0}
      in_svc && $0 ~ ("^      " key ":") {
        sub(/^[^:]+:[[:space:]]*/, "", $0)
        gsub(/["'\'' ]/, "", $0)
        print $0
        exit
      }
    ' "${CFG_TMP}"
  )"
  if [[ "${got}" != "${expected}" ]]; then
    echo "FAIL ${svc}.${key}: attendu=${expected} obtenu=${got:-<absent>}" >&2
    exit 1
  fi
  echo "OK ${svc}.${key}=${got}"
}

echo "== Vérif DSN / flags / topics =="
for svc in tracking-kafka-consumer tracking-processed-fanout kafka-dlq-consumer; do
  echo "--- ${svc} ---"
  awk -v svc="$svc" '
    $0 ~ "^  "svc":" {in_svc=1; next}
    in_svc && /^  [a-zA-Z]/ {in_svc=0}
    in_svc && /DATABASE_URL:|SQLALCHEMY_DATABASE_URI:|PRIMARY_DATABASE_URL:|REPLICA_DATABASE_URL:|REPLICA_DATABASE_URLS:|POSTGRES_HOST:|POSTGRES_PORT:|POSTGRES_DB:|POSTGRES_USER:/ {
      print
    }
  ' "${CFG_TMP}"
done

# Pas d'URL interpolée user/password
if grep -E 'postgresql\+psycopg://\$\{POSTGRES_(USER|PASSWORD)' "${CFG_TMP}" >/dev/null 2>&1; then
  echo "FAIL: URL postgresql+psycopg interpolée détectée dans le config fusionné" >&2
  exit 1
fi

assert_env_flag tracking-kafka-consumer TRACKING_PERSIST_WITH_OUTBOX false
assert_env_flag tracking-kafka-consumer TRACKING_INGEST_PERSIST_ENABLED true
assert_env_flag tracking-kafka-consumer TRACKING_INGEST_ALLOW_REPUBLISH_ONLY false
assert_env_flag tracking-kafka-consumer TRACKING_INGEST_SEEK_TO_END_ON_START false
assert_env_flag tracking-kafka-consumer TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE false
assert_env_flag tracking-processed-fanout TRACKING_PROCESSED_FANOUT_ENABLED false

# Topics *.v2 obligatoires (noms effectifs après interpolation) — fail-hard
assert_env_flag tracking-kafka-consumer KAFKA_TOPIC_DRIVER_LOCATION_RAW driver.location.raw.v2
assert_env_flag tracking-kafka-consumer KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed.v2
assert_env_flag tracking-kafka-consumer KAFKA_TOPIC_DRIVER_LOCATION_DLQ driver.location.dlq.v2
assert_env_flag tracking-processed-fanout KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed.v2
assert_env_flag kafka-dlq-consumer KAFKA_TOPIC_DRIVER_LOCATION_DLQ driver.location.dlq.v2

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — aucun stop / up / recreate. Config OK."
  exit 0
fi

if [[ "${EXECUTE_P0_RECREATE}" != "YES" ]]; then
  echo "Refuse l'exécution destructive : définir EXECUTE_P0_RECREATE=YES (ou DRY_RUN=1)." >&2
  exit 1
fi

echo "== Stop fanout legacy (tous replicas) =="
compose stop tracking-processed-fanout
# Échec si un replica tourne encore
fanout_state="$(compose ps --status running --services 2>/dev/null | grep -E '^tracking-processed-fanout$' || true)"
if [[ -n "${fanout_state}" ]]; then
  echo "FAIL: tracking-processed-fanout encore running après stop" >&2
  compose ps tracking-processed-fanout >&2 || true
  exit 1
fi
compose ps tracking-processed-fanout || true

echo "== Stop ingest pendant patch =="
compose stop tracking-kafka-consumer

echo "== Recreate ingest uniquement =="
compose up -d --no-deps --force-recreate tracking-kafka-consumer

echo "== Préflight SELECT 1 + assert flags (sans imprimer le DSN) =="
compose exec -T tracking-kafka-consumer python - <<'PY'
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from config import _build_database_url_safe

raw = _build_database_url_safe()
url = make_url(raw)
print({
    "driver": url.drivername,
    "host": url.host,
    "port": url.port,
    "database": url.database,
    "user_present": bool(url.username),
    "password_present": bool(url.password),
})
assert url.host == "pgbouncer", url.host
assert url.port == 6432, url.port
with create_engine(raw).connect() as conn:
    assert conn.execute(text("SELECT 1")).scalar() == 1
print("SELECT 1 OK")

def flag(name: str) -> str:
    return (os.getenv(name) or "").strip().lower()

assert flag("TRACKING_PERSIST_WITH_OUTBOX") == "false"
assert flag("TRACKING_INGEST_PERSIST_ENABLED") == "true"
assert flag("TRACKING_INGEST_ALLOW_REPUBLISH_ONLY") == "false"
assert flag("TRACKING_INGEST_SEEK_TO_END_ON_START") == "false"
assert flag("TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE") == "false"
print("flags OK")
PY

echo "== Fanout doit rester exited =="
fanout_running="$(compose ps --status running --services 2>/dev/null | grep -E '^tracking-processed-fanout$' || true)"
if [[ -n "${fanout_running}" ]]; then
  echo "FAIL: tracking-processed-fanout running après recreate ingest" >&2
  exit 1
fi
compose ps tracking-processed-fanout || true
echo "DONE — lancer ensuite la gate E2E ×3 (nouvelles positions, pas de reset offsets)."
echo "Recréer kafka-dlq-consumer seulement après gate ingest OK."
