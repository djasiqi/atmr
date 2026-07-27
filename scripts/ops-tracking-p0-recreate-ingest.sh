#!/usr/bin/env bash
# P0 ops — stop fanout legacy + recreate ingest seul (pas de up global).
# Utiliser EXACTEMENT les mêmes -f / --env-file que le déploiement.
#
# Exemple production :
#   export COMPOSE_FILES="-f docker-compose.production.yml -f docker-compose.kafka.yml -f docker-compose.kafka.atmr-network.yml"
#   export ENV_FILE="--env-file .env.production"
#   ./scripts/ops-tracking-p0-recreate-ingest.sh
#
set -euo pipefail

COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.yml -f docker-compose.kafka.yml}"
ENV_FILE="${ENV_FILE:-}"
PROFILE="${PROFILE:---profile kafka}"

compose() {
  # shellcheck disable=SC2086
  docker compose ${ENV_FILE} ${COMPOSE_FILES} ${PROFILE} "$@"
}

echo "== Préflight config fusionné (extraits DSN) =="
compose config > /tmp/atmr-effective-p0.yml
for svc in tracking-kafka-consumer tracking-processed-fanout kafka-dlq-consumer; do
  echo "--- ${svc} ---"
  # Ne jamais greper le mot de passe
  awk -v svc="$svc" '
    $0 ~ "^  "svc":" {in_svc=1; next}
    in_svc && /^  [a-zA-Z]/ {in_svc=0}
    in_svc && /DATABASE_URL:|SQLALCHEMY_DATABASE_URI:|PRIMARY_DATABASE_URL:|POSTGRES_HOST:|POSTGRES_PORT:|POSTGRES_DB:|POSTGRES_USER:/ {
      print
    }
  ' /tmp/atmr-effective-p0.yml
done

echo "== Stop fanout legacy (tous replicas) =="
compose stop tracking-processed-fanout || true
compose ps tracking-processed-fanout || true

echo "== Stop ingest pendant patch =="
compose stop tracking-kafka-consumer || true

echo "== Recreate ingest uniquement =="
compose up -d --no-deps --force-recreate tracking-kafka-consumer

echo "== Préflight SELECT 1 (sans imprimer le DSN) =="
compose exec -T tracking-kafka-consumer python - <<'PY'
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
PY

echo "== Fanout doit rester stoppé =="
compose ps tracking-processed-fanout || true
echo "DONE — lancer ensuite la gate E2E ×3 (nouvelles positions, pas de reset offsets)."
echo "Recréer kafka-dlq-consumer seulement après gate ingest OK."
