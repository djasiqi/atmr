#!/usr/bin/env bash
# Vérification post-hotfix persistance GPS (flotte-large).
#
# Usage (sur le serveur prod, depuis /srv/atmr) :
#   HOTFIX_TS=$(cat /tmp/atmr-hotfix-persist-ts.txt) bash scripts/verify-tracking-persist-hotfix.sh
# Boucle T+5/10/15 :
#   bash scripts/verify-tracking-persist-hotfix-loop.sh
set -euo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

TS_FILE="/tmp/atmr-hotfix-persist-ts.txt"
if [[ -z "${HOTFIX_TS:-}" ]] && [[ -f "${TS_FILE}" ]]; then
  HOTFIX_TS="$(cat "${TS_FILE}")"
fi
HOTFIX_TS="${HOTFIX_TS:-$(date -u +'%Y-%m-%d %H:%M:%S+00')}"
COMPOSE=(docker compose -f docker-compose.production.yml)
SERVICE=tracking-kafka-consumer
METRICS_PORT="${TRACKING_CONSUMER_METRICS_PORT:-9115}"
PASS=1

echo "=== Vérification persistance GPS (hotfix depuis ${HOTFIX_TS}) ==="

echo "--- Conteneur consumer ---"
cid="$("${COMPOSE[@]}" --profile kafka ps -q tracking-kafka-consumer 2>/dev/null | head -n1 || true)"
if [[ -n "${cid}" ]]; then
  docker exec "${cid}" env | grep -E '^TRACKING_INGEST_' || true
  if docker exec "${cid}" env | grep -q '^TRACKING_INGEST_PERSIST_ENABLED=true'; then
    echo "[OK] TRACKING_INGEST_PERSIST_ENABLED=true dans le conteneur"
  else
    echo "[FAIL] TRACKING_INGEST_PERSIST_ENABLED!=true dans le conteneur"
    PASS=0
  fi
  docker logs "${cid}" --tail=20 2>&1 || true
else
  echo "[WARN] tracking-kafka-consumer absent"
  PASS=0
fi

echo "--- R11 Métriques consumer (tracking_invalid_config_total) ---"
if [[ -n "${cid}" ]]; then
  metrics="$(docker exec "${cid}" sh -c "wget -qO- http://127.0.0.1:${METRICS_PORT}/metrics 2>/dev/null || curl -sf http://127.0.0.1:${METRICS_PORT}/metrics 2>/dev/null" || true)"
  if echo "${metrics}" | grep -q 'tracking_invalid_config_total'; then
    echo "${metrics}" | grep 'tracking_invalid_config_total' || true
  else
    echo "[WARN] tracking_invalid_config_total absente (PR1 pas encore déployée ?)"
  fi
fi

echo "--- SQL flotte (via backend container) ---"
"${COMPOSE[@]}" exec -T backend python - <<PY
import os

hotfix = os.environ.get("HOTFIX_TS", "${HOTFIX_TS}")
print(f"hotfix_ts={hotfix}")

from app import create_app
from ext import db

app = create_app()
with app.app_context():
    row = db.session.execute(
        db.text(
            """
            SELECT COUNT(*) AS active,
                   COUNT(*) FILTER (
                     WHERE last_position_update > NOW() - INTERVAL '10 minutes'
                   ) AS fresh_10m
            FROM driver WHERE is_active = true
            """
        )
    ).one()
    print(f"drivers_active={row.active} fresh_10m={row.fresh_10m}")

    tt = db.session.execute(
        db.text(
            """
            SELECT COUNT(*) AS rows_post_hotfix
            FROM trip_tracking
            WHERE timestamp > CAST(:ts AS timestamptz)
            """
        ),
        {"ts": hotfix},
    ).one()
    print(f"trip_tracking_post_hotfix={tt.rows_post_hotfix}")

    jozsef = db.session.execute(
        db.text(
            """
            SELECT id, last_position_update, latitude, longitude
            FROM driver WHERE id = 3
            """
        )
    ).one_or_none()
    if jozsef:
        print(
            f"jozsef driver_id=3 last_position_update={jozsef.last_position_update} "
            f"lat={jozsef.latitude} lon={jozsef.longitude}"
        )
PY

echo "--- Critères R5 ---"
echo "  fresh_10m >= 1"
echo "  trip_tracking_post_hotfix > 0"
echo "  TRACKING_INGEST_PERSIST_ENABLED=true (conteneur)"
echo "--- Prometheus (externe) ---"
echo "  rate(tracking_kafka_persist_total[5m])>0, lag<200"
echo "  Alertes R10 : TrackingInvalidConfig, TrackingPersistStalledWhileIngesting, TrackingKafkaConsumerRestartLoop"

if [[ "${PASS}" -eq 0 ]]; then
  exit 1
fi
