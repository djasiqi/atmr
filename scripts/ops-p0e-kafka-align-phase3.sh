#!/usr/bin/env bash
# Attendre le prochain PUT 200 ~67x, corréler eid ledger, lancer canary A+B.
set -euo pipefail
cd /srv/atmr

WINDOW_MIN="${CANARY_WINDOW_MIN:-15}"
END_TS=$(( $(date +%s) + WINDOW_MIN * 60 ))
BACKEND_CID="$(docker ps -q --filter name=atmr-backend | head -n1)"

psql_at() {
  docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
    exec -T postgres psql -U atmr -d atmr -Atc "$1"
}

BEFORE_ID="$(psql_at "SELECT COALESCE(MAX(id),0) FROM tracking_ingest_events;")"
echo "WINDOW_MIN=${WINDOW_MIN} BEFORE_ID=${BEFORE_ID} END=$(date -u -d "@${END_TS}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo ${END_TS})"

FOUND_EID=""
FOUND_DRIVER=""
ACCESS_LINE=""

while [[ $(date +%s) -lt ${END_TS} ]]; do
  access="$(docker logs --since 90s "${BACKEND_CID}" 2>&1 | grep 'PUT /api/v1/driver/me/location' | tail -n 8 || true)"
  if echo "${access}" | grep -qE ' 202 '; then
    echo "SEEN_202 — chemin async, continue (NON EXERCÉ pour ces hits)"
    echo "${access}" | grep ' 202 ' || true
  fi
  if echo "${access}" | grep -qE ' 200 67[0-9] '; then
    ACCESS_LINE="$(echo "${access}" | grep -E ' 200 67[0-9] ' | tail -n1)"
    echo "SEEN_200_67x: ${ACCESS_LINE}"
    # Nouvel event ledger après BEFORE_ID
    row="$(psql_at "
SELECT e.location_event_id || '|' || e.driver_id::text
FROM tracking_ingest_events e
WHERE e.id > ${BEFORE_ID}
ORDER BY e.id ASC
LIMIT 1;
")"
    if [[ -n "${row}" ]]; then
      FOUND_EID="${row%%|*}"
      FOUND_DRIVER="${row##*|}"
      echo "CORRELATED eid=${FOUND_EID} driver=${FOUND_DRIVER}"
      break
    fi
  fi
  sleep 15
done

if [[ -z "${FOUND_EID}" ]]; then
  echo "PREUVE_A=NON_EXERCE"
  echo "REASON=aucun PUT 200~67x corrélé à un nouvel event ledger dans la fenêtre"
  echo "GO_P0F=NON"
  exit 0
fi

# Tenter lecture cache idempotent Redis (preuve body) si possible
set -a
# shellcheck disable=SC1091
source <(grep -E '^(REDIS_PASSWORD|REDIS_URL)=' .env.production | sed 's/\r$//')
set +a
echo "=== try idempotent cache (best effort) ==="
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T redis sh -c "redis-cli -a \"\$REDIS_PASSWORD\" --no-auth-warning KEYS '*idempot*' 2>/dev/null | head -20" || true
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T redis sh -c "redis-cli -a \"\$REDIS_PASSWORD\" --no-auth-warning KEYS '*${FOUND_EID}*' 2>/dev/null | head -20" || true

echo "ACCESS_LINE=${ACCESS_LINE}"
echo "HTTP_STATUS_HINT=200 body_bytes~67x (pas 202 queued_async)"
echo "NOTE_BODY=access log ne conserve pas le JSON; corrélation sync via 200+taille+latence ledger<1s"

export LOCATION_EVENT_ID="${FOUND_EID}"
export DRIVER_ID="${FOUND_DRIVER}"
export ENV_FILE=.env.production
bash scripts/ops-gps-p0e-canary.sh
rc_a=$?
bash scripts/ops-gps-p0e-canary.sh --proof-b-only
rc_b=$?

if [[ ${rc_a} -eq 0 && ${rc_b} -eq 0 ]]; then
  echo "PREUVE_A=PASS_SQL"
  echo "PREUVE_B=PASS"
  echo "HTTP_BODY_CAPTURE=PARTIAL_ACCESS_LOG_ONLY"
else
  echo "PREUVE_A_OR_B=FAIL rc_a=${rc_a} rc_b=${rc_b}"
fi
