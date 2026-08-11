#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr

fetch_ack() {
  local eid="$1" did="$2"
  docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
    exec -T -e EID="${eid}" -e DID="${did}" backend python -c '
import json, os
from services.geolocation.driver_location_http import get_idempotent_response, _idem_key
eid = os.environ["EID"]
driver_id = int(os.environ["DID"])
print("idem_key", _idem_key(driver_id, eid))
resp = get_idempotent_response(driver_id, eid)
print(json.dumps(resp, indent=2, default=str) if resp else "NONE")
'
}

echo "=== prior canary eid ==="
fetch_ack trk_1786460027941_pl595416 3 || true

BEFORE="$(
  docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
    exec -T postgres psql -U atmr -d atmr -Atc \
    "SELECT COALESCE(MAX(id),0) FROM tracking_ingest_events;"
)"
echo "BEFORE=${BEFORE}"

for i in $(seq 1 12); do
  sleep 12
  ROW="$(
    docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
      exec -T postgres psql -U atmr -d atmr -Atc \
      "SELECT location_event_id || '|' || driver_id::text FROM tracking_ingest_events WHERE id > ${BEFORE} ORDER BY id ASC LIMIT 1;"
  )"
  if [[ -n "${ROW}" ]]; then
    EID="${ROW%%|*}"
    DID="${ROW##*|}"
    echo "NEW=${ROW}"
    BACKEND_CID="$(docker ps -q --filter name=atmr-backend | head -n1)"
    docker logs --since 2m "${BACKEND_CID}" 2>&1 | grep 'PUT /api/v1/driver/me/location' | tail -n 5 || true
    fetch_ack "${EID}" "${DID}"
    export LOCATION_EVENT_ID="${EID}" DRIVER_ID="${DID}" ENV_FILE=.env.production
    bash scripts/ops-gps-p0e-canary.sh
    bash scripts/ops-gps-p0e-canary.sh --proof-b-only
    exit 0
  fi
done
echo "NO_NEW_EVENT"
exit 1
