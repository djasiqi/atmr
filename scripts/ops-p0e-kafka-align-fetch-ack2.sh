#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
EID="${1:?eid}"
DRIVER_ID="${2:-3}"
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T backend python - <<PY
import json
from services.geolocation.driver_location_http import get_idempotent_response, _idem_key
import os
eid = "${EID}"
driver_id = int("${DRIVER_ID}")
print("idem_key", _idem_key(driver_id, eid))
resp = get_idempotent_response(driver_id, eid)
print(json.dumps(resp, indent=2, default=str) if resp else "NONE")
PY
