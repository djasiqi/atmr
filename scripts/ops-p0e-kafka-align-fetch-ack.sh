#!/usr/bin/env bash
# Récupérer le body ACK depuis le cache idempotent Redis (clé location_event_id).
set -euo pipefail
cd /srv/atmr
EID="${1:?LOCATION_EVENT_ID}"
DRIVER_ID="${2:-3}"

docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T backend python - <<PY
import json, os
from security import idempotency
# tenter plusieurs patterns de clé
eid = "${EID}"
driver_id = int("${DRIVER_ID}")
candidates = []
try:
    from security.idempotency import get_idempotent_response
    for key in (eid,):
        resp = get_idempotent_response(driver_id, key)
        print("KEY", key, "=>", json.dumps(resp, default=str) if resp else None)
except Exception as e:
    print("ERR", type(e).__name__, e)

# scan redis brut
try:
    import redis
    from urllib.parse import urlparse
    url = os.environ.get("REDIS_URL") or ""
    password = os.environ.get("REDIS_PASSWORD") or ""
    host = os.environ.get("REDIS_HOST", "redis")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    r = redis.Redis(host=host, port=port, password=password or None, decode_responses=True)
    for pat in (f"*{eid}*", f"*idempot*{driver_id}*", f"*location*{driver_id}*"):
        keys = list(r.scan_iter(match=pat, count=100))[:30]
        print("PAT", pat, "KEYS", keys)
        for k in keys[:5]:
            val = r.get(k)
            print("VAL", k, (val[:500] if isinstance(val, str) else val))
except Exception as e:
    print("REDIS_ERR", type(e).__name__, e)
PY
