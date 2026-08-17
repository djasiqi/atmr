#!/usr/bin/env bash
# P0-E Phase 1 — E smoke (flag OFF)
set -euo pipefail
cd /srv/atmr
DRIVER_ID="${P0E_DRIVER_ID:-20135}"

echo "=== E SMOKE ==="
BH=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' atmr-backend-1)
CH=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' atmr-tracking-kafka-consumer-1)
RH=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' atmr-redis)
PH=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' atmr-postgres)
echo "backend_health=${BH}"
echo "consumer_health=${CH}"
echo "redis_health=${RH}"
echo "postgres_health=${PH}"
test "${BH}" = "healthy"
test "${RH}" = "healthy"
test "${PH}" = "healthy"

echo "=== API ready ==="
# internal health if available
docker exec atmr-backend-1 python - <<'PY' || true
import urllib.request
for url in ("http://127.0.0.1:5000/api/v1/health", "http://127.0.0.1:5000/health", "http://127.0.0.1:5000/"):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            print("HEALTH", url, r.status)
            break
    except Exception as e:
        print("HEALTH_TRY", url, type(e).__name__)
PY

echo "=== code + flag ==="
docker exec atmr-backend-1 test -f /app/services/tracking/location_candidate.py
PF=$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo false)
echo "PG_FIRST=${PF}"
test "${PF}" = "false"

echo "=== probe LOC/Redis via python ==="
docker cp /tmp/_p0e_smoke_e.py atmr-backend-1:/tmp/_p0e_smoke_e.py
docker exec atmr-backend-1 python /tmp/_p0e_smoke_e.py

echo "=== recent logs errors ==="
# new tracebacks last 3 min — best-effort
docker logs atmr-backend-1 --since 3m 2>&1 | grep -E "Traceback|ERROR|IntegrityError|UndefinedColumn" | grep -v EventletDeprecation | tail -20 || true
docker logs atmr-tracking-kafka-consumer-1 --since 3m 2>&1 | grep -E "Traceback|ERROR|IntegrityError|UndefinedColumn" | grep -v EventletDeprecation | tail -20 || true

echo "E_SMOKE_DONE"
