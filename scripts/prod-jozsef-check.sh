#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
echo "=== trip_tracking depuis restart consumer (2026-06-24 13:31 UTC) ==="
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'SQL'
SELECT driver_id, COUNT(*) AS n
FROM trip_tracking
WHERE timestamp > '2026-06-24 13:31:00+00'
GROUP BY driver_id
ORDER BY n DESC
LIMIT 15;
SQL
echo "=== heartbeats Jozsef aujourd'hui ==="
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "SELECT COUNT(*) AS hb_today FROM driver_device_health_events WHERE driver_id=3 AND created_at > CURRENT_DATE;" 2>/dev/null || echo "(table absente ou erreur)"
