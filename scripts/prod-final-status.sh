#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
echo "=== R5 verify once ==="
bash scripts/verify-tracking-persist-hotfix.sh --once 2>&1 || true

echo ""
echo "=== prod-persist-status-check ==="
bash scripts/prod-persist-status-check.sh 2>&1 || true

echo ""
echo "=== Jozsef device health + bookings ==="
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'SQL'
SELECT event_type, created_at, payload::text
FROM driver_device_health_events
WHERE driver_id=3
ORDER BY created_at DESC LIMIT 8;

SELECT b.id, b.status, b.scheduled_pickup_time, b.driver_id
FROM booking b
WHERE b.driver_id=3
ORDER BY b.scheduled_pickup_time DESC NULLS LAST
LIMIT 5;

SELECT COUNT(*) AS drivers_fresh_1h
FROM driver
WHERE is_active=true AND last_position_update > NOW()-INTERVAL '1 hour';
SQL

echo ""
echo "=== Consumer env + uptime ==="
docker compose -f docker-compose.production.yml --profile kafka ps tracking-kafka-consumer
docker inspect atmr-tracking-kafka-consumer-1 --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep TRACKING_INGEST || true
