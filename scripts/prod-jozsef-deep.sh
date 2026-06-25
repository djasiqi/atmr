#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
echo "=== Driver 3 (Jozsef) vs 6858 (actif) ==="
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'SQL'
SELECT id, user_id, is_active, last_position_update, 
       ST_Y(last_known_location::geometry) AS lat,
       ST_X(last_known_location::geometry) AS lon
FROM driver WHERE id IN (3, 6858);

SELECT d.id, d.last_position_update,
       (SELECT COUNT(*) FROM trip_tracking tt WHERE tt.driver_id=d.id AND tt.timestamp > NOW()-INTERVAL '24 hours') AS tt_24h
FROM driver d WHERE d.id IN (3, 6858);

SELECT id, driver_id, status, started_at, ended_at, location_mode
FROM mission 
WHERE driver_id IN (3, 6858) 
ORDER BY started_at DESC NULLS LAST
LIMIT 10;

SELECT id, driver_id, status, pickup_time, dropoff_time
FROM booking_assignment ba
JOIN booking b ON b.id = ba.booking_id
WHERE ba.driver_id IN (3, 6858)
ORDER BY b.pickup_time DESC NULLS LAST
LIMIT 8;
SQL

echo "=== Logs consumer driver 3 (dernières lignes) ==="
docker compose -f docker-compose.production.yml --profile kafka logs tracking-kafka-consumer --since 4h 2>&1 | grep -i 'driver.*3\b\|driver_id=3\|driver_id: 3' | tail -20 || echo "(aucun log driver 3)"

echo "=== Logs API location driver 3 ==="
docker compose -f docker-compose.production.yml logs api --since 4h 2>&1 | grep -E 'driver.*(id=3|/3/|driver_id=3)' | tail -15 || echo "(aucun log API driver 3)"
