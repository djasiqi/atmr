#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
echo "Consumer started: $(docker inspect atmr-tracking-kafka-consumer-1 --format '{{.State.StartedAt}}')"
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'SQL'
SELECT COUNT(*) AS trip_24h FROM trip_tracking WHERE timestamp > NOW() - INTERVAL '24 hours';
SELECT COUNT(*) AS jozsef_since_21jun FROM trip_tracking WHERE driver_id=3 AND timestamp > '2026-06-21 06:40:00';
SELECT MAX(timestamp) AS jozsef_max FROM trip_tracking WHERE driver_id=3;
SELECT COUNT(*) AS active,
       COUNT(*) FILTER (WHERE last_position_update > NOW() - INTERVAL '24 hours') AS fresh_24h,
       COUNT(*) FILTER (WHERE last_position_update > NOW() - INTERVAL '10 minutes') AS fresh_10m
FROM driver WHERE is_active = true;
SELECT id, last_position_update, latitude, longitude FROM driver WHERE id=3;
SQL
docker exec atmr-tracking-kafka-consumer-1 wget -qO- http://127.0.0.1:9115/metrics 2>/dev/null | grep -E 'tracking_kafka_persist_total|tracking_http_accepted_async' | head -15 || echo "(metrics unavailable)"
