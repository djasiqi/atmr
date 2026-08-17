#!/usr/bin/env bash
# Diagnostiquer gel DLE vs PUT 202 / consumer
set -euo pipefail

echo "=== effective flags ==="
echo -n "consumer_OUTBOX="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX
echo -n "consumer_PG_FIRST="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
echo -n "backend_ASYNC="
docker exec atmr-backend-1 printenv TRACKING_INGEST_ASYNC_ENABLED || echo UNSET

echo "=== consumer logs 10m (persist/kafka) ==="
docker logs atmr-tracking-kafka-consumer-1 --since 10m 2>&1 \
  | grep -E 'persist|outbox|p5b|driver_id=20135|20135|ERROR|WARNING|Traceback|committed|skipped' \
  | grep -v EventletDeprecation \
  | tail -50 || true

echo "=== backend location structured 5m ==="
docker logs atmr-backend-1 --since 5m 2>&1 \
  | grep -E 'driver_id.?=.?20135|20135|accepted_async|ingested_non|location_event|capture_id|async' \
  | grep -v EventletDeprecation \
  | tail -40 || true

echo "=== global DLE max ==="
docker cp /tmp/_p0e_phase2_dle_check.py atmr-backend-1:/tmp/_p0e_phase2_dle_check.py
docker exec atmr-backend-1 python /tmp/_p0e_phase2_dle_check.py 2>/dev/null \
  | grep -vE 'Eventlet|OpenTelemetry|SOCKET|Socket|Security|deprecated|Message queue'
