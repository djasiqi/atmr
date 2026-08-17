#!/usr/bin/env bash
set -euo pipefail
echo "=== flags ==="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX
echo "=== promote/persist logs 20m ==="
docker logs atmr-tracking-kafka-consumer-1 --since 20m 2>&1 \
  | grep -E 'p5b_promote|promotion canonical|_maybe_promote|persist_kafka|WARNING|ERROR|Traceback' \
  | grep -v Eventlet | tail -40 || true
echo "=== PUT last 10 ==="
docker logs atmr-backend-1 --since 5m 2>&1 \
  | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | tail -10 || true
echo "=== probe redis+pg ==="
docker cp /tmp/_p0e_phase2_probe.py atmr-backend-1:/tmp/_p0e_phase2_probe.py
docker exec atmr-backend-1 python /tmp/_p0e_phase2_probe.py 2>&1 \
  | grep -vE 'Eventlet|OpenTelemetry|SOCKET|Socket|Security|deprecated|Message queue|Flask|eventlet|strongly|framework|patcher'
