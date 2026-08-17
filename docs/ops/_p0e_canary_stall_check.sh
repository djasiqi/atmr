#!/usr/bin/env bash
set -euo pipefail
echo "=== flags/health ==="
echo -n "PG_FIRST="; docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
echo -n "OUTBOX="; docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX
docker inspect -f 'backend={{.State.Health.Status}}' atmr-backend-1
docker inspect -f 'consumer={{.State.Health.Status}}' atmr-tracking-kafka-consumer-1
echo "=== PUT 3m ==="
docker logs atmr-backend-1 --since 3m 2>&1 \
  | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin \
  | awk '{print $9}' | sort | uniq -c | sort -rn || true
echo "=== TB ==="
echo -n "tb="; docker logs atmr-tracking-kafka-consumer-1 --since 5m 2>&1 | grep -c Traceback || true
echo "=== DLQ types 3m ==="
docker logs atmr-tracking-kafka-consumer-1 --since 3m 2>&1 \
  | grep 'DLQ confirmed' | sed -n 's/.*type=\([^ ]*\).*/\1/p' \
  | sort | uniq -c | sort -rn || true
echo "=== python snap ==="
docker cp /tmp/_p0e_canary_snap.py atmr-backend-1:/tmp/_p0e_canary_snap.py
docker exec atmr-backend-1 python /tmp/_p0e_canary_snap.py 2>&1 \
  | grep -vE 'Eventlet|OpenTelemetry|SOCKET|Socket|Security|deprecated|Message queue|Flask|eventlet|strongly|framework|patcher|Sentry|Waiting|Press'
