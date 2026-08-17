#!/usr/bin/env bash
set -euo pipefail
echo "=== PUT status codes 5m ==="
docker logs atmr-backend-1 --since 5m 2>&1 \
  | grep 'PUT /api/v1/driver/me/location' \
  | grep -v Darwin \
  | awk '{print $9}' \
  | sort | uniq -c | sort -rn || true

echo "=== PUT sample lines ==="
docker logs atmr-backend-1 --since 3m 2>&1 \
  | grep 'PUT /api/v1/driver/me/location' \
  | grep -v Darwin \
  | tail -12 || true

echo "=== app logs location accept ==="
docker logs atmr-backend-1 --since 5m 2>&1 \
  | grep -E 'accepted_async|ingested_non_persisted|accepted_canonical|409|session_conflict|ownership' \
  | grep -v EventletDeprecation \
  | tail -40 || true

echo "=== consumer p5b / errors ==="
docker logs atmr-tracking-kafka-consumer-1 --since 5m 2>&1 \
  | grep -E 'p5b_promote|promotion canonical|Traceback|UndefinedColumn|IntegrityError' \
  | grep -v EventletDeprecation \
  | tail -30 || true

echo "=== TB counts 5m ==="
echo -n "backend_tb="
docker logs atmr-backend-1 --since 5m 2>&1 | grep -c Traceback || true
echo -n "consumer_tb="
docker logs atmr-tracking-kafka-consumer-1 --since 5m 2>&1 | grep -c Traceback || true
