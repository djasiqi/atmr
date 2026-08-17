#!/usr/bin/env bash
# P0-E Phase 2 — preflight + enable PG_FIRST canary (flag ON)
set -euo pipefail
cd /srv/atmr

echo "=== PREFLIGHT ==="
test "$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)" = "healthy"
test "$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)" = "healthy"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX)" = "true"
test "$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo false)" = "false"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo false)" = "false"
docker exec atmr-backend-1 test -f /app/services/tracking/location_candidate.py
echo PREFLIGHT_PASS

echo "=== ENABLE PG_FIRST=true ==="
cp -a .env.production ".env.production.bak-p0e-p5b-phase2-$(date +%Y%m%d%H%M%S)"
if grep -q "^TRACKING_PG_FIRST_CANONICAL_ENABLED=" .env.production; then
  sed -i "s|^TRACKING_PG_FIRST_CANONICAL_ENABLED=.*|TRACKING_PG_FIRST_CANONICAL_ENABLED=true|" .env.production
else
  echo "TRACKING_PG_FIRST_CANONICAL_ENABLED=true" >> .env.production
fi
# keep outbox true
if grep -q "^TRACKING_PERSIST_WITH_OUTBOX=" .env.production; then
  sed -i "s|^TRACKING_PERSIST_WITH_OUTBOX=.*|TRACKING_PERSIST_WITH_OUTBOX=true|" .env.production
else
  echo "TRACKING_PERSIST_WITH_OUTBOX=true" >> .env.production
fi
grep -E "^(TRACKING_PG_FIRST_CANONICAL_ENABLED|TRACKING_PERSIST_WITH_OUTBOX|DOCKER_TAG)=" .env.production

echo "=== RECREATE backend + consumer ==="
docker compose -f docker-compose.production.yml --env-file .env.production \
  up -d --no-deps --force-recreate backend
docker compose -f docker-compose.production.yml --profile kafka --env-file .env.production \
  up -d --no-deps --force-recreate tracking-kafka-consumer

for i in $(seq 1 60); do
  BH=$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1 2>/dev/null || echo missing)
  CH=$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1 2>/dev/null || echo missing)
  echo "t=${i} backend=${BH} consumer=${CH}"
  if [ "${BH}" = "healthy" ] && [ "${CH}" = "healthy" ]; then
    break
  fi
  if [ "${BH}" = "unhealthy" ] || [ "${CH}" = "unhealthy" ]; then
    echo STOP_UNHEALTHY
    exit 1
  fi
  sleep 3
done
test "$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)" = "healthy"
test "$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)" = "healthy"

echo -n "backend_pg_first="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
echo -n "consumer_pg_first="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
echo -n "consumer_outbox="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX
test "$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)" = "true"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)" = "true"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX)" = "true"
echo PHASE2_ENABLE_PASS
