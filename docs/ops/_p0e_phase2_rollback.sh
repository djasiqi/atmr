#!/usr/bin/env bash
# P0-E Phase 2 — rollback canary : PG_FIRST=false (pas de rollback migration)
set -euo pipefail
cd /srv/atmr

echo "=== ROLLBACK PG_FIRST=false ==="
cp -a .env.production ".env.production.bak-p0e-p5b-phase2-rollback-$(date +%Y%m%d%H%M%S)"
if grep -q "^TRACKING_PG_FIRST_CANONICAL_ENABLED=" .env.production; then
  sed -i "s|^TRACKING_PG_FIRST_CANONICAL_ENABLED=.*|TRACKING_PG_FIRST_CANONICAL_ENABLED=false|" .env.production
else
  echo "TRACKING_PG_FIRST_CANONICAL_ENABLED=false" >> .env.production
fi
# garder OUTBOX
if grep -q "^TRACKING_PERSIST_WITH_OUTBOX=" .env.production; then
  sed -i "s|^TRACKING_PERSIST_WITH_OUTBOX=.*|TRACKING_PERSIST_WITH_OUTBOX=true|" .env.production
else
  echo "TRACKING_PERSIST_WITH_OUTBOX=true" >> .env.production
fi
grep -E "^(TRACKING_PG_FIRST_CANONICAL_ENABLED|TRACKING_PERSIST_WITH_OUTBOX|DOCKER_TAG)=" .env.production

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
  sleep 3
done
test "$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)" = "false"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)" = "false"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX)" = "true"
echo PHASE2_ROLLBACK_PASS
