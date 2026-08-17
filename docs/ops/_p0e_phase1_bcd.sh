#!/usr/bin/env bash
# P0-E Phase 1 — B/C/D : tag d5694d8 + PG_FIRST=false + recreate backend/consumer
set -euo pipefail
cd /srv/atmr

TAG="sha-d5694d8e7cec"
FULL="d5694d8e7cec190978098db6eb20f242226784a8"
EXPECTED_DIGEST="sha256:5e58f61bf3393ee3883dff55dd04affe688f7bce71021896fa922d633ef2af00"

echo "=== B/C: backup + env patch ==="
cp -a .env.production ".env.production.bak-p0e-p5b-$(date +%Y%m%d%H%M%S)"

set_kv() {
  local key="$1"
  local val="$2"
  if grep -q "^${key}=" .env.production; then
    sed -i "s|^${key}=.*|${key}=${val}|" .env.production
  else
    echo "${key}=${val}" >> .env.production
  fi
}

set_kv DOCKER_TAG "${TAG}"
set_kv GIT_SHA "${FULL}"
set_kv SENTRY_RELEASE "${FULL}"
set_kv TRACKING_PG_FIRST_CANONICAL_ENABLED "false"

# Prefer tag path: comment out BACKEND_IMAGE_REF if present
if grep -q "^BACKEND_IMAGE_REF=" .env.production; then
  sed -i "s|^BACKEND_IMAGE_REF=|#BACKEND_IMAGE_REF=|" .env.production
fi

echo "VERIFY_ENV:"
grep -E "^(DOCKER_TAG|GIT_SHA|SENTRY_RELEASE|TRACKING_PG_FIRST_CANONICAL_ENABLED)=" .env.production
grep -E "^BACKEND_IMAGE_REF=" .env.production || echo "BACKEND_IMAGE_REF=absent_ok"

echo "=== ensure image ==="
docker image inspect "djasiqi/atmr-backend:${TAG}" >/dev/null
# Confirm local digest matches expected RepoDigest if available
RD=$(docker image inspect "djasiqi/atmr-backend:${TAG}" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)
echo "RepoDigest=${RD}"
echo "${RD}" | grep -Fq "${EXPECTED_DIGEST}" && echo DIGEST_OK=yes || echo DIGEST_WARN=check_manually

echo "=== D: recreate backend ==="
docker compose -f docker-compose.production.yml --env-file .env.production \
  up -d --no-deps --force-recreate backend

echo "=== D: recreate tracking-kafka-consumer ==="
docker compose -f docker-compose.production.yml --profile kafka --env-file .env.production \
  up -d --no-deps --force-recreate tracking-kafka-consumer

echo "=== wait health ==="
BH="starting"
CH="starting"
for i in $(seq 1 60); do
  BH=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' atmr-backend-1 2>/dev/null || echo missing)
  CH=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' atmr-tracking-kafka-consumer-1 2>/dev/null || echo missing)
  echo "t=${i} backend=${BH} consumer=${CH}"
  if [ "${BH}" = "healthy" ] && { [ "${CH}" = "healthy" ] || [ "${CH}" = "running" ]; }; then
    break
  fi
  if [ "${BH}" = "unhealthy" ] || [ "${CH}" = "unhealthy" ]; then
    echo "STOP unhealthy"
    docker compose -f docker-compose.production.yml logs backend --tail=40 || true
    docker compose -f docker-compose.production.yml --profile kafka logs tracking-kafka-consumer --tail=40 || true
    exit 1
  fi
  sleep 3
done

if [ "${BH}" != "healthy" ]; then
  echo "STOP backend not healthy: ${BH}"
  exit 1
fi

echo "=== post-recreate checks ==="
docker inspect atmr-backend-1 --format 'backend_image={{.Config.Image}}'
docker inspect atmr-tracking-kafka-consumer-1 --format 'consumer_image={{.Config.Image}}'
echo -n "backend_pg_first="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo UNSET
echo -n "consumer_pg_first="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo UNSET
echo -n "backend_git="
docker exec atmr-backend-1 printenv GIT_SHA || true
docker exec atmr-backend-1 test -f /app/services/tracking/location_candidate.py && echo BACKEND_HAS_LC=yes || { echo BACKEND_HAS_LC=no; exit 1; }
docker exec atmr-tracking-kafka-consumer-1 test -f /app/services/tracking/location_candidate.py && echo CONSUMER_HAS_LC=yes || { echo CONSUMER_HAS_LC=no; exit 1; }
MC=$(docker exec atmr-backend-1 grep -c _maybe_promote_after_pg /app/services/tracking/persist_kafka_outbox.py || true)
echo "maybe_promote_count=${MC}"
test "${MC}" -ge 1

# Flag must be false or unset (treat empty as ok if code defaults false; we set false)
PF=$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo false)
if [ "${PF}" != "false" ] && [ "${PF}" != "0" ]; then
  echo "STOP PG_FIRST not false: ${PF}"
  exit 1
fi
echo "BCD_PASS"
