#!/usr/bin/env bash
# Watch logs 15 min post-recreate (Phase 1c).
set -euo pipefail
cd /srv/atmr
echo "LOG_WATCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
sleep 900
echo "==== LOG_WATCH_15M $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
for svc in tracking-kafka-consumer tracking-outbox-publisher; do
  echo "--- ${svc} ---"
  cid="$(docker ps -q --filter "name=${svc}" | head -n1)"
  docker logs --since 20m "${cid}" 2>&1 \
    | grep -Ei 'UniqueViolation|IntegrityError|Name or service not known|OperationalError|Traceback' \
    | tail -n 50 || echo "OK_no_targeted_errors"
done
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T postgres psql -U atmr -d atmr -Atc \
  "SELECT COUNT(*) FROM tracking_event_outbox WHERE published_at IS NULL;"
docker compose -p atmr --env-file .env.production \
  -f docker-compose.production.yml -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml \
  --profile kafka exec -T kafka-broker-1 \
  kafka-consumer-groups --bootstrap-server kafka-broker-1:29092 \
  --describe --group tracking-ingest-consumer-group 2>&1 | head -n 20 || true
echo "LOG_WATCH_DONE"
