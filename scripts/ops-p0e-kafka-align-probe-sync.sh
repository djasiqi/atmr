#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
BACKEND_CID="$(docker ps -q --filter name=atmr-backend | head -n1)"
echo "BACKEND_CID=${BACKEND_CID}"
echo "=== access location last 15 ==="
docker logs --since 30m "${BACKEND_CID}" 2>&1 | grep 'PUT /api/v1/driver/me/location' | tail -n 20
echo "=== app logs durability ==="
docker logs --since 30m "${BACKEND_CID}" 2>&1 | grep -E 'persisted_sync|queued_async|ledger_persisted|sync_ledger' | tail -n 40 || echo none
echo "=== redis circuit / heartbeat ==="
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T redis redis-cli GET tracking:consumer:ingest:circuit || true
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T redis redis-cli GET tracking:consumer:ingest:heartbeat || true
echo "=== recent ledger latency recorded vs received ==="
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T postgres psql -U atmr -d atmr -c "
SELECT location_event_id, driver_id, source,
       EXTRACT(EPOCH FROM (received_at - recorded_at)) AS recv_minus_rec_s,
       received_at
FROM tracking_ingest_events
WHERE received_at > NOW() - INTERVAL '15 minutes'
ORDER BY received_at DESC
LIMIT 10;
"
