#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
nohup bash /srv/atmr/scripts/ops-p0e-kafka-align-logwatch.sh > /tmp/p0e-logwatch-15m.txt 2>&1 &
echo "LOGWATCH_PID=$!"
grep -E '^(TRACKING_INGEST_ASYNC_ENABLED|KAFKA_ENABLED|TRACKING_INGEST_PERSIST_ENABLED)=' .env.production || true
docker inspect atmr-backend-1 --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(TRACKING_INGEST_ASYNC_ENABLED|KAFKA_ENABLED)=' || true
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T postgres psql -U atmr -d atmr -c "
SELECT driver_id, location_event_id, sequence_id, source, received_at
FROM tracking_ingest_events
WHERE received_at > NOW() - INTERVAL '10 minutes'
ORDER BY received_at DESC
LIMIT 15;
"
