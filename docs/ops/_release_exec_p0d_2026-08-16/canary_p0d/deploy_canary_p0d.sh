#!/usr/bin/env bash
# Canary P0-D : hot-patch fichiers dans backend + tracking-kafka-consumer (pas d'image rebuild).
set -euo pipefail
STAMP="p0d-canary-$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="/tmp/atmr-${STAMP}-backup"
mkdir -p "$BACKUP"

for c in atmr-backend-1 atmr-tracking-kafka-consumer-1; do
  echo "=== backup $c ==="
  docker cp "$c:/app/services/tracking/persist_with_outbox.py" "$BACKUP/${c}.persist_with_outbox.py"
  docker cp "$c:/app/routes/driver.py" "$BACKUP/${c}.driver.py" || true
done

echo "=== install location_idempotency.py ==="
docker cp /tmp/p0d_canary/location_idempotency.py atmr-backend-1:/app/services/tracking/location_idempotency.py
docker cp /tmp/p0d_canary/location_idempotency.py atmr-tracking-kafka-consumer-1:/app/services/tracking/location_idempotency.py

echo "=== install persist_with_outbox.py ==="
docker cp /tmp/p0d_canary/persist_with_outbox.py atmr-backend-1:/app/services/tracking/persist_with_outbox.py
docker cp /tmp/p0d_canary/persist_with_outbox.py atmr-tracking-kafka-consumer-1:/app/services/tracking/persist_with_outbox.py

echo "=== install driver.py (API only) ==="
docker cp /tmp/p0d_canary/driver.py atmr-backend-1:/app/routes/driver.py

echo "=== compile check ==="
docker exec atmr-backend-1 python -m py_compile \
  /app/services/tracking/location_idempotency.py \
  /app/services/tracking/persist_with_outbox.py \
  /app/routes/driver.py
docker exec atmr-tracking-kafka-consumer-1 python -m py_compile \
  /app/services/tracking/location_idempotency.py \
  /app/services/tracking/persist_with_outbox.py

echo "=== restart services ==="
docker restart atmr-backend-1 atmr-tracking-kafka-consumer-1
sleep 8

echo "=== health ==="
docker inspect atmr-backend-1 --format 'backend={{.State.Status}} health={{.State.Health.Status}}'
docker inspect atmr-tracking-kafka-consumer-1 --format 'consumer={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'

echo "=== verify module ==="
docker exec atmr-backend-1 python -c "from services.tracking.location_idempotency import DuplicateDecision, resolve_client_recorded_at; print('API', DuplicateDecision.DUPLICATE_LEGACY_EQUIVALENT.value, resolve_client_recorded_at({'timestamp':'2026-08-16T13:57:08.992Z'}))"
docker exec atmr-tracking-kafka-consumer-1 python -c "from services.tracking.location_idempotency import DuplicateDecision; print('CONSUMER', DuplicateDecision.DUPLICATE_LEGACY_EQUIVALENT.value)"
docker exec atmr-backend-1 python -c "import services.tracking.persist_with_outbox as p; import inspect; src=inspect.getsource(p.persist_location_event_with_outbox); print('HAS_COMPARE', 'compare_persisted_event' in src); print('HAS_CAPTURE_COL', 'capture_id' in open('/app/services/tracking/persist_with_outbox.py').read().split('INSERT INTO tracking_ingest')[1][:400])"

echo "BACKUP=$BACKUP"
echo "STAMP=$STAMP"
echo "CANARY_DEPLOY_OK"
