#!/usr/bin/env bash
set -euo pipefail
BACKUP=$(ls -d /tmp/atmr-p0d-canary-*-backup 2>/dev/null | sort | tail -1)
echo "BACKUP=$BACKUP"
test -n "$BACKUP"
test -f "$BACKUP/atmr-backend-1.persist_with_outbox.py"
test -f "$BACKUP/atmr-backend-1.driver.py"
test -f "$BACKUP/atmr-tracking-kafka-consumer-1.persist_with_outbox.py"

echo "=== restore backend ==="
docker cp "$BACKUP/atmr-backend-1.persist_with_outbox.py" atmr-backend-1:/app/services/tracking/persist_with_outbox.py
docker cp "$BACKUP/atmr-backend-1.driver.py" atmr-backend-1:/app/routes/driver.py
docker exec atmr-backend-1 rm -f /app/services/tracking/location_idempotency.py
docker exec atmr-backend-1 sh -c 'rm -f /app/services/tracking/__pycache__/location_idempotency*.pyc /app/services/tracking/__pycache__/persist_with_outbox*.pyc /app/routes/__pycache__/driver*.pyc' || true

echo "=== restore consumer ==="
docker cp "$BACKUP/atmr-tracking-kafka-consumer-1.persist_with_outbox.py" atmr-tracking-kafka-consumer-1:/app/services/tracking/persist_with_outbox.py
docker exec atmr-tracking-kafka-consumer-1 rm -f /app/services/tracking/location_idempotency.py
docker exec atmr-tracking-kafka-consumer-1 sh -c 'rm -f /app/services/tracking/__pycache__/location_idempotency*.pyc /app/services/tracking/__pycache__/persist_with_outbox*.pyc' || true

echo "=== compile ==="
docker exec atmr-backend-1 python -m py_compile /app/services/tracking/persist_with_outbox.py /app/routes/driver.py
docker exec atmr-tracking-kafka-consumer-1 python -m py_compile /app/services/tracking/persist_with_outbox.py

echo "=== restart ==="
docker restart atmr-backend-1 atmr-tracking-kafka-consumer-1
sleep 12
docker inspect atmr-backend-1 --format 'backend={{.State.Status}} health={{.State.Health.Status}}'
docker inspect atmr-tracking-kafka-consumer-1 --format 'consumer={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'

echo "=== verify ==="
docker exec atmr-backend-1 grep -c compare_persisted_event /app/services/tracking/persist_with_outbox.py || true
docker exec atmr-backend-1 test ! -f /app/services/tracking/location_idempotency.py && echo IDEM_GONE=yes
docker exec atmr-tracking-kafka-consumer-1 grep -c compare_persisted_event /app/services/tracking/persist_with_outbox.py || true
docker exec atmr-tracking-kafka-consumer-1 test ! -f /app/services/tracking/location_idempotency.py && echo CONSUMER_IDEM_GONE=yes
docker exec atmr-backend-1 grep -c 'recorded_at manquant, défaut=ts ou now' /app/routes/driver.py || true
docker inspect atmr-backend-1 --format 'IMAGE={{.Config.Image}}'
wc -l /app/services/tracking/persist_with_outbox.py 2>/dev/null || docker exec atmr-backend-1 wc -l /app/services/tracking/persist_with_outbox.py
echo "ROLLBACK_OK"
