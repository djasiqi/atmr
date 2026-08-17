#!/usr/bin/env bash
# Gate pré-Phase2 — tourne sur l'hôte (DLQ via docker logs)
set -euo pipefail
DRIVER_ID="${P0E_DRIVER_ID:-20135}"
BASE_ID="${P0E_BASE_DLE_ID:-5903}"
OLD_SESSION="${P0E_OLD_SESSION:-trk_sess_1786965149557_7lkzgzna}"
MIN_NEW="${P0E_MIN_NEW:-3}"
DLQ_WINDOW="${P0E_DLQ_WINDOW:-2m}"

echo "=== FLAGS / HEALTH ==="
echo -n "PG_FIRST="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
echo -n "OUTBOX="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX
BH=$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)
CH=$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)
echo "backend=${BH} consumer=${CH}"
test "${BH}" = "healthy"
test "${CH}" = "healthy"
test "$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)" = "false"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX)" = "true"

echo "=== DLQ conflict ${DLQ_WINDOW} ==="
DLQ_N=$(docker logs atmr-tracking-kafka-consumer-1 --since "${DLQ_WINDOW}" 2>&1 \
  | grep -c event_id_payload_conflict || true)
echo "DLQ_conflict=${DLQ_N}"

echo "=== PUT codes ${DLQ_WINDOW} ==="
docker logs atmr-backend-1 --since "${DLQ_WINDOW}" 2>&1 \
  | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin \
  | awk '{print $9}' | sort | uniq -c | sort -rn || true

echo "=== PYTHON GATE ==="
docker cp /tmp/_p0e_gate_clean_session.py atmr-backend-1:/tmp/_p0e_gate_clean_session.py
docker exec \
  -e "P0E_DRIVER_ID=${DRIVER_ID}" \
  -e "P0E_BASE_DLE_ID=${BASE_ID}" \
  -e "P0E_OLD_SESSION=${OLD_SESSION}" \
  -e "P0E_MIN_NEW=${MIN_NEW}" \
  -e "P0E_DLQ_WINDOW=${DLQ_WINDOW}" \
  -e "P0E_DLQ_COUNT_OVERRIDE=${DLQ_N}" \
  atmr-backend-1 python /tmp/_p0e_gate_clean_session.py
