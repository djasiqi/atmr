#!/usr/bin/env bash
# Soft-gate DLQ : tolère UNIQUEMENT event_id_payload_conflict post-persist.
# STOP si autre reason, ou conflit sur eid jamais persisté, ou PG gelé.
set -euo pipefail
DRIVER_ID="${P0E_DRIVER_ID:-20135}"
WINDOW="${P0E_DLQ_WINDOW:-3m}"
SAMPLE_N="${P0E_DLQ_SAMPLE:-8}"

echo "=== SOFT GATE DLQ (${WINDOW}) ==="
echo -n "PG_FIRST="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || true
echo -n "OUTBOX="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX || true
BH=$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)
CH=$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)
echo "backend=${BH} consumer=${CH}"
test "${BH}" = "healthy"
test "${CH}" = "healthy"

TB=$(docker logs atmr-tracking-kafka-consumer-1 --since "${WINDOW}" 2>&1 | grep -c Traceback || true)
echo "consumer_traceback_${WINDOW}=${TB}"
test "${TB}" = "0"

# Counts by DLQ type
echo "=== DLQ types ${WINDOW} ==="
docker logs atmr-tracking-kafka-consumer-1 --since "${WINDOW}" 2>&1 \
  | grep 'DLQ confirmed' \
  | sed -n 's/.*type=\([^ ]*\).*/\1/p' \
  | sort | uniq -c | sort -rn || true

OTHER=$(docker logs atmr-tracking-kafka-consumer-1 --since "${WINDOW}" 2>&1 \
  | grep 'DLQ confirmed' \
  | grep -v 'type=event_id_payload_conflict' \
  | wc -l | tr -d ' ' || true)
OTHER="${OTHER:-0}"
echo "DLQ_other_reasons=${OTHER}"
if [ "${OTHER}" != "0" ]; then
  echo "STOP DLQ reason other than event_id_payload_conflict"
  exit 1
fi

# Sample recent conflicts and verify each eid already in PG (via backend python)
docker cp /tmp/_p0e_soft_dlq_check.py atmr-backend-1:/tmp/_p0e_soft_dlq_check.py
docker cp /tmp/_p0e_phase2_dlq_sample.py atmr-tracking-kafka-consumer-1:/tmp/_p0e_phase2_dlq_sample.py
SAMPLES=$(docker exec atmr-tracking-kafka-consumer-1 \
  python /tmp/_p0e_phase2_dlq_sample.py 2>/dev/null | grep location_event_id= || true)
echo "DLQ_SAMPLES"
echo "${SAMPLES}"

EIDS=$(echo "${SAMPLES}" | sed -n 's/.*location_event_id=//p' | tr '\n' ',' || true)
docker exec \
  -e "P0E_DRIVER_ID=${DRIVER_ID}" \
  -e "P0E_DLQ_EIDS=${EIDS}" \
  atmr-backend-1 python /tmp/_p0e_soft_dlq_check.py

echo "SOFT_GATE_DLQ_PASS"
