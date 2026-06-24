#!/usr/bin/env bash
# Verification test terrain GPS (P0-2) — lecture seule
# Usage: DRIVER_ID=6858 T0='2026-06-24 22:00:00+00' ./scripts/verify-tracking-field-test.sh
set -euo pipefail
cd /srv/atmr

DRIVER_ID="${DRIVER_ID:?DRIVER_ID requis}"
T0="${T0:-$(date -u +%Y-%m-%d\ %H:%M:%S+00)}"

echo "===== Test terrain GPS — driver_id=$DRIVER_ID depuis T0=$T0 ====="
echo ""

echo "--- driver.last_position_update ---"
docker exec atmr-postgres psql -U atmr -d atmr -c \
  "SELECT id, latitude, longitude, last_position_update FROM driver WHERE id = ${DRIVER_ID};"

echo "--- trip_tracking (nouvelles lignes depuis T0) ---"
docker exec atmr-postgres psql -U atmr -d atmr -c \
  "SELECT count(*) AS new_rows, max(\"timestamp\") AS last_ts FROM trip_tracking WHERE driver_id = ${DRIVER_ID} AND \"timestamp\" > '${T0}';"

echo "--- driver_device_health_events (depuis T0) ---"
docker exec atmr-postgres psql -U atmr -d atmr -c \
  "SELECT count(*) AS new_rows, max(recorded_at) AS last_recorded FROM driver_device_health_events WHERE driver_id = ${DRIVER_ID} AND recorded_at > '${T0}';"

echo "--- Metriques Prometheus (rates 10m) ---"
docker exec atmr-prometheus wget -qO- \
  'http://localhost:9090/api/v1/query?query=sum(rate(tracking_kafka_persist_total%5B10m%5D))' 2>/dev/null | head -c 400
echo ""
docker exec atmr-prometheus wget -qO- \
  'http://localhost:9090/api/v1/query?query=sum(rate(tracking_http_accepted_async_total%5B10m%5D))' 2>/dev/null | head -c 400
echo ""

echo "--- Kafka lag fanout ---"
docker exec atmr-kafka-broker-1 kafka-consumer-groups \
  --bootstrap-server kafka-broker-1:29092 \
  --describe --group tracking-processed-fanout-group 2>/dev/null | grep processed.v2 | awk '{print $3,$4,$5,$6}'

echo ""
echo "===== Checklist carte (manuel) ====="
echo "[ ] Chauffeur visible web dispatch"
echo "[ ] Chauffeur visible mobile entreprise iPhone"
echo "[ ] Delai apparition marqueur < 10 s"
echo "[ ] Aucune disparition icone pendant 10 min"
echo ""
echo "===== fanout errors (leader_epoch) ====="
docker logs --tail 500 atmr-tracking-processed-fanout-1 2>&1 | grep -c leader_epoch || echo "0"
