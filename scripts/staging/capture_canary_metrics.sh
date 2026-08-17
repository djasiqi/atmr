#!/usr/bin/env bash
# Snapshot Prometheus / Kafka pour canary GPS réel (staging isolé).
set -euo pipefail

PROM="${STAGING_PROM_URL:-http://127.0.0.1:19090}"
OUT="${1:-staging/output/canary-metrics-$(date -u +%Y%m%dT%H%M%SZ).txt}"
mkdir -p "$(dirname "$OUT")"

query() {
  local q="$1"
  curl -fsS --get "$PROM/api/v1/query" --data-urlencode "query=$q" \
    || echo "{\"error\":\"query_failed\",\"q\":$q}"
}

{
  echo "# canary GPS metrics $(date -u +%FT%TZ)"
  echo "## kafka lag"
  query 'sum(tracking_kafka_consumer_lag)'
  echo "## kafka dlq"
  query 'sum(tracking_kafka_dlq_messages_total)'
  echo "## kafka dlq force commit"
  query 'sum(tracking_kafka_dlq_force_commit_total)'
  echo "## kafka publish errors"
  query 'sum(tracking_kafka_publish_errors_total)'
  echo "## http 202 async"
  query 'sum(tracking_http_accepted_async_total)'
  echo "## batch rate limited"
  query 'sum(driver_location_batch_rate_limited_total)'
  echo "## firewall observe"
  query 'sum by (reason,would_block,enforced,mode) (tracking_mission_firewall_total)'
  echo "## delivery results"
  query 'sum by (mode,transport,result) (tracking_delivery_result_total)'
  echo "## e2e p95"
  query 'histogram_quantile(0.95, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))'
  echo "## batch latency p95"
  query 'histogram_quantile(0.95, sum(rate(driver_location_batch_latency_seconds_bucket[5m])) by (le))'
} | tee "$OUT"

echo "écrit $OUT"
echo "Rappel : UniqueViolation / crashes / SQLite / false-live = logs mobiles + backend + Sentry, pas uniquement Prom."
