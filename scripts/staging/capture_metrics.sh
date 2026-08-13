#!/usr/bin/env bash
# Capture les 4 familles de preuves depuis Prometheus staging (localhost:19090).
set -euo pipefail

PROM="${STAGING_PROM_URL:-http://127.0.0.1:19090}"
OUT="${1:-staging/output/metrics-$(date -u +%Y%m%dT%H%M%SZ).txt}"
mkdir -p "$(dirname "$OUT")"

query() {
  local q="$1"
  curl -fsS --get "$PROM/api/v1/query" --data-urlencode "query=$q" || echo "{\"error\":\"query_failed\",\"q\":$q}"
}

{
  echo "# staging metrics snapshot $(date -u +%FT%TZ)"
  echo "## firewall"
  query 'sum by (reason,transport,would_block,enforced,mode) (tracking_mission_firewall_total)'
  echo "## watchdog"
  query 'sum by (reason) (tracking_stale_fix_watchdog_kick_total)'
  echo "## kafka lag"
  query 'tracking_kafka_consumer_lag'
  echo "## postgres activity"
  query 'pg_stat_activity_count'
  echo "## redis"
  query 'redis_connected_clients'
} | tee "$OUT"

echo "écrit $OUT"
