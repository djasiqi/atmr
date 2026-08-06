#!/usr/bin/env bash
# Phase 0 — capture preuves GPS (lecture seule). À exécuter sur l'hôte prod.
# Usage: ./scripts/ops-gps-phase0-capture.sh [OUT_DIR]
set -euo pipefail

OUT_DIR="${1:-reports/gps-phase0-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT_DIR"
COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.production.yml}"
ENV_FILE="${ENV_FILE:-.env.production}"

echo "== Phase 0 capture → $OUT_DIR =="

{
  echo "# GPS Phase 0 capture $(date -u -Iseconds)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
} >"$OUT_DIR/meta.txt"

# Conteneurs tracking / kafka
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' \
  | tee "$OUT_DIR/docker-ps.txt" || true

# Exit reasons fanout / DLQ
for c in atmr-tracking-processed-fanout tracking-processed-fanout \
  atmr-kafka-dlq-consumer kafka-dlq-consumer tracking-kafka-consumer; do
  if docker inspect "$c" >/dev/null 2>&1; then
    docker inspect "$c" --format '{{.Name}} State={{.State.Status}} Exit={{.State.ExitCode}} Finished={{.State.FinishedAt}} Error={{.State.Error}}' \
      | tee -a "$OUT_DIR/container-exits.txt"
  fi
done

# Env effectifs (sans secrets)
if docker compose $COMPOSE_FILES --env-file "$ENV_FILE" ps backend >/dev/null 2>&1; then
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T backend \
    sh -c 'env | grep -E "^(RATELIMIT_|TRACKING_|KAFKA_|HTTP_DRIVER_LOCATION_|WS_DRIVER_LOCATION_)" | sort' \
    >"$OUT_DIR/backend-env-tracking.txt" 2>/dev/null || true
fi

# Redis : buckets Flask-Limiter GPS + canonical + http_rate
if docker compose $COMPOSE_FILES --env-file "$ENV_FILE" ps redis >/dev/null 2>&1; then
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T redis \
    sh -c 'redis-cli --scan --pattern "*driver_driver_location*" | head -50' \
    >"$OUT_DIR/redis-flask-limiter-gps-keys.txt" 2>/dev/null || true
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T redis \
    sh -c 'redis-cli --scan --pattern "http_rate:driver_location*" | head -20; echo ---; redis-cli --scan --pattern "driver:*:loc:canonical" | wc -l' \
    >"$OUT_DIR/redis-http-rate-and-canonical.txt" 2>/dev/null || true
fi

# Access log 401/429 location (dernières 24h si log dispo)
ACCESS_LOG="${ACCESS_LOG:-/var/log/nginx/access.log}"
if [[ -f "$ACCESS_LOG" ]]; then
  grep -E "driver/me/location|/api/v1/driver/me/location" "$ACCESS_LOG" \
    | awk '{print $NF}' | sort | uniq -c | sort -rn \
    >"$OUT_DIR/access-location-status-counts.txt" || true
  grep -E "driver/me/location" "$ACCESS_LOG" | grep " 429 " | wc -l \
    >"$OUT_DIR/access-location-429-count.txt" || true
  grep -E "driver/me/location" "$ACCESS_LOG" | grep " 401 " | wc -l \
    >"$OUT_DIR/access-location-401-count.txt" || true
fi

# Kafka offsets (si kafka CLI dispo)
if command -v kafka-consumer-groups.sh >/dev/null 2>&1 || docker ps --format '{{.Names}}' | grep -q kafka; then
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T kafka \
    kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --all-groups \
    >"$OUT_DIR/kafka-consumer-groups.txt" 2>/dev/null || true
fi

# Métriques consumer
curl -sf "http://127.0.0.1:9115/metrics" 2>/dev/null | grep -E "tracking_|gps_" \
  >"$OUT_DIR/consumer-metrics-9115.txt" || echo "metrics:9115 unavailable" >"$OUT_DIR/consumer-metrics-9115.txt"

# Disque DLQ
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T kafka-dlq-consumer \
  sh -c 'df -h /app/data 2>/dev/null; ls -lah /app/data 2>/dev/null; wc -l /app/data/kafka_dlq_events.jsonl 2>/dev/null' \
  >"$OUT_DIR/dlq-disk.txt" 2>/dev/null || echo "dlq container unavailable" >"$OUT_DIR/dlq-disk.txt"

echo "Capture terminée : $OUT_DIR"
ls -la "$OUT_DIR"
