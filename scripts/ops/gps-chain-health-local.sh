#!/usr/bin/env bash
# Santé chaîne GPS locale (Docker + API localhost).
# Usage:
#   bash scripts/ops/gps-chain-health-local.sh
#   DRIVER_ID=7135 bash scripts/ops/gps-chain-health-local.sh
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:5000}"
DRIVER_ID="${DRIVER_ID:-}"
DLQ_WINDOW_MIN="${DLQ_WINDOW_MIN:-10}"
FAIL=0

ok() { printf "[OK] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
fail() { printf "[FAIL] %s\n" "$*"; FAIL=1; }

section() { printf "\n========== %s ==========\n" "$*"; }

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

section "1. Conteneurs Docker (socle tracking)"
REQUIRED=(
  atmr-atmr_api
  atmr-redis-1
  atmr-kafka-broker-1
  atmr-tracking-kafka-consumer-1
  atmr-tracking-processed-fanout-1
)
for c in "${REQUIRED[@]}"; do
  if container_running "$c"; then
    ok "running: $c"
  else
    fail "absent ou arrêté: $c"
  fi
done

section "2. API locale"
if curl -sf "${API_BASE}/health" | grep -q '"status"'; then
  ok "API health ${API_BASE}/health"
else
  fail "API health inaccessible (${API_BASE})"
fi

READY="$(curl -sf "${API_BASE}/api/v1/ready" 2>/dev/null || echo '{}')"
if echo "${READY}" | grep -q '"status":"ready"'; then
  ok "API ready (db+redis)"
else
  warn "API non ready: ${READY}"
fi

section "3. Flags Kafka API"
if container_running atmr-atmr_api; then
  API_ENV="$(docker exec atmr-atmr_api env 2>/dev/null || true)"
  echo "${API_ENV}" | grep -E '^KAFKA_ENABLED=|^TRACKING_INGEST_ASYNC_ENABLED=' || true
  if echo "${API_ENV}" | grep -q '^KAFKA_ENABLED=true'; then
    ok "KAFKA_ENABLED=true sur API"
  else
    warn "KAFKA_ENABLED=false — chemin sync uniquement (merge docker-compose.kafka.dev.yml requis)"
  fi
fi

section "4. Workers tracking (env critique)"
if container_running atmr-tracking-kafka-consumer-1; then
  CE="$(docker exec atmr-tracking-kafka-consumer-1 env 2>/dev/null || true)"
  for var in FLASK_CONFIG APP_ENCRYPTION_KEY_B64 REDIS_URL TRACKING_INGEST_PERSIST_ENABLED; do
    if echo "${CE}" | grep -q "^${var}="; then
      ok "consumer: ${var} défini"
    else
      fail "consumer: ${var} manquant"
    fi
  done
  if echo "${CE}" | grep -q '^FLASK_CONFIG=development'; then
    ok "consumer FLASK_CONFIG=development"
  elif echo "${CE}" | grep -q '^FLASK_CONFIG=production'; then
    if echo "${CE}" | grep -q '^SOCKETIO_CORS_ORIGINS='; then
      ok "consumer production + CORS défini"
    else
      fail "consumer FLASK_CONFIG=production sans SOCKETIO_CORS_ORIGINS"
    fi
  fi
  DLQ_COUNT="$(docker logs atmr-tracking-kafka-consumer-1 --since "${DLQ_WINDOW_MIN}m" 2>&1 | grep -c 'DLQ confirmed' || true)"
  DLQ_COUNT="${DLQ_COUNT//$'\r'/}"
  if [[ "${DLQ_COUNT:-0}" -eq 0 ]]; then
    ok "aucune DLQ consumer (${DLQ_WINDOW_MIN} min)"
  else
    fail "${DLQ_COUNT} DLQ confirmée(s) (${DLQ_WINDOW_MIN} min) — voir data/kafka-dlq/kafka_dlq_events.jsonl"
  fi
fi

if container_running atmr-tracking-processed-fanout-1; then
  FE="$(docker exec atmr-tracking-processed-fanout-1 env 2>/dev/null || true)"
  if echo "${FE}" | grep -q '^TRACKING_PROCESSED_FANOUT_ENABLED=true'; then
    ok "fanout TRACKING_PROCESSED_FANOUT_ENABLED=true"
  else
    fail "fanout TRACKING_PROCESSED_FANOUT_ENABLED absent/false"
  fi
  if echo "${FE}" | grep -q '^REDIS_URL='; then
    ok "fanout REDIS_URL défini (message_queue Socket.IO)"
  else
    fail "fanout REDIS_URL manquant"
  fi
fi

section "5. Redis position chauffeur"
if [[ -n "${DRIVER_ID}" ]] && container_running atmr-redis-1; then
  LOC="$(docker exec atmr-redis-1 redis-cli HGETALL "driver:${DRIVER_ID}:loc" 2>/dev/null || true)"
  if [[ -z "${LOC}" ]]; then
    warn "driver:${DRIVER_ID}:loc vide"
  else
    echo "${LOC}"
    if echo "${LOC}" | grep -q '^recorded_at$'; then
      ok "driver:${DRIVER_ID}:loc présent"
    fi
  fi
else
  warn "DRIVER_ID non défini — skip Redis (ex: DRIVER_ID=7135)"
fi

section "6. Métriques GPS (instance API)"
METRICS="$(curl -sf "${API_BASE}/api/v1/prometheus/metrics" 2>/dev/null || true)"
if [[ -n "${METRICS}" ]]; then
  recv="$(echo "${METRICS}" | grep -E '^driver_location_received_total' | awk '{s+=$2} END {print s+0}')"
  persist="$(echo "${METRICS}" | grep -E '^tracking_kafka_persist_total' | awk '{s+=$2} END {print s+0}')"
  fanout="$(echo "${METRICS}" | grep -E '^tracking_fanout_emit_total' | awk '{s+=$2} END {print s+0}')"
  echo "driver_location_received_total (sum): ${recv}"
  echo "tracking_kafka_persist_total (sum): ${persist}"
  echo "tracking_fanout_emit_total (sum): ${fanout}"
  ok "métriques Prometheus accessibles"
else
  warn "métriques Prometheus indisponibles"
fi

section "Verdict"
if [[ "${FAIL}" -eq 0 ]]; then
  ok "Chaîne locale GPS — configuration saine"
  echo "Stack Kafka dev: docker compose -f docker-compose.yml -f docker-compose.kafka.dev.yml up -d"
else
  fail "Chaîne locale GPS — problèmes détectés (voir [FAIL] ci-dessus)"
  exit 1
fi
