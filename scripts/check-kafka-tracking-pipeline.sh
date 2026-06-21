#!/usr/bin/env bash
# Checklist automatisée partielle — pipeline tracking Kafka (T1–T13).
# Usage : ATMR_DEPLOY_ROOT=/srv/atmr ./scripts/check-kafka-tracking-pipeline.sh
# Prérequis : stack Kafka ON, backend healthy, jq optionnel pour latence Prometheus.
set -uo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"
export ATMR_ENV_FILE="${ENV_FILE}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

PASS=0
FAIL=0
WARN=0

ok() { echo "[OK] $*"; PASS=$((PASS + 1)); }
ko() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }

echo "=== Checklist pipeline tracking Kafka ==="
echo "Compose: ${KAFKA_COMPOSE_FILE}"

# T3 — lag consumers
first="$(kafka_bootstrap_first)"
lag_max="$(kafka_docker_compose exec -T kafka-broker-1 kafka-consumer-groups.sh \
  --bootstrap-server "${first}" --all-groups --describe 2>/dev/null \
  | awk 'NR>1 && $6 ~ /^[0-9]+$/ {print $6}' | sort -n | tail -1 || true)"
if [[ -z "${lag_max}" ]] || [[ "${lag_max}" == "0" ]]; then
  ok "T3 consumer lag = 0 (ou groups vides)"
else
  warn "T3 lag max=${lag_max} — vérifier manuellement"
fi

# T9 — broker health
kafka_refresh_broker_containers
healthy=0
for c in "${KAFKA_BROKER_CONTAINERS[@]}"; do
  st="$(docker inspect "${c}" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' 2>/dev/null || echo missing)"
  if [[ "${st}" == "healthy" ]] || [[ "${st}" == "running" ]]; then
    healthy=$((healthy + 1))
  fi
done
if ((healthy >= 1)); then
  ok "T9 broker(s) healthy/running (${healthy}/${#KAFKA_BROKER_CONTAINERS[@]})"
else
  ko "T9 aucun broker healthy"
fi

# T10/T11 — RAM stack Kafka (documenter, pas de seuil automatique strict)
if docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' 2>/dev/null | grep -E 'kafka|zookeeper' >/tmp/kafka-stats.$$.txt; then
  ok "T10/T11 docker stats Kafka capturé ($(wc -l < /tmp/kafka-stats.$$.txt) lignes)"
  cat /tmp/kafka-stats.$$.txt
  rm -f /tmp/kafka-stats.$$.txt
else
  warn "T10/T11 docker stats indisponible"
fi

# Smoke = proxy T1 pipeline broker
if kafka_check_functional_smoke; then
  ok "T1 smoke producer→broker→consumer"
else
  ko "T1 smoke Kafka KO"
fi

# T13 — latence E2E (consumer ingest + Prometheus)
metrics_url="${PROMETHEUS_METRICS_URL:-http://localhost:9090}"
consumer_metrics_port="${TRACKING_CONSUMER_METRICS_PORT:-9115}"
consumer_e2e_count="$(
  docker exec atmr-backend-1 curl -sf --max-time 5 \
    "http://tracking-kafka-consumer:${consumer_metrics_port}/metrics" 2>/dev/null \
    | grep -E '^tracking_kafka_e2e_latency_seconds_count' \
    | awk '{s+=$2} END {print s+0}' || true
)"
if [[ -n "${consumer_e2e_count}" && "${consumer_e2e_count}" != "0" ]]; then
  ok "T13 consumer tracking_kafka_e2e count=${consumer_e2e_count}"
else
  warn "T13 consumer metrics absentes (deploy image + recreate tracking-kafka-consumer + scrape Prometheus)"
fi
e2e_p95="$(curl -sf "${metrics_url}/api/v1/query" \
  --data-urlencode 'query=histogram_quantile(0.95, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))' 2>/dev/null \
  | grep -oE '"value":\[[^]]+\]' | tail -1 | grep -oE '[0-9.]+$' || true)"
if [[ -n "${e2e_p95}" && "${e2e_p95}" != "NaN" ]]; then
  if awk -v v="${e2e_p95}" 'BEGIN{exit (v+0 > 2)}'; then
    ok "T13 tracking_kafka_e2e P95=${e2e_p95}s (< 2s)"
  else
    ko "T13 tracking_kafka_e2e P95=${e2e_p95}s (> 2s)"
  fi
elif [[ -n "${consumer_e2e_count}" && "${consumer_e2e_count}" != "0" ]]; then
  warn "T13 P95 Prometheus indisponible (attendre 1–2 scrapes après activité GPS)"
else
  warn "T13 latence Prometheus non mesurée (PROMETHEUS_METRICS_URL ou métriques absentes)"
fi

echo ""
echo "Résumé : OK=${PASS} FAIL=${FAIL} WARN=${WARN}"
echo "Tests manuels requis : T2,T4,T5,T6,T7,T8,T12,T13 (10 envois carte)"
echo "Voir docs/ops/kafka-optimization-lirie.md"

if ((FAIL > 0)); then
  exit 1
fi
exit 0
