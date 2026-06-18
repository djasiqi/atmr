#!/usr/bin/env bash
set -euo pipefail

# Démarre la stack Kafka et les consumers ATMR profile « kafka » sur atmr-network.
# Voir docs/ops/kafka-optimization-lirie.md pour Phase 1 / Phase 2.
#
# Usage :
#   INIT_TOPICS=1 scripts/deploy-kafka-production.sh
#   KAFKA_COMPOSE_FILE=docker-compose.kafka.single.yml INIT_TOPICS=1 scripts/deploy-kafka-production.sh
#
# Variables :
#   KAFKA_COMPOSE_FILE — défaut docker-compose.kafka.yml ; Phase 2 : docker-compose.kafka.single.yml
#   KAFKA_UI_ENABLED   — si 1, démarre kafka-ui (profile kafka-ui)
#   INIT_TOPICS        — si 1, exécute kafka-init-topics-compose.sh

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"
export ATMR_ENV_FILE="${ENV_FILE}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export KAFKA_COMPOSE_FILE="${KAFKA_COMPOSE_FILE:-docker-compose.kafka.yml}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

kafka_discover_broker_services() {
  kafka_docker_compose config --services 2>/dev/null | grep '^kafka-broker-' || true
}

kafka_discover_zookeeper_services() {
  kafka_docker_compose config --services 2>/dev/null | grep -E '^zookeeper(-[0-9]+)?$' || true
}

KAFKA_BROKER_SERVICES=()
while IFS= read -r svc; do
  [[ -n "${svc}" ]] && KAFKA_BROKER_SERVICES+=("${svc}")
done < <(kafka_discover_broker_services)

KAFKA_ZK_SERVICES=()
while IFS= read -r svc; do
  [[ -n "${svc}" ]] && KAFKA_ZK_SERVICES+=("${svc}")
done < <(kafka_discover_zookeeper_services)

KAFKA_INFRA_SERVICES=("${KAFKA_ZK_SERVICES[@]}" "${KAFKA_BROKER_SERVICES[@]}" redis-failover)

KAFKA_CONSUMER_AND_UI_SERVICES=(
  tracking-kafka-consumer
  tracking-processed-fanout
  kafka-dlq-consumer
)

if [[ "${KAFKA_UI_ENABLED:-0}" == "1" ]]; then
  KAFKA_CONSUMER_AND_UI_SERVICES+=(kafka-ui)
fi

echo "=== Kafka deploy : ${ROOT} (env=${ENV_FILE}, compose=${KAFKA_COMPOSE_FILE}) ==="

echo "--- Phase 1/5 : preflight Kafka ON ---"
PREFLIGHT_OK=1
kafka_check_flags_all_true || PREFLIGHT_OK=0
kafka_check_compose_files || PREFLIGHT_OK=0
kafka_check_atmr_network || PREFLIGHT_OK=0
kafka_check_compose_resolution || PREFLIGHT_OK=0
kafka_check_replication_factors || PREFLIGHT_OK=0

if ((PREFLIGHT_OK == 0)); then
  if [[ "${FORCE:-0}" == "1" ]]; then
    log_force_override
    echo "[WARN] preflight KO mais FORCE=1 — poursuite (bootstrap initial uniquement)." >&2
  else
    echo "Refus : preflight Kafka ON KO. Corriger ${ENV_FILE} ou FORCE=1 (exceptionnel)." >&2
    exit 2
  fi
fi

echo "--- Phase 2/5 : up brokers (${#KAFKA_BROKER_SERVICES[@]} broker(s)) ---"
kafka_docker_compose up -d "${KAFKA_INFRA_SERVICES[@]}" "$@"

echo "--- Phase 3/5 : wait brokers healthy ---"
if ! kafka_wait_brokers_healthy 180; then
  echo "FAIL : brokers non healthy — abandon avant init topics." >&2
  kafka_summary
  exit 3
fi

if [[ "${INIT_TOPICS:-0}" == "1" ]]; then
  echo "--- Phase 4/5 : init topics ---"
  if ! ATMR_DEPLOY_ROOT="${ROOT}" KAFKA_COMPOSE_FILE="${KAFKA_COMPOSE_FILE}" "${SCRIPT_DIR}/kafka-init-topics-compose.sh"; then
    echo "FAIL : init topics — abandon avant up consumers." >&2
    kafka_summary
    exit 3
  fi
else
  echo "--- Phase 4/5 : init topics (skip — INIT_TOPICS!=1) ---"
fi

echo "--- Phase 5a/5 : up consumers (profile kafka) ---"
if [[ "${KAFKA_UI_ENABLED:-0}" == "1" ]]; then
  kafka_docker_compose --profile kafka-ui up -d "${KAFKA_CONSUMER_AND_UI_SERVICES[@]}"
else
  kafka_docker_compose up -d "${KAFKA_CONSUMER_AND_UI_SERVICES[@]}"
fi

echo "--- Phase 5b/5 : validations post-deploy ---"
POST_OK=1
kafka_check_dns_from_atmr_network || POST_OK=0
kafka_check_broker_api || POST_OK=0
kafka_check_topics_exist || POST_OK=0
kafka_check_consumers_running || POST_OK=0
kafka_check_functional_smoke || POST_OK=0

kafka_summary

if ((POST_OK == 0)); then
  echo "FAIL post-deploy : au moins un check a échoué (voir résumé)." >&2
  exit 3
fi

echo "OK Kafka déployé et validé de bout en bout."
echo "Vérifier avec : scripts/check-kafka-production.sh on"
