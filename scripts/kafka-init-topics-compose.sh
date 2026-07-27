#!/usr/bin/env bash
set -euo pipefail

# Crée les topics Kafka depuis le conteneur kafka-broker-1 (binaires Confluent inclus).
#
# Usage (répertoire = racine déploiement) :
#   ./scripts/kafka-init-topics-compose.sh
#
# Variables : ATMR_DEPLOY_ROOT, KAFKA_COMPOSE_FILE, BOOTSTRAP_SERVERS,
#   KAFKA_TOPIC_REPLICATION_FACTOR, KAFKA_DEFAULT_PARTITIONS, KAFKA_CREATE_INACTIVE_TOPICS, etc.

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

KAFKA_COMPOSE_FILE="${KAFKA_COMPOSE_FILE:-docker-compose.kafka.yml}"
KAFKA_NETWORK_FILE="${KAFKA_NETWORK_FILE:-}"

if [[ -z "${KAFKA_NETWORK_FILE}" ]]; then
  if [[ "${KAFKA_COMPOSE_FILE}" == *single* ]]; then
    KAFKA_NETWORK_FILE="docker-compose.kafka.atmr-network.single.yml"
  else
    KAFKA_NETWORK_FILE="docker-compose.kafka.atmr-network.yml"
  fi
fi

for f in docker-compose.production.yml "${KAFKA_COMPOSE_FILE}" "${KAFKA_NETWORK_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Fichier manquant : ${ROOT}/${f}" >&2
    exit 1
  fi
done

BOOTSTRAP_SERVERS="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
# Prod 3 brokers : RF=3 / minISR=2 (plan Phase 0B). Dev mono-broker : surcharger RF=1.
REPLICATION_FACTOR="${KAFKA_TOPIC_REPLICATION_FACTOR:-${REPLICATION_FACTOR:-3}}"

if [[ -n "${KAFKA_MIN_INSYNC_REPLICAS:-}" ]] || [[ -n "${MIN_INSYNC_REPLICAS:-}" ]]; then
  MIN_INSYNC_REPLICAS="${KAFKA_MIN_INSYNC_REPLICAS:-${MIN_INSYNC_REPLICAS:-2}}"
else
  if [[ "${REPLICATION_FACTOR}" -ge 3 ]]; then
    MIN_INSYNC_REPLICAS=2
  else
    MIN_INSYNC_REPLICAS=1
  fi
fi

if [[ "${REPLICATION_FACTOR}" -ge 3 ]] && [[ "${MIN_INSYNC_REPLICAS}" -lt 2 ]]; then
  echo "ERREUR : RF=${REPLICATION_FACTOR} exige min.insync.replicas>=2 (reçu ${MIN_INSYNC_REPLICAS})" >&2
  exit 1
fi

echo "Kafka topics init : RF=${REPLICATION_FACTOR} minISR=${MIN_INSYNC_REPLICAS}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/kafka_topics_init.sh
source "${SCRIPT_DIR}/lib/kafka_topics_init.sh"

compose_exec() {
  docker compose \
    -f docker-compose.production.yml \
    -f "${KAFKA_COMPOSE_FILE}" \
    -f "${KAFKA_NETWORK_FILE}" \
    exec -T kafka-broker-1 "$@"
}

for _ in $(seq 1 60); do
  if compose_exec kafka-broker-api-versions --bootstrap-server "${BOOTSTRAP_SERVERS}" >/dev/null 2>&1; then
    break
  fi
  echo "En attente du broker Kafka (${BOOTSTRAP_SERVERS})…"
  sleep 2
done

if ! compose_exec kafka-broker-api-versions --bootstrap-server "${BOOTSTRAP_SERVERS}" >/dev/null 2>&1; then
  echo "Échec : impossible de contacter ${BOOTSTRAP_SERVERS} après le délai." >&2
  exit 1
fi

create_topic() {
  local topic="$1"
  local partitions="$2"
  local retention_ms="$3"
  local cleanup_policy="$4"

  compose_exec kafka-topics --bootstrap-server "${BOOTSTRAP_SERVERS}" \
    --create \
    --if-not-exists \
    --topic "${topic}" \
    --partitions "${partitions}" \
    --replication-factor "${REPLICATION_FACTOR}"

  compose_exec kafka-configs --bootstrap-server "${BOOTSTRAP_SERVERS}" \
    --alter \
    --entity-type topics \
    --entity-name "${topic}" \
    --add-config "min.insync.replicas=${MIN_INSYNC_REPLICAS},retention.ms=${retention_ms},cleanup.policy=${cleanup_policy}"
}

kafka_topics_create_all

echo "Kafka topics OK — bootstrap=${BOOTSTRAP_SERVERS} replication=${REPLICATION_FACTOR}"
