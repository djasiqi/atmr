#!/usr/bin/env bash
set -euo pipefail

# Crée les topics Kafka depuis le conteneur kafka-broker-1 (binaires Confluent inclus).
# Pas besoin d’installer kafka-topics sur l’hôte ; le broker doit être up sur le réseau Compose.
#
# Usage (répertoire = racine déploiement, avec les 3 YAML présents) :
#   ./scripts/kafka-init-topics-compose.sh
#
# Variables optionnelles :
#   ATMR_DEPLOY_ROOT   — défaut : parent de scripts/
#   BOOTSTRAP_SERVERS  — défaut : kafka-broker-1:29092
#   KAFKA_TOPIC_REPLICATION_FACTOR / REPLICATION_FACTOR — défaut : 2

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

for f in docker-compose.production.yml docker-compose.kafka.yml docker-compose.kafka.atmr-network.yml; do
  if [[ ! -f "${f}" ]]; then
    echo "Fichier manquant : ${ROOT}/${f}" >&2
    exit 1
  fi
done

BOOTSTRAP_SERVERS="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
REPLICATION_FACTOR="${KAFKA_TOPIC_REPLICATION_FACTOR:-${REPLICATION_FACTOR:-2}}"

if [[ -n "${KAFKA_MIN_INSYNC_REPLICAS:-}" ]] || [[ -n "${MIN_INSYNC_REPLICAS:-}" ]]; then
  MIN_INSYNC_REPLICAS="${KAFKA_MIN_INSYNC_REPLICAS:-${MIN_INSYNC_REPLICAS:-1}}"
else
  if [[ "${REPLICATION_FACTOR}" -ge 3 ]]; then
    MIN_INSYNC_REPLICAS=2
  else
    MIN_INSYNC_REPLICAS=1
  fi
fi

compose_exec() {
  docker compose \
    -f docker-compose.production.yml \
    -f docker-compose.kafka.yml \
    -f docker-compose.kafka.atmr-network.yml \
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

create_topic "driver.location.raw" "36" "7200000" "delete"
create_topic "driver.location.processed" "36" "259200000" "delete"
create_topic "driver.location.dlq" "36" "259200000" "delete"

create_topic "notifications.push" "36" "259200000" "delete"
create_topic "notifications.sms" "36" "259200000" "delete"
create_topic "notifications.email" "36" "259200000" "delete"
create_topic "notifications.dlq" "36" "259200000" "delete"

create_topic "mission.events" "36" "259200000" "delete"
create_topic "notification.events" "36" "259200000" "delete"
create_topic "dispatch.events" "36" "259200000" "delete"

echo "Kafka topics OK — bootstrap=${BOOTSTRAP_SERVERS} replication=${REPLICATION_FACTOR}"
