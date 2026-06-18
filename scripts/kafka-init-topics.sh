#!/usr/bin/env bash
set -euo pipefail

# Initialise explicitement les topics critiques Kafka.
# Usage:
#   BOOTSTRAP_SERVERS=kafka-broker-1:29092 ./scripts/kafka-init-topics.sh
#
# Variables (voir env.kafka.production.example) :
#   KAFKA_TOPIC_REPLICATION_FACTOR / REPLICATION_FACTOR — défaut 2
#   KAFKA_DEFAULT_PARTITIONS — défaut 6
#   KAFKA_DLQ_PARTITIONS — défaut 3
#   KAFKA_CREATE_INACTIVE_TOPICS — défaut false (topics notifications/mission)
#   KAFKA_TOPIC_DRIVER_LOCATION_* — noms de topics (suffixe .v2 en Phase 1 prod)

BOOTSTRAP_SERVERS="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
REPLICATION_FACTOR="${KAFKA_TOPIC_REPLICATION_FACTOR:-${REPLICATION_FACTOR:-2}}"

if [ -n "${KAFKA_MIN_INSYNC_REPLICAS:-}" ] || [ -n "${MIN_INSYNC_REPLICAS:-}" ]; then
  MIN_INSYNC_REPLICAS="${KAFKA_MIN_INSYNC_REPLICAS:-${MIN_INSYNC_REPLICAS:-1}}"
else
  if [ "${REPLICATION_FACTOR}" -ge 3 ]; then
    MIN_INSYNC_REPLICAS=2
  else
    MIN_INSYNC_REPLICAS=1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/kafka_topics_init.sh
source "${SCRIPT_DIR}/lib/kafka_topics_init.sh"

export BOOTSTRAP_SERVERS

for _ in $(seq 1 40); do
  if kafka-broker-api-versions --bootstrap-server "${BOOTSTRAP_SERVERS}" >/dev/null 2>&1; then
    break
  fi
  echo "En attente du broker Kafka sur ${BOOTSTRAP_SERVERS}…"
  sleep 2
done
if ! kafka-broker-api-versions --bootstrap-server "${BOOTSTRAP_SERVERS}" >/dev/null 2>&1; then
  echo "Échec : impossible de contacter ${BOOTSTRAP_SERVERS} après le délai d'attente." >&2
  exit 1
fi

create_topic() {
  local topic="$1"
  local partitions="$2"
  local retention_ms="$3"
  local cleanup_policy="$4"

  kafka-topics --bootstrap-server "${BOOTSTRAP_SERVERS}" \
    --create \
    --if-not-exists \
    --topic "${topic}" \
    --partitions "${partitions}" \
    --replication-factor "${REPLICATION_FACTOR}"

  kafka-configs --bootstrap-server "${BOOTSTRAP_SERVERS}" \
    --alter \
    --entity-type topics \
    --entity-name "${topic}" \
    --add-config "min.insync.replicas=${MIN_INSYNC_REPLICAS},retention.ms=${retention_ms},cleanup.policy=${cleanup_policy}"
}

kafka_topics_create_all

echo "Kafka topics init done on ${BOOTSTRAP_SERVERS} (replication factor=${REPLICATION_FACTOR})"
