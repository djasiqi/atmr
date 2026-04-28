#!/usr/bin/env bash
set -euo pipefail

# Initialise explicitement les topics critiques Kafka.
# Usage:
#   BOOTSTRAP_SERVERS=kafka-broker-1:29092 ./scripts/kafka-init-topics.sh
#
# Réplication : par défaut 2 (valide avec 2 ou 3+ brokers). RF=3 échoue si seuls
# 2 brokers répondent au cluster (InvalidReplicationFactorException).
# Pour 3 brokers stables, définir KAFKA_TOPIC_REPLICATION_FACTOR=3.

BOOTSTRAP_SERVERS="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
# Préférer KAFKA_TOPIC_REPLICATION_FACTOR (compose) ; REPLICATION_FACTOR reste un alias.
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

export BOOTSTRAP_SERVERS

for _ in $(seq 1 40); do
  if kafka-broker-api-versions --bootstrap-server "${BOOTSTRAP_SERVERS}" >/dev/null 2>&1; then
    break
  fi
  echo "En attente du broker Kafka sur ${BOOTSTRAP_SERVERS}…"
  sleep 2
done
if ! kafka-broker-api-versions --bootstrap-server "${BOOTSTRAP_SERVERS}" >/dev/null 2>&1; then
  echo "Échec : impossible de contacter ${BOOTSTRAP_SERVERS} après le délai d’attente." >&2
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

# Tracking (2h sur raw, 72h sur processed/dlq)
create_topic "driver.location.raw" "36" "7200000" "delete"
create_topic "driver.location.processed" "36" "259200000" "delete"
create_topic "driver.location.dlq" "36" "259200000" "delete"

# Notifications (optionnelles si profile kafka-notifications activé)
create_topic "notifications.push" "36" "259200000" "delete"
create_topic "notifications.sms" "36" "259200000" "delete"
create_topic "notifications.email" "36" "259200000" "delete"
create_topic "notifications.dlq" "36" "259200000" "delete"

# Topics events explicitement contrôlés
create_topic "mission.events" "36" "259200000" "delete"
create_topic "notification.events" "36" "259200000" "delete"
create_topic "dispatch.events" "36" "259200000" "delete"

echo "Kafka topics init done on ${BOOTSTRAP_SERVERS} (replication factor=${REPLICATION_FACTOR})"
