#!/usr/bin/env bash
# shellcheck shell=bash
# Logique partagée de création des topics Kafka (sourcée par kafka-init-topics*.sh).

kafka_topics_read_env() {
  KAFKA_DEFAULT_PARTITIONS="${KAFKA_DEFAULT_PARTITIONS:-6}"
  KAFKA_DLQ_PARTITIONS="${KAFKA_DLQ_PARTITIONS:-3}"
  KAFKA_SMOKE_PARTITIONS="${KAFKA_SMOKE_PARTITIONS:-1}"
  KAFKA_CREATE_INACTIVE_TOPICS="${KAFKA_CREATE_INACTIVE_TOPICS:-false}"

  KAFKA_TOPIC_DRIVER_LOCATION_RAW="${KAFKA_TOPIC_DRIVER_LOCATION_RAW:-driver.location.raw}"
  KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED="${KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED:-driver.location.processed}"
  KAFKA_TOPIC_DRIVER_LOCATION_DLQ="${KAFKA_TOPIC_DRIVER_LOCATION_DLQ:-driver.location.dlq}"
  KAFKA_TOPIC_NOTIFICATIONS_DLQ="${KAFKA_TOPIC_NOTIFICATIONS_DLQ:-notifications.dlq}"
  KAFKA_OPS_SMOKE_TOPIC="${KAFKA_OPS_SMOKE_TOPIC:-atmr.ops.smoke}"
}

kafka_topics_create_all() {
  # Fonction create_topic(topic partitions retention_ms cleanup_policy) doit être définie par l'appelant.
  if ! declare -F create_topic >/dev/null 2>&1; then
    echo "kafka_topics_create_all : create_topic non défini" >&2
    return 1
  fi

  kafka_topics_read_env

  create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_RAW}" "${KAFKA_DEFAULT_PARTITIONS}" "7200000" "delete"
  create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED}" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
  create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_DLQ}" "${KAFKA_DLQ_PARTITIONS}" "259200000" "delete"
  create_topic "${KAFKA_TOPIC_NOTIFICATIONS_DLQ}" "${KAFKA_DLQ_PARTITIONS}" "259200000" "delete"
  create_topic "${KAFKA_OPS_SMOKE_TOPIC}" "${KAFKA_SMOKE_PARTITIONS}" "259200000" "delete"

  if [[ "${KAFKA_CREATE_INACTIVE_TOPICS}" == "true" ]]; then
    create_topic "notifications.push" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "notifications.sms" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "notifications.email" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "mission.events" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "notification.events" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "dispatch.events" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
  fi
}
