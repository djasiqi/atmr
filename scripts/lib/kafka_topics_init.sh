#!/usr/bin/env bash
# shellcheck shell=bash
# Logique partagée de création des topics Kafka (sourcée par kafka-init-topics*.sh).

kafka_topics_read_env() {
  _kt_env() {
    local name="$1"
    local default="$2"
    # Variable déjà exportée dans le shell (prioritaire)
    if [[ -n "${!name-}" ]]; then
      printf '%s' "${!name}"
      return
    fi
    local envf="${ATMR_ENV_FILE:-}"
    if [[ -n "${envf}" ]] && [[ -f "${envf}" ]]; then
      local v=""
      v="$(grep -E "^${name}=" "${envf}" 2>/dev/null | tail -n1 | cut -d'=' -f2-)"
      v="${v//\'/}"
      v="${v//\"/}"
      if [[ -n "${v}" ]]; then
        printf '%s' "${v}"
        return
      fi
    fi
    printf '%s' "${default}"
  }

  KAFKA_DEFAULT_PARTITIONS="$(_kt_env KAFKA_DEFAULT_PARTITIONS 6)"
  KAFKA_DLQ_PARTITIONS="$(_kt_env KAFKA_DLQ_PARTITIONS 3)"
  KAFKA_SMOKE_PARTITIONS="$(_kt_env KAFKA_SMOKE_PARTITIONS 1)"
  KAFKA_CREATE_INACTIVE_TOPICS="$(_kt_env KAFKA_CREATE_INACTIVE_TOPICS false)"

  KAFKA_TOPIC_DRIVER_LOCATION_RAW="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_RAW driver.location.raw)"
  KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed)"
  KAFKA_TOPIC_DRIVER_LOCATION_DLQ="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_DLQ driver.location.dlq)"
  # Contrat Kafka v3 (RF=3 / minISR=2) — suffixe = version contrat, pas version plan
  KAFKA_TOPIC_DRIVER_LOCATION_RAW_V3="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_RAW_V3 driver.location.raw.v3)"
  KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED_V3="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED_V3 driver.location.processed.v3)"
  KAFKA_TOPIC_DRIVER_LOCATION_DLQ_V3="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_DLQ_V3 driver.location.dlq.v3)"
  KAFKA_TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3 driver.location.raw.shadow.v3)"
  KAFKA_TOPIC_DRIVER_LOCATION_DIRECT_OBSERVED_V3="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_DIRECT_OBSERVED_V3 driver.location.direct.observed.v3)"
  KAFKA_TOPIC_DRIVER_LOCATION_ENRICHED_V3="$(_kt_env KAFKA_TOPIC_DRIVER_LOCATION_ENRICHED_V3 driver.location.enriched.v3)"
  KAFKA_CREATE_V3_TOPICS="$(_kt_env KAFKA_CREATE_V3_TOPICS true)"
  KAFKA_TOPIC_NOTIFICATIONS_DLQ="$(_kt_env KAFKA_TOPIC_NOTIFICATIONS_DLQ notifications.dlq)"
  KAFKA_OPS_SMOKE_TOPIC="$(_kt_env KAFKA_OPS_SMOKE_TOPIC atmr.ops.smoke)"
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

  # Topics contrat v3 — création nouvelle (ne pas compter sur --if-not-exists pour changer RF)
  if [[ "${KAFKA_CREATE_V3_TOPICS}" == "true" ]]; then
    # RAW rétention dimensionnée : défaut 72 h (259200000 ms) — gate capacité ops avant prod
    create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_RAW_V3}" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED_V3}" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_DLQ_V3}" "${KAFKA_DLQ_PARTITIONS}" "259200000" "delete"
    create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3}" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_DIRECT_OBSERVED_V3}" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "${KAFKA_TOPIC_DRIVER_LOCATION_ENRICHED_V3}" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
  fi

  if [[ "${KAFKA_CREATE_INACTIVE_TOPICS}" == "true" ]]; then
    create_topic "notifications.push" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "notifications.sms" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "notifications.email" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "mission.events" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "notification.events" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
    create_topic "dispatch.events" "${KAFKA_DEFAULT_PARTITIONS}" "259200000" "delete"
  fi
}
