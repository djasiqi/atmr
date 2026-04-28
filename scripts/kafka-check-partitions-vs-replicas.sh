#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_SERVERS="${BOOTSTRAP_SERVERS:-kafka-broker-1:29092}"
TRACKING_TOPIC="${TRACKING_TOPIC:-driver.location.raw}"
REPLICAS_REQUIRED="${REPLICAS_REQUIRED:-3}"

partitions=$(
  kafka-topics --bootstrap-server "${BOOTSTRAP_SERVERS}" --describe --topic "${TRACKING_TOPIC}" \
    | awk -F'PartitionCount:' 'NF>1{print $2}' \
    | awk '{print $1}' \
    | head -n 1
)

if [[ -z "${partitions}" ]]; then
  echo "Unable to read partitions for topic ${TRACKING_TOPIC}" >&2
  exit 1
fi

if (( partitions < REPLICAS_REQUIRED )); then
  echo "FAIL: ${TRACKING_TOPIC} partitions=${partitions} < replicas=${REPLICAS_REQUIRED}" >&2
  exit 2
fi

echo "OK: ${TRACKING_TOPIC} partitions=${partitions} >= replicas=${REPLICAS_REQUIRED}"
