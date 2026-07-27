#!/usr/bin/env bash
# Gate RF=3 / minISR=2 pour topics contrat .v3 (Phase 0B / 2 / 3).
set -euo pipefail

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-kafka-broker-1:9092}"
TOPICS=(
  "driver.location.raw.v3"
  "driver.location.processed.v3"
  "driver.location.enriched.v3"
  "driver.location.raw.shadow.v3"
  "driver.location.dlq.v3"
)

# Dev mono-broker : ALLOW_RF1_LOCAL=true accepte RF=1 (gate prod reste RF=3)
ALLOW_RF1_LOCAL="${ALLOW_RF1_LOCAL:-false}"
EXPECTED_RF=3
if [[ "${ALLOW_RF1_LOCAL}" == "true" ]]; then
  EXPECTED_RF=1
  echo "NOTE: ALLOW_RF1_LOCAL=true — gate assouplie pour cluster local"
fi

fail=0
for topic in "${TOPICS[@]}"; do
  echo "== describe $topic =="
  out="$(kafka-topics --bootstrap-server "$BOOTSTRAP" --describe --topic "$topic" 2>&1 || true)"
  echo "$out"
  if ! echo "$out" | grep -q "ReplicationFactor: ${EXPECTED_RF}"; then
    echo "FAIL: $topic ReplicationFactor != ${EXPECTED_RF}" >&2
    fail=1
  fi
  if [[ "${EXPECTED_RF}" -ge 3 ]]; then
    if ! echo "$out" | grep -E "Isr: .*,.*" >/dev/null; then
      echo "WARN: $topic ISR peut être < 2" >&2
    fi
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "Gate RF=${EXPECTED_RF} FAILED" >&2
  exit 1
fi
echo "Gate RF=${EXPECTED_RF} PASS (topics .v3)"
