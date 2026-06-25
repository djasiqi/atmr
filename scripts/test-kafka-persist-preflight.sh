#!/usr/bin/env bash
# Tests locaux preflight TRACKING_INGEST_PERSIST_ENABLED (PR2/B — R12/R13).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

write_env() {
  : >"${TMP}/.env.test"
  while [[ $# -ge 2 ]]; do
    echo "$1=$2" >>"${TMP}/.env.test"
    shift 2
  done
  export ATMR_ENV_FILE="${TMP}/.env.test"
}

run_full_preflight() {
  kafka_check_flags_all_true && kafka_check_tracking_persist_coherence
}

run_case() {
  local label="$1"
  local expect_ok="$2"
  shift 2
  write_env "$@"
  echo "=== Cas : ${label} ==="
  if run_full_preflight; then
    rc=0
  else
    rc=1
  fi
  if [[ "${expect_ok}" == "ok" ]] && [[ "${rc}" -eq 0 ]]; then
    echo "[PASS] attendu OK"
  elif [[ "${expect_ok}" == "fail" ]] && [[ "${rc}" -ne 0 ]]; then
    echo "[PASS] attendu FAIL"
  else
    echo "[FAIL] résultat inattendu (rc=${rc}, attendu=${expect_ok})"
    exit 1
  fi
}

run_coherence_only() {
  local label="$1"
  local expect_ok="$2"
  shift 2
  write_env "$@"
  echo "=== Cas cohérence seule : ${label} ==="
  if kafka_check_tracking_persist_coherence; then
    rc=0
  else
    rc=1
  fi
  if [[ "${expect_ok}" == "ok" ]] && [[ "${rc}" -eq 0 ]]; then
    echo "[PASS] cohérence OK"
  elif [[ "${expect_ok}" == "fail" ]] && [[ "${rc}" -ne 0 ]]; then
    echo "[PASS] cohérence FAIL attendu"
  else
    echo "[FAIL] cohérence inattendue (rc=${rc})"
    exit 1
  fi
}

# R12 — deploy normal
run_case "absent PERSIST" fail \
  KAFKA_ENABLED true \
  TRACKING_INGEST_ASYNC_ENABLED true \
  TRACKING_PROCESSED_FANOUT_ENABLED true \
  WS_KAFKA_CONSUMER_ENABLED true

run_case "PERSIST vide" fail \
  KAFKA_ENABLED true \
  TRACKING_INGEST_ASYNC_ENABLED true \
  TRACKING_PROCESSED_FANOUT_ENABLED true \
  WS_KAFKA_CONSUMER_ENABLED true \
  TRACKING_INGEST_PERSIST_ENABLED ""

run_case "PERSIST=true" ok \
  KAFKA_ENABLED true \
  TRACKING_INGEST_ASYNC_ENABLED true \
  TRACKING_PROCESSED_FANOUT_ENABLED true \
  WS_KAFKA_CONSUMER_ENABLED true \
  TRACKING_INGEST_PERSIST_ENABLED true

# R12 cas 4 — rollback republish-only : preflight complet FAIL, cohérence OK
run_case "republish-only (preflight complet)" fail \
  KAFKA_ENABLED true \
  TRACKING_INGEST_ASYNC_ENABLED true \
  TRACKING_PROCESSED_FANOUT_ENABLED true \
  WS_KAFKA_CONSUMER_ENABLED true \
  TRACKING_INGEST_PERSIST_ENABLED false \
  TRACKING_INGEST_ALLOW_REPUBLISH_ONLY true

run_coherence_only "republish-only (cohérence seule)" ok \
  KAFKA_ENABLED true \
  TRACKING_INGEST_ASYNC_ENABLED true \
  TRACKING_PROCESSED_FANOUT_ENABLED true \
  WS_KAFKA_CONSUMER_ENABLED true \
  TRACKING_INGEST_PERSIST_ENABLED false \
  TRACKING_INGEST_ALLOW_REPUBLISH_ONLY true

# R13 — rollback Kafka total (tous flags false)
echo "=== Cas : rollback Kafka total (flags_all_false) ==="
write_env \
  KAFKA_ENABLED false \
  TRACKING_INGEST_ASYNC_ENABLED false \
  TRACKING_PROCESSED_FANOUT_ENABLED false \
  WS_KAFKA_CONSUMER_ENABLED false \
  TRACKING_INGEST_PERSIST_ENABLED false
if kafka_check_flags_all_false; then
  echo "[PASS] kafka_check_flags_all_false OK avec PERSIST=false"
else
  echo "[FAIL] kafka_check_flags_all_false devrait accepter tous flags false"
  exit 1
fi

echo "Tous les cas preflight sont PASS (R12 + R13)."
