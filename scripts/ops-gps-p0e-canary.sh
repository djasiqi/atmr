#!/usr/bin/env bash
# Canary P0-E — preuve chemin SYNC (pas 202 queued_async).
#
# Prérequis :
#   - Un PUT /driver/me/location a répondu HTTP 200 avec :
#       ack_status=persisted
#       durability=persisted_sync
#       ledger_persisted=true
#       ledger_reason=inserted
#       location_event_id = <EID>
#   - Si la réponse était 202 queued_async → NI PASS NI FAIL : rejouer en SYNC.
#   - Ne pas casser Kafka / circuit pour forcer le fallback.
#
# Usage :
#   LOCATION_EVENT_ID=<uuid> DRIVER_ID=<id> bash scripts/ops-gps-p0e-canary.sh
#   LOCATION_EVENT_ID=<uuid> DRIVER_ID=<id> bash scripts/ops-gps-p0e-canary.sh --proof-b-only
#
# Variables :
#   ENV_FILE (défaut .env.production)
#   COMPOSE_FILES (défaut -f docker-compose.production.yml)
#   POSTGRES_SERVICE (défaut postgres)
#   PROOF_B_WINDOW (défaut 1 hour)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-.env.production}"
COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.production.yml}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
PROOF_B_WINDOW="${PROOF_B_WINDOW:-1 hour}"
PROOF_B_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --proof-b-only) PROOF_B_ONLY=1 ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
  esac
done

if [[ -f "${ROOT}/.local.deploy.env" ]]; then
  # shellcheck source=/dev/null
  set -a
  source "${ROOT}/.local.deploy.env"
  set +a
fi

compose() {
  # shellcheck disable=SC2086
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" "$@"
}

psql_q() {
  local sql="$1"
  compose exec -T "$POSTGRES_SERVICE" \
    psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" -tA -c "$sql"
}

echo "== Canary P0-E (SYNC ledger) =="

# --- Preuve B : cohérence globale DLE → ledger (fenêtre) ---
proof_b() {
  echo "-- Preuve B : driver_location_events sans tracking_ingest_events (${PROOF_B_WINDOW})"
  local count
  count="$(psql_q "
    SELECT COUNT(*)::text
    FROM driver_location_events d
    LEFT JOIN tracking_ingest_events t
      ON t.driver_id = d.driver_id
     AND t.location_event_id = d.location_event_id
    WHERE d.recorded_at > NOW() - INTERVAL '${PROOF_B_WINDOW}'
      AND t.location_event_id IS NULL;
  ")"
  count="$(echo "$count" | tr -d '[:space:]')"
  echo "dle_sans_ledger=${count}"
  if [[ "$count" != "0" ]]; then
    echo "FAIL Preuve B: anomalies DLE sans ledger = ${count}" >&2
    return 1
  fi
  echo "OK Preuve B"
  return 0
}

if [[ "$PROOF_B_ONLY" -eq 1 ]]; then
  proof_b
  exit $?
fi

LOCATION_EVENT_ID="${LOCATION_EVENT_ID:-}"
DRIVER_ID="${DRIVER_ID:-}"

if [[ -z "$LOCATION_EVENT_ID" || -z "$DRIVER_ID" ]]; then
  echo "ABORT: LOCATION_EVENT_ID et DRIVER_ID requis pour Preuve A." >&2
  echo "Si ACK = 202 queued_async → hors verdict P0-E (chemin async non exercé)." >&2
  echo "Pour Preuve B seule: bash $0 --proof-b-only" >&2
  exit 2
fi

echo "-- Preuve A : ACK canary → tables (eid=${LOCATION_EVENT_ID} driver=${DRIVER_ID})"
echo "Attendu HTTP côté mobile/API (à confirmer manuellement avant ce script):"
echo "  200 ack_status=persisted durability=persisted_sync"
echo "  ledger_persisted=true ledger_reason=inserted"
echo "  location_event_id echo identique"
echo "  (202 queued_async = NI PASS NI FAIL — rejouer en SYNC)"

eid_sql="${LOCATION_EVENT_ID//\'/\'\'}"

ledger="$(psql_q "
  SELECT COUNT(*)::text FROM tracking_ingest_events
  WHERE driver_id = ${DRIVER_ID}::int AND location_event_id = '${eid_sql}';
")"
dle="$(psql_q "
  SELECT COUNT(*)::text FROM driver_location_events
  WHERE driver_id = ${DRIVER_ID}::int AND location_event_id = '${eid_sql}';
")"
session_hit="$(psql_q "
  SELECT COUNT(*)::text
  FROM tracking_session_state s
  JOIN tracking_ingest_events e
    ON e.driver_id = s.driver_id
   AND e.tracking_session_id = s.tracking_session_id
  WHERE e.driver_id = ${DRIVER_ID}::int
    AND e.location_event_id = '${eid_sql}'
    AND s.max_seen_sequence >= e.sequence_id;
")"
outbox="$(psql_q "
  SELECT COUNT(*)::text FROM tracking_event_outbox
  WHERE driver_id = ${DRIVER_ID}::int
    AND location_event_id = '${eid_sql}';
")"

ledger="$(echo "$ledger" | tr -d '[:space:]')"
dle="$(echo "$dle" | tr -d '[:space:]')"
session_hit="$(echo "$session_hit" | tr -d '[:space:]')"
outbox="$(echo "$outbox" | tr -d '[:space:]')"

echo "tracking_ingest_events=${ledger}"
echo "driver_location_events=${dle}"
echo "tracking_session_state_watermark=${session_hit}"
echo "tracking_event_outbox=${outbox}"

fail=0
[[ "$ledger" == "1" || "$ledger" -ge 1 ]] || { echo "FAIL Preuve A: ledger manquant" >&2; fail=1; }
[[ "$dle" == "1" || "$dle" -ge 1 ]] || { echo "FAIL Preuve A: driver_location_events manquant" >&2; fail=1; }
[[ "$session_hit" == "1" || "$session_hit" -ge 1 ]] || { echo "FAIL Preuve A: watermark session non avancé" >&2; fail=1; }
[[ "$outbox" == "1" || "$outbox" -ge 1 ]] || { echo "FAIL Preuve A: outbox manquante" >&2; fail=1; }

if [[ "$fail" -ne 0 ]]; then
  echo "FAIL Preuve A" >&2
  exit 1
fi
echo "OK Preuve A"

proof_b || exit 1

echo ""
echo "GO Phase 1 (SQL) : preuves A+B OK pour eid=${LOCATION_EVENT_ID}"
echo "Compléter manuellement: 0×429 pilote ; 409/503 sans purge SQLite prématurée."
