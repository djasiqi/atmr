#!/usr/bin/env bash
# Phase 0 — Diagnostic pipeline notifications push (production / staging).
# Usage:
#   ./scripts/push-notifications-diagnostic.sh [driver_id] [booking_id]
#
# Prérequis: accès docker (atmr-postgres, atmr-celery-worker) ou variables DATABASE_URL.
set -euo pipefail

DRIVER_ID="${1:-}"
BOOKING_ID="${2:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${PUSH_DIAG_OUT:-/tmp/push-notifications-diagnostic/${TS}}"
mkdir -p "$OUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$ROOT_DIR/.env.production" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.production"
  set +a
fi

PSQL_CMD=""
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'atmr-postgres'; then
  PSQL_CMD="docker exec -i atmr-postgres psql -U ${POSTGRES_USER:-atmr} -d ${POSTGRES_DB:-atmr} -t -A"
elif [ -n "${DATABASE_URL:-}" ]; then
  PSQL_CMD="psql \"${DATABASE_URL}\" -t -A"
else
  echo "WARN: ni atmr-postgres ni DATABASE_URL — requêtes SQL ignorées" | tee "$OUT_DIR/warn.txt"
fi

run_sql() {
  local label="$1"
  local sql="$2"
  local out="$OUT_DIR/${label}.txt"
  if [ -z "$PSQL_CMD" ]; then
    echo "(skipped — no psql)" >"$out"
    return
  fi
  # shellcheck disable=SC2086
  echo "$sql" | $PSQL_CMD >"$out" 2>&1 || echo "SQL error — voir $out" >&2
}

echo "=== Push notifications diagnostic ===" | tee "$OUT_DIR/summary.txt"
echo "Output: $OUT_DIR" | tee -a "$OUT_DIR/summary.txt"
echo "" | tee -a "$OUT_DIR/summary.txt"

# V1 — Tokens actifs par chauffeur
run_sql "v1_active_tokens_per_driver" "
SELECT driver_id, COUNT(*) AS active_tokens
FROM device_tokens
WHERE is_active = true AND driver_id IS NOT NULL
GROUP BY driver_id
ORDER BY active_tokens ASC, driver_id ASC
LIMIT 200;
"

# V2 — provider / platform / is_active
run_sql "v2_provider_platform" "
SELECT provider, platform, is_active, COUNT(*) AS cnt
FROM device_tokens
GROUP BY provider, platform, is_active
ORDER BY provider, platform, is_active;
"

# Chauffeurs sans token actif (top 50)
run_sql "v1b_drivers_without_active_token" "
SELECT d.id AS driver_id
FROM driver d
WHERE NOT EXISTS (
  SELECT 1 FROM device_tokens dt
  WHERE dt.driver_id = d.id AND dt.is_active = true
)
ORDER BY d.id
LIMIT 50;
"

# V4 — Détail chauffeur test
if [ -n "$DRIVER_ID" ]; then
  run_sql "v4_driver_tokens" "
SELECT id, provider, platform, is_active,
       consecutive_push_failures, last_push_error_code,
       last_push_failure_at, last_push_success_at, updated_at
FROM device_tokens
WHERE driver_id = ${DRIVER_ID}
ORDER BY updated_at DESC;
"
fi

# V3 — Erreurs FCM dans logs Celery
FCM_LOG="$OUT_DIR/v3_fcm_errors.txt"
: >"$FCM_LOG"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'atmr-celery-worker'; then
  docker logs atmr-celery-worker --since 72h 2>&1 \
    | grep -iE 'UnregisteredError|SenderIdMismatchError|token_invalid|device_not_registered' \
    | tail -200 >>"$FCM_LOG" || true
  echo "count=$(wc -l <"$FCM_LOG" | tr -d ' ')" >>"$FCM_LOG"
else
  echo "(skipped — atmr-celery-worker absent)" >>"$FCM_LOG"
fi

# V5 — Trace pipeline driver_push pour booking ou driver
V5_LOG="$OUT_DIR/v5_pipeline_trace.txt"
: >"$V5_LOG"
grep_sources() {
  local pattern="$1"
  local file="$2"
  if [ -f "$file" ]; then
    grep -E '"stage": *"driver_push\.' "$file" | grep -E "$pattern" | tail -100 >>"$V5_LOG" || true
  fi
}
if [ -n "$BOOKING_ID" ]; then
  grep_sources "\"booking_id\": *${BOOKING_ID}" "$ROOT_DIR/backend.log"
  docker logs atmr-celery-worker --since 24h 2>&1 \
    | grep -E '"stage": *"driver_push\.' \
    | grep -E "\"booking_id\": *${BOOKING_ID}" \
    | tail -100 >>"$V5_LOG" 2>/dev/null || true
elif [ -n "$DRIVER_ID" ]; then
  grep_sources "\"driver_id\": *${DRIVER_ID}" "$ROOT_DIR/backend.log"
  docker logs atmr-celery-worker --since 24h 2>&1 \
    | grep -E '"stage": *"driver_push\.' \
    | grep -E "\"driver_id\": *${DRIVER_ID}" \
    | tail -100 >>"$V5_LOG" 2>/dev/null || true
else
  echo "Indiquer driver_id et/ou booking_id pour V5 (ex: ./scripts/push-notifications-diagnostic.sh 42 1234)" >>"$V5_LOG"
fi

# Interprétation automatique (heuristique)
REC="$OUT_DIR/recommendation.txt"
{
  echo "=== Recommandation Correction #1 (commits token lifecycle) ==="
  echo ""
  if [ -f "$OUT_DIR/v3_fcm_errors.txt" ] && [ "$(wc -l <"$OUT_DIR/v3_fcm_errors.txt" | tr -d ' ')" -gt 5 ]; then
    echo "SIGNAL: Erreurs FCM UnregisteredError/SenderIdMismatchError détectées (voir v3_fcm_errors.txt)."
    echo "ACTION: Envisager Correction #1 (rollback + cleanup async) après revue manuelle des ids."
  else
    echo "SIGNAL: Peu ou pas d'erreurs FCM token_invalid dans les 72h."
    echo "ACTION: Ne PAS implémenter Correction #1 tant que ce signal ne se confirme pas."
  fi
  echo ""
  if [ -f "$OUT_DIR/v1b_drivers_without_active_token.txt" ] && grep -qE '^[0-9]+$' "$OUT_DIR/v1b_drivers_without_active_token.txt" 2>/dev/null; then
    echo "SIGNAL: Chauffeurs sans token actif (v1b_drivers_without_active_token.txt)."
    echo "ACTION: Prioriser Correction #2 (retry + AsyncStorage mobile) — ROI maximal."
  fi
  echo ""
  echo "Chaîne V5 attendue: driver_push.publish -> handler -> notify -> enqueue -> task_start -> task_done"
  echo "Si la chaîne s'arrête avant enqueue: investiguer event bus / handlers (hors scope mobile P0/P1)."
} >"$REC"

cat "$REC" | tee -a "$OUT_DIR/summary.txt"
echo ""
echo "Done. Artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
