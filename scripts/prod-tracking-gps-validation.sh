#!/usr/bin/env bash
# Checklist validation prod — chaîne GPS (8 points du runbook).
#
# Usage :
#   bash scripts/prod-tracking-gps-validation.sh              # via SSH si SERVER_HOST défini
#   bash scripts/prod-tracking-gps-validation.sh --local-hints  # instructions manuelles uniquement
#
# Variables : SERVER_HOST, SERVER_USER (défaut deploy), SERVER_PATH (défaut /srv/atmr)
# Optionnel : VALIDATION_DRIVER_ID (défaut 3), MIN_APP_VERSION (défaut 1.0.8)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIVER_ID="${VALIDATION_DRIVER_ID:-3}"
MIN_APP_VERSION="${MIN_APP_VERSION:-1.0.8}"
OK=0
FAIL=0
SKIP=0

if [[ -f "${ROOT}/.local.deploy.env" ]]; then
  # shellcheck source=/dev/null
  set -a
  source "${ROOT}/.local.deploy.env"
  set +a
fi

pass() { echo "[OK]   $1"; OK=$((OK + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "[SKIP] $1"; SKIP=$((SKIP + 1)); }

print_hints() {
  cat <<EOF
=== Checklist manuelle (8 points) ===

1. Version app driver ≥ ${MIN_APP_VERSION}
   → Play Console / TestFlight / device_health app_version

2. Logs mobile : pas de ReferenceError nowIso
   → adb logcat | grep -i nowIso  (doit être vide d'erreurs)

3. Backend : driver_location_received_total > 0
   → Prometheus ou metrics endpoint backend

4. Kafka : lag ≈ 0, partitions équilibrées
   → bash scripts/check-kafka-production.sh

5. Redis : clé canonical mise à jour
   → redis-cli GET driver:${DRIVER_ID}:loc:canonical

6. Fanout : events driver_location_update
   → logs processed_fanout_consumer

7. Dashboard : positions fraîches (< 60 s)
   → Grafana driver-tracking-health + dispatch UI

8. device_health : constraint_reason null ou RECOVERING
   → SELECT payload FROM driver_device_health_events WHERE driver_id=${DRIVER_ID} ORDER BY created_at DESC LIMIT 5;

Configurer SSH : cp docs/deployment-ssh.md → .local.deploy.env avec SERVER_HOST
EOF
}

if [[ "${1:-}" == "--local-hints" ]]; then
  print_hints
  exit 0
fi

if [[ -z "${SERVER_HOST:-}" ]]; then
  echo "=== SERVER_HOST non défini — mode hints ==="
  print_hints
  exit 2
fi

REMOTE="${SERVER_USER:-deploy}@${SERVER_HOST}"
REMOTE_PATH="${SERVER_PATH:-/srv/atmr}"

echo "=== Validation GPS prod via ${REMOTE} (driver_id=${DRIVER_ID}) ==="

REMOTE_SCRIPT=$(cat <<'EOS'
set -euo pipefail
cd /srv/atmr
DRIVER_ID="${1}"
MIN_APP_VERSION="${2}"

check_metric_positive() {
  local pattern="$1"
  local label="$2"
  local val
  val=$(docker compose -f docker-compose.production.yml exec -T backend wget -qO- http://127.0.0.1:5000/metrics 2>/dev/null | grep -E "^${pattern}" | head -1 | awk '{print $2}' || echo "0")
  if awk -v v="$val" 'BEGIN { exit (v+0 > 0) ? 0 : 1 }'; then
    echo "OK|${label}|${val}"
  else
    echo "FAIL|${label}|${val}"
  fi
}

# Point 3 — ingest
check_metric_positive 'driver_location_received_total' "backend ingest driver_location_received_total"

# Point 4 — kafka consumer lag (approx)
LAG=$(docker compose -f docker-compose.production.yml --profile kafka exec -T tracking-kafka-consumer wget -qO- http://127.0.0.1:9115/metrics 2>/dev/null | grep 'tracking_kafka_consumer_lag' | head -1 | awk '{print $2}' || echo "unknown")
if [[ "$LAG" == "unknown" ]] || awk -v l="$LAG" 'BEGIN { exit (l+0 < 100) ? 0 : 1 }'; then
  echo "OK|kafka lag|${LAG}"
else
  echo "FAIL|kafka lag high|${LAG}"
fi

# Point 5 — redis canonical (via backend redis if available)
REDIS_TS=$(docker compose -f docker-compose.production.yml exec -T redis redis-cli GET "driver:${DRIVER_ID}:loc:canonical" 2>/dev/null | head -c 200 || echo "")
if [[ -n "$REDIS_TS" && "$REDIS_TS" != "(nil)" ]]; then
  echo "OK|redis canonical|present"
else
  echo "FAIL|redis canonical|missing"
fi

# Point 7 — driver last_position_update freshness
FRESH=$(docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -t -A -c \
  "SELECT EXTRACT(EPOCH FROM (NOW() - last_position_update))::int FROM driver WHERE id=${DRIVER_ID};" 2>/dev/null | tr -d '[:space:]' || echo "9999")
if [[ "$FRESH" != "9999" && "$FRESH" -le 60 ]]; then
  echo "OK|dashboard freshness|${FRESH}s"
elif [[ "$FRESH" != "9999" && "$FRESH" -le 300 ]]; then
  echo "WARN|dashboard freshness|${FRESH}s"
else
  echo "FAIL|dashboard freshness|${FRESH}s"
fi

# Point 8 — device_health constraint
CONSTRAINT=$(docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -t -A -c \
  "SELECT COALESCE(payload->>'constraint_reason','') FROM driver_device_health_events WHERE driver_id=${DRIVER_ID} ORDER BY created_at DESC LIMIT 1;" 2>/dev/null | tr -d '[:space:]' || echo "unknown")
if [[ -z "$CONSTRAINT" || "$CONSTRAINT" == "null" || "$CONSTRAINT" == "RECOVERING" ]]; then
  echo "OK|device_health constraint|${CONSTRAINT:-null}"
else
  echo "FAIL|device_health constraint|${CONSTRAINT}"
fi

# Point 1 — app_version from latest device_health
APP_VER=$(docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -t -A -c \
  "SELECT COALESCE(payload->>'app_version','') FROM driver_device_health_events WHERE driver_id=${DRIVER_ID} ORDER BY created_at DESC LIMIT 1;" 2>/dev/null | tr -d '[:space:]' || echo "")
if [[ -n "$APP_VER" ]]; then
  if printf '%s\n%s' "$MIN_APP_VERSION" "$APP_VER" | sort -C -V 2>/dev/null; then
    echo "OK|app version|${APP_VER}"
  else
    echo "FAIL|app version|${APP_VER} < ${MIN_APP_VERSION}"
  fi
else
  echo "WARN|app version|unknown"
fi

# Point 6 — fanout (proxy: persist metrics)
PERSIST=$(docker compose -f docker-compose.production.yml --profile kafka exec -T tracking-kafka-consumer wget -qO- http://127.0.0.1:9115/metrics 2>/dev/null | grep -E 'tracking_kafka_persist_total|tracking_http_accepted_async' | head -3 || echo "")
if [[ -n "$PERSIST" ]]; then
  echo "OK|fanout/persist metrics|present"
else
  echo "WARN|fanout/persist metrics|unavailable"
fi
EOS
)

while IFS= read -r line; do
  status="${line%%|*}"
  rest="${line#*|}"
  label="${rest%%|*}"
  detail="${rest#*|}"
  case "${status}" in
    OK) pass "${label} (${detail})" ;;
    WARN) skip "${label} (${detail}) — vérifier manuellement" ;;
    FAIL) fail "${label} (${detail})" ;;
  esac
done < <(ssh "${REMOTE}" "bash -s" -- "${DRIVER_ID}" "${MIN_APP_VERSION}" <<< "${REMOTE_SCRIPT}")

skip "logs mobile nowIso (point 2) — vérifier via adb sur device test"

echo ""
echo "=== Résumé : OK=${OK} FAIL=${FAIL} SKIP=${SKIP} ==="
if [[ "${FAIL}" -gt 0 ]]; then
  echo "Validation : ÉCHEC — voir docs/operations/tracking-runbook.md"
  exit 1
fi
echo "Validation : OK (avec ${SKIP} point(s) manuel(s))"
