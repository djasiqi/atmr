#!/usr/bin/env bash
# Diagnostic GPS bout-en-bout — mobile (adb) + backend (Prometheus API publique).
# Usage:
#   bash scripts/ops/gps-pipeline-diagnostic.sh
#   DRIVER_ID=7514 bash scripts/ops/gps-pipeline-diagnostic.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
API_BASE="${PUBLIC_BASE_URL:-https://api.lirie.ch}"
DRIVER_ID="${DRIVER_ID:-7514}"
PKG="ch.liri.operations"
REPORT="${ROOT}/tmp_gps_diagnostic_$(date +%Y%m%d_%H%M%S).txt"

exec > >(tee "${REPORT}") 2>&1

section() { printf "\n========== %s ==========\n" "$*"; }
ok() { printf "[OK] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
fail() { printf "[FAIL] %s\n" "$*"; }

section "1. Device Android (adb)"
if command -v adb >/dev/null 2>&1; then
  adb devices -l || true
  if adb shell pm path "${PKG}" >/dev/null 2>&1; then
    adb shell dumpsys package "${PKG}" | grep -E "versionName|versionCode|lastUpdateTime" || true
    if adb shell dumpsys activity services "${PKG}" 2>/dev/null | grep -q LocationTaskService; then
      ok "LocationTaskService (FGS) présent"
      adb shell dumpsys activity services "${PKG}" 2>/dev/null | grep -E "isForeground|startForegroundCount|lastActivity" | head -5
    else
      warn "LocationTaskService absent — pas de FGS tracking"
    fi
    RN_LINES="$(adb logcat -d -t 3000 -s ReactNativeJS:V 2>/dev/null | grep -cE 'driver-telemetry.*tracking' || true)"
    RN_LINES="${RN_LINES//$'\r'/}"
    if [[ "${RN_LINES:-0}" -gt 0 ]]; then
      ok "Événements tracking dans logcat (${RN_LINES} lignes)"
      adb logcat -d -t 3000 -s ReactNativeJS:V 2>/dev/null | grep -E 'driver-telemetry.*tracking' | tail -15
    else
      warn "Aucun événement driver-telemetry tracking — mission active EN_ROUTE requise"
    fi
    if adb logcat -d -t 5000 2>/dev/null | grep -qiE 'ReferenceError.*nowIso|nowIso is not defined'; then
      fail "ReferenceError nowIso détecté"
    else
      ok "Pas de ReferenceError nowIso dans buffer logcat"
    fi
  else
    warn "App ${PKG} non installée"
  fi
else
  warn "adb absent — section mobile ignorée"
fi

section "2. Backend API"
READY="$(curl -sf "${API_BASE}/api/v1/ready" 2>/dev/null || echo '{}')"
echo "ready: ${READY}"
if echo "${READY}" | grep -q '"status":"ready"'; then
  ok "API ready (db+redis)"
else
  fail "API non ready"
fi

section "3. Prometheus métriques GPS (instance courante)"
METRICS="$(curl -sf "${API_BASE}/api/v1/prometheus/metrics" 2>/dev/null || true)"
if [[ -z "${METRICS}" ]]; then
  fail "Impossible de lire /api/v1/prometheus/metrics"
  exit 1
fi

_count_metric() {
  local name="$1"
  echo "${METRICS}" | grep -E "^${name}" | awk '{s+=$2} END {print s+0}'
}

recv="$(_count_metric driver_location_received_total)"
batch_pts="$(_count_metric driver_location_batch_points_received_total)"
persist="$(_count_metric tracking_kafka_persist_total)"
fanout="$(_count_metric tracking_fanout_emit_total)"
http_async="$(_count_metric tracking_http_accepted_async_total)"
e2e_count="$(echo "${METRICS}" | grep '^tracking_kafka_e2e_latency_seconds_count' | awk '{print $2}' | tail -1)"
fix_stale="$(echo "${METRICS}" | grep 'constraint_reason="fix_stale"' | awk '{s+=$2} END {print s+0}')"
hb_total="$(echo "${METRICS}" | grep '^driver_device_health_reports_total' | awk '{s+=$2} END {print s+0}')"
inv_viol="$(echo "${METRICS}" | grep '^tracking_invariant_violation_total' | awk '{s+=$2} END {print s+0}')"

echo "driver_location_received_total (sum labels): ${recv}"
echo "driver_location_batch_points_received_total: ${batch_pts}"
echo "tracking_http_accepted_async_total: ${http_async}"
echo "tracking_kafka_persist_total: ${persist}"
echo "tracking_fanout_emit_total: ${fanout}"
echo "tracking_kafka_e2e_latency_seconds_count: ${e2e_count}"
echo "device_health heartbeats (reports_total sum): ${hb_total}"
echo "fix_stale heartbeats (sum): ${fix_stale}"
echo "invariant_violations: ${inv_viol}"

if [[ "${recv}" == "0" && "${batch_pts}" == "0" ]]; then
  fail "Aucune position reçue sur cette instance backend (counters=0 depuis restart)"
else
  ok "Ingest GPS actif sur instance courante"
fi

section "4. Heartbeats app_version (échantillon)"
echo "${METRICS}" | grep 'driver_device_health_reports_total' | grep app_version | head -10

section "5. Verdict & actions"
echo ""
if [[ "${recv}" == "0" && "${batch_pts}" == "0" ]]; then
  echo "→ Pipeline aval probablement sain mais AFFAMÉ : aucun point GPS ingéré."
  echo "→ Mobile : vérifier mission EN_ROUTE + logs [driver-telemetry] tracking.watch.started / tracking.bridge.health"
  echo "→ Si mission active sans telemetry : bug bridge ou flags."
  echo "→ Si telemetry OK mais recv=0 : socket/HTTP ou auth driver."
fi
if [[ "${fix_stale}" != "0" ]]; then
  echo "→ fix_stale présent (${fix_stale}) : self-heal phase 1 doit réduire après mission active 1.0.8"
fi
echo ""
echo "Rapport sauvé : ${REPORT}"
echo "Checklist SSH complète : SERVER_HOST=... bash scripts/prod-tracking-gps-validation.sh"
