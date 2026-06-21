#!/usr/bin/env bash
# validate-prod-deploy.sh — Runbook consolidé Phase 2.5 → 3.5 du plan
# « Validation prod tracking BG » (cf. plans/validation_prod_tracking_bg).
#
# À exécuter SUR LE SERVEUR DE PRODUCTION (accès docker + atmr-network).
#
# Couvre :
#   - Phase 2.5 : kafka-consumer-groups --describe + lag
#   - Phase 2.6 : test propagation Kafka E2E (PUT /driver/me/location)
#   - Phase 2.7 : STOP GATE Kafka (refuse Phase 3 si KO)
#   - Phase 3.2 : Redis DB verify (CONFIG GET databases + INFO keyspace)
#   - Phase 3.1 + 3.3 : POST device-health + SCAN dual-write + HGETALL + TTL
#   - Phase 3.4 : driver_device_health_reports_total
#   - Phase 3.5 : TrackingStaleHigh + silent_push_wake_total (lecture)
#
# Usage :
#   scripts/validate-prod-deploy.sh consumer-groups
#   scripts/validate-prod-deploy.sh propagation        # nécessite DRIVER_TEST_JWT
#   scripts/validate-prod-deploy.sh stop-gate
#   scripts/validate-prod-deploy.sh redis-verify
#   scripts/validate-prod-deploy.sh heartbeat-replay   # nécessite DRIVER_TEST_JWT, DRIVER_ID
#   scripts/validate-prod-deploy.sh redis-keys         # nécessite DRIVER_ID
#   scripts/validate-prod-deploy.sh metrics
#   scripts/validate-prod-deploy.sh stale-alert
#   scripts/validate-prod-deploy.sh silent-wake
#   scripts/validate-prod-deploy.sh phase3             # 3.2 → 3.5 enchaînés
#   scripts/validate-prod-deploy.sh all                # Phase 2.5 → 3.5, STOP GATE inclus
#
# Variables :
#   ATMR_DEPLOY_ROOT   répertoire des YAML (défaut : parent de scripts/)
#   ATMR_ENV_FILE      .env (défaut : ${ROOT}/.env.production)
#   BACKEND_URL        URL backend interne (défaut : http://localhost:5000)
#   PUBLIC_BASE_URL    URL publique HTTPS (défaut : https://api.lirie.ch)
#   DRIVER_TEST_JWT    JWT chauffeur de test (requis : propagation, heartbeat-replay)
#   DRIVER_ID          ID numérique chauffeur de test (requis : redis-keys, heartbeat-replay)
#   REDIS_PASSWORD     mot de passe Redis (lu depuis .env.production si absent)
#   LAG_THRESHOLD      lag max accepté par partition (défaut : 100)
#   PROPAGATION_WAIT_S délai propagation après PUT location (défaut : 10)
#
# Codes de sortie :
#   0   tout OK
#   1   au moins un check FAIL
#   2   variable obligatoire manquante / sous-commande inconnue

set -uo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"
export ATMR_ENV_FILE="${ENV_FILE}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

BACKEND_URL="${BACKEND_URL:-http://localhost:5000}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-https://api.lirie.ch}"
LAG_THRESHOLD="${LAG_THRESHOLD:-100}"
PROPAGATION_WAIT_S="${PROPAGATION_WAIT_S:-10}"

REDIS_CONTAINER="atmr-redis"
BACKEND_CONTAINER="atmr-backend-1"
CELERY_CONTAINER="atmr-celery-worker"
KAFKA_BROKER_CONTAINER="atmr-kafka-broker-1"

CONSUMER_GROUPS=(
  tracking-ingest-consumer-group
  tracking-processed-fanout-group
  kafka-dlq-consumer-group
  ws-service-shared
)

# -- Logging --------------------------------------------------------------
log_info() { printf "  [OK]   %s\n" "$*"; }
log_warn() { printf "  [WARN] %s\n" "$*" >&2; }
log_fail() { printf "  [FAIL] %s\n" "$*" >&2; }
section() { printf "\n=== %s ===\n" "$*"; }

# Charge REDIS_PASSWORD depuis .env.production si non défini (sans logger la valeur).
load_redis_password() {
  if [[ -z "${REDIS_PASSWORD:-}" && -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set +u
    REDIS_PASSWORD="$(grep -E '^REDIS_PASSWORD=' "${ENV_FILE}" | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
    set -u
    export REDIS_PASSWORD
  fi
  if [[ -z "${REDIS_PASSWORD:-}" ]]; then
    log_fail "REDIS_PASSWORD introuvable (ni env, ni .env.production)"
    return 1
  fi
  return 0
}

require_var() {
  local var="$1"
  if [[ -z "${!var:-}" ]]; then
    log_fail "Variable manquante : ${var}"
    return 2
  fi
  return 0
}

# -- Phase 2.5 : consumer groups + lag ------------------------------------
cmd_consumer_groups() {
  section "Phase 2.5 — kafka-consumer-groups --describe (lag < ${LAG_THRESHOLD})"
  local ok=1
  local g
  for g in "${CONSUMER_GROUPS[@]}"; do
    printf "  → group=%s\n" "${g}"
    local out
    if ! out="$(docker exec "${KAFKA_BROKER_CONTAINER}" \
        kafka-consumer-groups --bootstrap-server kafka-broker-1:29092 \
        --describe --group "${g}" 2>&1)"; then
      log_fail "describe failed for ${g} : ${out}"
      ok=0
      continue
    fi
    echo "${out}" | sed 's/^/      /'

    # Vérifier qu'aucune partition n'a CONSUMER-ID = '-' et que LAG < seuil.
    # Header: GROUP TOPIC PARTITION CURRENT-OFFSET LOG-END-OFFSET LAG CONSUMER-ID HOST CLIENT-ID
    local bad_consumer
    bad_consumer="$(echo "${out}" | awk 'NR>1 && $7=="-" {print $2"/"$3}')"
    if [[ -n "${bad_consumer}" ]]; then
      log_fail "${g} : partitions sans consumer assigné : ${bad_consumer}"
      ok=0
    fi

    local high_lag
    high_lag="$(echo "${out}" | awk -v thr="${LAG_THRESHOLD}" \
      'NR>1 && $6 ~ /^[0-9]+$/ && $6+0 >= thr {print $2"/"$3":"$6}')"
    if [[ -n "${high_lag}" ]]; then
      log_fail "${g} : lag élevé : ${high_lag}"
      ok=0
    else
      log_info "${g} : tous les consumers assignés, lag < ${LAG_THRESHOLD}"
    fi
  done
  return $((ok == 1 ? 0 : 1))
}

# -- Phase 2.6 : propagation Kafka E2E ------------------------------------
_kafka_offset() {
  local topic="$1"
  docker exec "${KAFKA_BROKER_CONTAINER}" kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list kafka-broker-1:29092 --topic "${topic}" --time -1 2>/dev/null \
    | awk -F: '{s+=$3} END {print s+0}'
}

_metric_value() {
  local pattern="$1"
  curl -sS --max-time 10 "${BACKEND_URL}/api/v1/prometheus/metrics" \
    | grep -E "^${pattern}" \
    | awk '{s+=$2} END {print s+0}'
}

cmd_propagation() {
  section "Phase 2.6 — propagation Kafka E2E (PUT /driver/me/location)"
  require_var DRIVER_TEST_JWT || return 2

  local topic_raw topic_processed
  topic_raw="$(kafka_read_env_value KAFKA_TOPIC_DRIVER_LOCATION_RAW driver.location.raw)"
  topic_processed="$(kafka_read_env_value KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed)"

  local raw_before processed_before
  raw_before="$(_kafka_offset "${topic_raw}")"
  processed_before="$(_kafka_offset "${topic_processed}")"
  printf "  Topics         : raw=%s processed=%s\n" "${topic_raw}" "${topic_processed}"
  printf "  Offsets AVANT  : raw=%s processed=%s\n" "${raw_before}" "${processed_before}"

  local rec_before ing_before proc_before fan_before
  rec_before="$(_metric_value 'driver_location_received_total')"
  ing_before="$(_metric_value 'driver_location_ingested_total')"
  proc_before="$(_metric_value 'driver_location_processed_total\\{accept_status="accepted"')"
  fan_before="$(_metric_value 'driver_location_fanout_events_total')"
  printf "  Compteurs AVANT: received=%s ingested=%s processed=%s fanout=%s\n" \
    "${rec_before}" "${ing_before}" "${proc_before}" "${fan_before}"

  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "  → PUT %s/api/v1/driver/me/location (timestamp=%s)\n" "${PUBLIC_BASE_URL}" "${ts}"
  local code
  code="$(curl -sS -o /tmp/_put_loc_resp.json -w '%{http_code}' \
    -X PUT "${PUBLIC_BASE_URL}/api/v1/driver/me/location" \
    -H "Authorization: Bearer ${DRIVER_TEST_JWT}" \
    -H "Content-Type: application/json" \
    --max-time 15 \
    -d "{\"latitude\":46.2044,\"longitude\":6.1432,\"accuracy\":10.0,\"speed\":0.0,\"heading\":0.0,\"timestamp\":\"${ts}\"}" \
    || echo "000")"
  if [[ "${code}" != "200" && "${code}" != "201" && "${code}" != "202" && "${code}" != "204" ]]; then
    log_fail "PUT location HTTP ${code} — corps :"
    sed 's/^/        /' /tmp/_put_loc_resp.json >&2 || true
    return 1
  fi
  log_info "PUT location HTTP ${code}"

  printf "  → attente propagation %ss (raw → ingest → processed → fanout)\n" "${PROPAGATION_WAIT_S}"
  sleep "${PROPAGATION_WAIT_S}"

  local raw_after processed_after rec_after ing_after proc_after fan_after
  raw_after="$(_kafka_offset "${topic_raw}")"
  processed_after="$(_kafka_offset "${topic_processed}")"
  rec_after="$(_metric_value 'driver_location_received_total')"
  ing_after="$(_metric_value 'driver_location_ingested_total')"
  proc_after="$(_metric_value 'driver_location_processed_total\\{accept_status="accepted"')"
  fan_after="$(_metric_value 'driver_location_fanout_events_total')"

  printf "  Offsets APRÈS  : raw=%s processed=%s\n" "${raw_after}" "${processed_after}"
  printf "  Compteurs APRÈS: received=%s ingested=%s processed=%s fanout=%s\n" \
    "${rec_after}" "${ing_after}" "${proc_after}" "${fan_after}"

  local ok=1
  local d_raw=$((raw_after - raw_before))
  local d_processed=$((processed_after - processed_before))
  local d_rec=$((rec_after - rec_before))
  local d_ing=$((ing_after - ing_before))
  local d_proc=$((proc_after - proc_before))
  local d_fan=$((fan_after - fan_before))

  printf "  Δ raw=%d processed=%d received=%d ingested=%d processed_metric=%d fanout=%d\n" \
    "${d_raw}" "${d_processed}" "${d_rec}" "${d_ing}" "${d_proc}" "${d_fan}"

  ((d_raw       >= 1)) && log_info  "raw offset +${d_raw}"               || { log_fail "raw offset n'a pas progressé"; ok=0; }
  ((d_processed >= 1)) && log_info  "processed offset +${d_processed}"   || { log_fail "processed offset n'a pas progressé (consumer raw KO ou tout en DLQ)"; ok=0; }
  ((d_rec       >= 1)) && log_info  "received_total +${d_rec}"           || { log_fail "received_total ne bouge pas (backend ne reçoit pas le PUT)"; ok=0; }
  ((d_ing       >= 1)) && log_info  "ingested_total +${d_ing}"           || { log_fail "ingested_total ne bouge pas (producer Kafka backend KO)"; ok=0; }
  ((d_proc      >= 1)) && log_info  "processed_total{accepted} +${d_proc}" || log_warn "processed_total{accepted} ne bouge pas (point dédupliqué ou rejeté ?)"
  ((d_fan       >= 1)) && log_info  "fanout_events_total +${d_fan}"      || { log_fail "fanout_events_total ne bouge pas (tracking-processed-fanout n'a pas consommé)"; ok=0; }

  return $((ok == 1 ? 0 : 1))
}

# -- Phase 2.7 : STOP GATE ------------------------------------------------
cmd_stop_gate() {
  section "Phase 2.7 — STOP GATE Kafka"
  local ok=1
  echo "  [1/4] check-kafka-production.sh on"
  if ! "${ROOT}/scripts/check-kafka-production.sh" on; then
    log_fail "check-kafka-production.sh on a échoué"
    ok=0
  fi
  echo "  [2/4] consumer groups + lag"
  cmd_consumer_groups || ok=0
  echo "  [3/4] propagation E2E (DRIVER_TEST_JWT requis)"
  if [[ -n "${DRIVER_TEST_JWT:-}" ]]; then
    cmd_propagation || ok=0
  else
    log_warn "DRIVER_TEST_JWT non défini — propagation E2E SKIP (à exécuter avant Phase 3)"
    ok=0
  fi
  echo "  [4/4] erreurs DNS récentes (5 dernières minutes)"
  local dns_errs
  dns_errs="$(docker logs --since 5m atmr-tracking-kafka-consumer-1 atmr-kafka-dlq-consumer atmr-ws-service atmr-backend-1 2>&1 \
    | grep -Ei 'kafka-broker-[0-9].*(no such host|name resolution|getaddrinfo)' | wc -l | tr -d ' ' || echo 0)"
  if [[ "${dns_errs}" != "0" ]]; then
    log_fail "${dns_errs} erreurs DNS Kafka dans les 5 dernières min"
    ok=0
  else
    log_info "0 erreur DNS Kafka sur 5 min"
  fi

  if (( ok == 0 )); then
    section "STOP GATE : KO — NE PAS DÉMARRER LA PHASE 3"
    return 1
  fi
  section "STOP GATE : OK — PHASE 3 AUTORISÉE"
  return 0
}

# -- Phase 3.2 : Redis DB verify ------------------------------------------
cmd_redis_verify() {
  section "Phase 3.2 — Redis DB verify (CONFIG GET databases + INFO keyspace)"
  load_redis_password || return 2
  local ok=1

  echo "  → Redis CONFIG GET databases"
  docker exec "${REDIS_CONTAINER}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning \
    CONFIG GET databases 2>/dev/null | sed 's/^/      /'

  echo "  → Redis INFO keyspace"
  local ks
  ks="$(docker exec "${REDIS_CONTAINER}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning \
    INFO keyspace 2>/dev/null)"
  echo "${ks}" | sed 's/^/      /'

  if echo "${ks}" | grep -q '^db0:'; then
    log_info "db0 contient des clés (cohérent avec l'audit db0:keys=9446)"
  else
    log_warn "db0 vide — vérifier que le backend pointe bien sur la même DB"
  fi

  echo "  → REDIS_DB côté conteneurs"
  local backend_db celery_db
  backend_db="$(docker exec "${BACKEND_CONTAINER}" sh -c 'echo "${REDIS_DB:-?}"' 2>/dev/null || echo '?')"
  celery_db="$(docker exec "${CELERY_CONTAINER}" sh -c 'echo "${REDIS_DB:-?}"' 2>/dev/null || echo '?')"
  printf "      backend REDIS_DB=%s ; celery-worker REDIS_DB=%s\n" "${backend_db}" "${celery_db}"
  if [[ "${backend_db}" == "0" && "${celery_db}" == "0" ]]; then
    log_info "backend et celery sur DB 0 (cohérent avec compose)"
  else
    log_fail "DB Redis incohérente (backend=${backend_db}, celery=${celery_db})"
    ok=0
  fi
  return $((ok == 1 ? 0 : 1))
}

# -- Phase 3.1 : POST device-health x3 ------------------------------------
cmd_heartbeat_replay() {
  section "Phase 3.1 — POST /driver/me/device-health × 3"
  require_var DRIVER_TEST_JWT || return 2
  local i code body
  for i in 1 2 3; do
    printf "  → POST #%d\n" "${i}"
    code="$(curl -sS -o /tmp/_dh_resp.json -w '%{http_code}' \
      -X POST "${PUBLIC_BASE_URL}/api/v1/driver/me/device-health" \
      -H "Authorization: Bearer ${DRIVER_TEST_JWT}" \
      -H "Content-Type: application/json" \
      --max-time 15 \
      -d '{"manufacturer":"Test","fgs_running":true,"battery_optimized":false,"fg_permission":"granted","bg_permission":"granted","gps_provider_enabled":true,"fix_success_rate_last_5min":1.0}' \
      || echo "000")"
    body="$(cat /tmp/_dh_resp.json 2>/dev/null || true)"
    if [[ "${code}" != "200" ]]; then
      log_fail "POST #${i} HTTP ${code} : ${body}"
      return 1
    fi
    if ! echo "${body}" | grep -q '"ok"[[:space:]]*:[[:space:]]*true'; then
      log_fail "POST #${i} : ok!=true : ${body}"
      return 1
    fi
    log_info "POST #${i} HTTP ${code} ok=true"
    sleep 5
  done
  return 0
}

# -- Phase 3.3 : SCAN dual-write keys + HGETALL + TTL --------------------
cmd_redis_keys() {
  section "Phase 3.3 — SCAN driver:*:health & driver:*:device_health"
  load_redis_password || return 2
  local ok=1

  echo "  → SCAN driver:*:health (db 0)"
  local hk_count
  hk_count="$(docker exec "${REDIS_CONTAINER}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning \
    -n 0 --scan --pattern 'driver:*:health' 2>/dev/null | tee /tmp/_dh_keys.txt | wc -l | tr -d ' ')"
  sed 's/^/      /' /tmp/_dh_keys.txt | head -10
  if (( hk_count >= 1 )); then
    log_info "driver:*:health : ${hk_count} clé(s)"
  else
    log_fail "driver:*:health : 0 clé"
    ok=0
  fi

  echo "  → SCAN driver:*:device_health (db 0)"
  local dh_count
  dh_count="$(docker exec "${REDIS_CONTAINER}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning \
    -n 0 --scan --pattern 'driver:*:device_health' 2>/dev/null | tee /tmp/_dh_keys2.txt | wc -l | tr -d ' ')"
  sed 's/^/      /' /tmp/_dh_keys2.txt | head -10
  if (( dh_count >= 1 )); then
    log_info "driver:*:device_health : ${dh_count} clé(s)"
  else
    log_fail "driver:*:device_health : 0 clé"
    ok=0
  fi

  if [[ -n "${DRIVER_ID:-}" ]]; then
    echo "  → HGETALL driver:${DRIVER_ID}:health"
    docker exec "${REDIS_CONTAINER}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning \
      -n 0 HGETALL "driver:${DRIVER_ID}:health" 2>/dev/null | sed 's/^/      /'
    local ttl1 ttl2
    ttl1="$(docker exec "${REDIS_CONTAINER}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning \
      -n 0 TTL "driver:${DRIVER_ID}:health" 2>/dev/null || echo -2)"
    ttl2="$(docker exec "${REDIS_CONTAINER}" redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning \
      -n 0 TTL "driver:${DRIVER_ID}:device_health" 2>/dev/null || echo -2)"
    printf "      TTL health=%s ; TTL device_health=%s\n" "${ttl1}" "${ttl2}"
    if (( ttl1 > 0 )); then log_info "TTL driver:${DRIVER_ID}:health = ${ttl1}s"; else log_fail "TTL driver:${DRIVER_ID}:health invalide"; ok=0; fi
    if (( ttl2 > 0 )); then log_info "TTL driver:${DRIVER_ID}:device_health = ${ttl2}s"; else log_fail "TTL driver:${DRIVER_ID}:device_health invalide"; ok=0; fi
  else
    log_warn "DRIVER_ID non défini — HGETALL/TTL skipped (à fournir pour vérification fine)"
  fi
  return $((ok == 1 ? 0 : 1))
}

# -- Phase 3.4 : metrics device-health ------------------------------------
cmd_metrics() {
  section "Phase 3.4 — metrics device-health (Prometheus)"
  echo "  → driver_device_health_reports_total"
  curl -sS --max-time 10 "${BACKEND_URL}/api/v1/prometheus/metrics" \
    | grep -E '^driver_device_health_reports_total|^driver_device_stale_fix_total' \
    | sed 's/^/      /'
  local rep stale
  rep="$(_metric_value '^driver_device_health_reports_total')"
  stale="$(_metric_value '^driver_device_stale_fix_total')"
  printf "  Σ reports=%s ; Σ stale_fix=%s\n" "${rep}" "${stale}"
  if (( rep >= 1 )); then
    log_info "driver_device_health_reports_total = ${rep}"
    return 0
  fi
  log_fail "driver_device_health_reports_total = 0 (aucun POST device-health vu par le backend)"
  return 1
}

# -- Phase 3.5 : alerte stale + silent wake -------------------------------
cmd_stale_alert() {
  section "Phase 3.5 — TrackingStaleHigh status"
  local stale rep ratio
  stale="$(_metric_value '^driver_device_stale_fix_total')"
  rep="$(_metric_value '^driver_device_health_reports_total')"
  if (( rep == 0 )); then
    log_warn "reports=0 — ratio non significatif (envoyer des heartbeats avant)"
    return 0
  fi
  ratio="$(awk -v s="${stale}" -v r="${rep}" 'BEGIN{ if (r==0) {print 0} else {printf "%.4f", s/r} }')"
  printf "  stale=%s reports=%s → ratio=%s (seuil alerte %.2f)\n" "${stale}" "${rep}" "${ratio}" "0.20"
  if awk -v r="${ratio}" 'BEGIN{ exit !(r > 0.20) }'; then
    log_fail "ratio > 0.20 — TrackingStaleHigh va firing (à investiguer : drivers réellement stale OU faux positif métrique)"
    return 1
  fi
  log_info "ratio ≤ 0.20 — TrackingStaleHigh devrait se résoudre après for: 10m"
  return 0
}

cmd_silent_wake() {
  section "Phase 3.5 — silent_push_wake_total (lecture pure)"
  local sent acked ratio
  sent="$(_metric_value '^silent_push_wake_total\\{result="sent"')"
  acked="$(_metric_value '^silent_push_wake_total\\{result="acked"')"
  printf "  sent=%s ; acked=%s\n" "${sent}" "${acked}"
  if (( sent == 0 )); then
    log_warn "sent = 0 — déclencher un silent push (Phase 3.5 procédure manuelle iPhone) avant d'évaluer le ratio"
    return 1
  fi
  ratio="$(awk -v s="${sent}" -v a="${acked}" 'BEGIN{ if (s==0) {print 0} else {printf "%.4f", a/s} }')"
  printf "  ratio acked/sent = %s (seuil ≥ 0.80)\n" "${ratio}"
  if awk -v r="${ratio}" 'BEGIN{ exit !(r >= 0.80) }'; then
    log_info "ratio ≥ 0.80 — TrackingSilentWakeLow non firing"
    return 0
  fi
  log_fail "ratio < 0.80 — TrackingSilentWakeLow firing (handler mobile à vérifier)"
  return 1
}

# -- Phase 3 enchaînée ----------------------------------------------------
cmd_phase3() {
  local ok=1
  cmd_redis_verify     || ok=0
  cmd_heartbeat_replay || ok=0
  cmd_redis_keys       || ok=0
  cmd_metrics          || ok=0
  cmd_stale_alert      || ok=0
  cmd_silent_wake      || ok=0
  return $((ok == 1 ? 0 : 1))
}

# -- All ------------------------------------------------------------------
cmd_all() {
  local ok=1
  cmd_stop_gate || { ok=0; section "ABANDON : STOP GATE Kafka KO"; return 1; }
  cmd_phase3    || ok=0
  return $((ok == 1 ? 0 : 1))
}

# -- Dispatch -------------------------------------------------------------
usage() {
  sed -n '2,30p' "$0"
}

MODE="${1:-}"
case "${MODE}" in
  consumer-groups)   cmd_consumer_groups ;;
  propagation)       cmd_propagation ;;
  stop-gate)         cmd_stop_gate ;;
  redis-verify)      cmd_redis_verify ;;
  heartbeat-replay)  cmd_heartbeat_replay ;;
  redis-keys)        cmd_redis_keys ;;
  metrics)           cmd_metrics ;;
  stale-alert)       cmd_stale_alert ;;
  silent-wake)       cmd_silent_wake ;;
  phase3)            cmd_phase3 ;;
  all)               cmd_all ;;
  ""|-h|--help)      usage; exit 2 ;;
  *)                 echo "Mode inconnu : ${MODE}" >&2; usage >&2; exit 2 ;;
esac
