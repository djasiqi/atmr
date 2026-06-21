#!/usr/bin/env bash
# Test T13 production — 10 envois GPS E2E + latence HTTP + offsets v2 + P95 Prometheus.
# À exécuter SUR LE SERVEUR (docker + réseau atmr).
#
# Usage :
#   ATMR_DEPLOY_ROOT=/srv/atmr ./scripts/run-t13-prod-gps-test.sh
#   DRIVER_TEST_JWT=... ./scripts/run-t13-prod-gps-test.sh   # optionnel si JWT connu
#
# Codes : 0 OK, 1 échec propagation/latence, 2 prérequis manquant

set -uo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production.kafka-effective}"
if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${ROOT}/.env.production"
fi
export ATMR_ENV_FILE="${ENV_FILE}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-https://api.lirie.ch}"
BACKEND_URL="${BACKEND_URL:-http://localhost:5000}"
PROMETHEUS_URL="${PROMETHEUS_METRICS_URL:-http://localhost:9090}"
SEND_COUNT="${T13_SEND_COUNT:-10}"
PROPAGATION_WAIT_S="${PROPAGATION_WAIT_S:-15}"
KAFKA_BROKER_CONTAINER="${KAFKA_BROKER_CONTAINER:-atmr-kafka-broker-1}"
BACKEND_CONTAINER="${BACKEND_CONTAINER:-atmr-backend-1}"

TOPIC_RAW="$(kafka_read_env_value KAFKA_TOPIC_DRIVER_LOCATION_RAW driver.location.raw)"
TOPIC_PROCESSED="$(kafka_read_env_value KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed)"

log() { printf '[T13] %s\n' "$*"; }
fail() { log "FAIL: $*"; exit 1; }

_kafka_offset() {
  local topic="$1"
  docker exec "${KAFKA_BROKER_CONTAINER}" kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list kafka-broker-1:29092 --topic "${topic}" --time -1 2>/dev/null \
    | awk -F: '{s+=$3} END {print s+0}'
}

_prom_p95_e2e() {
  curl -sf "${PROMETHEUS_URL}/api/v1/query" \
    --data-urlencode 'query=histogram_quantile(0.95, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))' 2>/dev/null \
    | grep -oE '"value":\[[^]]+\]' | tail -1 | grep -oE '[0-9.]+$' || true
}

_metric_count_e2e() {
  curl -sf "${BACKEND_URL}/api/v1/prometheus/metrics" 2>/dev/null \
    | grep -E '^tracking_kafka_e2e_latency_seconds_count' \
    | awk '{s+=$2} END {print s+0}'
}

_ensure_jwt() {
  if [[ -n "${DRIVER_TEST_JWT:-}" ]]; then
    return 0
  fi
  local pyfile
  pyfile="$(mktemp /tmp/t13-jwt.XXXXXX.py)"
  cat >"${pyfile}" <<'PY'
from app import create_app
from flask_jwt_extended import create_access_token
from models.driver import Driver

app = create_app("production")
with app.app_context():
    d = Driver.query.filter_by(is_active=True).first()
    if not d or not d.user:
        raise SystemExit("NO_DRIVER")
    tok = create_access_token(
        identity=str(d.user.public_id),
        additional_claims={
            "role": "driver",
            "company_id": d.company_id,
            "driver_id": d.id,
            "aud": "atmr-api",
        },
    )
    print(f"DRIVER_ID={d.id}")
    print(f"JWT={tok}")
PY
  docker cp "${pyfile}" "${BACKEND_CONTAINER}:/tmp/t13_jwt.py" >/dev/null
  rm -f "${pyfile}"
  local out
  out="$(docker exec "${BACKEND_CONTAINER}" python3 /tmp/t13_jwt.py 2>/dev/null)" || fail "génération JWT impossible"
  if grep -q '^NO_DRIVER' <<<"${out}"; then
    fail "aucun chauffeur actif en base"
  fi
  DRIVER_ID="$(grep '^DRIVER_ID=' <<<"${out}" | cut -d= -f2-)"
  DRIVER_TEST_JWT="$(grep '^JWT=' <<<"${out}" | cut -d= -f2-)"
  export DRIVER_ID DRIVER_TEST_JWT
  log "JWT généré pour driver_id=${DRIVER_ID}"
}

log "Topics : raw=${TOPIC_RAW} processed=${TOPIC_PROCESSED}"
log "Env    : ${ENV_FILE}"

_ensure_jwt

raw_before="$(_kafka_offset "${TOPIC_RAW}")"
proc_before="$(_kafka_offset "${TOPIC_PROCESSED}")"
e2e_count_before="$(_metric_count_e2e)"
log "Offsets AVANT  : raw=${raw_before} processed=${proc_before} e2e_count=${e2e_count_before}"

declare -a http_times=()
base_lat="46.2044"
base_lon="6.1432"
ok_puts=0

for i in $(seq 1 "${SEND_COUNT}"); do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Légère variation pour éviter déduplication stricte
  lat="$(awk -v b="${base_lat}" -v n="${i}" 'BEGIN{printf "%.6f", b + n * 0.00001}')"
  lon="$(awk -v b="${base_lon}" -v n="${i}" 'BEGIN{printf "%.6f", b + n * 0.00001}')"
  idem_key="t13-$(date +%s)-${i}-$$"
  resp_file="/tmp/t13_put_${i}.json"
  code="$(curl -sS -o "${resp_file}" -w '%{http_code} %{time_total}' \
    -X PUT "${PUBLIC_BASE_URL}/api/v1/driver/me/location" \
    -H "Authorization: Bearer ${DRIVER_TEST_JWT}" \
    -H "Content-Type: application/json" \
    -H "Idempotency-Key: ${idem_key}" \
    --max-time 15 \
    -d "{\"latitude\":${lat},\"longitude\":${lon},\"accuracy\":10.0,\"speed\":0.0,\"heading\":0.0,\"timestamp\":\"${ts}\",\"location_mode\":\"mission_live\"}" \
    2>/dev/null || echo "000 0")"
  http_code="${code%% *}"
  http_time="${code##* }"
  http_times+=("${http_time}")
  if [[ "${http_code}" == "200" || "${http_code}" == "201" || "${http_code}" == "202" || "${http_code}" == "204" ]]; then
    ok_puts=$((ok_puts + 1))
    log "PUT ${i}/${SEND_COUNT} HTTP ${http_code} time=${http_time}s queued=$(grep -o '"queued":[^,}]*' "${resp_file}" 2>/dev/null || true)"
  else
    log "PUT ${i}/${SEND_COUNT} HTTP ${http_code} — $(tr -d '\n' <"${resp_file}" | head -c 200)"
  fi
  sleep 0.3
done

log "Attente propagation ${PROPAGATION_WAIT_S}s..."
sleep "${PROPAGATION_WAIT_S}"

raw_after="$(_kafka_offset "${TOPIC_RAW}")"
proc_after="$(_kafka_offset "${TOPIC_PROCESSED}")"
e2e_count_after="$(_metric_count_e2e)"
d_raw=$((raw_after - raw_before))
d_proc=$((proc_after - proc_before))
d_e2e=$((e2e_count_after - e2e_count_before))

log "Offsets APRÈS : raw=${raw_after} processed=${proc_after} (Δ raw=${d_raw} processed=${d_proc})"
log "e2e_count Δ=${d_e2e} (avant=${e2e_count_before} après=${e2e_count_after})"

# P95 Prometheus (peut être NaN si fenêtre rate trop courte — on retente)
e2e_p95=""
for _ in 1 2 3; do
  e2e_p95="$(_prom_p95_e2e)"
  if [[ -n "${e2e_p95}" && "${e2e_p95}" != "NaN" ]]; then
    break
  fi
  sleep 5
done

# Stats HTTP (temps réponse API, proxy latence perçue carte côté admission)
max_http="0"
sum_http="0"
for t in "${http_times[@]}"; do
  sum_http="$(awk -v s="${sum_http}" -v t="${t}" 'BEGIN{printf "%.6f", s+t}')"
  max_http="$(awk -v m="${max_http}" -v t="${t}" 'BEGIN{if(t+0>m+0) print t; else print m}')"
done
avg_http="$(awk -v s="${sum_http}" -v n="${#http_times[@]}" 'BEGIN{if(n>0) printf "%.3f", s/n; else print "0"}')"

log "HTTP admission : ok=${ok_puts}/${SEND_COUNT} avg=${avg_http}s max=${max_http}s"
if [[ -n "${e2e_p95}" && "${e2e_p95}" != "NaN" ]]; then
  log "Prometheus P95 tracking_kafka_e2e = ${e2e_p95}s"
else
  log "Prometheus P95 indisponible (fenêtre rate ou métrique absente)"
fi

exit_code=0
if ((ok_puts < SEND_COUNT)); then
  log "Échec : moins de ${SEND_COUNT} PUT acceptés"
  exit_code=1
fi
if ((d_raw < SEND_COUNT)); then
  log "Échec : raw offset +${d_raw} < ${SEND_COUNT}"
  exit_code=1
fi
if ((d_proc < SEND_COUNT)); then
  log "Échec : processed offset +${d_proc} < ${SEND_COUNT}"
  exit_code=1
fi

# Seuil T13 : P95 E2E < 2s OU si P95 indisponible, max HTTP admission < 2s + pipeline OK
if [[ -n "${e2e_p95}" && "${e2e_p95}" != "NaN" ]]; then
  if awk -v v="${e2e_p95}" 'BEGIN{exit !(v+0 > 2)}'; then
    log "Échec T13 : P95 E2E ${e2e_p95}s > 2s"
    exit_code=1
  else
    log "OK T13 : P95 E2E ${e2e_p95}s < 2s"
  fi
elif awk -v m="${max_http}" 'BEGIN{exit !(m+0 > 2)}'; then
  log "WARN T13 : P95 Prometheus absent, max HTTP ${max_http}s > 2s"
  exit_code=1
else
  log "OK T13 (proxy) : pipeline E2E OK, max HTTP admission ${max_http}s < 2s (P95 Prometheus à confirmer sur Grafana)"
fi

printf '\n=== RÉSUMÉ T13 ===\n'
printf 'driver_id=%s puts_ok=%d/%d d_raw=%d d_proc=%d http_avg=%ss http_max=%ss p95_e2e=%s\n' \
  "${DRIVER_ID:-?}" "${ok_puts}" "${SEND_COUNT}" "${d_raw}" "${d_proc}" "${avg_http}" "${max_http}" "${e2e_p95:-n/a}"

exit "${exit_code}"
