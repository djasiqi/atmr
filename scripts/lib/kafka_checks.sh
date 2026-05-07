#!/usr/bin/env bash
# shellcheck shell=bash
# Bibliothèque sourceable — pas de `set -e` (géré par l’appelant si besoin).
# Contrat : définir ATMR_ENV_FILE (ou ROOT + défaut .env.production) avant source.

KAFKA_REQUIRED_FLAGS=(
  KAFKA_ENABLED
  TRACKING_INGEST_ASYNC_ENABLED
  TRACKING_PROCESSED_FANOUT_ENABLED
  WS_KAFKA_CONSUMER_ENABLED
)

KAFKA_REQUIRED_COMPOSE_FILES=(
  docker-compose.production.yml
  docker-compose.kafka.yml
  docker-compose.kafka.atmr-network.yml
)

KAFKA_EXPECTED_TOPICS=(
  driver.location.raw
  driver.location.processed
  driver.location.dlq
  notifications.push
  notifications.sms
  notifications.email
  notifications.dlq
  mission.events
  notification.events
  dispatch.events
)

# Consumers du profile `kafka` (docker-compose.production.yml)
KAFKA_CONSUMER_SERVICES=(
  tracking-kafka-consumer
  tracking-processed-fanout
  kafka-dlq-consumer
)

KAFKA_BROKER_CONTAINERS=(
  atmr-kafka-broker-1
  atmr-kafka-broker-2
  atmr-kafka-broker-3
)

# Commande docker compose « Kafka ON » (3 YAML + profile kafka)
kafka_docker_compose() {
  docker compose \
    -f docker-compose.production.yml \
    -f docker-compose.kafka.yml \
    -f docker-compose.kafka.atmr-network.yml \
    --profile kafka \
    "$@"
}

read_env_flag() {
  local name="$1"
  local default="${2:-false}"
  local v=""
  local envf="${ATMR_ENV_FILE:-}"
  if [[ -n "${envf}" ]] && [[ -f "${envf}" ]]; then
    v="$(grep -E "^${name}=" "${envf}" 2>/dev/null | tail -n1 | cut -d'=' -f2-)"
    v="${v//\'/}"
    v="${v//\"/}"
    v="${v// /}"
    v="$(printf '%s' "${v}" | tr '[:upper:]' '[:lower:]')"
    if [[ -n "${v}" ]]; then
      printf '%s\n' "${v}"
      return
    fi
  fi
  local indirect="${name}"
  printf '%s' "${!indirect:-${default}}" | tr '[:upper:]' '[:lower:]'
}

# Première adresse bootstrap (hôte:port) — ne pas passer par read_env_flag (casse les URLs).
kafka_bootstrap_first() {
  if [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    local b="${KAFKA_BOOTSTRAP_SERVERS%%,*}"
    printf '%s\n' "${b}"
    return
  fi
  local envf="${ATMR_ENV_FILE:-}"
  local v=""
  if [[ -n "${envf}" ]] && [[ -f "${envf}" ]]; then
    v="$(grep -E "^KAFKA_BOOTSTRAP_SERVERS=" "${envf}" 2>/dev/null | tail -n1 | cut -d'=' -f2-)"
    v="${v//\'/}"
    v="${v//\"/}"
    v="${v// /}"
    if [[ -n "${v}" ]]; then
      printf '%s\n' "${v%%,*}"
      return
    fi
  fi
  printf '%s\n' "kafka-broker-1:29092"
}

log_info() { echo "[OK] $*"; }
log_warn() { echo "[WARN] $*" >&2; }
log_fail() { echo "[FAIL] $*" >&2; }

log_force_override() {
  local stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local user="${USER:-${USERNAME:-unknown}}"
  echo "[FORCE_OVERRIDE] ts=${stamp} user=${user} script=${0##*/} env_file=${ATMR_ENV_FILE:-unset}"
  if [[ -n "${ATMR_FORCE_AUDIT_LOG:-}" ]] && [[ -w "$(dirname "${ATMR_FORCE_AUDIT_LOG}")" ]]; then
    echo "ts=${stamp} user=${user} script=${0##*/}" >>"${ATMR_FORCE_AUDIT_LOG}"
  fi
}

kafka_check_flags_all_true() {
  local incoherent=()
  local f v
  for f in "${KAFKA_REQUIRED_FLAGS[@]}"; do
    v="$(read_env_flag "${f}")"
    if [[ "${v}" != "true" ]]; then
      incoherent+=("${f}=${v:-<empty>}")
    fi
  done
  if ((${#incoherent[@]})); then
    log_fail "flags Kafka : attendu true pour les 4 — détail : ${incoherent[*]}"
    return 1
  fi
  log_info "flags Kafka (4/4 = true)"
  return 0
}

kafka_check_flags_all_false() {
  local bad=()
  local f v
  for f in "${KAFKA_REQUIRED_FLAGS[@]}"; do
    v="$(read_env_flag "${f}")"
    if [[ "${v}" == "true" ]]; then
      bad+=("${f}=true")
    fi
  done
  if ((${#bad[@]})); then
    log_fail "mode OFF : les flags suivants doivent être false ou absents — ${bad[*]}"
    return 1
  fi
  log_info "flags Kafka OFF (aucun true)"
  return 0
}

kafka_check_compose_files() {
  local f
  for f in "${KAFKA_REQUIRED_COMPOSE_FILES[@]}"; do
    if [[ ! -f "${f}" ]]; then
      log_fail "fichier Compose manquant : ${f}"
      return 1
    fi
  done
  log_info "fichiers Compose Kafka (3/3 présents)"
  return 0
}

kafka_check_atmr_network() {
  if docker network inspect atmr-network >/dev/null 2>&1; then
    log_info "réseau Docker atmr-network présent"
    return 0
  fi
  if [[ "${ATMR_AUTO_CREATE_NETWORK:-1}" == "1" ]]; then
    log_warn "création du réseau atmr-network (ATMR_AUTO_CREATE_NETWORK=1)"
    docker network create atmr-network
    log_info "réseau atmr-network créé"
    return 0
  fi
  log_fail "réseau atmr-network absent et ATMR_AUTO_CREATE_NETWORK!=1 — créer : docker network create atmr-network"
  return 1
}

kafka_check_compose_resolution() {
  local services
  services="$(kafka_docker_compose config --services 2>/dev/null || true)"
  local svc
  for svc in kafka-broker-1 kafka-broker-2 kafka-broker-3 "${KAFKA_CONSUMER_SERVICES[@]}"; do
    if ! grep -qx "${svc}" <<<"${services}"; then
      log_fail "service Compose « ${svc} » absent du merge (oubli -f / profile ?)"
      return 1
    fi
  done
  log_info "résolution Compose : brokers + consumers (profile kafka)"
  return 0
}

kafka_wait_brokers_healthy() {
  local timeout_s="${1:-180}"
  local deadline=$((SECONDS + timeout_s))
  local c status healthy
  while ((SECONDS < deadline)); do
    healthy=0
    for c in "${KAFKA_BROKER_CONTAINERS[@]}"; do
      status="$(docker inspect "${c}" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>/dev/null || echo missing)"
      if [[ "${status}" == "healthy" ]]; then
        healthy=$((healthy + 1))
      fi
    done
    if ((healthy == 3)); then
      log_info "brokers Kafka healthy (3/3) via docker inspect"
      return 0
    fi
    sleep 5
  done
  log_fail "timeout brokers healthy (${timeout_s}s) — docker inspect ${KAFKA_BROKER_CONTAINERS[0]} …"
  return 1
}

kafka_check_dns_from_atmr_network() {
  if docker compose -f docker-compose.production.yml exec -T backend getent hosts kafka-broker-1 >/dev/null 2>&1; then
    log_info "DNS kafka-broker-1 depuis backend (atmr-network)"
    return 0
  fi
  if docker exec atmr-kafka-broker-1 getent hosts kafka-broker-1 >/dev/null 2>&1; then
    log_info "DNS kafka-broker-1 depuis atmr-kafka-broker-1"
    return 0
  fi
  log_fail "DNS kafka-broker-1 introuvable sur atmr-network — vérifier docker-compose.kafka.atmr-network.yml"
  return 1
}

kafka_check_broker_api() {
  local first
  first="$(kafka_bootstrap_first)"
  if ! kafka_docker_compose exec -T kafka-broker-1 kafka-broker-api-versions --bootstrap-server "${first}" >/dev/null 2>&1; then
    log_fail "kafka-broker-api-versions KO (bootstrap=${first})"
    return 1
  fi
  log_info "kafka-broker-api-versions OK (${first})"
  return 0
}

kafka_check_topics_exist() {
  local first
  first="$(kafka_bootstrap_first)"
  local listed
  if ! listed="$(kafka_docker_compose exec -T kafka-broker-1 kafka-topics --bootstrap-server "${first}" --list 2>/dev/null)"; then
    log_fail "impossible de lister les topics Kafka"
    return 1
  fi
  local missing=()
  local t
  for t in "${KAFKA_EXPECTED_TOPICS[@]}"; do
    if ! grep -qx "${t}" <<<"${listed}"; then
      missing+=("${t}")
    fi
  done
  if ((${#missing[@]})); then
    log_fail "topics manquants : ${missing[*]}"
    return 1
  fi
  log_info "topics attendus présents (${#KAFKA_EXPECTED_TOPICS[@]})"
  return 0
}

kafka_check_consumers_running() {
  local svc
  for svc in "${KAFKA_CONSUMER_SERVICES[@]}"; do
    local ids
    ids="$(kafka_docker_compose ps -q --status running "${svc}" 2>/dev/null || true)"
    if [[ -z "${ids}" ]]; then
      log_fail "consumer non running : ${svc}"
      return 1
    fi
  done
  log_info "consumers profile kafka running (${#KAFKA_CONSUMER_SERVICES[@]})"
  return 0
}

kafka_check_no_consumers_running() {
  local found=()
  local st
  st="$(docker inspect atmr-kafka-dlq-consumer --format '{{.State.Status}}' 2>/dev/null || echo absent)"
  if [[ "${st}" == "running" ]]; then
    found+=("atmr-kafka-dlq-consumer")
  fi
  local svc
  for svc in tracking-kafka-consumer tracking-processed-fanout kafka-consumer; do
    local n
    n="$(docker ps -q --filter "label=com.docker.compose.service=${svc}" --filter "status=running" 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${n}" != "0" ]]; then
      found+=("${svc}(${n})")
    fi
  done
  if ((${#found[@]})); then
    log_fail "consumers Kafka encore actifs (mode OFF) : ${found[*]}"
    return 1
  fi
  log_info "aucun consumer Kafka actif (OFF)"
  return 0
}

kafka_check_backend_healthy() {
  local cid
  cid="$(docker compose -f docker-compose.production.yml ps -q backend 2>/dev/null | head -n1)"
  if [[ -z "${cid}" ]]; then
    log_fail "conteneur backend introuvable (docker compose ps -q backend)"
    return 1
  fi
  local hs
  hs="$(docker inspect "${cid}" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>/dev/null || echo missing)"
  if [[ "${hs}" != "healthy" ]]; then
    log_fail "backend health=${hs} (attendu healthy)"
    return 1
  fi
  log_info "backend healthy"
  return 0
}

kafka_summary() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Résumé checks Kafka — repérer [OK] / [FAIL] ci-dessus."
  echo "Env lu : ATMR_ENV_FILE=${ATMR_ENV_FILE:-<non défini>}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}
