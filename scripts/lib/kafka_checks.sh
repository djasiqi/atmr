#!/usr/bin/env bash
# shellcheck shell=bash
# Bibliothèque sourceable — pas de `set -e` (géré par l’appelant si besoin).
# Contrat : définir ATMR_ENV_FILE (ou ROOT + défaut .env.production) avant source.

KAFKA_REQUIRED_FLAGS=(
  KAFKA_ENABLED
  TRACKING_INGEST_ASYNC_ENABLED
  TRACKING_PROCESSED_FANOUT_ENABLED
  WS_KAFKA_CONSUMER_ENABLED
  TRACKING_INGEST_PERSIST_ENABLED
)

KAFKA_REQUIRED_COMPOSE_FILES=(
  docker-compose.production.yml
)

# Fichier compose Kafka (3 brokers par défaut ; docker-compose.kafka.single.yml en Phase 2)
KAFKA_COMPOSE_FILE="${KAFKA_COMPOSE_FILE:-docker-compose.kafka.yml}"
KAFKA_NETWORK_FILE="${KAFKA_NETWORK_FILE:-}"

kafka_resolve_network_file() {
  if [[ -n "${KAFKA_NETWORK_FILE}" ]]; then
    printf '%s\n' "${KAFKA_NETWORK_FILE}"
    return
  fi
  if [[ "${KAFKA_COMPOSE_FILE}" == *single* ]] || [[ "${KAFKA_COMPOSE_FILE}" == *kraft* ]]; then
    printf '%s\n' "docker-compose.kafka.atmr-network.single.yml"
  else
    printf '%s\n' "docker-compose.kafka.atmr-network.yml"
  fi
}

# Topics actifs — noms résolus depuis .env (suffixe .v2 en Phase 1 prod)
KAFKA_EXPECTED_TOPICS=()

kafka_read_env_value() {
  local name="$1"
  local default="${2:-}"
  local v=""
  local envf="${ATMR_ENV_FILE:-}"
  if [[ -n "${envf}" ]] && [[ -f "${envf}" ]]; then
    v="$(grep -E "^${name}=" "${envf}" 2>/dev/null | tail -n1 | cut -d'=' -f2-)"
    v="${v//\'/}"
    v="${v//\"/}"
    if [[ -n "${v}" ]]; then
      printf '%s\n' "${v}"
      return
    fi
  fi
  printf '%s\n' "${default}"
}

kafka_build_expected_topics() {
  KAFKA_EXPECTED_TOPICS=(
    "$(kafka_read_env_value KAFKA_TOPIC_DRIVER_LOCATION_RAW driver.location.raw)"
    "$(kafka_read_env_value KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed)"
    "$(kafka_read_env_value KAFKA_TOPIC_DRIVER_LOCATION_DLQ driver.location.dlq)"
    "$(kafka_read_env_value KAFKA_TOPIC_NOTIFICATIONS_DLQ notifications.dlq)"
    "$(kafka_read_env_value KAFKA_OPS_SMOKE_TOPIC atmr.ops.smoke)"
  )
  local create_inactive
  create_inactive="$(kafka_read_env_value KAFKA_CREATE_INACTIVE_TOPICS false | tr '[:upper:]' '[:lower:]')"
  if [[ "${create_inactive}" == "true" ]]; then
    KAFKA_EXPECTED_TOPICS+=(
      notifications.push
      notifications.sms
      notifications.email
      mission.events
      notification.events
      dispatch.events
    )
  fi
}

# Consumers du profile `kafka` (docker-compose.production.yml)
KAFKA_CONSUMER_SERVICES=(
  tracking-kafka-consumer
  tracking-processed-fanout
  kafka-dlq-consumer
)

KAFKA_BROKER_CONTAINERS=()

kafka_refresh_broker_containers() {
  KAFKA_BROKER_CONTAINERS=()
  local svc
  while IFS= read -r svc; do
    [[ -z "${svc}" ]] && continue
    KAFKA_BROKER_CONTAINERS+=("atmr-${svc}")
  done < <(kafka_docker_compose config --services 2>/dev/null | grep '^kafka-broker-' || true)
}

kafka_count_running_brokers() {
  kafka_refresh_broker_containers
  local c running=0
  for c in "${KAFKA_BROKER_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' | grep -qx "${c}"; then
      running=$((running + 1))
    fi
  done
  printf '%s\n' "${running}"
}

kafka_check_brokers_running() {
  local expected running
  kafka_refresh_broker_containers
  expected="${#KAFKA_BROKER_CONTAINERS[@]}"
  running="$(kafka_count_running_brokers | tr -d '[:space:]')"
  if [[ ! "${running}" =~ ^[0-9]+$ ]] || [[ "${running}" -lt 1 ]]; then
    log_fail "aucun broker Kafka actif (0/${expected}) — lancer scripts/deploy-kafka-production.sh"
    return 1
  fi
  log_info "brokers Kafka actifs (${running}/${expected})"
  return 0
}

kafka_dns_probe_ok() {
  local probe="$1"
  docker exec "${probe}" getent hosts kafka-broker-1 >/dev/null 2>&1 \
    && docker exec "${probe}" getent hosts kafka-broker-2 >/dev/null 2>&1
}

kafka_dns_ephemeral_probe_ok() {
  docker network inspect atmr-network >/dev/null 2>&1 || return 1
  docker run --rm --network atmr-network alpine:3.20 getent hosts kafka-broker-1 >/dev/null 2>&1 \
    && docker run --rm --network atmr-network alpine:3.20 getent hosts kafka-broker-2 >/dev/null 2>&1
}

# Commande docker compose « Kafka ON » (production + kafka + réseau + profile kafka)
kafka_docker_compose() {
  local net
  net="$(kafka_resolve_network_file)"
  docker compose \
    -f docker-compose.production.yml \
    -f "${KAFKA_COMPOSE_FILE}" \
    -f "${net}" \
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
    log_fail "flags Kafka : attendu true pour les ${#KAFKA_REQUIRED_FLAGS[@]} — détail : ${incoherent[*]}"
    return 1
  fi
  log_info "flags Kafka (${#KAFKA_REQUIRED_FLAGS[@]}/${#KAFKA_REQUIRED_FLAGS[@]} = true)"
  return 0
}

kafka_check_tracking_persist_coherence() {
  local async persist allow
  async="$(read_env_flag TRACKING_INGEST_ASYNC_ENABLED)"
  persist="$(read_env_flag TRACKING_INGEST_PERSIST_ENABLED)"
  allow="$(read_env_flag TRACKING_INGEST_ALLOW_REPUBLISH_ONLY)"
  if [[ "${async}" != "true" ]]; then
    log_info "ingest async OFF — cohérence persist non requise"
    return 0
  fi
  if [[ "${persist}" == "true" ]]; then
    log_info "TRACKING_INGEST_PERSIST_ENABLED=true (mode normal)"
    return 0
  fi
  if [[ "${allow}" == "true" ]]; then
    log_warn "MODE REPUBLISH-ONLY : PERSIST=false avec ALLOW_REPUBLISH_ONLY=true"
    return 0
  fi
  log_fail "TRACKING_INGEST_ASYNC_ENABLED=true exige TRACKING_INGEST_PERSIST_ENABLED=true ou TRACKING_INGEST_ALLOW_REPUBLISH_ONLY=true (actuel PERSIST=${persist:-<empty>})"
  return 1
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
  local f net
  for f in "${KAFKA_REQUIRED_COMPOSE_FILES[@]}"; do
    if [[ ! -f "${f}" ]]; then
      log_fail "fichier Compose manquant : ${f}"
      return 1
    fi
  done
  if [[ ! -f "${KAFKA_COMPOSE_FILE}" ]]; then
    log_fail "fichier Compose Kafka manquant : ${KAFKA_COMPOSE_FILE}"
    return 1
  fi
  net="$(kafka_resolve_network_file)"
  if [[ ! -f "${net}" ]]; then
    log_fail "fichier réseau Kafka manquant : ${net}"
    return 1
  fi
  log_info "fichiers Compose Kafka présents (compose=${KAFKA_COMPOSE_FILE}, network=${net})"
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

kafka_read_replication_factor() {
  local name="$1"
  local default="${2:-2}"
  local v=""
  local envf="${ATMR_ENV_FILE:-}"
  if [[ -n "${envf}" ]] && [[ -f "${envf}" ]]; then
    v="$(grep -E "^${name}=" "${envf}" 2>/dev/null | tail -n1 | cut -d'=' -f2-)"
    v="${v//\'/}"
    v="${v//\"/}"
    v="${v// /}"
    if [[ -n "${v}" ]] && [[ "${v}" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "${v}"
      return
    fi
  fi
  printf '%s\n' "${default}"
}

kafka_count_broker_services() {
  local count
  # grep -c imprime « 0 » puis exit 1 si aucune correspondance — ne pas « || echo 0 » (double « 0 »).
  count="$(kafka_docker_compose config --services 2>/dev/null | grep -c '^kafka-broker-' 2>/dev/null || true)"
  printf '%s\n' "${count:-0}"
}

kafka_check_replication_factors() {
  local broker_count topic_rf broker_rf
  broker_count="$(kafka_count_broker_services | tr -d '[:space:]')"
  topic_rf="$(kafka_read_replication_factor KAFKA_TOPIC_REPLICATION_FACTOR 2)"
  broker_rf="$(kafka_read_replication_factor KAFKA_BROKER_REPLICATION_FACTOR 2)"
  if [[ ! "${broker_count}" =~ ^[0-9]+$ ]]; then
    log_fail "comptage brokers Kafka invalide (merge Compose ?) : « ${broker_count} »"
    return 1
  fi
  if [[ "${broker_count}" -lt 1 ]]; then
    log_fail "aucun service kafka-broker-* dans le merge Compose"
    return 1
  fi
  if ((topic_rf > broker_count)); then
    log_fail "KAFKA_TOPIC_REPLICATION_FACTOR=${topic_rf} > broker_count=${broker_count}"
    return 1
  fi
  if ((broker_rf > broker_count)); then
    log_fail "KAFKA_BROKER_REPLICATION_FACTOR=${broker_rf} > broker_count=${broker_count}"
    return 1
  fi
  log_info "réplication OK (brokers=${broker_count}, topic_rf=${topic_rf}, broker_rf=${broker_rf})"
  return 0
}

kafka_check_functional_smoke() {
  local first rf topic
  first="$(kafka_bootstrap_first)"
  rf="$(kafka_read_replication_factor KAFKA_TOPIC_REPLICATION_FACTOR 2)"
  topic="$(kafka_read_env_value KAFKA_OPS_SMOKE_TOPIC atmr.ops.smoke)"
  if ! kafka_docker_compose exec -T kafka-broker-1 kafka-topics \
    --bootstrap-server "${first}" \
    --create --if-not-exists \
    --topic "${topic}" \
    --partitions 1 \
    --replication-factor "${rf}" >/dev/null 2>&1; then
    log_fail "création topic smoke ${topic} KO"
    return 1
  fi
  local marker="atmr-kafka-smoke-$(date +%s)"
  if ! printf '%s\n' "${marker}" | kafka_docker_compose exec -T kafka-broker-1 kafka-console-producer \
    --bootstrap-server "${first}" \
    --topic "${topic}" >/dev/null 2>&1; then
    log_fail "publish smoke message KO"
    return 1
  fi
  local consumed
  consumed="$(kafka_docker_compose exec -T kafka-broker-1 kafka-console-consumer \
    --bootstrap-server "${first}" \
    --topic "${topic}" \
    --from-beginning \
    --timeout-ms 10000 \
    --max-messages 50 2>/dev/null | grep -F "${marker}" | tail -n1 || true)"
  if [[ -z "${consumed}" ]]; then
    log_fail "consume smoke message KO (marker=${marker})"
    return 1
  fi
  log_info "smoke Kafka producer→broker→consumer OK (${topic})"
  return 0
}

kafka_check_compose_resolution() {
  local services
  services="$(kafka_docker_compose config --services 2>/dev/null || true)"
  local svc
  local broker_count=0
  while IFS= read -r svc; do
    [[ -z "${svc}" ]] && continue
    if [[ "${svc}" == kafka-broker-* ]]; then
      if ! grep -qx "${svc}" <<<"${services}"; then
        log_fail "service broker « ${svc} » absent du merge"
        return 1
      fi
      broker_count=$((broker_count + 1))
    fi
  done < <(kafka_docker_compose config --services 2>/dev/null | grep '^kafka-broker-' || true)
  if ((broker_count < 1)); then
    log_fail "aucun kafka-broker-* dans le merge Compose"
    return 1
  fi
  for svc in "${KAFKA_CONSUMER_SERVICES[@]}"; do
    if ! grep -qx "${svc}" <<<"${services}"; then
      log_fail "service Compose « ${svc} » absent du merge (oubli -f / profile ?)"
      return 1
    fi
  done
  log_info "résolution Compose : ${broker_count} broker(s) + consumers (profile kafka)"
  return 0
}

kafka_wait_brokers_healthy() {
  local timeout_s="${1:-180}"
  local deadline=$((SECONDS + timeout_s))
  kafka_refresh_broker_containers
  local expected="${#KAFKA_BROKER_CONTAINERS[@]}"
  if ((expected < 1)); then
    log_fail "aucun conteneur broker à surveiller"
    return 1
  fi
  local c status healthy
  while ((SECONDS < deadline)); do
    healthy=0
    for c in "${KAFKA_BROKER_CONTAINERS[@]}"; do
      status="$(docker inspect "${c}" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>/dev/null || echo missing)"
      if [[ "${status}" == "healthy" ]]; then
        healthy=$((healthy + 1))
      fi
    done
    if ((healthy == expected)); then
      log_info "brokers Kafka healthy (${healthy}/${expected}) via docker inspect"
      return 0
    fi
    sleep 5
  done
  log_fail "timeout brokers healthy (${timeout_s}s) — attendu ${expected}, healthy partiel"
  return 1
}

kafka_check_dns_from_atmr_network() {
  local probe=""
  local probes=(
    backend atmr-backend-1
    atmr-kafka-broker-1 atmr-kafka-broker-2
    kafka-dlq-consumer atmr-kafka-dlq-consumer
    tracking-kafka-consumer atmr-tracking-kafka-consumer-1
  )
  for probe in "${probes[@]}"; do
    if docker ps --format '{{.Names}}' | grep -qx "${probe}"; then
      if kafka_dns_probe_ok "${probe}"; then
        log_info "DNS kafka-broker-1/2 depuis ${probe} (atmr-network)"
        return 0
      fi
    fi
  done
  if docker compose -f docker-compose.production.yml ps backend --status running -q 2>/dev/null | grep -q . \
    && docker compose -f docker-compose.production.yml exec -T backend getent hosts kafka-broker-1 >/dev/null 2>&1 \
    && docker compose -f docker-compose.production.yml exec -T backend getent hosts kafka-broker-2 >/dev/null 2>&1; then
    log_info "DNS kafka-broker-1/2 depuis backend (atmr-network)"
    return 0
  fi
  # Stack prod arrêtée (ex. après rollback) : sonde éphémère sur atmr-network
  if kafka_dns_ephemeral_probe_ok; then
    log_info "DNS kafka-broker-1/2 via sonde éphémère alpine (atmr-network)"
    return 0
  fi
  kafka_refresh_broker_containers
  local on_network=0 c net_containers
  if docker network inspect atmr-network >/dev/null 2>&1; then
    net_containers="$(docker network inspect atmr-network --format '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || true)"
    for c in "${KAFKA_BROKER_CONTAINERS[@]}"; do
      if grep -qw "${c}" <<<"${net_containers}"; then
        on_network=$((on_network + 1))
      fi
    done
  fi
  if ((${#KAFKA_BROKER_CONTAINERS[@]} > 0)) && ((on_network == 0)); then
    log_fail "brokers Kafka actifs mais absents de atmr-network — relancer deploy-kafka-production.sh (merge docker-compose.kafka.atmr-network.yml)"
    return 1
  fi
  log_fail "DNS kafka-broker-* introuvable depuis atmr-network — déployer Kafka (scripts/deploy-kafka-production.sh) ou vérifier le réseau"
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
  kafka_build_expected_topics
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
  echo "Compose : KAFKA_COMPOSE_FILE=${KAFKA_COMPOSE_FILE}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Charge KAFKA_COMPOSE_FILE depuis ATMR_ENV_FILE si défini (avant deploy/check).
kafka_load_compose_file_from_env() {
  local envf="${ATMR_ENV_FILE:-}"
  local v=""
  if [[ -n "${envf}" ]] && [[ -f "${envf}" ]]; then
    v="$(grep -E '^KAFKA_COMPOSE_FILE=' "${envf}" 2>/dev/null | tail -n1 | cut -d'=' -f2-)"
    v="${v//\'/}"
    v="${v//\"/}"
    v="${v// /}"
    if [[ -n "${v}" ]]; then
      KAFKA_COMPOSE_FILE="${v}"
    fi
  fi
}

kafka_load_compose_file_from_env
