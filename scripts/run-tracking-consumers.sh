#!/usr/bin/env bash
set -euo pipefail

# Garde-fou : ce script scale tracking-kafka-consumer — exiger ingest async + persist activés.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"

read_env_flag() {
  local name="$1"
  local default="${2:-false}"
  local v=""
  if [[ -f "${ENV_FILE}" ]]; then
    v="$(grep -E "^${name}=" "${ENV_FILE}" 2>/dev/null | tail -n1 | cut -d'=' -f2-)"
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

TRACKING_REQUIRED_FLAGS=(
  KAFKA_ENABLED
  TRACKING_INGEST_ASYNC_ENABLED
  TRACKING_INGEST_PERSIST_ENABLED
)
INCOHERENT=()
for flag in "${TRACKING_REQUIRED_FLAGS[@]}"; do
  v="$(read_env_flag "${flag}")"
  echo "🔒 ${flag}=${v} (${ENV_FILE})"
  if [[ "${v}" != "true" ]]; then
    INCOHERENT+=("${flag}=${v}")
  fi
done

async="$(read_env_flag TRACKING_INGEST_ASYNC_ENABLED)"
persist="$(read_env_flag TRACKING_INGEST_PERSIST_ENABLED)"
allow="$(read_env_flag TRACKING_INGEST_ALLOW_REPUBLISH_ONLY)"
if [[ "${async}" == "true" ]] && [[ "${persist}" != "true" ]] && [[ "${allow}" != "true" ]]; then
  INCOHERENT+=("coherence=ASYNC_without_PERSIST_or_ALLOW")
fi

if [[ ${#INCOHERENT[@]} -gt 0 ]] && [[ "${FORCE:-0}" != "1" ]]; then
  echo "❌ Refus : flags requis absents ou incohérents pour tracking-kafka-consumer."
  printf '   %s\n' "${INCOHERENT[@]}"
  echo "   Corriger ${ENV_FILE} ou FORCE=1 (exceptionnel)."
  exit 2
fi

# Fichier Kafka (2ᵉ) — mêmes valeurs par défaut que l’habitude du repo
COMPOSE_KAFKA="${1:-docker-compose.kafka.yml}"
SCALE="${2:-3}"
# Compose principal (1ᵉʳ) : requis pour pgbouncer/postgres sur le réseau partagé atmr-stack
COMPOSE_MAIN="${3:-docker-compose.yml}"

echo "Scaling tracking-kafka-consumer to ${SCALE} (main: ${COMPOSE_MAIN} + ${COMPOSE_KAFKA}, force-recreate)"
docker compose -f "${COMPOSE_MAIN}" -f "${COMPOSE_KAFKA}" up -d --force-recreate --scale "tracking-kafka-consumer=${SCALE}" tracking-kafka-consumer
