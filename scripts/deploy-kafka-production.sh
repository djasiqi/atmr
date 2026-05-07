#!/usr/bin/env bash
set -euo pipefail

# Démarre la stack Kafka (ZooKeeper + brokers) et les consumers ATMR profile « kafka »
# sur le réseau externe atmr-network, en fusionnant les trois fichiers Compose du dépôt.
#
# Prérequis :
#   - À la racine du déploiement : docker-compose.production.yml, docker-compose.kafka.yml,
#     docker-compose.kafka.atmr-network.yml (copier depuis le repo si absents sur le serveur).
#   - Réseau Docker atmr-network (créé automatiquement si manquant).
#   - Dans .env / .env.production : les 4 flags Kafka à true (voir env.kafka.production.example).
#
# Usage :
#   ATMR_DEPLOY_ROOT=/srv/atmr ./scripts/deploy-kafka-production.sh
#   INIT_TOPICS=1 ATMR_DEPLOY_ROOT=/srv/atmr ./scripts/deploy-kafka-production.sh
#
# Variables :
#   ATMR_DEPLOY_ROOT — répertoire contenant les YAML (défaut : parent de scripts/)
#   INIT_TOPICS      — si 1, exécute scripts/kafka-init-topics-compose.sh après le up
#   ATMR_ENV_FILE    — fichier .env à lire pour le garde-fou (défaut : ${ROOT}/.env.production)
#   FORCE            — si 1, ignore le garde-fou flags (bootstrap initial uniquement)

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

echo "📂 Répertoire déploiement : ${ROOT}"

# Garde-fou : les 4 flags Kafka doivent être explicitement true avant --profile kafka.
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

echo "🔒 Vérification flags Kafka (fichier : ${ENV_FILE})"
REQUIRED_FLAGS=(
  KAFKA_ENABLED
  TRACKING_INGEST_ASYNC_ENABLED
  TRACKING_PROCESSED_FANOUT_ENABLED
  WS_KAFKA_CONSUMER_ENABLED
)
INCOHERENT=()
for flag in "${REQUIRED_FLAGS[@]}"; do
  v="$(read_env_flag "${flag}")"
  echo "   ${flag}=${v}"
  if [[ "${v}" != "true" ]]; then
    INCOHERENT+=("${flag}=${v}")
  fi
done

if [[ ${#INCOHERENT[@]} -gt 0 ]] && [[ "${FORCE:-0}" != "1" ]]; then
  echo "❌ Refus : flags Kafka incohérents. Détail :"
  printf '   %s\n' "${INCOHERENT[@]}"
  echo "   Corriger ${ENV_FILE} ou lancer avec FORCE=1 (bootstrap initial uniquement)."
  exit 2
fi

for f in docker-compose.production.yml docker-compose.kafka.yml docker-compose.kafka.atmr-network.yml; do
  if [[ ! -f "${f}" ]]; then
    echo "❌ Fichier manquant : ${ROOT}/${f}"
    echo "   → Copier depuis le dépôt Git atmr/ à cet emplacement (ou définir ATMR_DEPLOY_ROOT)."
    exit 1
  fi
done

if ! docker network inspect atmr-network >/dev/null 2>&1; then
  echo "📡 Création du réseau Docker atmr-network…"
  docker network create atmr-network
else
  echo "✅ Réseau atmr-network présent"
fi

COMPOSE=(
  docker compose
  -f docker-compose.production.yml
  -f docker-compose.kafka.yml
  -f docker-compose.kafka.atmr-network.yml
)

echo "🚀 docker compose … --profile kafka up -d"
"${COMPOSE[@]}" --profile kafka up -d "$@"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Étapes suivantes :"
echo "  1) Vérifier DNS depuis un service sur atmr-network, ex. backend :"
echo "       docker compose -f docker-compose.production.yml exec backend getent hosts kafka-broker-1"
echo "  2) Si besoin, créer les topics (replication 3 si 3 brokers stables) :"
echo "       KAFKA_TOPIC_REPLICATION_FACTOR=3 ./scripts/kafka-init-topics-compose.sh"
echo "       (premier déploiement : INIT_TOPICS=1 ./scripts/deploy-kafka-production.sh lance aussi les topics)"
echo "  3) Redémarrer backend / ws-service si .env vient d’être mis à jour (KAFKA_ENABLED, etc.)."
echo "Variables .env : voir env.kafka.production.example à la racine du repo."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "${INIT_TOPICS:-0}" == "1" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  ATMR_DEPLOY_ROOT="${ROOT}" "${SCRIPT_DIR}/kafka-init-topics-compose.sh"
fi
