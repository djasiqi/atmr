#!/usr/bin/env bash
set -euo pipefail

# Démarre la stack Kafka (ZooKeeper + brokers) et les consumers ATMR profile « kafka »
# sur le réseau externe atmr-network, en fusionnant les trois fichiers Compose du dépôt.
#
# Prérequis :
#   - À la racine du déploiement : docker-compose.production.yml, docker-compose.kafka.yml,
#     docker-compose.kafka.atmr-network.yml (copier depuis le repo si absents sur le serveur).
#   - Réseau Docker atmr-network (créé si ATMR_AUTO_CREATE_NETWORK=1, défaut).
#   - Dans .env.production : les 4 flags Kafka à true (voir env.kafka.production.example).
#
# Usage :
#   ATMR_DEPLOY_ROOT=/srv/atmr ./scripts/deploy-kafka-production.sh
#   INIT_TOPICS=1 ATMR_DEPLOY_ROOT=/srv/atmr ./scripts/deploy-kafka-production.sh
#
# Variables :
#   ATMR_DEPLOY_ROOT — répertoire contenant les YAML (défaut : parent de scripts/)
#   INIT_TOPICS      — si 1, exécute scripts/kafka-init-topics-compose.sh après le up
#   ATMR_ENV_FILE    — fichier .env pour les garde-fous (défaut : ${ROOT}/.env.production)
#   FORCE            — si 1, ignore uniquement le preflight flags/réseau/compose (bootstrap initial)
#
# Codes de sortie :
#   0 — succès (preflight + up + post-deploy)
#   2 — preflight refusé
#   3 — post-deploy en échec
#   1 — erreur shell / usage

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"
export ATMR_ENV_FILE="${ENV_FILE}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

echo "=== Kafka deploy : ${ROOT} (env=${ENV_FILE}) ==="

echo "--- Phase 1/3 : preflight Kafka ON ---"
PREFLIGHT_OK=1
kafka_check_flags_all_true || PREFLIGHT_OK=0
kafka_check_compose_files || PREFLIGHT_OK=0
kafka_check_atmr_network || PREFLIGHT_OK=0
kafka_check_compose_resolution || PREFLIGHT_OK=0

if ((PREFLIGHT_OK == 0)); then
  if [[ "${FORCE:-0}" == "1" ]]; then
    log_force_override
    echo "[WARN] preflight KO mais FORCE=1 — poursuite (bootstrap initial uniquement)." >&2
  else
    echo "Refus : preflight Kafka ON KO. Corriger ${ENV_FILE} ou FORCE=1 (exceptionnel)." >&2
    exit 2
  fi
fi

echo "--- Phase 2/3 : docker compose up (profile kafka) ---"
kafka_docker_compose up -d "$@"

if [[ "${INIT_TOPICS:-0}" == "1" ]]; then
  echo "--- Phase 2bis : init topics ---"
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  ATMR_DEPLOY_ROOT="${ROOT}" "${SCRIPT_DIR}/kafka-init-topics-compose.sh"
fi

echo "--- Phase 3/3 : validations post-deploy ---"
POST_OK=1
kafka_wait_brokers_healthy 180 || POST_OK=0
kafka_check_dns_from_atmr_network || POST_OK=0
kafka_check_broker_api || POST_OK=0
kafka_check_topics_exist || POST_OK=0
kafka_check_consumers_running || POST_OK=0

kafka_summary

if ((POST_OK == 0)); then
  echo "FAIL post-deploy : au moins un check a échoué (voir résumé)." >&2
  exit 3
fi

echo "OK Kafka déployé et validé de bout en bout."
echo "Vérifier avec : scripts/check-kafka-production.sh on"
