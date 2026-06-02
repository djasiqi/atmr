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
#   0 — succès (preflight + up brokers + init topics + up consumers + post-deploy)
#   2 — preflight refusé
#   3 — post-deploy en échec (brokers non healthy, init topics KO, validations KO)
#   1 — erreur shell / usage
#
# Ordonnancement (P0.2) :
#   Le script ne fait PAS un `up -d --profile kafka` global, qui démarrerait les
#   consumers en parallèle des brokers — ces derniers joindraient leur groupe AVANT
#   que `kafka-topics-init` ait créé les topics, aboutissant à un assignment vide
#   sticky (cf. retex 2026-05-07). On sépare donc explicitement :
#     1. up zookeeper + brokers (+ redis-failover, sans profile kafka)
#     2. wait brokers healthy
#     3. init topics (idempotent, via kafka-init-topics-compose.sh si INIT_TOPICS=1)
#     4. up consumers + kafka-ui (profile kafka)
#     5. validations finales

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"
export ATMR_ENV_FILE="${ENV_FILE}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

# Services démarrés en phase « brokers » (pas de profile, requis avant init topics).
KAFKA_BROKER_SERVICES=(
  zookeeper
  zookeeper-2
  zookeeper-3
  kafka-broker-1
  kafka-broker-2
  kafka-broker-3
  redis-failover
)

# Services démarrés en phase « consumers » (profile kafka), après init topics.
# kafka-ui est sans profile mais on l’aligne ici pour qu’il arrive après les brokers.
KAFKA_CONSUMER_AND_UI_SERVICES=(
  tracking-kafka-consumer
  tracking-processed-fanout
  kafka-dlq-consumer
  kafka-ui
)

echo "=== Kafka deploy : ${ROOT} (env=${ENV_FILE}) ==="

echo "--- Phase 1/5 : preflight Kafka ON ---"
PREFLIGHT_OK=1
kafka_check_flags_all_true || PREFLIGHT_OK=0
kafka_check_compose_files || PREFLIGHT_OK=0
kafka_check_atmr_network || PREFLIGHT_OK=0
kafka_check_compose_resolution || PREFLIGHT_OK=0
kafka_check_replication_factors || PREFLIGHT_OK=0

if ((PREFLIGHT_OK == 0)); then
  if [[ "${FORCE:-0}" == "1" ]]; then
    log_force_override
    echo "[WARN] preflight KO mais FORCE=1 — poursuite (bootstrap initial uniquement)." >&2
  else
    echo "Refus : preflight Kafka ON KO. Corriger ${ENV_FILE} ou FORCE=1 (exceptionnel)." >&2
    exit 2
  fi
fi

echo "--- Phase 2/5 : up brokers (zookeeper + kafka-broker-1/2/3 + redis-failover) ---"
kafka_docker_compose up -d "${KAFKA_BROKER_SERVICES[@]}" "$@"

echo "--- Phase 3/5 : wait brokers healthy (avant init topics) ---"
if ! kafka_wait_brokers_healthy 180; then
  echo "FAIL : brokers non healthy dans le délai imparti — abandon avant init topics." >&2
  kafka_summary
  exit 3
fi

if [[ "${INIT_TOPICS:-0}" == "1" ]]; then
  echo "--- Phase 4/5 : init topics ---"
  if ! ATMR_DEPLOY_ROOT="${ROOT}" "${SCRIPT_DIR}/kafka-init-topics-compose.sh"; then
    echo "FAIL : init topics — abandon avant up consumers." >&2
    kafka_summary
    exit 3
  fi
else
  echo "--- Phase 4/5 : init topics (skip — INIT_TOPICS!=1) ---"
fi

echo "--- Phase 5a/5 : up consumers + kafka-ui (profile kafka) ---"
kafka_docker_compose up -d "${KAFKA_CONSUMER_AND_UI_SERVICES[@]}"

echo "--- Phase 5b/5 : validations post-deploy ---"
POST_OK=1
kafka_check_dns_from_atmr_network || POST_OK=0
kafka_check_broker_api || POST_OK=0
kafka_check_topics_exist || POST_OK=0
kafka_check_consumers_running || POST_OK=0
kafka_check_functional_smoke || POST_OK=0

kafka_summary

if ((POST_OK == 0)); then
  echo "FAIL post-deploy : au moins un check a échoué (voir résumé)." >&2
  exit 3
fi

echo "OK Kafka déployé et validé de bout en bout."
echo "Vérifier avec : scripts/check-kafka-production.sh on"
