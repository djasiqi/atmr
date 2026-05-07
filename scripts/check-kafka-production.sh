#!/usr/bin/env bash
# Diagnostic Kafka production (read-only : pas de up/down).
set -uo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT}"

ENV_FILE="${ATMR_ENV_FILE:-${ROOT}/.env.production}"
export ATMR_ENV_FILE="${ENV_FILE}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/kafka_checks.sh"

# Ne jamais créer atmr-network implicitement depuis ce diagnostic (read-only).
export ATMR_AUTO_CREATE_NETWORK=0

MODE="${1:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <mode>

Modes :
  off            Vérifie Kafka OFF : 4 flags≠true + aucun consumer actif + backend healthy.
  on             Vérifie Kafka ON  : 4 flags=true + preflight + brokers + DNS + API + topics + consumers.
  preflight-on   Uniquement pré-requis avant deploy-kafka-production.sh (flags + fichiers + réseau + compose).

Variables :
  ATMR_DEPLOY_ROOT, ATMR_ENV_FILE (défaut : .env.production à la racine du déploiement)

Exit codes :
  0   tous les checks du mode sont OK
  1   au moins un check a échoué
  2   mode inconnu ou argument manquant
EOF
}

if [[ -z "${MODE}" ]]; then
  usage >&2
  exit 2
fi

GLOBAL_OK=1

case "${MODE}" in
  off)
    echo "=== check-kafka-production.sh : mode off ==="
    kafka_check_flags_all_false || GLOBAL_OK=0
    kafka_check_no_consumers_running || GLOBAL_OK=0
    kafka_check_backend_healthy || GLOBAL_OK=0
    ;;
  on)
    echo "=== check-kafka-production.sh : mode on ==="
    kafka_check_flags_all_true || GLOBAL_OK=0
    kafka_check_compose_files || GLOBAL_OK=0
    kafka_check_atmr_network || GLOBAL_OK=0
    kafka_check_compose_resolution || GLOBAL_OK=0
    kafka_wait_brokers_healthy 30 || GLOBAL_OK=0
    kafka_check_dns_from_atmr_network || GLOBAL_OK=0
    kafka_check_broker_api || GLOBAL_OK=0
    kafka_check_topics_exist || GLOBAL_OK=0
    kafka_check_consumers_running || GLOBAL_OK=0
    ;;
  preflight-on)
    echo "=== check-kafka-production.sh : mode preflight-on ==="
    kafka_check_flags_all_true || GLOBAL_OK=0
    kafka_check_compose_files || GLOBAL_OK=0
    kafka_check_atmr_network || GLOBAL_OK=0
    kafka_check_compose_resolution || GLOBAL_OK=0
    ;;
  *)
    echo "Mode inconnu : ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

kafka_summary

if ((GLOBAL_OK == 0)); then
  exit 1
fi
exit 0
