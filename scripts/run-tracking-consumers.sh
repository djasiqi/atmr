#!/usr/bin/env bash
set -euo pipefail

# Fichier Kafka (2ᵉ) — mêmes valeurs par défaut que l’habitude du repo
COMPOSE_KAFKA="${1:-docker-compose.kafka.yml}"
SCALE="${2:-3}"
# Compose principal (1ᵉʳ) : requis pour pgbouncer/postgres sur le réseau partagé atmr-stack
COMPOSE_MAIN="${3:-docker-compose.yml}"

echo "Scaling tracking-kafka-consumer to ${SCALE} (main: ${COMPOSE_MAIN} + ${COMPOSE_KAFKA})"
docker compose -f "${COMPOSE_MAIN}" -f "${COMPOSE_KAFKA}" up -d --scale "tracking-kafka-consumer=${SCALE}" tracking-kafka-consumer
