#!/usr/bin/env bash
# Applique les migrations Alembic dans le conteneur atmr_api (réseau Docker correct).
# Prérequis : stack démarrée (docker compose up -d). Ne pas scale atmr_api > 1 (voir docker-compose.yml).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
COMPOSE="${COMPOSE_FILE:-docker-compose.yml}"
exec docker compose -f "$COMPOSE" exec -T \
  -e DISABLE_EVENTLET=1 \
  atmr_api alembic -c /app/migrations/alembic.ini upgrade head
