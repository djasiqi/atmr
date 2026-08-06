#!/usr/bin/env bash
# DEPLOY-A — restauration synchrone protégée (limiteur GPS + async OFF).
# Prérequis : image PR1 déjà buildée/pushée ; Phase 0 capture faite.
# Usage: COMPOSE_FILES="..." ENV_FILE=.env.production IMAGE_TAG=sha-xxx ./scripts/ops-gps-deploy-a.sh
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env.production}"
COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.production.yml}"
BACKEND_SERVICE="${BACKEND_SERVICE:-backend}"

echo "== DEPLOY-A : sync protégé =="

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Fichier env manquant: $ENV_FILE" >&2
  exit 1
fi

# 1) async OFF (pas de hausse RATELIMIT_DEFAULT_LIMITS)
if grep -q '^TRACKING_INGEST_ASYNC_ENABLED=' "$ENV_FILE"; then
  sed -i.bak 's/^TRACKING_INGEST_ASYNC_ENABLED=.*/TRACKING_INGEST_ASYNC_ENABLED=false/' "$ENV_FILE"
else
  echo 'TRACKING_INGEST_ASYNC_ENABLED=false' >>"$ENV_FILE"
fi

echo "TRACKING_INGEST_ASYNC_ENABLED=$(grep '^TRACKING_INGEST_ASYNC_ENABLED=' "$ENV_FILE")"

# 2) Rolling restart backend (et ws si présent)
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" up -d --no-deps --force-recreate "$BACKEND_SERVICE"
if docker compose $COMPOSE_FILES --env-file "$ENV_FILE" ps ws-service >/dev/null 2>&1; then
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" up -d --no-deps --force-recreate ws-service || true
fi

# 3) Purge ciblée buckets Flask-Limiter GPS (pas purge globale)
echo "Purge ciblée LIMITS…driver_driver_location…"
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T redis \
  sh -c 'redis-cli --scan --pattern "*driver_driver_location*" | while read -r k; do redis-cli DEL "$k"; done' \
  || echo "WARN: purge Redis partielle / indisponible"

echo "DEPLOY-A terminé. Enchaîner canary Gate 1 (chauffeur pilote)."
echo "Attendu: HTTP 200 + durability=persisted_sync, Redis canonical, 429 GPS = 0."
