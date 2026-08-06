#!/usr/bin/env bash
# Gate 3 — canary async contrôlé (après PR5).
# Prérequis : DEPLOY-A + Gate 1/2 verts ; fanout/DLQ up ; circuit partagé OK.
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env.production}"
COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.production.yml}"

echo "== Gate 3 canary async =="

# 1) Arrêt volontaire consumer → aucune 202
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" --profile kafka \
  stop tracking-kafka-consumer || true
sleep 5
echo "Vérifier manuellement: PUT location → 200 persisted_sync (pas 202)."
echo "Circuit Redis tracking:consumer:ingest:circuit doit être open."

# 2) Redémarrage consumer
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" --profile kafka \
  up -d tracking-kafka-consumer
echo "Attendre healthy_since + CIRCUIT_OK_THRESHOLD avant async."

# 3) Activer async seulement après circuit closed
echo "Ensuite seulement: TRACKING_INGEST_ASYNC_ENABLED=true + restart backend"
echo "Puis canary limité + 10k events de charge contrôlée."
