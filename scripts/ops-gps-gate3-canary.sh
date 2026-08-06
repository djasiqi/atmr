#!/usr/bin/env bash
# Gate 3 — canary async contrôlé (après PR5 / P0.1).
# Prérequis : DEPLOY-A + Gate 1/2 verts ; fanout/DLQ up ; circuit partagé OK.
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env.production}"
COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.production.yml}"
# TTL heartbeat + marge : open_circuit_immediate doit rendre le circuit open tout de suite
WAIT_AFTER_STOP_SEC="${WAIT_AFTER_STOP_SEC:-3}"

echo "== Gate 3 canary async =="

# 1) Arrêt volontaire consumer → circuit open immédiat → aucune 202
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" --profile kafka \
  stop tracking-kafka-consumer || true
sleep "$WAIT_AFTER_STOP_SEC"

REDIS_PASSWORD="$(grep -E '^REDIS_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
circuit="$(
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T \
    -e REDISCLI_AUTH="${REDIS_PASSWORD}" \
    redis sh -c 'redis-cli GET tracking:consumer:ingest:circuit' 2>/dev/null || true
)"
echo "circuit_after_stop=${circuit}"
if ! echo "$circuit" | grep -q '"state": "open"\|"state":"open"'; then
  echo "ABORT: circuit pas open après stop consumer (attendu open_circuit_immediate)" >&2
  exit 1
fi
echo "OK: circuit open. Vérifier manuellement: PUT location → 200 persisted_sync (pas 202)."

# 2) Redémarrage consumer
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" --profile kafka \
  up -d tracking-kafka-consumer
echo "Attendre CIRCUIT_OPEN_MIN_SEC + OK_THRESHOLD avant circuit closed."

# 3) Activer async seulement après circuit closed
echo "Ensuite seulement: TRACKING_INGEST_ASYNC_ENABLED=true + restart backend"
echo "Puis canary limité + 10k events de charge contrôlée."
