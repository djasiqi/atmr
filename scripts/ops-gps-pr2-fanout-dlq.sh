#!/usr/bin/env bash
# PR2-Ops — démarrage fanout/DLQ avec manifests prod réels (async reste OFF).
# Usage:
#   COMPOSE_FILES="-f docker-compose.production.yml -f ..." \
#   ENV_FILE=.env.production ./scripts/ops-gps-pr2-fanout-dlq.sh
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env.production}"
COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.production.yml}"

echo "== PR2-Ops fanout/DLQ =="
echo "Manifests: $COMPOSE_FILES"

# 1) config dry-run
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" --profile kafka config >/tmp/atmr-kafka-config.yml
echo "compose config OK → /tmp/atmr-kafka-config.yml"

# Vérifs basiques
grep -E "tracking-processed-fanout|kafka-dlq-consumer" /tmp/atmr-kafka-config.yml >/dev/null \
  || { echo "Services fanout/DLQ absents des manifests" >&2; exit 1; }

# 2) Confirmer async OFF
ASYNC_VAL=$(grep '^TRACKING_INGEST_ASYNC_ENABLED=' "$ENV_FILE" | cut -d= -f2- || echo "")
if [[ "$ASYNC_VAL" != "false" ]]; then
  echo "REFUS: TRACKING_INGEST_ASYNC_ENABLED=$ASYNC_VAL (doit être false avant fanout)" >&2
  exit 1
fi

# 3) Fanout offset policy
echo "Vérifier KAFKA_PROCESSED_FANOUT_OFFSET_RESET et groupe consumer avant start."
echo "Si groupe existant avec offsets anciens: reset explicite ou nouveau group ID."

# 4) Disque DLQ
df -h . | tee /tmp/atmr-dlq-disk-pre.txt || true

# 5) Start ciblé
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" --profile kafka up -d \
  tracking-processed-fanout kafka-dlq-consumer

docker compose $COMPOSE_FILES --env-file "$ENV_FILE" --profile kafka ps \
  tracking-processed-fanout kafka-dlq-consumer

echo "PR2-Ops démarré. Async reste OFF jusqu'à Gate 3 + PR5."
