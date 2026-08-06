#!/usr/bin/env bash
# DEPLOY-A — restauration synchrone protégée (limiteur GPS + async OFF).
# Usage:
#   BACKEND_IMAGE_REF=registry/...@sha256:abcdef \
#   COMPOSE_FILES="-f docker-compose.production.yml" \
#   ENV_FILE=.env.production \
#   ./scripts/ops-gps-deploy-a.sh
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env.production}"
COMPOSE_FILES="${COMPOSE_FILES:--f docker-compose.production.yml}"
BACKEND_SERVICE="${BACKEND_SERVICE:-backend}"

echo "== DEPLOY-A : sync protégé =="

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Fichier env manquant: $ENV_FILE" >&2
  exit 1
fi

# Image digest obligatoire (@sha256:...) — refuse tag :sha- / latest
if [[ -z "${BACKEND_IMAGE_REF:-}" ]]; then
  echo "REFUS: BACKEND_IMAGE_REF obligatoire (ex: registry/atmr-backend@sha256:...)" >&2
  exit 1
fi
if [[ "$BACKEND_IMAGE_REF" != *"@sha256:"* ]]; then
  echo "REFUS: BACKEND_IMAGE_REF doit être une ref digest (repo@sha256:...)" >&2
  echo "reçu: $BACKEND_IMAGE_REF" >&2
  exit 1
fi

export BACKEND_IMAGE_REF
if grep -q '^BACKEND_IMAGE_REF=' "$ENV_FILE"; then
  sed -i.bak "s|^BACKEND_IMAGE_REF=.*|BACKEND_IMAGE_REF=${BACKEND_IMAGE_REF}|" "$ENV_FILE"
else
  echo "BACKEND_IMAGE_REF=${BACKEND_IMAGE_REF}" >>"$ENV_FILE"
fi

# 1) async OFF (pas de hausse RATELIMIT_DEFAULT_LIMITS)
if grep -q '^TRACKING_INGEST_ASYNC_ENABLED=' "$ENV_FILE"; then
  sed -i.bak 's/^TRACKING_INGEST_ASYNC_ENABLED=.*/TRACKING_INGEST_ASYNC_ENABLED=false/' "$ENV_FILE"
else
  echo 'TRACKING_INGEST_ASYNC_ENABLED=false' >>"$ENV_FILE"
fi

echo "TRACKING_INGEST_ASYNC_ENABLED=$(grep '^TRACKING_INGEST_ASYNC_ENABLED=' "$ENV_FILE")"
echo "BACKEND_IMAGE_REF=$BACKEND_IMAGE_REF"

# 2) Rolling recreate backend (et ws si présent)
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" pull "$BACKEND_SERVICE" || true
docker compose $COMPOSE_FILES --env-file "$ENV_FILE" up -d --no-deps --force-recreate "$BACKEND_SERVICE"
if docker compose $COMPOSE_FILES --env-file "$ENV_FILE" ps ws-service >/dev/null 2>&1; then
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" up -d --no-deps --force-recreate ws-service || true
fi

# 3) Vérifier digest/image déployée
cid="$(docker compose $COMPOSE_FILES --env-file "$ENV_FILE" ps -q "$BACKEND_SERVICE" | head -1)"
if [[ -z "$cid" ]]; then
  echo "ABORT: conteneur $BACKEND_SERVICE introuvable après recreate" >&2
  exit 1
fi
inspect_ref="$(docker inspect --format '{{json .Image}} {{index .Config.Image}} {{range .RepoDigests}}{{.}} {{end}}' "$cid" 2>/dev/null || true)"
echo "Image inspectée: $inspect_ref"
# P0.2 : ne tester QUE inspect_ref (jamais concaténer BACKEND_IMAGE_REF → faux positif)
expected_token="$(echo "$BACKEND_IMAGE_REF" | grep -oE 'sha256:[a-f0-9]+' | head -1 || true)"
if [[ -z "$expected_token" ]]; then
  echo "ABORT: impossible d'extraire sha256:… depuis BACKEND_IMAGE_REF" >&2
  exit 1
fi
if ! echo "$inspect_ref" | grep -Fq "$expected_token"; then
  echo "ABORT: digest déployé incorrect (attendu $expected_token)" >&2
  echo "inspect_ref=$inspect_ref" >&2
  exit 1
fi
echo "OK: image backend correspond au digest $expected_token"

# 4) Purge ciblée buckets Flask-Limiter GPS (auth Redis si requirepass)
echo "Purge ciblée LIMITS…driver_driver_location…"
REDIS_PASSWORD="$(grep -E '^REDIS_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
export REDIS_PASSWORD
purged="$(
  docker compose $COMPOSE_FILES --env-file "$ENV_FILE" exec -T \
    -e REDISCLI_AUTH="${REDIS_PASSWORD}" \
    redis sh -c '
      n=0
      for k in $(redis-cli --scan --pattern "*driver_driver_location*" 2>/dev/null); do
        redis-cli DEL "$k" >/dev/null 2>&1 && n=$((n+1)) || true
      done
      echo "$n"
    ' 2>/dev/null || echo "0"
)"
echo "keys_purged=${purged}"
if [[ "${purged}" == "0" ]]; then
  echo "WARN: 0 clé purgée — vérifier REDISCLI_AUTH / pattern / déjà vide"
fi

echo "DEPLOY-A terminé. Enchaîner canary Gate 1 (chauffeur pilote)."
echo "Attendu: HTTP 200 + durability=persisted_sync + db_persisted, Redis canonical, 429 GPS = 0."
