#!/usr/bin/env bash
# Phase 1b — force-recreate tracking-outbox-publisher même Image ID / OCI que backend.
set -euo pipefail
cd /srv/atmr

TARGET_SHA="390076efc61ca71332c749a67aff1e6fc7c2d626"
EXPECTED_IMAGE_ID="sha256:780a166c04b928d3a24a7f773a83cf1835d03512b9ab1073d87ef395003ecc4d"

export BACKEND_IMAGE_REF=
export DOCKER_IMAGE=djasiqi/atmr-backend
export DOCKER_TAG=sha-390076efc61c

docker compose -p atmr --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  -f docker-compose.kafka.p0-hold.yml \
  --profile kafka \
  up -d --no-deps --pull never --force-recreate \
  tracking-outbox-publisher

sleep 3
OUTBOX_CID="$(docker compose -p atmr --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  -f docker-compose.kafka.p0-hold.yml \
  --profile kafka ps -q tracking-outbox-publisher)"

echo "OUTBOX_CID=${OUTBOX_CID}"
docker inspect "${OUTBOX_CID}" --format 'Config.Image={{.Config.Image}} ImageID={{.Image}} Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}'
img_id="$(docker inspect "${OUTBOX_CID}" --format '{{.Image}}')"
rev="$(docker image inspect "${img_id}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
echo "OCI_revision=${rev}"

if [[ "${img_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "FAIL Image ID outbox=${img_id} attendu=${EXPECTED_IMAGE_ID}" >&2
  exit 1
fi
if [[ "${rev}" != "${TARGET_SHA}" ]]; then
  echo "FAIL OCI outbox=${rev}" >&2
  exit 1
fi

echo "--- release env outbox ---"
docker inspect "${OUTBOX_CID}" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(GIT_SHA|SENTRY_RELEASE)=' || echo "WARN: GIT_SHA/SENTRY_RELEASE absents"

echo "PHASE1B_OK"
