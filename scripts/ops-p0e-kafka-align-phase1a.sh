#!/usr/bin/env bash
# Phase 1a — recreate tracking-kafka-consumer sur 390076ef (digest figé Phase 0).
set -euo pipefail
cd /srv/atmr

TARGET_SHA="390076efc61ca71332c749a67aff1e6fc7c2d626"
IMAGE_DIGEST="sha256:fb919878b7297417c0ed89c01a9f4ffc61dd9dd4c75f394ab227c36c79f41acf"

export BACKEND_IMAGE_REF=
export EXECUTE_P0_RECREATE=YES
export COMPOSE_PROJECT_NAME=atmr
export COMPOSE_FILES="-f docker-compose.production.yml -f docker-compose.kafka.yml -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml"
export ENV_FILE="--env-file .env.production"
export DOCKER_IMAGE=djasiqi/atmr-backend
export DOCKER_TAG=sha-390076efc61c
export SOURCE_SHA="${TARGET_SHA}"
export IMAGE_DIGEST
export EXPECTED_INGEST_REPLICAS=1

chmod +x scripts/ops-tracking-p0-recreate-ingest.sh
sed -i 's/\r$//' scripts/ops-tracking-p0-recreate-ingest.sh
./scripts/ops-tracking-p0-recreate-ingest.sh

echo
echo "======== Post-1a verify consumer ========"
cid="$(docker ps -aq --filter name=tracking-kafka-consumer | head -n1)"
img_id="$(docker inspect "${cid}" --format '{{.Image}}')"
cfg="$(docker inspect "${cid}" --format '{{.Config.Image}}')"
rev="$(docker image inspect "${img_id}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
echo "cid=${cid} Config.Image=${cfg} ImageID=${img_id} revision=${rev}"
docker inspect "${cid}" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(GIT_SHA|SENTRY_RELEASE)=' || true
echo "PHASE1A_OK"
