#!/usr/bin/env bash
# Phase 0 — préflight lecture seule (réalignement Kafka + Preuve A).
# Usage: cd /srv/atmr && bash scripts/ops-p0e-kafka-align-phase0.sh
set -euo pipefail
cd /srv/atmr

TARGET_SHA="390076efc61ca71332c749a67aff1e6fc7c2d626"

echo "======== 0.1 Triplet image backend ========"
BACKEND_IMAGE_ID="$(docker inspect atmr-backend-1 --format '{{.Image}}')"
echo "BACKEND_IMAGE_ID=${BACKEND_IMAGE_ID}"
echo "Config.Image=$(docker inspect atmr-backend-1 --format '{{.Config.Image}}')"
echo "--- RepoDigests ---"
docker image inspect "${BACKEND_IMAGE_ID}" --format '{{range .RepoDigests}}{{println .}}{{end}}'
OCI_REV="$(docker image inspect "${BACKEND_IMAGE_ID}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
echo "OCI_revision=${OCI_REV}"

REPO_DIGEST_LINE="$(
  docker image inspect "${BACKEND_IMAGE_ID}" --format '{{range .RepoDigests}}{{println .}}{{end}}' \
    | grep -E '^djasiqi/atmr-backend@sha256:[0-9a-f]{64}$' \
    | head -n1 || true
)"
if [[ -z "${REPO_DIGEST_LINE}" ]]; then
  echo "FAIL: RepoDigest djasiqi/atmr-backend@sha256:... introuvable" >&2
  exit 1
fi
IMAGE_DIGEST="${REPO_DIGEST_LINE#*@}"
echo "SELECTED_RepoDigest=${REPO_DIGEST_LINE}"
echo "IMAGE_DIGEST=${IMAGE_DIGEST}"

if [[ "${OCI_REV}" != "${TARGET_SHA}" ]]; then
  echo "STOP: OCI revision=${OCI_REV} != ${TARGET_SHA}" >&2
  exit 2
fi
echo "OK Phase 0.1 triplet immuable"

echo
echo "======== 0.2 Release / image env ========"
grep -E '^(GIT_SHA|SENTRY_RELEASE|BACKEND_IMAGE_REF|DOCKER_IMAGE|DOCKER_TAG)=' .env.production || true
if [[ -f .env.production.local ]]; then
  echo "--- .env.production.local (si présent) ---"
  grep -E '^(GIT_SHA|SENTRY_RELEASE|BACKEND_IMAGE_REF|DOCKER_IMAGE|DOCKER_TAG)=' .env.production.local || true
fi

echo
echo "======== 0.3 Dry-run recreate ingest ========"
export BACKEND_IMAGE_REF=
export DRY_RUN=1
export COMPOSE_PROJECT_NAME=atmr
export COMPOSE_FILES="-f docker-compose.production.yml -f docker-compose.kafka.yml -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml"
export ENV_FILE="--env-file .env.production"
export DOCKER_IMAGE=djasiqi/atmr-backend
export DOCKER_TAG=sha-390076efc61c
export SOURCE_SHA="${TARGET_SHA}"
export IMAGE_DIGEST
export EXPECTED_INGEST_REPLICAS=1
chmod +x scripts/ops-tracking-p0-recreate-ingest.sh
./scripts/ops-tracking-p0-recreate-ingest.sh

echo
echo "======== 0.4 Baseline ========"
echo "--- compose ps (kafka tracking) ---"
# shellcheck disable=SC2086
docker compose -p atmr --env-file .env.production \
  -f docker-compose.production.yml -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml \
  --profile kafka ps tracking-kafka-consumer tracking-outbox-publisher \
  tracking-processed-fanout kafka-dlq-consumer 2>/dev/null || true

echo "--- consumer/outbox OCI ---"
for name in tracking-kafka-consumer tracking-outbox-publisher; do
  cid="$(docker ps -aq --filter "name=${name}" | head -n1 || true)"
  if [[ -z "${cid}" ]]; then
    echo "${name}: ABSENT"
    continue
  fi
  img_id="$(docker inspect "${cid}" --format '{{.Image}}')"
  cfg="$(docker inspect "${cid}" --format '{{.Config.Image}}')"
  rev="$(docker image inspect "${img_id}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || echo unknown)"
  echo "${name}: cid=${cid} Config.Image=${cfg} ImageID=${img_id} revision=${rev}"
done

echo "--- outbox pending (read-only) ---"
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T postgres psql -U atmr -d atmr -c \
  "SELECT COUNT(*) AS pending FROM tracking_event_outbox WHERE published_at IS NULL;" 2>/dev/null \
  || echo "WARN: requête outbox pending échouée"

echo "--- kafka lag (best effort) ---"
docker compose -p atmr --env-file .env.production \
  -f docker-compose.production.yml -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml \
  --profile kafka exec -T kafka-broker-1 \
  kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group tracking-ingest-consumer-group 2>/dev/null \
  || echo "WARN: lag ingest non disponible"
docker compose -p atmr --env-file .env.production \
  -f docker-compose.production.yml -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml \
  --profile kafka exec -T kafka-broker-1 \
  kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group ws-service-shared 2>/dev/null \
  || echo "WARN: lag ws-service-shared non disponible"

echo
echo "PHASE0_OK BACKEND_IMAGE_ID=${BACKEND_IMAGE_ID} IMAGE_DIGEST=${IMAGE_DIGEST} OCI=${OCI_REV}"
