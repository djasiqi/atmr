#!/usr/bin/env bash
# Phase 1c — gates post-recreate (bloquants).
set -euo pipefail
cd /srv/atmr

TARGET_SHA="390076efc61ca71332c749a67aff1e6fc7c2d626"
EXPECTED_IMAGE_ID="sha256:780a166c04b928d3a24a7f773a83cf1835d03512b9ab1073d87ef395003ecc4d"
compose() {
  docker compose -p atmr --env-file .env.production \
    -f docker-compose.production.yml \
    -f docker-compose.kafka.yml \
    -f docker-compose.kafka.atmr-network.yml \
    -f docker-compose.kafka.p0-hold.yml \
    --profile kafka "$@"
}

fail=0

check_svc() {
  local name="$1" need_release="$2" match_backend_image="${3:-yes}"
  local cid img_id cfg rev
  cid="$(docker ps -aq --filter "name=${name}" | head -n1 || true)"
  [[ -n "${cid}" ]] || { echo "FAIL ${name}: absent"; fail=1; return; }
  img_id="$(docker inspect "${cid}" --format '{{.Image}}')"
  cfg="$(docker inspect "${cid}" --format '{{.Config.Image}}')"
  rev="$(docker image inspect "${img_id}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
  health="$(docker inspect "${cid}" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}')"
  echo "${name}: Config.Image=${cfg} ImageID=${img_id} revision=${rev} health=${health}"
  if [[ "${match_backend_image}" == "yes" ]]; then
    [[ "${img_id}" == "${EXPECTED_IMAGE_ID}" ]] || { echo "FAIL ${name} ImageID"; fail=1; }
  fi
  [[ "${rev}" == "${TARGET_SHA}" ]] || { echo "FAIL ${name} OCI"; fail=1; }
  [[ "${health}" == "healthy" || "${health}" == "n/a" ]] || { echo "FAIL ${name} health=${health}"; fail=1; }
  if [[ "${need_release}" == "yes" ]]; then
    local git_sha sentry
    git_sha="$(docker inspect "${cid}" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^GIT_SHA=' | head -n1 || true)"
    sentry="$(docker inspect "${cid}" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^SENTRY_RELEASE=' | head -n1 || true)"
    echo "  ${git_sha}"
    echo "  ${sentry}"
    [[ "${git_sha}" == "GIT_SHA=${TARGET_SHA}" ]] || { echo "FAIL ${name} GIT_SHA"; fail=1; }
    [[ "${sentry}" == "SENTRY_RELEASE=${TARGET_SHA}" ]] || { echo "FAIL ${name} SENTRY_RELEASE"; fail=1; }
  fi
}

echo "======== Backend / WS / consumer / outbox ========"
check_svc atmr-backend yes yes
# ws = image distincte (atmr-ws-service) — gate OCI + release, pas Image ID backend
ws_cid="$(docker ps -aq --filter name=ws-service | head -n1 || true)"
if [[ -n "${ws_cid}" ]]; then
  check_svc "$(docker inspect "${ws_cid}" --format '{{.Name}}' | sed 's#^/##')" yes no
else
  echo "WARN: ws-service container not found by name filter"
fi
check_svc tracking-kafka-consumer yes yes
check_svc tracking-outbox-publisher yes yes

echo
echo "======== compose ps targeted ========"
compose ps tracking-kafka-consumer tracking-outbox-publisher \
  tracking-processed-fanout kafka-dlq-consumer || true

echo
echo "======== outbox pending ========"
docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
  exec -T postgres psql -U atmr -d atmr -Atc \
  "SELECT COUNT(*) FROM tracking_event_outbox WHERE published_at IS NULL;"

echo
echo "======== lag (kafka broker internal) ========"
compose exec -T kafka-broker-1 \
  kafka-consumer-groups --bootstrap-server kafka-broker-1:29092 \
  --describe --group tracking-ingest-consumer-group 2>&1 | head -40 || echo "WARN lag ingest"
compose exec -T kafka-broker-1 \
  kafka-consumer-groups --bootstrap-server kafka-broker-1:29092 \
  --describe --group ws-service-shared 2>&1 | head -40 || echo "WARN lag ws"

echo
echo "======== recent errors (last 5 min window sample) ========"
since="${LOG_SINCE:-5m}"
for svc in tracking-kafka-consumer tracking-outbox-publisher; do
  echo "--- ${svc} ---"
  docker logs --since "${since}" "$(compose ps -q "${svc}")" 2>&1 \
    | grep -Ei 'UniqueViolation|IntegrityError|restart|DNS|Name or service not known|OperationalError|Traceback' \
    | tail -n 30 || echo "(aucune erreur ciblée)"
done

if [[ "${fail}" -ne 0 ]]; then
  echo "PHASE1C_FAIL"
  exit 1
fi
echo "PHASE1C_IMMEDIATE_OK"
