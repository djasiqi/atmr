#!/usr/bin/env bash
# P0 ops — stop fanout + DLQ legacy + recreate ingest seul (pas de up global).
# Aucun défaut pour COMPOSE_FILES / ENV_FILE / COMPOSE_PROJECT_NAME /
# DOCKER_IMAGE / DOCKER_TAG / SOURCE_SHA / IMAGE_DIGEST / EXPECTED_INGEST_REPLICAS.
#
# Dry-run (aucun stop / up / pull serveur) :
#   DRY_RUN=1 COMPOSE_PROJECT_NAME=atmr \
#     COMPOSE_FILES="-f docker-compose.production.yml -f docker-compose.kafka.yml -f docker-compose.kafka.atmr-network.yml -f docker-compose.kafka.p0-hold.yml" \
#     ENV_FILE="--env-file .env.production" \
#     DOCKER_IMAGE=... DOCKER_TAG=... SOURCE_SHA=... IMAGE_DIGEST=sha256:... \
#     EXPECTED_INGEST_REPLICAS=1 \
#     ./scripts/ops-tracking-p0-recreate-ingest.sh
#
# Exécution réelle :
#   EXECUTE_P0_RECREATE=YES ... (mêmes vars) ./scripts/ops-tracking-p0-recreate-ingest.sh
#
set -euo pipefail

: "${COMPOSE_FILES:?COMPOSE_FILES obligatoire}"
: "${ENV_FILE:?ENV_FILE obligatoire (ex: --env-file .env.production)}"
: "${COMPOSE_PROJECT_NAME:?COMPOSE_PROJECT_NAME obligatoire}"
: "${DOCKER_IMAGE:?DOCKER_IMAGE obligatoire}"
: "${DOCKER_TAG:?DOCKER_TAG obligatoire}"
: "${SOURCE_SHA:?SOURCE_SHA obligatoire (40 hex)}"
: "${IMAGE_DIGEST:?IMAGE_DIGEST obligatoire (sha256:...)}"
: "${EXPECTED_INGEST_REPLICAS:?EXPECTED_INGEST_REPLICAS obligatoire}"

PROFILE="${PROFILE:---profile kafka}"
DRY_RUN="${DRY_RUN:-0}"
EXECUTE_P0_RECREATE="${EXECUTE_P0_RECREATE:-}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-120}"

[[ "${SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "SOURCE_SHA doit être un SHA Git complet de 40 caractères" >&2
  exit 1
}
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  echo "IMAGE_DIGEST invalide (attendu sha256: + 64 hex)" >&2
  exit 1
}
[[ "${EXPECTED_INGEST_REPLICAS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EXPECTED_INGEST_REPLICAS invalide" >&2
  exit 1
}

compose() {
  # shellcheck disable=SC2086
  docker compose -p "${COMPOSE_PROJECT_NAME}" ${ENV_FILE} ${COMPOSE_FILES} ${PROFILE} "$@"
}

umask 077
CFG_TMP="$(mktemp /tmp/atmr-p0-compose.XXXXXX)"
cleanup_cfg() {
  rm -f "${CFG_TMP}"
}
trap cleanup_cfg EXIT

GPS_SERVICES=(tracking-kafka-consumer tracking-processed-fanout kafka-dlq-consumer)
EXPECTED_COMPOSE_BASENAMES=(
  docker-compose.production.yml
  docker-compose.kafka.yml
  docker-compose.kafka.atmr-network.yml
  docker-compose.kafka.p0-hold.yml
)

echo "== Préflight config fusionné (mktemp) =="
compose config > "${CFG_TMP}"

assert_env_flag() {
  local svc="$1"
  local key="$2"
  local expected="$3"
  local got
  got="$(
    awk -v svc="$svc" -v key="$key" '
      $0 ~ "^  "svc":" {in_svc=1; next}
      in_svc && /^  [a-zA-Z]/ {in_svc=0}
      in_svc && $0 ~ ("^      " key ":") {
        sub(/^[^:]+:[[:space:]]*/, "", $0)
        gsub(/["'\'' ]/, "", $0)
        print $0
        exit
      }
    ' "${CFG_TMP}"
  )"
  if [[ "${got}" != "${expected}" ]]; then
    echo "FAIL ${svc}.${key}: attendu=${expected} obtenu=${got:-<absent>}" >&2
    exit 1
  fi
  echo "OK ${svc}.${key}=${got}"
}

echo "== Vérif DSN / flags / topics (config) =="
for svc in "${GPS_SERVICES[@]}"; do
  echo "--- ${svc} ---"
  awk -v svc="$svc" '
    $0 ~ "^  "svc":" {in_svc=1; next}
    in_svc && /^  [a-zA-Z]/ {in_svc=0}
    in_svc && /DATABASE_URL:|SQLALCHEMY_DATABASE_URI:|PRIMARY_DATABASE_URL:|REPLICA_DATABASE_URL:|REPLICA_DATABASE_URLS:|POSTGRES_HOST:|POSTGRES_PORT:|POSTGRES_DB:|POSTGRES_USER:/ {
      print
    }
  ' "${CFG_TMP}"
done

if grep -E 'postgresql\+psycopg://\$\{POSTGRES_(USER|PASSWORD)' "${CFG_TMP}" >/dev/null 2>&1; then
  echo "FAIL: URL postgresql+psycopg interpolée détectée dans le config fusionné" >&2
  exit 1
fi

assert_env_flag tracking-kafka-consumer TRACKING_PERSIST_WITH_OUTBOX true
assert_env_flag tracking-kafka-consumer TRACKING_INGEST_PERSIST_ENABLED true
assert_env_flag tracking-kafka-consumer TRACKING_INGEST_ALLOW_REPUBLISH_ONLY false
assert_env_flag tracking-kafka-consumer TRACKING_INGEST_SEEK_TO_END_ON_START false
assert_env_flag tracking-kafka-consumer TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE false
assert_env_flag tracking-processed-fanout TRACKING_PROCESSED_FANOUT_ENABLED false
assert_env_flag tracking-kafka-consumer KAFKA_TOPIC_DRIVER_LOCATION_RAW driver.location.raw.v2
assert_env_flag tracking-kafka-consumer KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed.v2
assert_env_flag tracking-kafka-consumer KAFKA_TOPIC_DRIVER_LOCATION_DLQ driver.location.dlq.v2
assert_env_flag tracking-processed-fanout KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED driver.location.processed.v2
assert_env_flag kafka-dlq-consumer KAFKA_TOPIC_DRIVER_LOCATION_DLQ driver.location.dlq.v2

# Fail-hard si consumer GPS hors projet atmr (avant toute mutation)
fail_hard_foreign_gps_consumers() {
  local cid name short_id svc project image
  local found=0
  while IFS= read -r cid; do
    [[ -z "${cid}" ]] && continue
    project="$(docker inspect "${cid}" --format '{{index .Config.Labels "com.docker.compose.project"}}' 2>/dev/null || true)"
    svc="$(docker inspect "${cid}" --format '{{index .Config.Labels "com.docker.compose.service"}}' 2>/dev/null || true)"
    if [[ "${project}" != "${COMPOSE_PROJECT_NAME}" ]]; then
      found=1
      name="$(docker inspect "${cid}" --format '{{.Name}}' | sed 's#^/##')"
      short_id="${cid:0:12}"
      image="$(docker inspect "${cid}" --format '{{.Config.Image}}')"
      echo "FAIL consumer GPS étranger :" >&2
      echo "  name=${name} id=${short_id} service=${svc} project=${project:-<none>} image=${image}" >&2
    fi
  done < <(
    for svc in "${GPS_SERVICES[@]}"; do
      docker ps -q --filter "label=com.docker.compose.service=${svc}" --filter "status=running" 2>/dev/null || true
    done
  )
  if ((found != 0)); then
    echo "Intervention ops manuelle requise — aucun stop automatique hors projet ${COMPOSE_PROJECT_NAME}." >&2
    exit 1
  fi
  echo "OK aucun consumer GPS étranger au projet ${COMPOSE_PROJECT_NAME}"
}

fail_hard_foreign_gps_consumers

# Inventaire informatif (pré-stop) — ne pas exiger EXPECTED_INGEST_REPLICAS
mapfile -t pre_ingest_cids < <(
  docker ps -q \
    --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
    --filter "label=com.docker.compose.service=tracking-kafka-consumer" \
    --filter "status=running" 2>/dev/null || true
)
echo "INFO replicas ingest projet ${COMPOSE_PROJECT_NAME} avant stop : ${#pre_ingest_cids[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — aucun pull / stop / up / recreate. Config OK."
  exit 0
fi

if [[ "${EXECUTE_P0_RECREATE}" != "YES" ]]; then
  echo "Refuse l'exécution destructive : définir EXECUTE_P0_RECREATE=YES (ou DRY_RUN=1)." >&2
  exit 1
fi

expected_ref="${DOCKER_IMAGE}@${IMAGE_DIGEST}"
expected_tag_ref="${DOCKER_IMAGE}:${DOCKER_TAG}"

echo "== Pull image exacte par digest =="
docker pull "${expected_ref}"
expected_id="$(docker image inspect "${expected_ref}" --format '{{.Id}}')"
repo_digests="$(docker image inspect "${expected_ref}" --format '{{join .RepoDigests "\n"}}')"
if ! grep -Fx "${DOCKER_IMAGE}@${IMAGE_DIGEST}" <<<"${repo_digests}" >/dev/null; then
  echo "FAIL RepoDigest manquant : attendu ${DOCKER_IMAGE}@${IMAGE_DIGEST}" >&2
  echo "${repo_digests}" >&2
  exit 1
fi
echo "OK RepoDigest ${DOCKER_IMAGE}@${IMAGE_DIGEST}"
echo "OK expected_image_id=${expected_id}"

echo "== Tag local ${expected_tag_ref} (après validation digest) =="
docker tag "${expected_ref}" "${expected_tag_ref}"

echo "== Stop fanout + DLQ (projet ${COMPOSE_PROJECT_NAME} uniquement) =="
compose stop tracking-processed-fanout kafka-dlq-consumer || true

echo "== Stop ingest =="
compose stop tracking-kafka-consumer || true

# Sentry release = SHA réel de l'image consumer (pas celui du dernier deploy backend).
export GIT_SHA="${SOURCE_SHA}"
export SENTRY_RELEASE="${SOURCE_SHA}"
echo "OK Sentry release consumer GIT_SHA=${GIT_SHA} SENTRY_RELEASE=${SENTRY_RELEASE}"

echo "== Recreate ingest (--pull never, scale=${EXPECTED_INGEST_REPLICAS}) =="
compose up -d --no-deps --pull never --force-recreate \
  --scale "tracking-kafka-consumer=${EXPECTED_INGEST_REPLICAS}" \
  tracking-kafka-consumer

wait_ingest_healthy() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
  local cid hs restart_count
  while ((SECONDS < deadline)); do
    mapfile -t ingest_cids < <(
      docker ps -q \
        --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
        --filter "label=com.docker.compose.service=tracking-kafka-consumer" \
        --filter "status=running" 2>/dev/null || true
    )
    if [[ "${#ingest_cids[@]}" -eq "${EXPECTED_INGEST_REPLICAS}" ]]; then
      local all_ok=1
      for cid in "${ingest_cids[@]}"; do
        restart_count="$(docker inspect "${cid}" --format '{{.RestartCount}}')"
        if [[ "${restart_count}" -gt 3 ]]; then
          echo "FAIL restart loop détectée (RestartCount=${restart_count}) cid=${cid:0:12}" >&2
          exit 1
        fi
        hs="$(docker inspect "${cid}" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}')"
        if [[ "${hs}" == "unhealthy" ]]; then
          all_ok=0
          break
        fi
        if [[ "${hs}" != "none" && "${hs}" != "healthy" ]]; then
          all_ok=0
          break
        fi
      done
      if ((all_ok == 1)); then
        echo "OK ${#ingest_cids[@]} replica(s) ingest running/healthy"
        return 0
      fi
    fi
    sleep 3
  done
  echo "FAIL timeout ${HEALTH_TIMEOUT_S}s : replicas healthy attendus=${EXPECTED_INGEST_REPLICAS}" >&2
  docker ps --filter "label=com.docker.compose.service=tracking-kafka-consumer" >&2 || true
  exit 1
}

wait_ingest_healthy

mapfile -t ingest_cids < <(
  docker ps -q \
    --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
    --filter "label=com.docker.compose.service=tracking-kafka-consumer" \
    --filter "status=running"
)
if [[ "${#ingest_cids[@]}" -ne "${EXPECTED_INGEST_REPLICAS}" ]]; then
  echo "FAIL replicas running=${#ingest_cids[@]} attendu=${EXPECTED_INGEST_REPLICAS}" >&2
  exit 1
fi

assert_compose_basenames() {
  local cid="$1"
  local config_files
  local -a basenames=()
  local file expected_files actual_files
  config_files="$(
    docker inspect "${cid}" \
      --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
  )"
  IFS=',' read -ra files <<<"${config_files}"
  for file in "${files[@]}"; do
    file="${file#"${file%%[![:space:]]*}"}"
    file="${file%"${file##*[![:space:]]}"}"
    [[ -z "${file}" ]] && continue
    basenames+=("$(basename "${file}")")
  done
  expected_files="$(printf '%s\n' "${EXPECTED_COMPOSE_BASENAMES[@]}" | sort)"
  actual_files="$(printf '%s\n' "${basenames[@]}" | sed '/^$/d' | sort -u)"
  if [[ "${actual_files}" != "${expected_files}" ]]; then
    echo "FAIL ensemble Compose runtime incohérent (cid=${cid:0:12})" >&2
    diff -u <(printf '%s\n' "${expected_files}") <(printf '%s\n' "${actual_files}") >&2 || true
    exit 1
  fi
  echo "OK config_files ensemble exact (cid=${cid:0:12})"
}

assert_runtime_cid() {
  local cid="$1"
  local actual_ref actual_id revision
  actual_ref="$(docker inspect "${cid}" --format '{{.Config.Image}}')"
  actual_id="$(docker inspect "${cid}" --format '{{.Image}}')"
  if [[ "${actual_ref}" != "${expected_tag_ref}" ]]; then
    echo "FAIL Config.Image=${actual_ref} attendu=${expected_tag_ref}" >&2
    exit 1
  fi
  if [[ "${actual_id}" != "${expected_id}" ]]; then
    echo "FAIL Image ID runtime=${actual_id} attendu=${expected_id}" >&2
    exit 1
  fi
  revision="$(docker inspect "${cid}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
  if [[ "${revision}" != "${SOURCE_SHA}" ]]; then
    echo "FAIL OCI revision=${revision:-<absent>} attendu=${SOURCE_SHA}" >&2
    exit 1
  fi
  assert_compose_basenames "${cid}"

  docker exec -i "${cid}" python - <<'PY'
import os
import pathlib
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from config import _build_database_url_safe

url_keys = (
    "DATABASE_URL",
    "SQLALCHEMY_DATABASE_URI",
    "PRIMARY_DATABASE_URL",
    "REPLICA_DATABASE_URL",
    "REPLICA_DATABASE_URLS",
)
for key in url_keys:
    value = os.getenv(key)
    assert value == "", f"{key} is not empty (got {value!r})"

raw = _build_database_url_safe()
url = make_url(raw)
print({
    "driver": url.drivername,
    "host": url.host,
    "port": url.port,
    "database": url.database,
    "user_present": bool(url.username),
    "password_present": bool(url.password),
})
assert url.host == "pgbouncer", url.host
assert url.port == 6432, url.port
with create_engine(raw).connect() as conn:
    assert conn.execute(text("SELECT 1")).scalar() == 1
print("SELECT 1 OK")

def flag(name: str) -> str:
    return (os.getenv(name) or "").strip().lower()

assert flag("TRACKING_PERSIST_WITH_OUTBOX") == "true"
assert flag("TRACKING_INGEST_PERSIST_ENABLED") == "true"
assert flag("TRACKING_INGEST_ALLOW_REPUBLISH_ONLY") == "false"
assert flag("TRACKING_INGEST_SEEK_TO_END_ON_START") == "false"
assert flag("TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE") == "false"
assert os.getenv("KAFKA_TOPIC_DRIVER_LOCATION_RAW") == "driver.location.raw.v2"
assert os.getenv("KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED") == "driver.location.processed.v2"
assert os.getenv("KAFKA_TOPIC_DRIVER_LOCATION_DLQ") == "driver.location.dlq.v2"
print("flags/topics OK")

cmd = pathlib.Path("/proc/1/cmdline").read_text()
assert "ingest_consumer" in cmd, cmd
print("PID1 ingest_consumer OK")
PY
  echo "OK runtime asserts cid=${cid:0:12}"
}

echo "== Assertions runtime sur chaque replica =="
for cid in "${ingest_cids[@]}"; do
  assert_runtime_cid "${cid}"
done

echo "== Fanout + DLQ projet atmr doivent rester stopped =="
for svc in tracking-processed-fanout kafka-dlq-consumer; do
  running="$(
    docker ps -q \
      --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
      --filter "label=com.docker.compose.service=${svc}" \
      --filter "status=running" 2>/dev/null || true
  )"
  if [[ -n "${running}" ]]; then
    echo "FAIL ${svc} encore running après recreate ingest" >&2
    exit 1
  fi
  echo "OK ${svc} stopped"
done

fail_hard_foreign_gps_consumers

echo "DONE — lancer ensuite la gate E2E ×3 (nouvelles positions, pas de reset offsets)."
echo "Recréer kafka-dlq-consumer seulement après gate ingest OK."
