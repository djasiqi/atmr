#!/usr/bin/env bash
# Phase 4 — gate GO/NO-GO avant canary P0-F.
set -euo pipefail
cd /srv/atmr
TARGET=390076efc61ca71332c749a67aff1e6fc7c2d626
EXPECTED=sha256:780a166c04b928d3a24a7f773a83cf1835d03512b9ab1073d87ef395003ecc4d

echo "=== OCI align ==="
for name in atmr-backend tracking-kafka-consumer tracking-outbox-publisher; do
  cid="$(docker ps -aq --filter "name=${name}" | head -n1)"
  img="$(docker inspect "${cid}" --format '{{.Image}}')"
  rev="$(docker image inspect "${img}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
  echo "${name}: ${rev} img=${img}"
done

echo "=== Sentry env (présence DSN, pas d’audit events) ==="
docker inspect atmr-backend-1 --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(SENTRY_DSN|SENTRY_RELEASE|GIT_SHA)=' \
  | sed 's/\(SENTRY_DSN=\).*/\1<present>/' || echo "SENTRY_DSN absent"

echo "=== Autopsy file ==="
test -f docs/ops/gps-p0e-seq3-autopsy.md && echo OK_autopsy || echo MISSING_autopsy

echo "PHASE4_CHECKS_DONE"
