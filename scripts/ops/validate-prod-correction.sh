#!/usr/bin/env bash
# Validation post-correction prod (Kafka / Celery / ws-service / Traefik).
# Usage : ATMR_DEPLOY_ROOT=/srv/atmr ./scripts/ops/validate-prod-correction.sh
set -euo pipefail

ROOT="${ATMR_DEPLOY_ROOT:-/srv/atmr}"
cd "${ROOT}"

echo "=== P1a Celery action= ==="
docker logs --since 10m atmr-celery-worker 2>&1 | grep -ci "unexpected keyword argument 'action'" || true
echo "(attendu: 0)"

echo "=== P1b Kafka brokers ==="
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -Ei 'kafka|zoo' || true
docker exec atmr-tracking-kafka-consumer-1 sh -lc \
  'getent hosts kafka-broker-1 kafka-broker-2 kafka-broker-3' 2>/dev/null || true

echo "=== P2b ws-service health ==="
ws_ip="$(docker inspect -f '{{(index .NetworkSettings.Networks "atmr-network").IPAddress}}' atmr-ws-service 2>/dev/null || echo "")"
if [[ -n "${ws_ip}" ]]; then
  curl -fsS "http://${ws_ip}:8001/health" | python3 -m json.tool || true
fi

echo "=== P2c Traefik ACME (externe) ==="
echo "Depuis un réseau externe : curl -v http://api.lirie.ch/.well-known/acme-challenge/test"
docker logs --since 10m traefik 2>&1 | grep -i acme | tail -n 20 || true

echo "=== Fin validation ==="
