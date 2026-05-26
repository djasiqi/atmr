#!/usr/bin/env bash
# Celery worker diagnostics snapshot (runbook push assignation).
# Usage: ./scripts/celery-diagnostics-snapshot.sh [label]
# label = baseline | post-p0b | oom-incident | post-p0c
set -euo pipefail

LABEL="${1:-snapshot}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DIR="/tmp/celery-diagnostics/${TS}_${LABEL}"
mkdir -p "$DIR"

echo "Writing diagnostics to $DIR"

docker stats atmr-celery-worker --no-stream >"$DIR/stats.txt" 2>&1 || true
docker exec atmr-celery-worker ps aux --sort=-rss | head -30 >"$DIR/ps-rss.txt" 2>&1 || true

docker exec atmr-celery-worker \
  celery -A celery_app.celery inspect stats >"$DIR/inspect-stats.txt" 2>&1 || true
docker exec atmr-celery-worker \
  celery -A celery_app.celery inspect active >"$DIR/inspect-active.txt" 2>&1 || true
docker exec atmr-celery-worker \
  celery -A celery_app.celery inspect reserved >"$DIR/inspect-reserved.txt" 2>&1 || true

# Profondeur queues Redis — LLEN une queue à la fois (OBLIGATOIRE).
# NE PAS utiliser : redis-cli LLEN default LLEN dispatch ... (syntaxe invalide).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$ROOT_DIR/.env.production" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.production"
  set +a
fi
: >"$DIR/queue-lengths.txt"
for Q in default dispatch realtime notifications billing rl geocoding dlq; do
  LEN="$(docker exec atmr-redis redis-cli -a "${REDIS_PASSWORD:-}" --no-auth-warning LLEN "$Q" 2>/dev/null || echo ERR)"
  echo "$Q=$LEN" >>"$DIR/queue-lengths.txt"
done

dmesg | grep -i kill | tail -20 >"$DIR/dmesg-oom.txt" 2>&1 || true
docker events --filter container=atmr-celery-worker --since 24h --until 0s \
  >"$DIR/docker-events-24h.txt" 2>&1 || true

echo "Done: $DIR"
ls -la "$DIR"
