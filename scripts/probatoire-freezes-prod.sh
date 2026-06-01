#!/usr/bin/env bash
# Phase probatoire — freezes Gunicorn (lecture seule)
# Usage sur le serveur prod :
#   cd /srv/atmr && bash scripts/probatoire-freezes-prod.sh
# Usage depuis poste local (SSH) :
#   export SERVER_HOST=... SERVER_USER=deploy
#   bash scripts/probatoire-freezes-prod.sh --remote
set -eu

REMOTE=0
OUT_DIR="${OUT_DIR:-/tmp/probatoire-freezes-$(date -u +%Y%m%dT%H%M%SZ)}"
SINCE="${SINCE:-48h}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"
BACKEND_CONTAINER="${BACKEND_CONTAINER:-atmr-backend-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE=1; shift ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --since) SINCE="$2"; shift 2 ;;
    *) echo "Usage: $0 [--remote] [--out DIR] [--since 48h]"; exit 1 ;;
  esac
done

run_remote() {
  : "${SERVER_HOST:?Définir SERVER_HOST (voir docs/deployment-ssh.md)}"
  local user="${SERVER_USER:-deploy}"
  local path="${SERVER_PATH:-/srv/atmr}"
  ssh "${user}@${SERVER_HOST}" "cd ${path} && OUT_DIR='${OUT_DIR}' SINCE='${SINCE}' bash scripts/probatoire-freezes-prod.sh"
}

if [[ "$REMOTE" -eq 1 ]]; then
  run_remote
  exit 0
fi

mkdir -p "$OUT_DIR"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT_DIR/run.log"; }

log "=== Etape 1 — Config runtime ==="
{
  echo "--- docker inspect Cmd/Args ---"
  docker inspect "$BACKEND_CONTAINER" --format '{{.Config.Cmd}} {{.Args}}' 2>/dev/null || true
  echo "--- ps gunicorn ---"
  docker exec "$BACKEND_CONTAINER" sh -c 'ps -ef | grep -i gunicorn | grep -v grep' 2>/dev/null || true
  echo "--- worker count ---"
  docker exec "$BACKEND_CONTAINER" sh -c 'pgrep -af "gunicorn.*worker" | wc -l' 2>/dev/null || true
  echo "--- env Gunicorn/SocketIO ---"
  docker exec "$BACKEND_CONTAINER" printenv GUNICORN_WORKERS GUNICORN_WORKER_CLASS GUNICORN_TIMEOUT SOCKETIO_ASYNC_MODE 2>/dev/null || true
  echo "--- DATABASE_URL (masque) ---"
  docker exec "$BACKEND_CONTAINER" sh -c 'printenv DATABASE_URL SQLALCHEMY_DATABASE_URI | sed -E "s#//[^@]*@#//***@#"' 2>/dev/null || true
  echo "--- drivername ---"
  docker exec "$BACKEND_CONTAINER" python -c "from ext import db; print(db.engine.url.drivername)" 2>/dev/null || true
  echo "--- psycogreen ---"
  docker exec "$BACKEND_CONTAINER" pip show psycogreen 2>/dev/null || echo "psycogreen ABSENT"
  echo "--- gunicorn.conf.py in container ---"
  docker exec "$BACKEND_CONTAINER" sh -c 'grep -n "worker_abort\|worker_int\|faulthandler" gunicorn.conf.py 2>/dev/null || echo ABSENT' 2>/dev/null || true
} > "$OUT_DIR/etape1_config.txt" 2>&1

log "=== Etape 2 — Dumps faulthandler (PRIORITE ABSOLUE) ==="
cd /srv/atmr 2>/dev/null || cd "$(dirname "$0")/.." || true

docker compose -f "$COMPOSE_FILE" logs backend --since "$SINCE" 2>/dev/null \
  | grep -nE 'WORKER TIMEOUT|WORKER ABORT|Current thread|File "' \
  > "$OUT_DIR/abort_dumps_index.txt" || true
wc -l "$OUT_DIR/abort_dumps_index.txt" | tee -a "$OUT_DIR/run.log"

docker compose -f "$COMPOSE_FILE" logs backend --since "$SINCE" 2>/dev/null \
  | awk '/WORKER ABORT/{flag=1;count=0} flag{print;count++} count==120{flag=0}' \
  > "$OUT_DIR/abort_stacks_full.txt" || true

docker compose -f "$COMPOSE_FILE" logs backend --since "$SINCE" 2>/dev/null \
  | awk '/WORKER TIMEOUT/{flag=1;count=0} flag{print;count++} count==120{flag=0}' \
  > "$OUT_DIR/timeout_stacks_full.txt" || true

ABORT_COUNT=$(grep -c 'WORKER ABORT' "$OUT_DIR/abort_stacks_full.txt" 2>/dev/null || echo 0)
TIMEOUT_COUNT=$(grep -c 'WORKER TIMEOUT' "$OUT_DIR/timeout_stacks_full.txt" 2>/dev/null || echo 0)
log "WORKER ABORT blocks: $ABORT_COUNT | WORKER TIMEOUT blocks: $TIMEOUT_COUNT"

log "=== Etape 4 — IP pollers /drivers ==="
docker compose -f "$COMPOSE_FILE" logs backend --since "$SINCE" 2>/dev/null \
  | grep -E 'GET /(api/v1/)?companies/me/drivers(/locations)?' \
  | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' \
  | sort | uniq -c | sort -rn | head -30 \
  > "$OUT_DIR/drivers_ip_counts.txt" || true

docker compose -f "$COMPOSE_FILE" logs backend --since "$SINCE" 2>/dev/null \
  | grep -E 'GET /(api/v1/)?companies/me/drivers(/locations)?.*" (504|499|500) ' \
  > "$OUT_DIR/drivers_errors_sample.txt" || true

log "=== Resume ==="
echo "OUT_DIR=$OUT_DIR" | tee "$OUT_DIR/summary.txt"
echo "abort_dumps_index lines: $(wc -l < "$OUT_DIR/abort_dumps_index.txt" 2>/dev/null || echo 0)" >> "$OUT_DIR/summary.txt"
echo "abort_stacks_full bytes: $(wc -c < "$OUT_DIR/abort_stacks_full.txt" 2>/dev/null || echo 0)" >> "$OUT_DIR/summary.txt"
log "Termine. Analyser: $OUT_DIR/abort_stacks_full.txt"
