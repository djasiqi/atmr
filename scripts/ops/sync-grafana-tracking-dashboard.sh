#!/usr/bin/env bash
# Synchronise le dashboard Grafana « Driver Tracking Health » vers monitoring/ + redémarre Grafana.
#
# Usage local (Docker monitoring) :
#   bash scripts/ops/sync-grafana-tracking-dashboard.sh local
#
# Usage prod (SSH) :
#   export SERVER_HOST=... SERVER_USER=deploy
#   bash scripts/ops/sync-grafana-tracking-dashboard.sh prod
#
# Import manuel : copier monitoring/grafana/dashboards/driver-tracking-health.json
# dans Grafana UI → Dashboards → Import (uid: driver-tracking-health).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CANONICAL="${ROOT}/monitoring/grafana/dashboards/driver-tracking-health.json"
SOURCE="${ROOT}/backend/alerts/grafana/driver_tracking_health.json"
TARGET_MODE="${1:-local}"

echo "=== Sync dashboard Grafana tracking ==="

if [[ ! -f "${CANONICAL}" ]]; then
  echo "[FAIL] Dashboard canonique absent : ${CANONICAL}"
  exit 1
fi

if [[ -f "${SOURCE}" ]]; then
  echo "[INFO] Source plan : ${SOURCE}"
  echo "[INFO] Canonique (provisioning) : ${CANONICAL}"
  echo "[OK] Le dashboard provisionné inclut les panels N3 (freshness, invariants, pipeline)."
else
  echo "[WARN] Source backend/alerts absente — utilisation du canonique uniquement."
fi

sync_local() {
  if ! docker ps --format '{{.Names}}' | grep -q '^atmr-grafana$'; then
    echo "[INFO] Grafana local non démarré — lancement monitoring profile..."
    cd "${ROOT}"
    docker compose -f docker-compose.monitoring.yml up -d grafana
    sleep 8
  fi
  echo "[INFO] Redémarrage Grafana pour recharger provisioning..."
  docker restart atmr-grafana
  sleep 5
  if docker exec atmr-grafana test -f /var/lib/grafana/dashboards/driver-tracking-health.json; then
    echo "[OK] Dashboard monté dans le container"
  else
    echo "[FAIL] Dashboard non monté — vérifier docker-compose.monitoring.yml volumes"
    exit 1
  fi
  echo "[OK] Grafana local : http://localhost:3000 (ou port mappé)"
  echo "     Dashboard uid=driver-tracking-health"
}

sync_prod() {
  if [[ -z "${SERVER_HOST:-}" ]]; then
    echo "[FAIL] SERVER_HOST requis pour prod (.local.deploy.env)"
    exit 1
  fi
  REMOTE="${SERVER_USER:-deploy}@${SERVER_HOST}"
  REMOTE_PATH="${SERVER_PATH:-/srv/atmr}"
  echo "[INFO] Sync vers ${REMOTE}:${REMOTE_PATH}"
  rsync -avz "${CANONICAL}" "${REMOTE}:${REMOTE_PATH}/monitoring/grafana/dashboards/driver-tracking-health.json"
  ssh "${REMOTE}" "cd ${REMOTE_PATH} && docker restart atmr-grafana && sleep 5 && docker exec atmr-grafana test -f /var/lib/grafana/dashboards/driver-tracking-health.json"
  echo "[OK] Grafana prod : https://grafana.lirie.ch/d/driver-tracking-health"
}

case "${TARGET_MODE}" in
  local) sync_local ;;
  prod) sync_prod ;;
  *)
    echo "Mode : local | prod"
    exit 1
    ;;
esac
