#!/usr/bin/env bash
# Runbook P0 — push assignation driver 6855 (serveur /srv/atmr)
# Exécuter manuellement avec accès SSH production.
set -euo pipefail

cd /srv/atmr

echo "=== P0a baseline ==="
chmod +x scripts/celery-diagnostics-snapshot.sh
./scripts/celery-diagnostics-snapshot.sh baseline

echo "=== P0b — définir CELERY_CONCURRENCY=4 dans .env.production puis redémarrer ==="
if ! grep -q '^CELERY_CONCURRENCY=4' .env.production 2>/dev/null; then
  echo "ATTENTION: ajouter CELERY_CONCURRENCY=4 à .env.production avant de continuer"
  exit 1
fi

docker compose -f docker-compose.production.yml up -d celery-worker
./scripts/celery-diagnostics-snapshot.sh post-p0b

echo "=== Surveillance 30 min — lancer dans un autre terminal ==="
echo "  docker events --filter container=atmr-celery-worker"
echo "  docker logs -f atmr-celery-worker --since 5m"

echo "=== P0e smoke test (après worker stable) ==="
docker exec atmr-celery-worker python -c "
from celery_app import celery
r = celery.send_task('tasks.health_tasks.celery_health_ping')
print('task_id=', r.id)
print('result=', r.get(timeout=15))
"

echo "=== P0c — effectuer assignation test driver 6855 puis ==="
echo "  ./scripts/celery-diagnostics-snapshot.sh post-p0c"
