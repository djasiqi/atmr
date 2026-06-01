#!/usr/bin/env bash
set -eu
echo "=== /app/docker-entrypoint.sh (lignes gunicorn) ==="
docker exec atmr-backend-1 sh -c "grep -n gunicorn /app/docker-entrypoint.sh | head -30"
echo "=== ls /app/gunicorn.conf.py ==="
docker exec atmr-backend-1 sh -c "ls -la /app/gunicorn.conf.py 2>&1 || true"
echo "=== Image digest ==="
docker inspect atmr-backend-1 --format '{{.Image}}' || true
docker inspect atmr-backend-1 --format '{{.Config.Image}}' || true
echo "=== Labels (VERSION/VCS_REF/BUILD_DATE) ==="
docker inspect atmr-backend-1 --format '{{json .Config.Labels}}' || true
echo "=== Section --preload dans entrypoint ==="
docker exec atmr-backend-1 sh -c "grep -n 'preload\|--config\|exec gunicorn' /app/docker-entrypoint.sh"
echo "=== Date du fichier entrypoint dans le conteneur ==="
docker exec atmr-backend-1 stat /app/docker-entrypoint.sh || true
