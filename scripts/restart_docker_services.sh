#!/bin/bash
# Script de redémarrage des services Docker corrigés en production
# Usage: ./scripts/restart_docker_services.sh

set -e

echo "🔄 Redémarrage des services Docker corrigés..."
echo ""

# Répertoires
ATMR_DIR="/srv/atmr"
ATMR_RL_DIR="$HOME/atmr-rl"

# 1. Redémarrer celery-beat (volume corrompu corrigé)
echo "1️⃣  Redémarrage de celery-beat..."
cd "$ATMR_DIR"
docker compose -f docker-compose.production.yml stop celery-beat || true
docker compose -f docker-compose.production.yml rm -f celery-beat || true
docker volume rm atmr_celery_beat_data 2>/dev/null || true
docker compose -f docker-compose.production.yml up -d celery-beat
echo "✅ celery-beat redémarré"
sleep 5

# 2. Redémarrer Flower (healthcheck corrigé)
echo ""
echo "2️⃣  Redémarrage de Flower..."
cd "$ATMR_DIR"
docker compose -f docker-compose.production.yml up -d --force-recreate --no-deps flower
echo "✅ Flower redémarré"
sleep 5

# 3. Redémarrer le backend (configuration Traefik ajoutée)
echo ""
echo "3️⃣  Redémarrage du backend avec configuration Traefik..."
cd "$ATMR_DIR"
docker compose -f docker-compose.production.yml up -d backend
echo "✅ Backend redémarré"
sleep 5

# 4. Redémarrer le worker RL (si le répertoire existe)
if [ -d "$ATMR_RL_DIR" ]; then
    echo ""
    echo "4️⃣  Redémarrage du worker RL..."
    cd "$ATMR_RL_DIR"
    docker compose -f docker-compose.rl.yml stop rl-worker || true
    docker compose -f docker-compose.rl.yml rm -f rl-worker || true
    docker compose -f docker-compose.rl.yml up -d rl-worker
    echo "✅ Worker RL redémarré"
    sleep 5
fi

# 5. Vérifier l'état des services
echo ""
echo "5️⃣  Vérification de l'état des services..."
docker ps | grep -E "atmr-backend|atmr-flower|atmr-celery-beat|atmr-rl-worker" || true

echo ""
echo "✅ Redémarrage terminé !"
echo ""
echo "📊 Vérifier les logs avec :"
echo "   docker logs atmr-backend --tail 50"
echo "   docker logs atmr-celery-beat --tail 50"
echo "   docker logs atmr-flower --tail 50"

