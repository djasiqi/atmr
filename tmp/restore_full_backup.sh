#!/bin/bash
# Restauration complète de la base de données

set -euo pipefail

cd /srv/atmr

echo "🛑 Arrêt du backend..."
docker compose -f docker-compose.production.yml stop backend celery-worker celery-beat flower

echo "🗄️ Restauration de la base de données..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr < /tmp/restore_full.sql

echo "✅ Restauration terminée"

echo "🚀 Redémarrage des services..."
docker compose -f docker-compose.production.yml start backend celery-worker celery-beat flower

echo "⏳ Attente du démarrage du backend..."
sleep 10

echo "🔍 Vérification du nombre d'utilisateurs..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c 'SELECT COUNT(*) as total_users FROM "user";'

echo "✅ Restauration complète terminée !"
