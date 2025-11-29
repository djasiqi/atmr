#!/bin/bash
# Script pour redémarrer tous les services de production
# Usage: ./scripts/restart_production_services.sh

SERVER="${1:-deploy@138.201.155.201}"

echo "🔄 Redémarrage des services de production..."
echo "Serveur: $SERVER"
echo ""

ssh "$SERVER" << 'EOF'
echo "📍 Répertoire de travail: /srv/atmr"
cd /srv/atmr || { echo "❌ Impossible d'accéder à /srv/atmr"; exit 1; }

echo ""
echo "🛑 1. Arrêt des services de production..."
echo "=========================================="
docker compose -f docker-compose.production.yml down

echo ""
echo "🔄 2. Redémarrage des services de production..."
echo "=============================================="
docker compose -f docker-compose.production.yml up -d

echo ""
echo "⏳ 3. Attente du démarrage (30 secondes)..."
echo "=========================================="
sleep 30

echo ""
echo "✅ 4. Vérification de l'état des services..."
echo "==========================================="
docker compose -f docker-compose.production.yml ps

echo ""
echo "📊 5. Statut des conteneurs principaux..."
echo "========================================"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "atmr-backend|atmr-celery|atmr-flower|atmr-postgres|atmr-redis" || echo "Aucun conteneur trouvé"

echo ""
echo "🔍 6. Logs récents du backend..."
echo "================================"
docker logs atmr-backend --tail 20 2>&1 | tail -10

echo ""
echo "✅ Redémarrage terminé!"
EOF

