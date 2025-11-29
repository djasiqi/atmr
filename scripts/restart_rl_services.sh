#!/bin/bash
# Script pour redémarrer tous les services RL
# Usage: ./scripts/restart_rl_services.sh

SERVER="${1:-deploy@138.201.155.201}"

echo "🔄 Redémarrage des services RL..."
echo "Serveur: $SERVER"
echo ""

ssh "$SERVER" << 'EOF'
echo "📍 Répertoire de travail: ~/atmr-rl"
cd ~/atmr-rl || { echo "❌ Impossible d'accéder à ~/atmr-rl"; exit 1; }

echo ""
echo "🛑 1. Arrêt des services RL..."
echo "=============================="
docker compose -f docker-compose.rl.yml down

echo ""
echo "🔄 2. Redémarrage des services RL..."
echo "===================================="
docker compose -f docker-compose.rl.yml up -d

echo ""
echo "⏳ 3. Attente du démarrage (30 secondes)..."
echo "=========================================="
sleep 30

echo ""
echo "✅ 4. Vérification de l'état des services RL..."
echo "=============================================="
docker compose -f docker-compose.rl.yml ps

echo ""
echo "📊 5. Statut des conteneurs RL..."
echo "================================="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "atmr-rl|atmr-optuna" || echo "Aucun conteneur RL trouvé"

echo ""
echo "🔍 6. Logs récents du worker RL..."
echo "=================================="
docker logs atmr-rl-worker --tail 20 2>&1 | tail -10

echo ""
echo "✅ Redémarrage RL terminé!"
EOF

