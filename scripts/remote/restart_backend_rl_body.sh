#!/usr/bin/env bash
# Exécuté sur le serveur (stdin bash -s) — utilisé par restart_backend_with_rl_network.ps1
# et logique alignée sur restart_backend_with_rl_network.sh
set -euo pipefail
cd /srv/atmr

echo "📦 Vérification de la configuration docker-compose..."
if [ ! -f docker-compose.production.yml ]; then
    echo "❌ Erreur: docker-compose.production.yml non trouvé"
    exit 1
fi

echo "🔍 Vérification que le réseau RL existe..."
if ! docker network inspect atmr-rl-network > /dev/null 2>&1; then
    echo "⚠️  Le réseau atmr-rl-network n'existe pas encore."
    echo "ℹ️  Il sera créé automatiquement au prochain démarrage de docker-compose.rl.yml"
    echo "   Ou créez-le manuellement avec: docker network create atmr-rl-network"
fi

echo "🛑 Arrêt du backend..."
docker compose -f docker-compose.production.yml stop backend

echo "🗑️  Suppression du conteneur backend (pour recréer avec nouvelle config réseau)..."
docker compose -f docker-compose.production.yml rm -f backend

echo "🚀 Démarrage du backend avec nouvelle configuration..."
docker compose -f docker-compose.production.yml up -d backend

echo "⏳ Attente de 10 secondes pour le démarrage..."
sleep 10

echo "✅ Vérification de l'état du backend..."
docker compose -f docker-compose.production.yml ps backend

echo ""
echo "🔍 Vérification des réseaux connectés au backend..."
BACKEND_CID=$(docker compose -f docker-compose.production.yml ps -q backend 2>/dev/null)
if [ -n "$BACKEND_CID" ]; then
  docker inspect "$BACKEND_CID" --format='{{range $net, $conf := .NetworkSettings.Networks}}{{printf "  - %s\n" $net}}{{end}}' | head -10
else
  echo "  (conteneur backend introuvable — docker compose ps backend)"
fi

echo ""
echo "✅ Backend redémarré avec la nouvelle configuration!"
echo "ℹ️  Le backend devrait maintenant être connecté aux réseaux:"
echo "   - atmr-network"
echo "   - traefik-network"
echo "   - atmr-rl-network (pour communiquer avec le worker RL)"
