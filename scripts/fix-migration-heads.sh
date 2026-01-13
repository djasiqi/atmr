#!/bin/bash
# Script pour résoudre le problème de heads multiples dans Alembic
# Usage: ./fix-migration-heads.sh

set -euo pipefail

cd /srv/atmr

echo "🔍 Diagnostic des migrations Alembic..."
echo ""

# Fonction helper pour docker compose exec
docker_exec() {
  docker compose -f docker-compose.production.yml exec -T backend "$@"
}

echo "📋 1. État actuel des migrations :"
docker_exec flask db current || echo "Aucune migration appliquée"
echo ""

echo "📋 2. Toutes les heads (branches) disponibles :"
docker_exec flask db heads
echo ""

echo "📋 3. Historique complet des migrations :"
docker_exec flask db history | head -30
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔧 SOLUTION : Merger les deux heads"
echo ""
echo "Nous avons deux options :"
echo ""
echo "Option A (RECOMMANDÉE) : Créer une migration de merge"
echo "----------------------------------------"
echo "Cette option crée une nouvelle migration qui merge les deux branches"
echo ""
read -p "Voulez-vous créer une migration de merge ? (oui/non) : " -r
echo ""

if [[ $REPLY =~ ^oui$ ]]; then
    echo "📝 Création d'une migration de merge..."
    echo ""
    
    # Créer une migration de merge
    docker_exec flask db merge heads -m "merge: résolution des heads multiples"
    
    echo ""
    echo "✅ Migration de merge créée"
    echo ""
    echo "📋 Nouvelles heads après merge :"
    docker_exec flask db heads
    echo ""
    
    echo "🔄 Application de la migration de merge..."
    docker_exec flask db upgrade heads
    
    echo ""
    echo "✅ Migration de merge appliquée avec succès !"
    echo ""
    echo "📋 État final :"
    docker_exec flask db current
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "🎉 Problème résolu !"
    echo ""
    echo "📝 N'oubliez pas de :"
    echo "   1. Récupérer la nouvelle migration de merge depuis le serveur"
    echo "   2. L'ajouter au repository Git"
    echo "   3. La committer et pusher"
    echo ""
    echo "Commandes :"
    echo "   docker compose -f docker-compose.production.yml cp backend:/app/migrations/versions/. backend/migrations/versions/"
    echo "   git add backend/migrations/versions/"
    echo "   git commit -m 'fix: merge migration heads'"
    echo "   git push"
    echo ""
else
    echo ""
    echo "Option B : Forcer l'upgrade sur un seul head"
    echo "----------------------------------------"
    echo ""
    echo "⚠️  Cette option force Alembic à n'utiliser qu'un seul head"
    echo ""
    read -p "Voulez-vous forcer sur le head bf7baf7a0f6f ? (oui/non) : " -r
    echo ""
    
    if [[ $REPLY =~ ^oui$ ]]; then
        echo "🔄 Upgrade vers bf7baf7a0f6f..."
        docker_exec flask db stamp bf7baf7a0f6f
        docker_exec flask db upgrade bf7baf7a0f6f
        
        echo ""
        echo "✅ Migration forcée"
        echo ""
        echo "⚠️  ATTENTION : Vous devrez peut-être appliquer l'autre head manuellement"
        echo ""
        echo "Pour appliquer l'autre head :"
        echo "   docker compose -f docker-compose.production.yml exec backend flask db stamp 311e1f6c9c9d"
        echo "   docker compose -f docker-compose.production.yml exec backend flask db upgrade 311e1f6c9c9d"
    else
        echo "❌ Opération annulée"
        exit 0
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔍 Vérification finale..."
echo ""

# Vérifier que le backend est healthy
echo "⏳ Attente du backend..."
for i in $(seq 1 30); do
  if curl -f -s --max-time 5 "http://localhost:5000/health" > /dev/null 2>&1; then
    echo "✅ Backend healthy"
    break
  fi
  sleep 1
done

echo ""
echo "✅ Tout est résolu !"
echo ""
