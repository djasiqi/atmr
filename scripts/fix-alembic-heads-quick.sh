#!/bin/bash
# Fix rapide pour le problème de heads multiples Alembic
# Ce script merge automatiquement les heads et applique les migrations

set -euo pipefail

cd /srv/atmr

echo "🔧 Fix rapide : Merge des heads Alembic multiples"
echo ""

# Fonction helper
docker_exec() {
  docker compose -f docker-compose.production.yml exec -T backend "$@"
}

# 1. Créer une migration de merge
echo "📝 Création de la migration de merge..."
docker_exec flask db merge heads -m "merge: fix overlapping migration heads (bf7baf7a0f6f + 311e1f6c9c9d)" || {
    echo "⚠️  La migration de merge existe peut-être déjà"
}

# 2. Appliquer toutes les migrations
echo ""
echo "🔄 Application de toutes les migrations..."
docker_exec flask db upgrade heads

# 3. Vérifier l'état final
echo ""
echo "✅ État final des migrations :"
docker_exec flask db current

echo ""
echo "🎉 Problème résolu !"
echo ""
echo "💡 Conseil : Récupérez la migration de merge et committez-la :"
echo "   docker compose -f docker-compose.production.yml cp backend:/app/migrations/versions/. backend/migrations/versions/"
echo "   git add backend/migrations/versions/"
echo "   git commit -m 'fix: merge overlapping migration heads'"
echo "   git push"
echo ""
