#!/bin/bash
set -euo pipefail

# Script pour synchroniser automatiquement le schéma DB avec les modèles SQLAlchemy
# en générant une migration automatique qui détecte les différences

COMPOSE_FILE="${1:-docker-compose.production.yml}"
MIGRATION_NAME="${2:-auto_sync_schema}"

echo "🔄 Synchronisation automatique du schéma de la base de données..."
echo ""

# 1. Vérifier la version actuelle
echo "📋 Version actuelle :"
docker compose -f "$COMPOSE_FILE" exec backend flask db current
echo ""

# 2. Générer une migration automatique pour détecter les différences
echo "🔍 Génération d'une migration automatique..."
docker compose -f "$COMPOSE_FILE" exec backend flask db revision --autogenerate -m "$MIGRATION_NAME"
echo ""

# 3. Appliquer la nouvelle migration
echo "⬆️  Application de la migration..."
docker compose -f "$COMPOSE_FILE" exec backend flask db upgrade heads
echo ""

# 4. Vérifier la nouvelle version
echo "✅ Migration appliquée"
echo "📋 Nouvelle version :"
docker compose -f "$COMPOSE_FILE" exec backend flask db current
echo ""

# 5. Redémarrer le backend
echo "🔄 Redémarrage du backend..."
docker compose -f "$COMPOSE_FILE" restart backend
echo ""

# 6. Attendre que le backend soit prêt
echo "⏳ Attente du backend..."
sleep 10
HEALTH=$(docker compose -f "$COMPOSE_FILE" exec backend curl -s http://localhost:5000/health 2>/dev/null || echo "{}")
echo "Healthcheck: $HEALTH"
echo ""

echo "✅ Terminé !"
echo ""
echo "💡 Astuce: Vérifiez le fichier de migration généré dans backend/migrations/versions/"
echo "   Si des modifications incorrectes ont été détectées, supprimez la migration et"
echo "   corrigez manuellement le schéma avant de relancer ce script."
