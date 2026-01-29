#!/usr/bin/env bash
# Vérification & exécution migration Alembic (idempotent, safe)
# Usage: ./scripts/verify_and_migrate.sh [container_name]
# Depuis la racine du projet (atmr/)
# Container par défaut: atmr-api-1 (ou premier argument)

set -e

CONTAINER="${1:-atmr-api-1}"

echo "=== 1. Vérification état des migrations (container: $CONTAINER) ==="
CURRENT=$(docker exec "$CONTAINER" flask db current 2>/dev/null | grep -E '^[a-f0-9]+' || true)
HEADS=$(docker exec "$CONTAINER" flask db heads 2>/dev/null | grep -E '^[a-f0-9]+' || true)

# Extraire uniquement le hash (ex: f2b0c6600828)
CURRENT_REV=$(echo "$CURRENT" | awk '{print $1}')
HEADS_REV=$(echo "$HEADS" | awk '{print $1}')

echo "  current: $CURRENT_REV"
echo "  heads:   $HEADS_REV"

if [ -z "$CURRENT_REV" ] || [ -z "$HEADS_REV" ]; then
  echo "❌ Impossible de lire l'état des migrations"
  exit 1
fi

if [ "$CURRENT_REV" = "$HEADS_REV" ]; then
  echo ""
  echo "✅ Migration déjà appliquée — aucune action requise"
  exit 0
fi

echo ""
echo "=== 2. Migration manquante → application ==="
docker exec "$CONTAINER" flask db upgrade head

echo ""
echo "=== 3. Vérification post-migration ==="
docker exec "$CONTAINER" flask db current

echo ""
echo "✅ Migration appliquée avec succès"
