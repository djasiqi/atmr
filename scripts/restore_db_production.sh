#!/bin/bash
# Script de restauration PostgreSQL pour ATMR (Production)
# Usage: ./scripts/restore_db_production.sh <backup_file> [--force]

set -euo pipefail

cd /srv/atmr

BACKUP_FILE="${1:-}"
FORCE="${2:-}"

# Variables d'environnement (depuis .env.production)
if [ -f .env.production ]; then
    source .env.production
fi

POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-atmr}"
POSTGRES_USER="${POSTGRES_USER:-atmr}"

# Vérifier argument
if [ -z "$BACKUP_FILE" ]; then
    echo "❌ Usage: $0 <backup_file> [--force]"
    echo ""
    echo "Exemples:"
    echo "  $0 backups/local_backup_20260113.sql"
    echo "  $0 backups/local_backup_20260113.dump"
    exit 1
fi

# Vérifier que le fichier existe
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Erreur: Fichier de backup non trouvé: $BACKUP_FILE"
    exit 1
fi

# Détecter le format
BACKUP_FORMAT="sql"
if [[ "$BACKUP_FILE" == *.dump ]]; then
    BACKUP_FORMAT="custom"
fi

echo "🔄 Restauration base de données PostgreSQL..."
echo "   Backup: $BACKUP_FILE"
echo "   Format: $BACKUP_FORMAT"
echo "   Database: $POSTGRES_DB"
echo ""

# Confirmation (sauf si --force)
if [ "$FORCE" != "--force" ]; then
    echo "⚠️  ATTENTION: Cette opération va écraser la base de données actuelle!"
    echo "   Toutes les données non sauvegardées seront perdues."
    echo ""
    read -p "Continuer? (tapez 'yes' pour confirmer): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "❌ Opération annulée."
        exit 0
    fi
fi

export PGPASSWORD="$POSTGRES_PASSWORD"

# Copier le backup dans le container
BACKUP_BASENAME=$(basename "$BACKUP_FILE")
echo "📦 Copie du backup dans le conteneur..."
docker cp "$BACKUP_FILE" atmr-postgres:/tmp/$BACKUP_BASENAME

if [ "$BACKUP_FORMAT" = "custom" ]; then
    echo "🔧 Restauration format custom (.dump)..."
    # Restauration format custom avec pg_restore
    docker compose -f docker-compose.production.yml exec -T postgres pg_restore \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --clean \
        --if-exists \
        --no-owner \
        --no-acl \
        --verbose \
        "/tmp/$BACKUP_BASENAME" 2>&1 | grep -v "WARNING\|NOTICE" || true
else
    echo "🔧 Restauration format SQL (.sql)..."
    # Restauration format SQL avec psql
    docker compose -f docker-compose.production.yml exec -T postgres psql \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -f "/tmp/$BACKUP_BASENAME" 2>&1 | grep -v "WARNING\|NOTICE" || true
fi

# Nettoyer
docker compose -f docker-compose.production.yml exec -T postgres rm -f "/tmp/$BACKUP_BASENAME"
unset PGPASSWORD

echo ""
echo "✅ Restauration terminée avec succès!"

# Vérifications post-restauration
echo ""
echo "🔍 Vérifications post-restauration..."

export PGPASSWORD="$POSTGRES_PASSWORD"
USER_COUNT=$(docker compose -f docker-compose.production.yml exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM public.user;" | tr -d ' ')
TABLE_COUNT=$(docker compose -f docker-compose.production.yml exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d ' ')
unset PGPASSWORD

echo "   📊 Tables trouvées: $TABLE_COUNT"
echo "   👥 Utilisateurs: $USER_COUNT"

if [ "$TABLE_COUNT" -gt 0 ] && [ "$USER_COUNT" -gt 0 ]; then
    echo "   ✅ Base de données restaurée correctement"
else
    echo "   ⚠️  Vérifier le contenu de la base"
fi

echo ""
echo "🔄 Redémarrage du backend pour recharger le schéma..."
docker compose -f docker-compose.production.yml restart backend

echo ""
echo "💡 Prochaines étapes:"
echo "   1. Vérifier santé API: curl http://localhost:5000/health"
echo "   2. Tester le login depuis le frontend"
echo ""
