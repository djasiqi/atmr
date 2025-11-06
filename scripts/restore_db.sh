#!/bin/bash
# Script de restauration PostgreSQL pour ATMR
# Usage: ./scripts/restore_db.sh <backup_file> [--force]

set -euo pipefail

BACKUP_FILE="${1:-}"
FORCE="${2:-}"

# Variables d'environnement par défaut
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-atmr}"
POSTGRES_USER="${POSTGRES_USER:-atmr}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-atmr}"

# Vérifier argument
if [ -z "$BACKUP_FILE" ]; then
    echo "❌ Usage: $0 <backup_file> [--force]"
    echo ""
    echo "Exemples:"
    echo "  $0 backups/atmr_backup_20250127_120000.dump"
    echo "  $0 backups/atmr_backup_20250127_120000.sql"
    echo "  $0 backups/latest.dump --force"
    exit 1
fi

# Vérifier que le fichier existe
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Erreur: Fichier de backup non trouvé: $BACKUP_FILE"
    exit 1
fi

# Détecter le format
BACKUP_FORMAT="unknown"
if [[ "$BACKUP_FILE" == *.dump ]] || file "$BACKUP_FILE" | grep -q "PostgreSQL custom database dump"; then
    BACKUP_FORMAT="custom"
    RESTORE_CMD="pg_restore"
elif [[ "$BACKUP_FILE" == *.sql ]] || file "$BACKUP_FILE" | grep -q "ASCII text\|UTF-8 Unicode"; then
    BACKUP_FORMAT="sql"
    RESTORE_CMD="psql"
else
    echo "⚠️  Format de backup non reconnu, tentative de détection automatique..."
    if head -1 "$BACKUP_FILE" | grep -q "PGDMP\|PostgreSQL database dump"; then
        BACKUP_FORMAT="custom"
        RESTORE_CMD="pg_restore"
    else
        BACKUP_FORMAT="sql"
        RESTORE_CMD="psql"
    fi
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

# Vérifier si on est dans docker-compose ou local
if command -v docker-compose &> /dev/null && docker-compose ps postgres &> /dev/null; then
    echo "   Mode: Docker Compose"
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Copier le backup dans le container
    BACKUP_BASENAME=$(basename "$BACKUP_FILE")
    docker-compose cp "$BACKUP_FILE" "postgres:/tmp/$BACKUP_BASENAME"
    
    if [ "$BACKUP_FORMAT" = "custom" ]; then
        # Restauration format custom
        docker-compose exec -T postgres pg_restore \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            --clean \
            --if-exists \
            --verbose \
            "/tmp/$BACKUP_BASENAME"
    else
        # Restauration format SQL
        docker-compose exec -T postgres psql \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            -f "/tmp/$BACKUP_BASENAME"
    fi
    
    # Nettoyer
    docker-compose exec -T postgres rm -f "/tmp/$BACKUP_BASENAME"
    unset PGPASSWORD
    
elif command -v pg_restore &> /dev/null || command -v psql &> /dev/null; then
    echo "   Mode: Local"
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    if [ "$BACKUP_FORMAT" = "custom" ]; then
        pg_restore \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            --clean \
            --if-exists \
            --verbose \
            "$BACKUP_FILE"
    else
        psql \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            -f "$BACKUP_FILE"
    fi
    
    unset PGPASSWORD
else
    echo "❌ Erreur: pg_restore/psql non trouvé et docker-compose non disponible"
    exit 1
fi

echo ""
echo "✅ Restauration terminée avec succès!"

# Vérifications post-restauration
echo ""
echo "🔍 Vérifications post-restauration..."

if command -v docker-compose &> /dev/null && docker-compose ps postgres &> /dev/null; then
    export PGPASSWORD="$POSTGRES_PASSWORD"
    TABLE_COUNT=$(docker-compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d ' ')
    unset PGPASSWORD
else
    export PGPASSWORD="$POSTGRES_PASSWORD"
    TABLE_COUNT=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d ' ')
    unset PGPASSWORD
fi

echo "   📊 Tables trouvées: $TABLE_COUNT"

if [ "$TABLE_COUNT" -gt 0 ]; then
    echo "   ✅ Base de données semble restaurée correctement"
else
    echo "   ⚠️  Aucune table trouvée - vérifier le backup"
fi

echo ""
echo "💡 Prochaines étapes:"
echo "   1. Vérifier santé API: curl http://localhost:5000/health"
echo "   2. Vérifier ready: curl http://localhost:5000/ready"
echo "   3. Tester une requête: curl http://localhost:5000/api/bookings"

