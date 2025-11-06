#!/bin/bash
# Script de backup PostgreSQL pour ATMR
# Usage: ./scripts/backup_db.sh [backup_dir]

set -euo pipefail

# Configuration
BACKUP_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/atmr_backup_$TIMESTAMP.sql"
BACKUP_FILE_CUSTOM="$BACKUP_DIR/atmr_backup_$TIMESTAMP.dump"

# Variables d'environnement par défaut (docker-compose)
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-atmr}"
POSTGRES_USER="${POSTGRES_USER:-atmr}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-atmr}"

# Créer le répertoire de backup
mkdir -p "$BACKUP_DIR"

echo "🔄 Backup base de données PostgreSQL..."
echo "   Database: $POSTGRES_DB"
echo "   Host: $POSTGRES_HOST:$POSTGRES_PORT"

# Vérifier si on est dans docker-compose ou local
if command -v docker-compose &> /dev/null && docker-compose ps postgres &> /dev/null; then
    echo "   Mode: Docker Compose"
    
    # Export PGPASSWORD pour éviter prompt
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Backup avec pg_dump (format custom pour restauration plus rapide)
    docker-compose exec -T postgres pg_dump \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -F c \
        -f "/tmp/atmr_backup_$TIMESTAMP.dump"
    
    # Copier depuis le container
    docker-compose cp postgres:/tmp/atmr_backup_$TIMESTAMP.dump "$BACKUP_FILE_CUSTOM"
    
    # Backup SQL également (format texte, plus lisible)
    docker-compose exec -T postgres pg_dump \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -F p \
        -f "/tmp/atmr_backup_$TIMESTAMP.sql"
    
    docker-compose cp postgres:/tmp/atmr_backup_$TIMESTAMP.sql "$BACKUP_FILE"
    
    # Nettoyer dans le container
    docker-compose exec -T postgres rm -f "/tmp/atmr_backup_$TIMESTAMP.dump" "/tmp/atmr_backup_$TIMESTAMP.sql"
    
    unset PGPASSWORD
    
elif command -v pg_dump &> /dev/null; then
    echo "   Mode: Local"
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Backup format custom (recommandé)
    pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -F c \
        -f "$BACKUP_FILE_CUSTOM"
    
    # Backup format SQL (texte)
    pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -F p \
        -f "$BACKUP_FILE"
    
    unset PGPASSWORD
else
    echo "❌ Erreur: pg_dump non trouvé et docker-compose non disponible"
    exit 1
fi

# Vérifier que le backup existe
if [ ! -f "$BACKUP_FILE_CUSTOM" ] && [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Erreur: Backup non créé"
    exit 1
fi

# Afficher informations
echo ""
echo "✅ Backup créé avec succès!"
if [ -f "$BACKUP_FILE_CUSTOM" ]; then
    SIZE_CUSTOM=$(du -h "$BACKUP_FILE_CUSTOM" | cut -f1)
    echo "   📦 Format custom: $BACKUP_FILE_CUSTOM ($SIZE_CUSTOM)"
fi
if [ -f "$BACKUP_FILE" ]; then
    SIZE_SQL=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "   📄 Format SQL: $BACKUP_FILE ($SIZE_SQL)"
fi

# Créer un lien symbolique vers le dernier backup
if [ -f "$BACKUP_FILE_CUSTOM" ]; then
    ln -sf "$(basename "$BACKUP_FILE_CUSTOM")" "$BACKUP_DIR/latest.dump" 2>/dev/null || true
fi
if [ -f "$BACKUP_FILE" ]; then
    ln -sf "$(basename "$BACKUP_FILE")" "$BACKUP_DIR/latest.sql" 2>/dev/null || true
fi

echo "   🔗 Liens: $BACKUP_DIR/latest.dump, $BACKUP_DIR/latest.sql"

