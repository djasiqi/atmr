#!/bin/bash
# =============================================================================
# Script de restauration PostgreSQL pour ATMR
# Usage: ./scripts/restore_db.sh <backup_file.dump>
# =============================================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}❌ Usage: $0 <backup_file.dump>${NC}"
    echo ""
    echo "Backups disponibles:"
    ls -lh ./backups/atmr_*.dump 2>/dev/null || echo "  (aucun dans ./backups/)"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "${BACKUP_FILE}" ]; then
    echo -e "${RED}❌ Fichier non trouvé: ${BACKUP_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}⚠️  ATTENTION: Cette opération va REMPLACER toutes les données de la base atmr${NC}"
echo -e "${YELLOW}    Fichier: ${BACKUP_FILE}${NC}"
read -p "Confirmer (y/N)? " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Annulé."
    exit 1
fi

# Vérifier que postgres est accessible
echo -e "${YELLOW}🔍 [Restore] Vérification de la connexion PostgreSQL...${NC}"
if ! docker compose exec -T postgres pg_isready -U atmr -d atmr > /dev/null 2>&1; then
    echo -e "${RED}❌ [Restore] PostgreSQL n'est pas accessible${NC}"
    exit 1
fi

# Copier le backup dans le container
BACKUP_NAME=$(basename "${BACKUP_FILE}")
echo -e "${YELLOW}📋 [Restore] Copie du backup vers le container...${NC}"
docker cp "${BACKUP_FILE}" "$(docker compose ps -q postgres):/tmp/${BACKUP_NAME}"

# Restaurer
echo -e "${YELLOW}💾 [Restore] Restauration en cours...${NC}"
docker compose exec -T postgres pg_restore -U atmr -d atmr \
    --clean \
    --if-exists \
    --verbose \
    "/tmp/${BACKUP_NAME}" || true  # pg_restore peut retourner des erreurs non-fatales

# Nettoyer
docker compose exec -T postgres rm -f "/tmp/${BACKUP_NAME}"

# Vérifier
echo -e "${YELLOW}🔍 [Restore] Vérification...${NC}"
USER_COUNT=$(echo 'SELECT COUNT(*) FROM "user";' | docker compose exec -T postgres psql -U atmr -d atmr -t | tr -d ' ')
echo -e "${GREEN}✅ [Restore] Restauration terminée. ${USER_COUNT} utilisateurs dans la base.${NC}"

echo -e "${YELLOW}⚠️  N'oubliez pas de redémarrer l'API: docker compose restart api${NC}"
