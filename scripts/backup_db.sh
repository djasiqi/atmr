#!/bin/bash
# =============================================================================
# Script de backup automatique PostgreSQL pour ATMR
# Usage: ./scripts/backup_db.sh
# Cron recommandé: 0 2 * * * /path/to/atmr/scripts/backup_db.sh
# =============================================================================

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="atmr_${TIMESTAMP}.dump"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}📦 [Backup] Démarrage du backup PostgreSQL...${NC}"

# Créer le dossier backups s'il n'existe pas
mkdir -p "${BACKUP_DIR}"

# Vérifier que postgres est accessible
echo -e "${YELLOW}🔍 [Backup] Vérification de la connexion PostgreSQL...${NC}"
if ! docker compose exec -T postgres pg_isready -U atmr -d atmr > /dev/null 2>&1; then
    echo -e "${RED}❌ [Backup] PostgreSQL n'est pas accessible${NC}"
    exit 1
fi

# Créer le backup dans le container
echo -e "${YELLOW}💾 [Backup] Création du dump...${NC}"
docker compose exec -T postgres pg_dump -U atmr -d atmr \
    --format=custom \
    --verbose \
    -f "/var/lib/postgresql/data/${BACKUP_NAME}"

# Copier le backup hors du container
echo -e "${YELLOW}📋 [Backup] Copie vers ${BACKUP_DIR}/${BACKUP_NAME}...${NC}"
docker cp "$(docker compose ps -q postgres):/var/lib/postgresql/data/${BACKUP_NAME}" "${BACKUP_DIR}/${BACKUP_NAME}"

# Supprimer le backup du container
docker compose exec -T postgres rm -f "/var/lib/postgresql/data/${BACKUP_NAME}"

# Vérifier que le fichier existe et a une taille > 0
if [ ! -s "${BACKUP_DIR}/${BACKUP_NAME}" ]; then
    echo -e "${RED}❌ [Backup] Le fichier backup est vide ou n'existe pas${NC}"
    exit 1
fi

# Afficher la taille
BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}" | cut -f1)
echo -e "${GREEN}✅ [Backup] Backup créé: ${BACKUP_DIR}/${BACKUP_NAME} (${BACKUP_SIZE})${NC}"

# Rotation : supprimer les backups de plus de RETENTION_DAYS jours
echo -e "${YELLOW}🔄 [Backup] Rotation des anciens backups (> ${RETENTION_DAYS} jours)...${NC}"
find "${BACKUP_DIR}" -name "atmr_*.dump" -type f -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true

# Lister les backups restants
echo -e "${GREEN}📋 [Backup] Backups disponibles:${NC}"
ls -lh "${BACKUP_DIR}"/atmr_*.dump 2>/dev/null || echo "  (aucun)"

echo -e "${GREEN}✅ [Backup] Terminé avec succès${NC}"
