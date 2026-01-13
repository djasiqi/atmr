#!/bin/bash
# Script de restauration de base de données depuis un backup
# Usage: ./restore-from-backup.sh [fichier_backup.sql]

set -euo pipefail

cd /srv/atmr

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔄 Script de restauration de base de données${NC}"
echo ""

# Vérifier les arguments
if [ $# -eq 0 ]; then
    echo "📋 Backups disponibles :"
    ls -lht /srv/atmr/backups/pre-deploy-*.sql 2>/dev/null || {
        echo -e "${RED}❌ Aucun backup trouvé dans /srv/atmr/backups/${NC}"
        exit 1
    }
    echo ""
    echo -e "${YELLOW}Usage: $0 <fichier_backup.sql>${NC}"
    echo "Exemple: $0 /srv/atmr/backups/pre-deploy-20260113-143022.sql"
    exit 1
fi

BACKUP_FILE="$1"

# Vérifier que le fichier existe
if [ ! -f "${BACKUP_FILE}" ]; then
    echo -e "${RED}❌ Erreur: Le fichier ${BACKUP_FILE} n'existe pas${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Fichier de backup trouvé: ${BACKUP_FILE}${NC}"
echo "   Taille: $(du -h "${BACKUP_FILE}" | cut -f1)"
echo ""

# Demander confirmation
echo -e "${RED}⚠️  ATTENTION: Cette opération va ÉCRASER toutes les données actuelles !${NC}"
echo "   La base de données sera complètement remplacée par le backup."
echo ""
read -p "Êtes-vous sûr de vouloir continuer ? (tapez 'oui' pour confirmer) : " -r
echo ""
if [[ ! $REPLY =~ ^oui$ ]]; then
    echo "❌ Restauration annulée"
    exit 1
fi

# Charger les variables d'environnement
if [ -f ".env.production" ]; then
    source .env.production
    echo -e "${GREEN}✅ Variables d'environnement chargées${NC}"
else
    echo -e "${RED}❌ Fichier .env.production non trouvé${NC}"
    exit 1
fi

# Vérifier que PostgreSQL est actif
echo "🔍 Vérification de PostgreSQL..."
if ! docker compose -f docker-compose.production.yml ps postgres --format json 2>/dev/null | grep -q '"State":"running"'; then
    echo -e "${RED}❌ PostgreSQL n'est pas actif${NC}"
    echo "   Démarrage de PostgreSQL..."
    docker compose -f docker-compose.production.yml up -d postgres
    
    echo "⏳ Attente de PostgreSQL (30 secondes max)..."
    for i in $(seq 1 30); do
        if docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER}" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PostgreSQL est prêt${NC}"
            break
        fi
        sleep 1
    done
fi

# Arrêter les services backend pour éviter les conflits
echo "🛑 Arrêt des services backend..."
docker compose -f docker-compose.production.yml stop backend celery-worker celery-beat flower 2>/dev/null || true

# Créer un backup de sécurité avant restauration
SAFETY_BACKUP="/srv/atmr/backups/before-restore-$(date +%Y%m%d-%H%M%S).sql"
echo "💾 Création d'un backup de sécurité avant restauration..."
echo "   ${SAFETY_BACKUP}"
docker compose -f docker-compose.production.yml exec -T postgres pg_dump -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > "${SAFETY_BACKUP}" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Impossible de créer le backup de sécurité (base peut-être vide)${NC}"
}

# Restaurer le backup
echo ""
echo "🔄 Restauration en cours..."
echo "   Cela peut prendre plusieurs minutes selon la taille du backup..."
echo ""

# Méthode 1 : DROP et CREATE (plus propre)
docker compose -f docker-compose.production.yml exec -T postgres psql -U "${POSTGRES_USER}" -d postgres <<EOF
-- Déconnecter tous les utilisateurs
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '${POSTGRES_DB}'
  AND pid <> pg_backend_pid();

-- Supprimer et recréer la base
DROP DATABASE IF EXISTS ${POSTGRES_DB};
CREATE DATABASE ${POSTGRES_DB} OWNER ${POSTGRES_USER};
EOF

# Restaurer les données
cat "${BACKUP_FILE}" | docker compose -f docker-compose.production.yml exec -T postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" 2>&1 | {
    grep -E "^(ERROR|ERREUR)" || true
}

echo ""
echo -e "${GREEN}✅ Restauration terminée${NC}"
echo ""

# Vérifier la restauration
echo "🔍 Vérification de la restauration..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" <<'EOF'
SELECT 
    'Utilisateurs' as table_name, COUNT(*) as count FROM "user"
UNION ALL
SELECT 'Companies', COUNT(*) FROM company
UNION ALL
SELECT 'Clients', COUNT(*) FROM client
UNION ALL
SELECT 'Drivers', COUNT(*) FROM driver
UNION ALL
SELECT 'Bookings', COUNT(*) FROM booking;
EOF

echo ""
echo -e "${GREEN}✅ Vérification terminée${NC}"
echo ""

# Redémarrer les services
echo "🔄 Redémarrage des services..."
docker compose -f docker-compose.production.yml up -d

echo ""
echo -e "${GREEN}✅ Restauration complète !${NC}"
echo ""
echo "📊 Récapitulatif :"
echo "   - Backup restauré : ${BACKUP_FILE}"
echo "   - Backup de sécurité : ${SAFETY_BACKUP}"
echo "   - Services redémarrés"
echo ""
echo "💡 Conseil : Testez l'application pour vérifier que tout fonctionne correctement"
echo ""
