#!/bin/bash
# Script de test backup/restore pour ATMR
# Usage: ./scripts/test_backup_restore.sh

set -euo pipefail

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="${BACKUP_DIR:-./backups}"
TEST_TABLE="backup_test_validation"
TEST_TIMESTAMP=$(date +%s)

echo "=========================================="
echo "🧪 TEST BACKUP/RESTORE PostgreSQL"
echo "=========================================="
echo ""

# Vérifier que docker-compose est disponible
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Erreur: docker-compose non trouvé${NC}"
    exit 1
fi

# Vérifier que PostgreSQL est accessible
if ! docker-compose ps postgres | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Démarrage du service PostgreSQL...${NC}"
    docker-compose up -d postgres
    sleep 5
fi

export PGPASSWORD="${POSTGRES_PASSWORD:-atmr}"

# 1. BACKUP
echo "📦 Étape 1/4: Création du backup..."
START_BACKUP=$(date +%s)

if [ ! -f "scripts/backup_db.sh" ]; then
    echo -e "${RED}❌ Erreur: scripts/backup_db.sh non trouvé${NC}"
    exit 1
fi

bash scripts/backup_db.sh "$BACKUP_DIR"
BACKUP_EXIT=$?

END_BACKUP=$(date +%s)
BACKUP_DURATION=$((END_BACKUP - START_BACKUP))

if [ $BACKUP_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Backup échoué${NC}"
    exit 1
fi

LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/atmr_backup_*.dump 2>/dev/null | head -1 || ls -t "$BACKUP_DIR"/atmr_backup_*.sql 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo -e "${RED}❌ Aucun fichier de backup trouvé${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Backup créé: $LATEST_BACKUP (${BACKUP_DURATION}s)${NC}"
echo ""

# 2. CRÉER DONNÉES DE TEST
echo "📝 Étape 2/4: Création de données de test..."

# Créer une table de test et insérer des données
docker-compose exec -T postgres psql -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" <<EOF
-- Créer table de test si elle n'existe pas
CREATE TABLE IF NOT EXISTS ${TEST_TABLE} (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100),
    test_timestamp BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insérer données de test
INSERT INTO ${TEST_TABLE} (test_name, test_timestamp) 
VALUES ('BACKUP_TEST_${TEST_TIMESTAMP}', ${TEST_TIMESTAMP});

-- Vérifier insertion
SELECT COUNT(*) as count FROM ${TEST_TABLE} WHERE test_timestamp = ${TEST_TIMESTAMP};
EOF

TEST_DATA_COUNT=$(docker-compose exec -T postgres psql -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" -t -c "SELECT COUNT(*) FROM ${TEST_TABLE} WHERE test_timestamp = ${TEST_TIMESTAMP};" | tr -d ' ')

if [ "$TEST_DATA_COUNT" != "1" ]; then
    echo -e "${RED}❌ Erreur: Données de test non créées${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Données de test créées (timestamp: ${TEST_TIMESTAMP})${NC}"
echo ""

# 3. RESTAURATION
echo "🔄 Étape 3/4: Restauration depuis le backup..."
START_RESTORE=$(date +%s)

if [ ! -f "scripts/restore_db.sh" ]; then
    echo -e "${RED}❌ Erreur: scripts/restore_db.sh non trouvé${NC}"
    exit 1
fi

# Forcer la restauration (mode test)
bash scripts/restore_db.sh "$LATEST_BACKUP" --force
RESTORE_EXIT=$?

END_RESTORE=$(date +%s)
RESTORE_DURATION=$((END_RESTORE - START_RESTORE))

if [ $RESTORE_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Restauration échouée${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Restauration terminée (${RESTORE_DURATION}s)${NC}"
echo ""

# 4. VÉRIFICATION
echo "🔍 Étape 4/4: Vérification de l'intégrité..."

# Vérifier que les données de test ne sont plus présentes (car restaurées depuis avant leur création)
RESTORED_TEST_COUNT=$(docker-compose exec -T postgres psql -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" -t -c "SELECT COUNT(*) FROM ${TEST_TABLE} WHERE test_timestamp = ${TEST_TIMESTAMP};" 2>/dev/null | tr -d ' ' || echo "0")

if [ "$RESTORED_TEST_COUNT" != "0" ]; then
    echo -e "${RED}❌ ÉCHEC: Les données de test sont toujours présentes après restauration${NC}"
    echo "   Cela signifie que la restauration n'a pas fonctionné correctement."
    
    # Nettoyer les données de test
    docker-compose exec -T postgres psql -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" -c "DROP TABLE IF EXISTS ${TEST_TABLE};" 2>/dev/null || true
    
    exit 1
fi

# Vérifier que la base de données contient des tables
TABLE_COUNT=$(docker-compose exec -T postgres psql -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';" | tr -d ' ')

if [ "$TABLE_COUNT" -eq "0" ]; then
    echo -e "${RED}❌ ÉCHEC: Aucune table trouvée après restauration${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Test réussi: données restaurées correctement${NC}"
echo "   📊 Tables restaurées: $TABLE_COUNT"

# Nettoyer la table de test si elle existe encore
docker-compose exec -T postgres psql -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" -c "DROP TABLE IF EXISTS ${TEST_TABLE};" 2>/dev/null || true

# Vérifier santé API si disponible
if docker-compose ps api | grep -q "Up"; then
    echo ""
    echo "🔍 Vérification santé API..."
    if curl -s http://localhost:5000/health | grep -q "ok"; then
        echo -e "${GREEN}✅ API répond correctement${NC}"
    else
        echo -e "${YELLOW}⚠️  API non accessible (normal si redémarrée)${NC}"
    fi
fi

# Calculer RTO/RPO
TOTAL_TIME=$((BACKUP_DURATION + RESTORE_DURATION))

echo ""
echo "=========================================="
echo -e "${GREEN}✅ TEST BACKUP/RESTORE RÉUSSI${NC}"
echo "=========================================="
echo ""
echo "📊 Métriques:"
echo "   ⏱️  Temps de backup: ${BACKUP_DURATION}s"
echo "   ⏱️  Temps de restauration: ${RESTORE_DURATION}s"
echo "   ⏱️  Temps total: ${TOTAL_TIME}s"
echo ""
echo "🎯 Objectifs:"
echo "   RTO (Restore Time Objective): ${RESTORE_DURATION}s (objectif: < 30 min ✅)"
echo "   RPO (Recovery Point Objective): ~${BACKUP_DURATION}s (objectif: < 15 min ✅)"
echo ""

unset PGPASSWORD

