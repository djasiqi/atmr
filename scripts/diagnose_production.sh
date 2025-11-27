#!/bin/bash
# Script de diagnostic pour le serveur de production
# À exécuter sur le serveur: bash scripts/diagnose_production.sh

set -o errexit -o nounset -o pipefail

echo "🔍 Diagnostic de production ATMR"
echo "=================================="
echo ""

cd /srv/atmr || { echo "❌ Répertoire /srv/atmr non trouvé"; exit 1; }

# 1. Vérifier les logs du celery-worker qui crash
echo "1️⃣ Logs du celery-worker (dernières 50 lignes):"
echo "-----------------------------------------------"
docker logs atmr-celery-worker --tail 50 2>&1 || true
echo ""

# 2. Vérifier les logs du backend
echo "2️⃣ Logs du backend (dernières 30 lignes):"
echo "------------------------------------------"
docker logs atmr-backend --tail 30 2>&1 || true
echo ""

# 3. Vérifier l'état des conteneurs ATMR
echo "3️⃣ État des conteneurs ATMR:"
echo "-----------------------------"
docker ps --filter "name=atmr-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# 4. Vérifier les conteneurs en erreur
echo "4️⃣ Conteneurs en erreur ou restarting:"
echo "----------------------------------------"
docker ps -a --filter "name=atmr-" --format "table {{.Names}}\t{{.Status}}" | grep -E "Restarting|Exited|Dead" || echo "Aucun conteneur en erreur"
echo ""

# 5. Vérifier PostgreSQL
echo "5️⃣ PostgreSQL:"
echo "--------------"
if docker exec atmr-postgres pg_isready -U "${POSTGRES_USER:-atmr_user}" -d "${POSTGRES_DB:-atmr_db}" > /dev/null 2>&1; then
    echo "✅ PostgreSQL est accessible"
else
    echo "❌ PostgreSQL n'est pas accessible"
fi
echo ""

# 6. Vérifier Redis
echo "6️⃣ Redis:"
echo "---------"
if docker exec atmr-redis redis-cli -a "${REDIS_PASSWORD:-}" ping > /dev/null 2>&1; then
    echo "✅ Redis est accessible"
else
    echo "❌ Redis n'est pas accessible"
fi
echo ""

# 7. Vérifier les variables d'environnement du celery-worker
echo "7️⃣ Variables d'environnement critiques (celery-worker):"
echo "----------------------------------------------------------"
docker exec atmr-celery-worker env | grep -E "POSTGRES_|REDIS_|CELERY_|DATABASE_URL|SQLALCHEMY" | sed 's/=.*/=***/' || echo "Impossible de lire les variables (conteneur peut-être en cours de redémarrage)"
echo ""

# 8. Vérifier les conflits de ports/réseaux
echo "8️⃣ Conflits potentiels (anciens conteneurs):"
echo "---------------------------------------------"
echo "Anciens conteneurs backend-* encore actifs:"
docker ps --filter "name=backend-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "Aucun ancien conteneur trouvé"
echo ""

# 9. Vérifier l'espace disque
echo "9️⃣ Espace disque:"
echo "-----------------"
df -h / | tail -1
echo ""

# 10. Recommandations
echo "=================================="
echo "💡 Recommandations:"
echo "=================================="
echo ""
echo "Si celery-worker crash:"
echo "  1. Vérifier les logs ci-dessus"
echo "  2. Vérifier que REDIS_PASSWORD est correct"
echo "  3. Vérifier que DATABASE_URL est correctement formaté"
echo ""
echo "Si anciens conteneurs sont actifs:"
echo "  1. Arrêter les anciens: docker stop backend-*"
echo "  2. Ou utiliser un autre docker-compose.yml"
echo ""

