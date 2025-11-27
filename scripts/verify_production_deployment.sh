#!/bin/bash
# Script de vérification du déploiement en production
# À exécuter sur le serveur de production après un déploiement

set -euo pipefail

echo "🔍 Vérification du déploiement en production..."
echo "=========================================="
echo ""

# 1. Vérifier l'état des conteneurs
echo "📦 1. État des conteneurs Docker..."
cd /srv/atmr || exit 1
docker compose -f docker-compose.production.yml ps
echo ""

# 2. Vérifier les healthchecks
echo "🏥 2. Healthchecks des services..."
echo "PostgreSQL:"
docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER:-atmr_user}" -d "${POSTGRES_DB:-atmr_db}" || echo "❌ PostgreSQL non prêt"
echo ""

echo "Redis:"
docker compose -f docker-compose.production.yml exec -T redis redis-cli -a "${REDIS_PASSWORD}" ping || echo "❌ Redis non accessible"
echo ""

echo "Backend:"
curl -f http://localhost:5000/health || echo "❌ Backend non accessible"
echo ""

# 3. Vérifier les migrations
echo "🗄️  3. État des migrations de base de données..."
docker compose -f docker-compose.production.yml exec -T \
  -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI}" \
  -e DATABASE_URL="${DATABASE_URL}" \
  backend flask db current || echo "❌ Impossible de vérifier les migrations"
echo ""

# 4. Vérifier les logs récents (dernières 20 lignes)
echo "📋 4. Logs récents du backend (dernières 20 lignes)..."
docker compose -f docker-compose.production.yml logs --tail=20 backend
echo ""

# 5. Vérifier les logs des workers Celery
echo "⚙️  5. Logs récents des workers Celery (dernières 10 lignes)..."
docker compose -f docker-compose.production.yml logs --tail=10 celery-worker
echo ""

# 6. Vérifier les volumes et permissions
echo "💾 6. Vérification des volumes Docker..."
docker volume ls | grep atmr
echo ""

# 7. Vérifier l'espace disque
echo "💿 7. Espace disque disponible..."
df -h / | tail -1
echo ""

# 8. Vérifier les répertoires de données
echo "📁 8. Vérification des répertoires de données..."
if [ -d "/var/lib/docker/volumes" ]; then
    echo "Volumes Docker trouvés"
    docker volume inspect atmr_backend_data 2>/dev/null | grep -i mountpoint || echo "⚠️  Volume backend_data non trouvé"
    docker volume inspect atmr_backend_logs 2>/dev/null | grep -i mountpoint || echo "⚠️  Volume backend_logs non trouvé"
    docker volume inspect atmr_backend_uploads 2>/dev/null | grep -i mountpoint || echo "⚠️  Volume backend_uploads non trouvé"
else
    echo "⚠️  Répertoire des volumes Docker non trouvé"
fi
echo ""

# 9. Vérifier les variables d'environnement critiques
echo "🔐 9. Vérification des variables d'environnement..."
if [ -f ".env.production" ]; then
    echo "✅ Fichier .env.production trouvé"
    # Vérifier que les variables critiques sont présentes (sans afficher les valeurs)
    MISSING_VARS=()
    source .env.production
    if [ -z "${DATABASE_URL:-}" ]; then MISSING_VARS+=("DATABASE_URL"); fi
    if [ -z "${REDIS_URL:-}" ]; then MISSING_VARS+=("REDIS_URL"); fi
    if [ -z "${SECRET_KEY:-}" ]; then MISSING_VARS+=("SECRET_KEY"); fi
    if [ -z "${JWT_SECRET_KEY:-}" ]; then MISSING_VARS+=("JWT_SECRET_KEY"); fi
    if [ -z "${APP_ENCRYPTION_KEY_B64:-}" ]; then MISSING_VARS+=("APP_ENCRYPTION_KEY_B64"); fi
    
    if [ ${#MISSING_VARS[@]} -eq 0 ]; then
        echo "✅ Toutes les variables critiques sont présentes"
    else
        echo "❌ Variables manquantes: ${MISSING_VARS[*]}"
    fi
else
    echo "❌ Fichier .env.production non trouvé"
fi
echo ""

# 10. Vérifier la connectivité réseau
echo "🌐 10. Vérification de la connectivité réseau..."
echo "PostgreSQL (depuis le backend):"
docker compose -f docker-compose.production.yml exec -T backend python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL').replace('postgresql+psycopg2://', 'postgresql://'))
    print('✅ Connexion PostgreSQL réussie')
    conn.close()
except Exception as e:
    print(f'❌ Erreur de connexion PostgreSQL: {e}')
" || echo "❌ Impossible de tester la connexion PostgreSQL"
echo ""

echo "Redis (depuis le backend):"
docker compose -f docker-compose.production.yml exec -T backend python -c "
import os
import redis
try:
    r = redis.from_url(os.getenv('REDIS_URL'))
    r.ping()
    print('✅ Connexion Redis réussie')
except Exception as e:
    print(f'❌ Erreur de connexion Redis: {e}')
" || echo "❌ Impossible de tester la connexion Redis"
echo ""

# 11. Vérifier les endpoints API
echo "🔌 11. Vérification des endpoints API..."
echo "Health endpoint:"
curl -s http://localhost:5000/health | jq '.' || curl -s http://localhost:5000/health
echo ""
echo ""

# 12. Résumé
echo "=========================================="
echo "✅ Vérification terminée"
echo ""
echo "📊 Résumé:"
echo "  - Conteneurs: $(docker compose -f docker-compose.production.yml ps --format json | jq -r '.[] | select(.State == "running") | .Name' | wc -l) en cours d'exécution"
echo "  - Backend health: $(curl -s http://localhost:5000/health | jq -r '.status // "unknown"' || echo "unknown")"
echo ""

