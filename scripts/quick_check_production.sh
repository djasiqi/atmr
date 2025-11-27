#!/bin/bash
# Vérification rapide de production - Commandes à exécuter sur le serveur
# Usage: Copier-coller ces commandes dans un terminal SSH sur le serveur

set -o errexit -o nounset -o pipefail

echo "🔍 Vérification rapide de production ATMR"
echo "=========================================="
echo ""

cd /srv/atmr || { echo "❌ Répertoire /srv/atmr non trouvé"; exit 1; }

# 1. État des conteneurs
echo "1️⃣ État des conteneurs:"
docker compose -f docker-compose.production.yml ps
echo ""

# 2. PostgreSQL
echo "2️⃣ PostgreSQL:"
if docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER:-atmr_user}" -d "${POSTGRES_DB:-atmr_db}" > /dev/null 2>&1; then
    echo "✅ PostgreSQL est accessible"
    docker compose -f docker-compose.production.yml exec -T \
        -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI:-}" \
        -e DATABASE_URL="${DATABASE_URL:-}" \
        -e POSTGRES_USER="${POSTGRES_USER:-atmr_user}" \
        -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}" \
        -e POSTGRES_DB="${POSTGRES_DB:-atmr_db}" \
        -e POSTGRES_HOST="postgres" \
        -e POSTGRES_PORT="5432" \
        backend flask db current 2>/dev/null | head -5 || echo "⚠️  Impossible de vérifier les migrations"
else
    echo "❌ PostgreSQL n'est pas accessible"
fi
echo ""

# 3. Redis
echo "3️⃣ Redis:"
if docker compose -f docker-compose.production.yml exec -T redis redis-cli -a "${REDIS_PASSWORD:-}" ping > /dev/null 2>&1; then
    echo "✅ Redis est accessible"
else
    echo "❌ Redis n'est pas accessible"
fi
echo ""

# 4. Backend API
echo "4️⃣ Backend API:"
if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Backend API répond"
    curl -s http://localhost:5000/health | head -3
else
    echo "❌ Backend API ne répond pas"
    echo "   Logs (dernières 10 lignes):"
    docker compose -f docker-compose.production.yml logs backend --tail 10 | tail -5
fi
echo ""

# 5. Celery Worker
echo "5️⃣ Celery Worker:"
if docker compose -f docker-compose.production.yml exec -T celery-worker celery -A celery_app.celery inspect ping > /dev/null 2>&1; then
    echo "✅ Celery worker répond"
else
    echo "⚠️  Celery worker ne répond pas"
fi
echo ""

# 6. Résumé
echo "=========================================="
echo "📊 Résumé:"
RUNNING=$(docker compose -f docker-compose.production.yml ps --format json | grep -c '"State":"running"' || echo "0")
echo "   Conteneurs running: $RUNNING"
echo ""

if [ "$RUNNING" -ge 5 ]; then
    echo "✅ Production semble opérationnelle"
else
    echo "⚠️  Certains services ne sont pas running"
fi

