#!/bin/bash
# Script de vérification de l'état de production
# Usage: ./scripts/verify_production.sh

set -o errexit -o nounset -o pipefail

echo "🔍 Vérification de l'état de production ATMR"
echo "=============================================="
echo ""

# Couleurs pour la sortie
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher un statut
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    else
        echo -e "${RED}❌ $message${NC}"
    fi
}

# 1. Vérifier que nous sommes dans le bon répertoire
echo "📁 Vérification du répertoire de travail..."
if [ -f "docker-compose.production.yml" ]; then
    print_status "OK" "docker-compose.production.yml trouvé"
    WORK_DIR=$(pwd)
    echo "   Répertoire: $WORK_DIR"
else
    print_status "ERROR" "docker-compose.production.yml non trouvé"
    echo "   Veuillez exécuter ce script depuis /srv/atmr"
    exit 1
fi
echo ""

# 2. Vérifier l'état des conteneurs Docker
echo "🐳 État des conteneurs Docker..."
docker compose -f docker-compose.production.yml ps
echo ""

# Vérifier chaque service individuellement
SERVICES=("postgres" "redis" "backend" "celery-worker" "celery-beat" "flower")
for service in "${SERVICES[@]}"; do
    echo "📊 Vérification du service: $service"
    STATUS=$(docker compose -f docker-compose.production.yml ps "$service" --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$STATUS" = "running" ]; then
        HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "atmr-$service" 2>/dev/null || echo "none")
        if [ "$HEALTH" = "healthy" ] || [ "$HEALTH" = "none" ]; then
            print_status "OK" "$service est running (health: $HEALTH)"
        else
            print_status "WARN" "$service est running mais unhealthy (health: $HEALTH)"
        fi
    elif [ "$STATUS" = "exited" ] || [ "$STATUS" = "dead" ]; then
        print_status "ERROR" "$service est $STATUS"
        echo "   Logs:"
        docker compose -f docker-compose.production.yml logs "$service" --tail 20 || true
    else
        print_status "WARN" "$service est $STATUS"
    fi
    echo ""
done

# 3. Vérifier la connexion PostgreSQL
echo "🐘 Vérification de PostgreSQL..."
if docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER:-atmr_user}" -d "${POSTGRES_DB:-atmr_db}" > /dev/null 2>&1; then
    print_status "OK" "PostgreSQL est accessible"
    
    # Vérifier les migrations
    echo "📋 Vérification des migrations de base de données..."
    CURRENT_REV=$(docker compose -f docker-compose.production.yml exec -T \
        -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI:-}" \
        -e DATABASE_URL="${DATABASE_URL:-}" \
        -e POSTGRES_USER="${POSTGRES_USER:-atmr_user}" \
        -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}" \
        -e POSTGRES_DB="${POSTGRES_DB:-atmr_db}" \
        -e POSTGRES_HOST="postgres" \
        -e POSTGRES_PORT="5432" \
        backend flask db current 2>/dev/null | grep -o '[a-f0-9]\{12\}' | head -1 || echo "unknown")
    
    if [ "$CURRENT_REV" != "unknown" ] && [ -n "$CURRENT_REV" ]; then
        print_status "OK" "Version de migration actuelle: $CURRENT_REV"
    else
        print_status "WARN" "Impossible de déterminer la version de migration"
    fi
    
    # Vérifier les heads
    HEADS=$(docker compose -f docker-compose.production.yml exec -T \
        -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI:-}" \
        -e DATABASE_URL="${DATABASE_URL:-}" \
        -e POSTGRES_USER="${POSTGRES_USER:-atmr_user}" \
        -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}" \
        -e POSTGRES_DB="${POSTGRES_DB:-atmr_db}" \
        -e POSTGRES_HOST="postgres" \
        -e POSTGRES_PORT="5432" \
        backend flask db heads 2>/dev/null | grep -o '[a-f0-9]\{12\}' || echo "")
    
    if [ -n "$HEADS" ]; then
        echo "   Heads disponibles: $HEADS"
    fi
else
    print_status "ERROR" "PostgreSQL n'est pas accessible"
fi
echo ""

# 4. Vérifier Redis
echo "🔴 Vérification de Redis..."
if docker compose -f docker-compose.production.yml exec -T redis redis-cli -a "${REDIS_PASSWORD:-}" ping > /dev/null 2>&1; then
    print_status "OK" "Redis est accessible et authentifié"
else
    print_status "ERROR" "Redis n'est pas accessible ou l'authentification a échoué"
fi
echo ""

# 5. Vérifier le backend (healthcheck)
echo "🌐 Vérification du backend API..."
if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
    print_status "OK" "Backend API répond sur /health"
    HEALTH_RESPONSE=$(curl -s http://localhost:5000/health 2>/dev/null || echo "{}")
    echo "   Réponse: $HEALTH_RESPONSE"
else
    print_status "ERROR" "Backend API ne répond pas sur /health"
    echo "   Logs backend (dernières 30 lignes):"
    docker compose -f docker-compose.production.yml logs backend --tail 30 || true
fi
echo ""

# 6. Vérifier les workers Celery
echo "⚙️  Vérification des workers Celery..."
if docker compose -f docker-compose.production.yml exec -T celery-worker celery -A celery_app.celery inspect ping > /dev/null 2>&1; then
    print_status "OK" "Celery worker répond"
    
    # Afficher les stats
    STATS=$(docker compose -f docker-compose.production.yml exec -T celery-worker celery -A celery_app.celery inspect stats 2>/dev/null | head -10 || echo "")
    if [ -n "$STATS" ]; then
        echo "   Stats:"
        echo "$STATS" | sed 's/^/      /'
    fi
else
    print_status "WARN" "Celery worker ne répond pas (peut être normal si aucun worker actif)"
fi
echo ""

# 7. Vérifier Flower (monitoring Celery)
echo "🌸 Vérification de Flower..."
if curl -f -s http://localhost:5555 > /dev/null 2>&1; then
    print_status "OK" "Flower est accessible sur http://localhost:5555"
else
    print_status "WARN" "Flower n'est pas accessible (optionnel)"
fi
echo ""

# 8. Vérifier les volumes et l'espace disque
echo "💾 Vérification de l'espace disque..."
df -h / | tail -1 | awk '{print "   Espace disque: " $4 " disponible sur " $2 " (utilisé: " $5 ")"}'
echo ""

# Vérifier les volumes Docker
echo "📦 Volumes Docker..."
docker volume ls | grep atmr || echo "   Aucun volume atmr trouvé"
echo ""

# 9. Vérifier les logs récents pour erreurs critiques
echo "📋 Vérification des logs récents (erreurs critiques)..."
ERROR_COUNT=0
for service in "${SERVICES[@]}"; do
    ERRORS=$(docker compose -f docker-compose.production.yml logs "$service" --tail 100 2>/dev/null | grep -i "error\|exception\|fatal\|critical" | wc -l || echo "0")
    if [ "$ERRORS" -gt 0 ]; then
        print_status "WARN" "$service: $ERRORS erreur(s) dans les 100 dernières lignes"
        ERROR_COUNT=$((ERROR_COUNT + ERRORS))
    fi
done

if [ "$ERROR_COUNT" -eq 0 ]; then
    print_status "OK" "Aucune erreur critique détectée dans les logs récents"
fi
echo ""

# 10. Résumé final
echo "=============================================="
echo "📊 Résumé de la vérification"
echo "=============================================="

# Compter les services running
RUNNING_COUNT=0
for service in "${SERVICES[@]}"; do
    STATUS=$(docker compose -f docker-compose.production.yml ps "$service" --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$STATUS" = "running" ]; then
        RUNNING_COUNT=$((RUNNING_COUNT + 1))
    fi
done

echo "   Services running: $RUNNING_COUNT/${#SERVICES[@]}"
echo "   Erreurs détectées: $ERROR_COUNT"

if [ "$RUNNING_COUNT" -eq "${#SERVICES[@]}" ] && [ "$ERROR_COUNT" -eq 0 ]; then
    print_status "OK" "Tous les services sont opérationnels"
    exit 0
elif [ "$RUNNING_COUNT" -ge 4 ]; then
    print_status "WARN" "La plupart des services sont opérationnels, mais certains problèmes ont été détectés"
    exit 1
else
    print_status "ERROR" "Plusieurs services ne sont pas opérationnels"
    exit 1
fi

