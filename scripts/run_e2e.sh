#!/bin/bash
# =============================================================================
# Script one-shot pour lancer les tests E2E avec migrations automatiques
# Usage: ./scripts/run_e2e.sh [options]
#
# Options:
#   -v, --verbose     Mode verbeux (pytest -vv)
#   -k PATTERN        Filtrer les tests par pattern (ex: -k "test_e2e_flow")
#   --keep-services   Ne pas arrêter les services après les tests
#   --no-rebuild      Ne pas reconstruire l'image Docker
#   -h, --help        Afficher l'aide
#
# Exemples:
#   ./scripts/run_e2e.sh                         # Tous les tests E2E
#   ./scripts/run_e2e.sh -k institution_flow     # Tests institution uniquement
#   ./scripts/run_e2e.sh --keep-services         # Garder DB/Redis après tests
# =============================================================================

set -euo pipefail

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration par défaut
VERBOSE=""
PATTERN=""
KEEP_SERVICES=false
NO_REBUILD=false
COMPOSE_FILE="docker-compose.test.yml"
TIMEOUT_POSTGRES=60
TIMEOUT_REDIS=30

# Variables d'environnement pour les tests
export DATABASE_URL_TEST="postgresql://test:test@localhost:5433/atmr_test"
export REDIS_URL="redis://localhost:6380/0"
export FLASK_CONFIG="testing"

# Parsing des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-vv"
            shift
            ;;
        -k)
            PATTERN="-k $2"
            shift 2
            ;;
        --keep-services)
            KEEP_SERVICES=true
            shift
            ;;
        --no-rebuild)
            NO_REBUILD=true
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Option inconnue: $1"
            echo "Utilisez -h pour l'aide"
            exit 1
            ;;
    esac
done

# Fonctions utilitaires
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

cleanup() {
    if [ "$KEEP_SERVICES" = false ]; then
        info "🧹 Nettoyage des services de test..."
        docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
    else
        warn "⚠️  Services conservés (--keep-services). Pour arrêter: docker compose -f $COMPOSE_FILE down -v"
    fi
}

# Trap pour nettoyage en cas d'erreur
trap cleanup EXIT

# Vérification des prérequis
step "1/5 - Vérification des prérequis"

if ! command -v docker &> /dev/null; then
    error "❌ Docker n'est pas installé"
    exit 1
fi
info "✅ Docker disponible"

if ! docker compose version &> /dev/null; then
    error "❌ Docker Compose n'est pas disponible"
    exit 1
fi
info "✅ Docker Compose disponible"

if [ ! -f "$COMPOSE_FILE" ]; then
    error "❌ Fichier $COMPOSE_FILE non trouvé"
    exit 1
fi
info "✅ Fichier $COMPOSE_FILE trouvé"

# Arrêter les services existants
step "2/5 - Démarrage des services (postgres_test, redis_test)"

info "🛑 Arrêt des services existants..."
docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true

info "🚀 Démarrage de PostgreSQL et Redis..."
docker compose -f "$COMPOSE_FILE" up -d postgres_test redis_test

# Attendre que PostgreSQL soit prêt
info "⏳ Attente de PostgreSQL (max ${TIMEOUT_POSTGRES}s)..."
SECONDS=0
until docker compose -f "$COMPOSE_FILE" exec -T postgres_test pg_isready -U test -d atmr_test > /dev/null 2>&1; do
    if [ $SECONDS -ge $TIMEOUT_POSTGRES ]; then
        error "❌ PostgreSQL n'est pas prêt après ${TIMEOUT_POSTGRES}s"
        docker compose -f "$COMPOSE_FILE" logs postgres_test
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo ""
info "✅ PostgreSQL prêt (${SECONDS}s)"

# Attendre que Redis soit prêt
info "⏳ Attente de Redis (max ${TIMEOUT_REDIS}s)..."
SECONDS=0
until docker compose -f "$COMPOSE_FILE" exec -T redis_test redis-cli ping > /dev/null 2>&1; do
    if [ $SECONDS -ge $TIMEOUT_REDIS ]; then
        error "❌ Redis n'est pas prêt après ${TIMEOUT_REDIS}s"
        docker compose -f "$COMPOSE_FILE" logs redis_test
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo ""
info "✅ Redis prêt (${SECONDS}s)"

# Build de l'image backend si nécessaire
step "3/5 - Build de l'image backend (target: testing)"

if [ "$NO_REBUILD" = true ]; then
    warn "⚠️  Skip du build (--no-rebuild)"
else
    info "🔨 Construction de l'image backend..."
    docker compose -f "$COMPOSE_FILE" build backend_tests
    info "✅ Image construite"
fi

# Appliquer les migrations
step "4/5 - Application des migrations Alembic"

# ✅ FIX: DISABLE_EVENTLET=1 est OBLIGATOIRE pour que les migrations fonctionnent.
# eventlet.monkey_patch() interfère avec les transactions Alembic/psycopg.
# Voir docs/migrations.md pour plus de détails.
info "🔄 Exécution de 'flask db upgrade heads' (DISABLE_EVENTLET=1)..."
docker compose -f "$COMPOSE_FILE" run --rm \
    -e DATABASE_URL=postgresql://test:test@postgres_test:5432/atmr_test \
    -e FLASK_APP=wsgi:app \
    -e FLASK_CONFIG=testing \
    -e DISABLE_EVENTLET=1 \
    backend_tests \
    flask db upgrade heads

if [ $? -eq 0 ]; then
    info "✅ Migrations appliquées avec succès"
else
    error "❌ Échec des migrations"
    exit 1
fi

# Vérifier l'état des migrations
info "📋 Vérification de l'état des migrations..."
docker compose -f "$COMPOSE_FILE" run --rm \
    -e DATABASE_URL=postgresql://test:test@postgres_test:5432/atmr_test \
    -e FLASK_APP=wsgi:app \
    -e FLASK_CONFIG=testing \
    -e DISABLE_EVENTLET=1 \
    backend_tests \
    flask db current

# Lancer les tests E2E
step "5/5 - Exécution des tests E2E"

info "🧪 Lancement des tests E2E..."
info "   Fichier: tests/e2e/test_e2e_institution_flow.py"
[ -n "$PATTERN" ] && info "   Pattern: $PATTERN"
[ -n "$VERBOSE" ] && info "   Mode: verbeux"

# Construire la commande pytest
PYTEST_CMD="python -m pytest tests/e2e/test_e2e_institution_flow.py -v ${VERBOSE} ${PATTERN} --tb=short"

docker compose -f "$COMPOSE_FILE" run --rm \
    -e DATABASE_URL=postgresql://test:test@postgres_test:5432/atmr_test \
    -e REDIS_URL=redis://redis_test:6379/0 \
    -e FLASK_CONFIG=testing \
    -e SECRET_KEY=test-secret-key \
    -e JWT_SECRET_KEY=test-jwt-secret \
    -e APP_ENCRYPTION_KEY_B64=MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY= \
    -e SKIP_E2E_MIGRATIONS=1 \
    -e SOCKETIO_ASYNC_MODE=threading \
    -e SKIP_SOCKETIO=true \
    backend_tests \
    $PYTEST_CMD

TEST_EXIT_CODE=$?

# Résumé
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ TOUS LES TESTS E2E ONT RÉUSSI${NC}"
else
    echo -e "${RED}❌ CERTAINS TESTS ONT ÉCHOUÉ (exit code: $TEST_EXIT_CODE)${NC}"
fi
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit $TEST_EXIT_CODE
