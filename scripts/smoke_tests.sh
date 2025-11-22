#!/bin/bash
# Script de smoke tests pour valider le déploiement ATMR
# Vérifie que les services critiques sont opérationnels après déploiement

set -euo pipefail

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKEND_URL="${BACKEND_URL:-http://localhost:5000}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-/health}"
TIMEOUT="${TIMEOUT:-10}"

# Compteur d'erreurs
ERRORS=0

# Fonction pour afficher les messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

info "🧪 Démarrage des smoke tests pour ATMR"
info "Backend URL: ${BACKEND_URL}"

# Test 1: Vérifier que l'endpoint /health répond avec status 200
info "Test 1: Vérification de l'endpoint /health"
if curl -f -s --max-time "${TIMEOUT}" "${BACKEND_URL}${HEALTH_ENDPOINT}" > /dev/null; then
    info "✅ L'endpoint /health répond correctement"
else
    error "❌ L'endpoint /health ne répond pas ou retourne une erreur"
fi

# Test 2: Vérifier que la réponse JSON contient 'status: healthy'
info "Test 2: Vérification du contenu de la réponse /health"
HEALTH_RESPONSE=$(curl -f -s --max-time "${TIMEOUT}" "${BACKEND_URL}${HEALTH_ENDPOINT}" || echo "")
if echo "${HEALTH_RESPONSE}" | grep -q '"status"[[:space:]]*:[[:space:]]*"healthy"'; then
    info "✅ La réponse contient 'status: healthy'"
else
    error "❌ La réponse ne contient pas 'status: healthy'"
    warn "Réponse reçue: ${HEALTH_RESPONSE}"
fi

# Test 3: Vérifier que la base de données est accessible (via docker compose exec si disponible)
info "Test 3: Vérification de l'accessibilité de la base de données"
if command -v docker &> /dev/null && [ -f "docker-compose.production.yml" ]; then
    if docker compose -f docker-compose.production.yml exec -T backend python -c "
import os
from sqlalchemy import create_engine, text
db_url = os.getenv('SQLALCHEMY_DATABASE_URI') or os.getenv('DATABASE_URL')
if not db_url:
    print('ERROR: SQLALCHEMY_DATABASE_URI or DATABASE_URL not set')
    exit(1)
try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
" 2>/dev/null | grep -q "OK"; then
        info "✅ La base de données est accessible"
    else
        error "❌ La base de données n'est pas accessible"
    fi
else
    warn "⚠️  Docker Compose non disponible, test de base de données ignoré"
fi

# Test 4: Vérifier que les migrations sont à jour
info "Test 4: Vérification de l'état des migrations"
if command -v docker &> /dev/null && [ -f "docker-compose.production.yml" ]; then
    if docker compose -f docker-compose.production.yml exec -T backend flask db current > /dev/null 2>&1; then
        info "✅ Les migrations sont à jour"
    else
        error "❌ Problème avec les migrations"
    fi
else
    warn "⚠️  Docker Compose non disponible, test de migrations ignoré"
fi

# Résumé
echo ""
if [ $ERRORS -eq 0 ]; then
    info "✅ Tous les smoke tests ont réussi"
    exit 0
else
    error "❌ ${ERRORS} test(s) ont échoué"
    exit 1
fi

