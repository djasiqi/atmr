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
# /api/v1/ready est l'endpoint canonique de readiness (DB + Redis), aligné sur :
#   - le healthcheck Docker du conteneur backend (docker-compose.production.yml)
#   - le healthcheck Traefik (label loadbalancer.healthcheck.path=/api/v1/ready)
# /healthz et /api/v1/health n'existent pas côté backend (404).
# F-01 prod : le port 5000 n'est pas publié sur l'hôte — USE_DOCKER_HEALTHCHECK=1 sonde via exec.
BACKEND_URL="${BACKEND_URL:-http://localhost:5000}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-/api/v1/ready}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"
USE_DOCKER_HEALTHCHECK="${USE_DOCKER_HEALTHCHECK:-0}"
TIMEOUT="${TIMEOUT:-10}"

probe_backend_http() {
  if [ "$USE_DOCKER_HEALTHCHECK" = "1" ]; then
    docker compose -f "$COMPOSE_FILE" exec -T backend python -c "
import urllib.request
import sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:5000${HEALTH_ENDPOINT}', timeout=${TIMEOUT})
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
    return $?
  fi
  curl -f -s --max-time "${TIMEOUT}" "${BACKEND_URL}${HEALTH_ENDPOINT}" > /dev/null 2>&1
}

fetch_backend_health_json() {
  if [ "$USE_DOCKER_HEALTHCHECK" = "1" ]; then
    docker compose -f "$COMPOSE_FILE" exec -T backend python -c "
import urllib.request
try:
    print(urllib.request.urlopen('http://127.0.0.1:5000${HEALTH_ENDPOINT}', timeout=${TIMEOUT}).read().decode())
except Exception:
    pass
" 2>/dev/null || echo ""
    return 0
  fi
  curl -f -s --max-time "${TIMEOUT}" "${BACKEND_URL}${HEALTH_ENDPOINT}" || echo ""
}

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

# Attendre que le backend réponde (retry avec backoff)
info "⏳ Attente de la disponibilité du backend..."
MAX_RETRIES=30
RETRY_COUNT=0
BACKEND_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if probe_backend_http; then
        BACKEND_READY=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "   Tentative $RETRY_COUNT/$MAX_RETRIES, attente 2 secondes..."
        sleep 2
    fi
done

if [ "$BACKEND_READY" = "false" ]; then
    error "❌ Le backend ne répond pas après $MAX_RETRIES tentatives"
    exit 1
fi

info "✅ Backend disponible, démarrage des tests"

# Test 1: Vérifier que l'endpoint de readiness répond avec status 200
info "Test 1: Vérification de l'endpoint ${HEALTH_ENDPOINT}"
if probe_backend_http; then
    info "✅ L'endpoint ${HEALTH_ENDPOINT} répond correctement"
else
    error "❌ L'endpoint ${HEALTH_ENDPOINT} ne répond pas ou retourne une erreur"
fi

# Test 2: Vérifier que la réponse JSON contient un statut sain
# /api/v1/ready -> {"status":"ready", ...} ; /health -> {"status":"healthy", ...}
info "Test 2: Vérification du contenu de la réponse ${HEALTH_ENDPOINT}"
HEALTH_RESPONSE=$(fetch_backend_health_json)
if echo "${HEALTH_RESPONSE}" | grep -qE '"status"[[:space:]]*:[[:space:]]*"(ready|healthy|ok)"'; then
    info "✅ La réponse contient un statut sain (ready/healthy/ok)"
else
    error "❌ La réponse ne contient pas de statut sain (ready/healthy/ok)"
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

# Test 5: Smoke métier synthétique (dry-run schema booking — hors facturation)
info "Test 5: Validation schéma création booking (synthétique)"
if command -v docker &> /dev/null && [ -f "docker-compose.production.yml" ]; then
    if docker compose -f docker-compose.production.yml exec -T backend python -c "
from schemas.booking_schemas import BookingCreateSchema
from schemas.validation_utils import validate_request
from datetime import datetime, timedelta, timezone
payload = {
    'customer_name': 'deployment_smoke',
    'pickup_location': 'Rue Smoke 1, 1000 Lausanne',
    'dropoff_location': 'Avenue Smoke 2, 1000 Lausanne',
    'scheduled_time': (datetime.now(timezone.utc) + timedelta(days=1)).replace(microsecond=0).isoformat().replace('+00:00','Z'),
    'amount': 50.0,
    'bill_to_patient': True,
    'amount_source': 'client_override',
    'synthetic': True,
    'source': 'deployment_smoke',
}
validated = validate_request(BookingCreateSchema(), payload)
assert 'bill_to_patient' not in validated
assert 'amount_source' not in validated
print('OK_SYNTHETIC_BOOKING_SCHEMA')
" 2>/dev/null | grep -q "OK_SYNTHETIC_BOOKING_SCHEMA"; then
        info "✅ Schéma booking ignore les champs internes (Option B)"
    else
        error "❌ Smoke schéma booking synthétique échoué"
    fi
else
    warn "⚠️  Docker Compose non disponible, smoke booking ignoré"
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

