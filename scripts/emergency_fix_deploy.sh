#!/bin/bash
# Script de réparation d'urgence du déploiement production
# À exécuter sur le serveur en cas d'échec du déploiement
#
# Usage: bash scripts/emergency_fix_deploy.sh [--skip-migrations]

set -e  # Arrêter en cas d'erreur
set -u  # Erreur sur variable non définie

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.production.yml"
BACKEND_SERVICE="backend"
POSTGRES_SERVICE="postgres"
REDIS_SERVICE="redis"
SKIP_MIGRATIONS=false

# Parser les arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-migrations)
            SKIP_MIGRATIONS=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Argument inconnu: $1${NC}"
            echo "Usage: $0 [--skip-migrations]"
            exit 1
            ;;
    esac
done

# Fonction d'affichage
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "$COMPOSE_FILE" ]; then
    log_error "Fichier $COMPOSE_FILE introuvable !"
    log_info "Assurez-vous d'être dans /srv/atmr"
    exit 1
fi

echo "=============================================="
echo "🚨 RÉPARATION D'URGENCE DU DÉPLOIEMENT"
echo "=============================================="
echo ""

# 1. Vérifier que les services sont en cours d'exécution
log_info "Vérification des services..."
if ! docker compose -f "$COMPOSE_FILE" ps | grep -q "$BACKEND_SERVICE"; then
    log_warning "Service backend non démarré, démarrage..."
    docker compose -f "$COMPOSE_FILE" up -d "$BACKEND_SERVICE"
    sleep 5
fi
log_success "Services vérifiés"
echo ""

# 2. Installer Flask-Limiter[redis]
log_info "Installation de Flask-Limiter[redis]..."
if docker compose -f "$COMPOSE_FILE" exec -T "$BACKEND_SERVICE" \
    pip install "Flask-Limiter[redis]>=3.0.0" > /dev/null 2>&1; then
    log_success "Flask-Limiter[redis] installé"
else
    log_error "Échec de l'installation de Flask-Limiter[redis]"
    exit 1
fi
echo ""

# 3. Vérifier l'installation
log_info "Vérification de l'installation..."
if docker compose -f "$COMPOSE_FILE" exec -T "$BACKEND_SERVICE" \
    python -c "import flask_limiter.storage; print('OK')" > /dev/null 2>&1; then
    log_success "flask_limiter.storage importé avec succès"
else
    log_error "Impossible d'importer flask_limiter.storage"
    exit 1
fi
echo ""

# 4. Diagnostic Alembic
log_info "Diagnostic des migrations Alembic..."
echo "----------------------------------------"
docker compose -f "$COMPOSE_FILE" exec -T "$BACKEND_SERVICE" bash -c "
    export FLASK_APP=backend.wsgi
    echo '=== Current Revision ==='
    flask db current 2>/dev/null || echo 'Aucune révision actuelle'
    echo ''
    echo '=== Available Heads ==='
    flask db heads 2>/dev/null || echo 'Aucun head disponible'
    echo ''
    echo '=== Recent History (last 15) ==='
    flask db history 2>/dev/null | head -30 || echo 'Aucun historique'
"
echo "----------------------------------------"
echo ""

# 5. Appliquer les migrations (si non skippé)
if [ "$SKIP_MIGRATIONS" = false ]; then
    log_info "Application des migrations..."
    
    # Tentative d'upgrade direct
    if docker compose -f "$COMPOSE_FILE" exec -T "$BACKEND_SERVICE" bash -c "
        export FLASK_APP=backend.wsgi
        flask db upgrade heads
    " 2>&1 | tee /tmp/migration_output.log; then
        log_success "Migrations appliquées avec succès"
    else
        log_warning "Échec de l'upgrade automatique"
        log_info "Vérification des conflits de merge..."
        
        # Vérifier si c'est un problème de merge
        if grep -q "overlaps" /tmp/migration_output.log; then
            log_warning "Conflit de merge détecté"
            echo ""
            echo "Options pour résoudre :"
            echo "  1. Appliquer manuellement les migrations parentes"
            echo "  2. Stamper les révisions (si le schéma existe déjà)"
            echo ""
            log_warning "Veuillez résoudre manuellement selon la procédure dans DEPLOY_FIX_PROCEDURE.md"
            exit 1
        else
            log_error "Erreur de migration non identifiée"
            cat /tmp/migration_output.log
            exit 1
        fi
    fi
else
    log_warning "Migrations skippées (--skip-migrations)"
fi
echo ""

# 6. Vérification de la santé de l'application
log_info "Vérification de la santé de l'application..."

# Attendre que l'application soit prête
sleep 3

# Test du healthcheck
if curl -s -f http://localhost:5000/health > /dev/null 2>&1; then
    log_success "Healthcheck OK"
    curl -s http://localhost:5000/health | python -m json.tool 2>/dev/null || true
else
    log_warning "Healthcheck échoué (l'application démarre peut-être encore)"
fi
echo ""

# 7. Vérifier les logs récents
log_info "Logs récents du backend (dernières 20 lignes)..."
echo "----------------------------------------"
docker compose -f "$COMPOSE_FILE" logs --tail 20 "$BACKEND_SERVICE"
echo "----------------------------------------"
echo ""

# 8. Résumé final
echo "=============================================="
echo "📋 RÉSUMÉ"
echo "=============================================="
log_success "Flask-Limiter[redis] installé"
if [ "$SKIP_MIGRATIONS" = false ]; then
    log_success "Migrations appliquées (ou vérifiées)"
fi
log_warning "Correction temporaire ! Redéployer avec l'image corrigée"
echo ""
echo "Prochaines étapes :"
echo "  1. ✅ Vérifier l'application : curl http://localhost:5000/health"
echo "  2. ✅ Vérifier les logs : docker compose -f $COMPOSE_FILE logs backend"
echo "  3. ⚠️  Redéployer avec l'image corrigée (voir DEPLOY_FIX_PROCEDURE.md)"
echo ""
echo "Pour plus de détails, consulter : DEPLOY_FIX_PROCEDURE.md"
echo "=============================================="
