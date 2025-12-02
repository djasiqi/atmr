#!/bin/bash

# Script de mise à jour des images Docker en développement
# Usage: ./scripts/update-docker-dev.sh [--no-cache]

set -e  # Arrêter en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions pour afficher les messages
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
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Vérifier que docker-compose est installé
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    error "docker-compose n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Détecter la commande docker-compose (v1 ou v2)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "docker-compose.yml" ]; then
    error "docker-compose.yml introuvable. Assurez-vous d'être dans le répertoire racine du projet."
    exit 1
fi

# Vérifier les arguments
NO_CACHE=""
if [ "$1" == "--no-cache" ]; then
    NO_CACHE="--no-cache"
    info "Reconstruction complète sans cache activée"
fi

echo ""
echo "=========================================="
echo "  Mise à jour des Images Docker - DEV"
echo "=========================================="
echo ""

# Étape 1 : Vérifier l'état actuel
step "1/6 : Vérification de l'état actuel"
if $DOCKER_COMPOSE ps | grep -q "Up"; then
    info "Des conteneurs sont en cours d'exécution"
    $DOCKER_COMPOSE ps
    echo ""
    read -p "Voulez-vous arrêter les conteneurs avant de continuer ? (O/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        step "Arrêt des conteneurs..."
        $DOCKER_COMPOSE down
        info "Conteneurs arrêtés"
    else
        warn "Les conteneurs continueront de tourner pendant la reconstruction"
    fi
else
    info "Aucun conteneur en cours d'exécution"
fi

# Étape 2 : Nettoyer les images obsolètes (optionnel)
step "2/6 : Nettoyage des images obsolètes"
read -p "Voulez-vous nettoyer les images Docker non utilisées ? (o/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[OoYy]$ ]]; then
    info "Nettoyage des images non utilisées..."
    docker image prune -f
    info "Nettoyage terminé"
else
    info "Nettoyage ignoré"
fi

# Étape 3 : Reconstruire les images
step "3/6 : Reconstruction des images Docker"
info "Reconstruction des services backend..."

if [ -n "$NO_CACHE" ]; then
    info "Reconstruction complète sans cache..."
    $DOCKER_COMPOSE build $NO_CACHE api celery-worker celery-beat flower
else
    info "Reconstruction avec cache..."
    $DOCKER_COMPOSE build api celery-worker celery-beat flower
fi

if [ $? -eq 0 ]; then
    info "✅ Images reconstruites avec succès"
else
    error "❌ Erreur lors de la reconstruction des images"
    exit 1
fi

# Étape 4 : Démarrer les services
step "4/6 : Démarrage des services"
info "Démarrage des services Docker..."
$DOCKER_COMPOSE up -d

if [ $? -eq 0 ]; then
    info "✅ Services démarrés"
else
    error "❌ Erreur lors du démarrage des services"
    exit 1
fi

# Étape 5 : Attendre que les services soient prêts
step "5/6 : Attente du démarrage des services"
info "Attente de 30 secondes pour que les services démarrent..."
sleep 30

# Vérifier l'état des services
info "État des services :"
$DOCKER_COMPOSE ps

# Étape 6 : Vérifications de santé
step "6/6 : Vérifications de santé"

# Vérifier l'API
info "Vérification de l'API..."
if command -v curl &> /dev/null; then
    if curl -f -s http://localhost:5000/health > /dev/null; then
        info "✅ API répond correctement"
    else
        warn "⚠️  L'API ne répond pas correctement. Vérifiez les logs :"
        $DOCKER_COMPOSE logs api | tail -20
    fi
else
    warn "curl n'est pas installé, impossible de vérifier l'API"
fi

# Vérifier les logs pour les erreurs
info "Vérification des logs pour les erreurs..."
ERRORS_FOUND=false

if $DOCKER_COMPOSE logs api 2>&1 | grep -i "error\|exception\|traceback" | tail -5; then
    warn "⚠️  Erreurs détectées dans les logs de l'API"
    ERRORS_FOUND=true
fi

if $DOCKER_COMPOSE logs celery-worker 2>&1 | grep -i "error\|exception\|traceback" | tail -5; then
    warn "⚠️  Erreurs détectées dans les logs du worker Celery"
    ERRORS_FOUND=true
fi

if [ "$ERRORS_FOUND" = false ]; then
    info "✅ Aucune erreur critique détectée dans les logs"
fi

# Résumé final
echo ""
echo "=========================================="
echo "  ✅ Mise à jour terminée"
echo "=========================================="
echo ""
info "Commandes utiles :"
echo "  - Voir les logs : $DOCKER_COMPOSE logs -f [service]"
echo "  - Voir l'état : $DOCKER_COMPOSE ps"
echo "  - Arrêter : $DOCKER_COMPOSE down"
echo "  - Redémarrer : $DOCKER_COMPOSE restart [service]"
echo ""
info "Services disponibles :"
echo "  - API : http://localhost:5000"
echo "  - Flower : http://localhost:5555"
echo "  - Prometheus : http://localhost:9090"
echo "  - Grafana : http://localhost:3001"
echo ""

