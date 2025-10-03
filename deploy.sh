#!/bin/bash

# Script de déploiement automatique pour ATMR
# Ce script arrête, reconstruit et redémarre tous les services Docker

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "  Déploiement ATMR - Docker Compose"
echo "=========================================="
echo ""

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier que docker-compose est installé
if ! command -v docker-compose &> /dev/null; then
    error "docker-compose n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "docker-compose.yml" ]; then
    error "docker-compose.yml introuvable. Assurez-vous d'être dans le répertoire racine du projet."
    exit 1
fi

# Demander confirmation
echo ""
warn "Ce script va :"
echo "  1. Arrêter tous les conteneurs"
echo "  2. Reconstruire les images"
echo "  3. Redémarrer tous les services"
echo ""
read -p "Voulez-vous continuer ? (o/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
    info "Déploiement annulé."
    exit 0
fi

# Étape 1 : Arrêter les conteneurs
info "Arrêt des conteneurs existants..."
docker-compose down || warn "Aucun conteneur à arrêter"

# Étape 2 : Reconstruire les images
info "Reconstruction des images Docker..."
docker-compose build --no-cache

# Étape 3 : Démarrer les services
info "Démarrage des services..."
docker-compose up -d

# Attendre que les services démarrent
info "Attente du démarrage des services (30 secondes)..."
sleep 30

# Étape 4 : Vérifier l'état des services
info "Vérification de l'état des services..."
docker-compose ps

# Étape 5 : Afficher les logs récents
echo ""
info "Logs récents de l'API :"
docker-compose logs --tail=20 api

echo ""
info "Logs récents de Celery Worker :"
docker-compose logs --tail=20 celery-worker

# Étape 6 : Vérifier la santé de l'API
echo ""
info "Vérification de la santé de l'API..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    info "✓ API accessible sur http://localhost:5000"
else
    warn "✗ API non accessible. Vérifiez les logs : docker-compose logs api"
fi

# Étape 7 : Vérifier Flower
info "Vérification de Flower..."
if curl -s http://localhost:5555 > /dev/null 2>&1; then
    info "✓ Flower accessible sur http://localhost:5555"
else
    warn "✗ Flower non accessible. Vérifiez les logs : docker-compose logs flower"
fi

# Résumé
echo ""
echo "=========================================="
info "Déploiement terminé !"
echo "=========================================="
echo ""
echo "Services disponibles :"
echo "  - API Flask      : http://localhost:5000"
echo "  - Flower (Celery): http://localhost:5555"
echo ""
echo "Commandes utiles :"
echo "  - Voir les logs        : docker-compose logs -f"
echo "  - Arrêter les services : docker-compose down"
echo "  - Redémarrer un service: docker-compose restart <service>"
echo "  - État des services    : docker-compose ps"
echo ""