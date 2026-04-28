#!/bin/bash

# Script de déploiement sur le serveur de production
# Utilise des variables d'environnement pour les informations sensibles

set -e  # Arrêter en cas d'erreur

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

# Configuration (obligatoire : SERVER_HOST — ne pas d’IP/hostname en dur, voir docs/deployment-ssh.md)
: "${SERVER_HOST:?Définir SERVER_HOST (adresse du serveur). Voir docs/deployment-ssh.md.}"
SERVER_USER="${SERVER_USER:-deploy}"
SERVER_PATH="${SERVER_PATH:-/home/deploy/atmr}"

# Utiliser docker-compose.production.yml par défaut, ou docker-compose.yml si spécifié
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"

info "Configuration du déploiement :"
echo "  Serveur    : ${SERVER_USER}@${SERVER_HOST}"
echo "  Chemin     : ${SERVER_PATH}"
echo "  Compose    : ${COMPOSE_FILE}"
echo ""

# Demander confirmation
warn "Ce script va se connecter au serveur et :"
echo "  1. Récupérer les dernières modifications (git pull)"
echo "  2. Reconstruire les images Docker"
echo "  3. Redémarrer les services"
echo ""
read -p "Voulez-vous continuer ? (o/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
    info "Déploiement annulé."
    exit 0
fi

# Commande complète à exécuter sur le serveur
info "Connexion au serveur ${SERVER_HOST} et déploiement..."
ssh ${SERVER_USER}@${SERVER_HOST} << EOF
    set -e
    cd ${SERVER_PATH}
    
    echo "📥 Récupération des dernières modifications..."
    git pull origin main
    
    echo "🔨 Reconstruction des images Docker..."
    docker-compose -f ${COMPOSE_FILE} build --no-cache
    
    echo "🚀 Redémarrage des services..."
    docker-compose -f ${COMPOSE_FILE} up -d --force-recreate
    
    echo "✅ Vérification de l'état des services..."
    docker-compose -f ${COMPOSE_FILE} ps
    
    echo ""
    echo "✅ Déploiement terminé !"
EOF

if [ $? -eq 0 ]; then
    info "Déploiement réussi !"
else
    error "Erreur lors du déploiement"
    exit 1
fi
