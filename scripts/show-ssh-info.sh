#!/usr/bin/env bash
# Script pour afficher les informations SSH de connexion au serveur
# Usage: ./scripts/show-ssh-info.sh

set -euo pipefail

echo "🔍 Informations SSH du serveur de production"
echo "=============================================="
echo ""

# Vérifier si le fichier .env.secrets existe
if [ -f ".env.secrets" ]; then
    echo "📝 Lecture depuis .env.secrets..."
    source .env.secrets
    
    echo "🌐 SSH_HOST: ${SSH_HOST:-'Non défini'}"
    echo "👤 SSH_USER: ${SSH_USER:-'Non défini'}"
    echo "🔌 SSH_PORT: ${SSH_PORT:-'22'}"
    echo ""
    echo "💡 Commande de connexion :"
    echo "   ssh ${SSH_USER:-user}@${SSH_HOST:-host} -p ${SSH_PORT:-22}"
    echo ""
elif [ -f "backend/.env.production" ]; then
    echo "📝 Lecture depuis backend/.env.production..."
    source backend/.env.production
    
    echo "🌐 SSH_HOST: ${SSH_HOST:-'Non défini'}"
    echo "👤 SSH_USER: ${SSH_USER:-'Non défini'}"
    echo "🔌 SSH_PORT: ${SSH_PORT:-'22'}"
    echo ""
    echo "💡 Commande de connexion :"
    echo "   ssh ${SSH_USER:-user}@${SSH_HOST:-host} -p ${SSH_PORT:-22}"
    echo ""
else
    echo "❌ Aucun fichier de configuration trouvé."
    echo ""
    echo "📍 Vos identifiants SSH sont dans :"
    echo "   1. GitHub Secrets : https://github.com/djasiqi/atmr/settings/secrets/actions"
    echo "   2. Votre fichier .env.secrets local (si vous l'avez créé)"
    echo ""
    echo "🔑 Secrets à vérifier :"
    echo "   - SSH_HOST  (adresse IP ou domaine du serveur)"
    echo "   - SSH_USER  (nom d'utilisateur SSH)"
    echo "   - SSH_PORT  (port SSH, généralement 22)"
    echo "   - SSH_KEY   (clé privée SSH)"
    echo ""
fi

echo "=============================================="
echo ""
echo "🔧 Commandes utiles une fois connecté :"
echo ""
echo "   # Voir les logs backend"
echo "   docker logs -f --tail 100 atmr-backend"
echo ""
echo "   # Voir tous les services"
echo "   cd /srv/atmr && docker compose -f docker-compose.production.yml ps"
echo ""
echo "   # Voir les erreurs récentes"
echo "   docker logs atmr-backend --since 10m 2>&1 | grep -iE '(error|exception|traceback)'"
echo ""
