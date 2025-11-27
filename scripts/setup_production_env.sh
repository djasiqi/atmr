#!/bin/bash
# Script pour configurer les variables d'environnement de production
# Usage: source scripts/setup_production_env.sh
# OU: ./scripts/setup_production_env.sh (pour vérifier seulement)

set -o errexit -o nounset -o pipefail

# Répertoire de travail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.production"

echo "🔧 Configuration des variables d'environnement de production"
echo "============================================================"
echo ""

# Vérifier si le fichier .env.production existe
if [ ! -f "$ENV_FILE" ]; then
    echo "⚠️  Fichier .env.production non trouvé: $ENV_FILE"
    echo ""
    echo "💡 Créez le fichier .env.production avec les variables suivantes:"
    echo ""
    cat << 'EOF'
# Variables de production (à remplir avec les valeurs réelles)
POSTGRES_USER=atmr
POSTGRES_PASSWORD=your_password_here
POSTGRES_DB=atmr
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

REDIS_PASSWORD=your_redis_password_here

SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
APP_ENCRYPTION_KEY_B64=your_encryption_key_base64_here

DOCKER_IMAGE=docker.io/djasiqi/atmr-backend
DOCKER_TAG=latest

# Optionnel
MAIL_PASSWORD=your_mail_password_here
SENTRY_DSN=your_sentry_dsn_here
EOF
    echo ""
    echo "❌ Impossible de continuer sans .env.production"
    exit 1
fi

# Charger les variables depuis .env.production
echo "📁 Chargement des variables depuis: $ENV_FILE"
set -a  # Auto-export toutes les variables
source "$ENV_FILE"
set +a

# Vérifier les variables critiques
echo ""
echo "🔍 Vérification des variables critiques..."
MISSING_VARS=()

if [ -z "${POSTGRES_USER:-}" ]; then MISSING_VARS+=("POSTGRES_USER"); fi
if [ -z "${POSTGRES_PASSWORD:-}" ]; then MISSING_VARS+=("POSTGRES_PASSWORD"); fi
if [ -z "${POSTGRES_DB:-}" ]; then MISSING_VARS+=("POSTGRES_DB"); fi
if [ -z "${REDIS_PASSWORD:-}" ]; then MISSING_VARS+=("REDIS_PASSWORD"); fi
if [ -z "${SECRET_KEY:-}" ]; then MISSING_VARS+=("SECRET_KEY"); fi
if [ -z "${JWT_SECRET_KEY:-}" ]; then MISSING_VARS+=("JWT_SECRET_KEY"); fi
if [ -z "${APP_ENCRYPTION_KEY_B64:-}" ]; then MISSING_VARS+=("APP_ENCRYPTION_KEY_B64"); fi
if [ -z "${DOCKER_IMAGE:-}" ]; then MISSING_VARS+=("DOCKER_IMAGE"); fi
if [ -z "${DOCKER_TAG:-}" ]; then MISSING_VARS+=("DOCKER_TAG"); fi

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "❌ Variables manquantes dans .env.production:"
    printf "   - %s\n" "${MISSING_VARS[@]}"
    exit 1
fi

echo "✅ Toutes les variables critiques sont définies"
echo ""

# Afficher un résumé (sans les valeurs sensibles)
echo "📊 Résumé des variables:"
echo "   POSTGRES_USER: ${POSTGRES_USER}"
echo "   POSTGRES_DB: ${POSTGRES_DB}"
echo "   POSTGRES_HOST: ${POSTGRES_HOST:-postgres}"
echo "   POSTGRES_PORT: ${POSTGRES_PORT:-5432}"
echo "   REDIS_PASSWORD: ${REDIS_PASSWORD:+***défini***}"
echo "   SECRET_KEY: ${SECRET_KEY:+***défini***}"
echo "   JWT_SECRET_KEY: ${JWT_SECRET_KEY:+***défini***}"
echo "   APP_ENCRYPTION_KEY_B64: ${APP_ENCRYPTION_KEY_B64:+***défini***}"
echo "   DOCKER_IMAGE: ${DOCKER_IMAGE}"
echo "   DOCKER_TAG: ${DOCKER_TAG}"
echo ""

# Si le script est sourcé (source setup_production_env.sh), les variables sont exportées
# Si le script est exécuté (./setup_production_env.sh), on affiche seulement
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "💡 Pour exporter les variables dans votre shell, utilisez:"
    echo "   source $0"
    echo ""
    echo "💡 Ou ajoutez cette ligne à votre ~/.bashrc:"
    echo "   source $PROJECT_ROOT/scripts/setup_production_env.sh"
else
    echo "✅ Variables d'environnement exportées dans le shell actuel"
    echo ""
    echo "💡 Vous pouvez maintenant exécuter:"
    echo "   cd $PROJECT_ROOT"
    echo "   docker compose -f docker-compose.production.yml up -d"
fi

