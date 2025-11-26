#!/bin/bash

# Script de validation du secret DATABASE_URL pour GitHub Actions
# Usage: ./scripts/validate_database_url.sh

set -euo pipefail

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Validation du format DATABASE_URL"
echo ""

# Récupérer les valeurs depuis les variables d'environnement ou les arguments
POSTGRES_USER="${1:-${POSTGRES_USER:-}}"
POSTGRES_PASSWORD="${2:-${POSTGRES_PASSWORD:-}}"
POSTGRES_DB="${3:-${POSTGRES_DB:-}}"
POSTGRES_HOST="${4:-${POSTGRES_HOST:-postgres}}"
POSTGRES_PORT="${5:-${POSTGRES_PORT:-5432}}"
DATABASE_URL="${DATABASE_URL:-}"

# Si DATABASE_URL est fourni directement, le valider
if [ -n "${DATABASE_URL}" ]; then
    echo "📋 Validation de DATABASE_URL fourni..."
    echo "  DATABASE_URL: ${DATABASE_URL:0:30}..." # Afficher les 30 premiers caractères
    
    # Vérifications de base
    if [[ ! "$DATABASE_URL" =~ ^postgresql:// ]]; then
        echo -e "${RED}❌ ERREUR: DATABASE_URL doit commencer par 'postgresql://'${NC}"
        exit 1
    fi
    
    if [[ ! "$DATABASE_URL" =~ @ ]]; then
        echo -e "${RED}❌ ERREUR: Format utilisateur:mot_de_passe@host manquant${NC}"
        exit 1
    fi
    
    if [[ ! "$DATABASE_URL" =~ /[^/]+$ ]]; then
        echo -e "${RED}❌ ERREUR: Nom de base de données manquant${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Format DATABASE_URL valide${NC}"
    echo ""
    echo "📝 DATABASE_URL recommandé pour GitHub Secrets:"
    echo "   ${DATABASE_URL}"
    exit 0
fi

# Sinon, construire depuis POSTGRES_*
if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DB" ]; then
    echo -e "${YELLOW}⚠️  Variables POSTGRES_* manquantes${NC}"
    echo ""
    echo "Usage:"
    echo "  ./scripts/validate_database_url.sh [POSTGRES_USER] [POSTGRES_PASSWORD] [POSTGRES_DB] [POSTGRES_HOST] [POSTGRES_PORT]"
    echo ""
    echo "Ou définissez les variables d'environnement:"
    echo "  export POSTGRES_USER=..."
    echo "  export POSTGRES_PASSWORD=..."
    echo "  export POSTGRES_DB=..."
    echo "  ./scripts/validate_database_url.sh"
    exit 1
fi

echo "📋 Construction de DATABASE_URL depuis POSTGRES_*..."
echo "  POSTGRES_USER: ${POSTGRES_USER}"
echo "  POSTGRES_DB: ${POSTGRES_DB}"
echo "  POSTGRES_HOST: ${POSTGRES_HOST}"
echo "  POSTGRES_PORT: ${POSTGRES_PORT}"
echo "  POSTGRES_PASSWORD: [masqué]"
echo ""

# Construire DATABASE_URL (Python pour l'échappement URL correct)
DATABASE_URL=$(python3 -c "
from urllib.parse import quote_plus
import sys

user = sys.argv[1]
password = sys.argv[2]
host = sys.argv[3]
port = sys.argv[4]
db = sys.argv[5]

password_escaped = quote_plus(password)
db_url = f'postgresql://{user}:{password_escaped}@{host}:{port}/{db}'
print(db_url)
" "$POSTGRES_USER" "$POSTGRES_PASSWORD" "$POSTGRES_HOST" "$POSTGRES_PORT" "$POSTGRES_DB")

echo -e "${GREEN}✅ DATABASE_URL construit avec succès${NC}"
echo ""
echo "📝 DATABASE_URL pour GitHub Secrets:"
echo "   ${DATABASE_URL}"
echo ""
echo "📋 Format décomposé:"
echo "   postgresql://[user]:[password]@[host]:[port]/[database]"
echo "   postgresql://${POSTGRES_USER}:[masqué]@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
echo ""
echo "💡 Copiez la valeur ci-dessus dans GitHub Secrets → DATABASE_URL"

