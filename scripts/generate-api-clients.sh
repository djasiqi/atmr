#!/bin/bash
# scripts/generate-api-clients.sh
# ✅ Tâche 2: Script pour générer les clients TypeScript depuis la spec OpenAPI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPEC_FILE="$PROJECT_ROOT/backend/docs/openapi.json"
FRONTEND_OUTPUT="$PROJECT_ROOT/frontend/src/generated/api"
MOBILE_OUTPUT="$PROJECT_ROOT/mobile/operations-app/src/generated/api"

# Vérifier que la spec existe
if [ ! -f "$SPEC_FILE" ]; then
    echo "❌ Erreur: $SPEC_FILE introuvable"
    echo "   Exécutez d'abord: docker-compose run --rm api python scripts/generate_openapi.py --output /app/docs/openapi.json"
    exit 1
fi

# Vérifier que openapi-generator est installé
if ! command -v openapi-generator-cli &> /dev/null; then
    echo "⚠️  openapi-generator-cli non trouvé. Installation..."
    npm install -g @openapitools/openapi-generator-cli
fi

echo "📦 Génération des clients TypeScript depuis $SPEC_FILE..."

# Générer le client pour le frontend web
echo "🔧 Génération client frontend web..."
mkdir -p "$FRONTEND_OUTPUT"
openapi-generator-cli generate \
    -i "$SPEC_FILE" \
    -g typescript-axios \
    -o "$FRONTEND_OUTPUT" \
    --additional-properties=supportsES6=true,withInterfaces=true,typescriptThreePlus=true

# Générer le client pour le mobile
echo "🔧 Génération client mobile..."
mkdir -p "$MOBILE_OUTPUT"
openapi-generator-cli generate \
    -i "$SPEC_FILE" \
    -g typescript-axios \
    -o "$MOBILE_OUTPUT" \
    --additional-properties=supportsES6=true,withInterfaces=true,typescriptThreePlus=true

echo "✅ Clients TypeScript générés:"
echo "   - Frontend: $FRONTEND_OUTPUT"
echo "   - Mobile: $MOBILE_OUTPUT"

