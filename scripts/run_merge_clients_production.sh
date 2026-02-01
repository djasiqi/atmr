#!/bin/bash
# Exécute la fusion des clients dupliqués Nicole Dettwiler sur la production
# Usage: ./scripts/run_merge_clients_production.sh

set -e

SERVER_HOST="${SERVER_HOST:-138.201.155.201}"
SERVER_USER="${SERVER_USER:-deploy}"
SERVER_PATH="${SERVER_PATH:-/srv/atmr}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT="$SCRIPT_DIR/merge_duplicate_clients_nicole_dettwiler.sh"

echo "🚀 Connexion à $SERVER_USER@$SERVER_HOST pour fusionner les clients dupliqués..."
echo ""

# Copier le script sur le serveur puis l'exécuter (interactif)
scp "$MERGE_SCRIPT" ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/merge_duplicate_clients_nicole_dettwiler.sh
ssh -t ${SERVER_USER}@${SERVER_HOST} "cd ${SERVER_PATH} && chmod +x merge_duplicate_clients_nicole_dettwiler.sh && ./merge_duplicate_clients_nicole_dettwiler.sh"
